import argparse, os, copy, random, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

from tqdm import *
import supcon

import dataset, utils, losses, net
from net.resnet import *
from scipy.optimize import linear_sum_assignment


def generate_dataset(dataset, index, index_target=None, target=None):
    dataset_ = copy.deepcopy(dataset)

    if target is not None:
        for i, v in enumerate(index_target):
            dataset_.ys[v] = target[i]

    for i, v in enumerate(index):
        j = v - i
        dataset_.I.pop(j)
        dataset_.ys.pop(j)
        dataset_.im_paths.pop(j)
    return dataset_


def merge_dataset(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    # if len(dataset_n.classes) > len(dataset_.classes):
    #     dataset_.classes = dataset_n.classes
    dataset_.I.extend(dataset_n.I)
    dataset_.im_paths.extend(dataset_n.im_paths)
    dataset_.ys.extend(dataset_n.ys)

    return dataset_

def find_reseverve_vectors_all():
    points = torch.randn(256, 256).cuda()
    points = normalize(points)
    points = torch.nn.Parameter(points)

    opt = torch.optim.SGD([points], lr=1)
        
    best_angle = 0
    tqdm_gen = tqdm(range(2500))

    for _ in tqdm_gen:
        
        sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
        l = torch.log(torch.exp(sim/1.0).sum(axis = 1)).sum() / points.shape[0]
            
        l.backward()
        opt.step()
        points.data = normalize(points.data)

        curr_angle, curr_angle_close = compute_angles(points.detach())
        if curr_angle > best_angle: 
            best_angle = curr_angle

        tqdm_gen.set_description(f"Loss = {l:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

    
    return points.data

def compute_angles(vectors):
    proto = vectors.cpu().numpy()
    dot = np.matmul(proto, proto.T)
    dot = dot.clip(min=0, max=1)
    theta = np.arccos(dot)
    np.fill_diagonal(theta, np.nan)
    theta = theta[~np.isnan(theta)].reshape(theta.shape[0], theta.shape[1] - 1)
        
    avg_angle_close = theta.min(axis = 1).mean()
    avg_angle = theta.mean()

    return np.rad2deg(avg_angle), np.rad2deg(avg_angle_close)

def normalize(tensor):
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    return tensor / norm

def assign_base_classifier(base_prototypes, rv):

    target_choice_ix = len(rv)
    base_prototypes_np = base_prototypes.detach().cpu().numpy()

    cost = cosine_similarity(base_prototypes_np, rv.cpu()[:target_choice_ix])
    col_ind = get_assignment(cost)
    new_fc_tensor = rv[col_ind]

    unassigned_rv = torch.stack([rv[i] for i in range(len(rv)) if i not in col_ind])

    # Return the assigned vectors
    return new_fc_tensor, unassigned_rv

def get_assignment(cost):
    """Tak array with cosine scores and return the output col ind """
    _, col_ind = linear_sum_assignment(cost, maximize=True)
    return col_ind


def update_projection_layer(projection_layer, dlod_tr_0, dlod_ev, base_prototypes, assigned_vectors):
    optimizer = torch.optim.Adam(projection_layer.parameters(), lr=0.0002)
    target_labels = len(base_prototypes)
    scl = supcon.SupConLoss()
    xent = torch.nn.CrossEntropyLoss()

    best_acc = 0
    bestbest_acc = 0
    best_projection_layer_state = None

    class_features = [[] for _ in range(target_labels)]
    class_labels = [[] for _ in range(target_labels)]

    with torch.enable_grad():
        tqdm_gen = tqdm(range(1))
        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            for idx, (x, y, z) in enumerate(dlod_tr_0):
                if torch.cuda.is_available():
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)

                feats = model(x.squeeze().cuda())
                projections = projection_layer(feats)
                projections = normalize(projections)



                logits = torch.matmul(projections, assigned_vectors.t())
                #label_rep = y.repeat(2)
                loss = xent(logits, y)

                ta.add(count_acc(logits, y))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                for label in range(target_labels):
                    if label in y.cpu().numpy():
                        indices = (y == label).nonzero(as_tuple=True)[0]
                        if len(indices) > 0:
                            feature = feats[indices[0]].detach().cpu().numpy()
                            if len(class_features[label]) < 5:
                                class_features[label].append(feature)
                                class_labels[label].append(label)

                out_string = f"Epoch: {epoch}|[{idx}/{len(dlod_tr_0)}], Training Accuracy (Projection): {ta.item() * 100:.3f}, Validation Accuracy (Projection): {best_acc * 100:.3f}"
                
                tqdm_gen.set_description(out_string)

            test_total_loss = 0
            test_ta = Averager()
            for test_idx, (test_x, test_y, test_z) in enumerate(dlod_ev):
                if torch.cuda.is_available():
                    test_x = test_x.cuda(non_blocking=True)
                    test_y = test_y.cuda(non_blocking=True)

                test_feats = model(test_x.squeeze().cuda())
                test_projections = projection_layer(test_feats)
                test_projections = normalize(test_projections)

                test_logits = torch.matmul(test_projections, assigned_vectors.t())

                test_loss = xent(test_logits, test_y)
                test_ta.add(count_acc(test_logits, test_y))

                test_total_loss += test_loss.item()

            test_loss = test_total_loss / len(dlod_ev)
            best_acc = test_ta.item()
            if best_acc > bestbest_acc:
               bestbest_acc = best_acc
            


    print(bestbest_acc)
    return class_features, class_labels

def update_projection_layer_new(projection_layer, processed_data, dlod_ev, selected_feats, selected_labels, assigned_vectors_new, assigned_vectors, class_features, class_labels):
    optimizer = torch.optim.Adam(projection_layer.parameters(), lr=0.001)
    #target_labels = len(base_prototypes)
    scl = supcon.SupConLoss()
    xent = torch.nn.CrossEntropyLoss()
    projection_layer_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_vectors = torch.cat([assigned_vectors_new, assigned_vectors], dim=0)
    
    #new_data = [(feat, label) for feat, label in zip(selected_feats, selected_labels)]
    #processed_data.extend(new_data)
    print(len(selected_feats))
    selected_feats = torch.stack(selected_feats)
    selected_feats = selected_feats.to(projection_layer_device)
    selected_labels = torch.tensor(selected_labels)

    new_data = list(zip(selected_feats.cpu().tolist(), selected_labels.tolist()))
    processed_data.extend(new_data)

    new_processed_data = []
    for label in range(len(class_features)):
        for feature, label_value in zip(class_features[label], class_labels[label]):
            new_processed_data.append((torch.tensor(feature).tolist(), torch.tensor(label_value).tolist()))
    new_processed_data.extend(processed_data)
    processed_data = new_processed_data
    
    
    print(len(selected_feats))

    best_acc = 0
    bestbest_acc = 0
    best_projection_layer_state = None

    with torch.enable_grad():
        tqdm_gen = tqdm(range(1))
        for epoch in tqdm_gen:
            total_loss = 0
            ta = Averager()
            batch_size = 128
            for i in range(0, len(processed_data), batch_size):
                batch = processed_data[i:i + batch_size]
                if not batch:
                   continue
                batch_x = [x for x, y in batch]
                batch_y = [y for x, y in batch]
                #if torch.cuda.is_available():
                    #x = x.cuda(non_blocking=True)
                    #y = y.cuda(non_blocking=True)
                feats = batch_x
                feats = torch.tensor(feats)
                feats = feats.to(projection_layer_device)
                projections = projection_layer(feats)
                projections = normalize(projections)

                logits = torch.matmul(projections, all_vectors.t())

                y_tensor = torch.tensor(batch_y, dtype=torch.long)
                y_tensor = y_tensor.to(logits.device)

                loss = xent(logits, y_tensor)
                #oss = loss + similarity_loss

                ta.add(count_acc_new(logits, y_tensor))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                out_string = f"Epoch: {epoch}|[{i}/{len(processed_data)}], Training Accuracy (Projection): {ta.item() * 100:.3f}, Validation Accuracy (Projection): {best_acc * 100:.3f}"
                tqdm_gen.set_description(out_string)

            test_total_loss = 0
            test_ta = Averager()
            results = []
            datacount = 0
            print(len(dlod_ev))
            for test_idx, (test_x, test_y, test_z) in enumerate(dlod_ev):
                if torch.cuda.is_available():
                   test_x = test_x.cuda(non_blocking=True)
                   test_y = test_y.cuda(non_blocking=True)

                test_feats = model(test_x.squeeze().cuda())
                test_projections = projection_layer(test_feats)
                test_projections = normalize(test_projections)

                test_logits = torch.matmul(test_projections, all_vectors.t())

                filtered_test_logits = []
                filtered_test_y = []
                filtered_test_logits_new = []
                filtered_test_y_new = []
                for i, label in enumerate(test_y):
                    if label < 140:
                       filtered_test_logits.append(test_logits[i])
                       filtered_test_y.append(label)
                    else:
                       filtered_test_logits_new.append(test_logits[i])
                       filtered_test_y_new.append(label)

                if filtered_test_y:
                   filtered_test_y_tensor = torch.stack(filtered_test_y)
                   filtered_test_logits_tensor = torch.stack(filtered_test_logits)
                   test_loss = xent(filtered_test_logits_tensor, filtered_test_y_tensor)
                   test_ta.add(count_acc_new(filtered_test_logits_tensor, filtered_test_y_tensor))
                   test_total_loss += test_loss.item()
                datacount = datacount + len(filtered_test_y_new)
                if filtered_test_y_new:
                   filtered_test_y_new_tensor = torch.stack(filtered_test_y_new)
                   filtered_test_logits_new_tensor = torch.stack(filtered_test_logits_new)
                   #test_loss_new = xent(filtered_test_logits_new_tensor, filtered_test_y_new_tensor)
                   result = count_acc_new_new(filtered_test_logits_new_tensor, filtered_test_y_new_tensor)
                   results.append(result)
                   
            processed_results = {}
            for result in results:
                for true_label, pseudo_label, count in result:
                    if true_label not in processed_results:
                       processed_results[true_label] = {}
                    if pseudo_label not in processed_results[true_label]:
                       processed_results[true_label][pseudo_label] = count
                    else:
                       processed_results[true_label][pseudo_label] += count
            final_results = {}
            for true_label, pseudo_label_counts in processed_results.items():
                max_count = 0
                max_pseudo_label = None
                for pseudo_label, count in pseudo_label_counts.items():
                    if count > max_count:
                       max_count = count
                       max_pseudo_label = pseudo_label
                final_results[true_label] = {'predicted_label': max_pseudo_label,'count': max_count}

            pseudo_label_counts = {}
            for true_label_result in final_results.values():
                pseudo_label = true_label_result['predicted_label']
                count = true_label_result['count']
                if pseudo_label not in pseudo_label_counts:
                   pseudo_label_counts[pseudo_label] = count
                else:
                   if count > pseudo_label_counts[pseudo_label]:
                      pseudo_label_counts[pseudo_label] = count
            total_count = 0
            for pseudo_label, max_count in pseudo_label_counts.items():
                if pseudo_label > 139:
                   total_count += max_count
            print(f" {total_count} / {datacount}")

            #test_total_loss = 0
            #test_ta = Averager()
            #for test_idx, (test_x, test_y, test_z) in enumerate(dlod_ev):
                #if torch.cuda.is_available():
                    #test_x = test_x.cuda(non_blocking=True)
                    #test_y = test_y.cuda(non_blocking=True)

                #test_feats = model(test_x.squeeze().cuda())
                #test_projections = projection_layer(test_feats)
                #test_projections = normalize(test_projections)

               # test_logits = torch.matmul(test_projections, assigned_vectors.t())
#
                #test_loss = xent(test_logits, test_y)
                #test_ta.add(count_acc(test_logits, test_y))

               # test_total_loss += test_loss.item()

           # test_loss = test_total_loss / len(dlod_ev)
            best_acc = test_ta.item()
            if best_acc > bestbest_acc:
               bestbest_acc = best_acc
            

            # Model Saving
            #test_out = test_pseudo_targets_projection(projection_layer, dlod_ev, epoch)
            #va = test_out["va"]

            #if best_acc is None or best_acc < va:
                #best_acc = va
                #best_projection_layer_state = deepcopy(projection_layer.state_dict())

            #out_string = f"Epoch: {epoch}, Training Accuracy (Projection): {ta.item() * 100:.3f}, Validation Accuracy (Projection): {va * 100:.3f}"
            #tqdm_gen.set_description(out_string)

    print(bestbest_acc)

def count_acc_new(logits, y_tensor):
    pred = torch.argmax(logits, dim=1)
    correct = (pred == y_tensor).sum().item()
    total = len(y_tensor)
    return correct / total if total > 0 else 0

def count_acc_new_new(logits, y_tensor):
    pred = torch.argmax(logits, dim=1)
    unique_labels = torch.unique(y_tensor)
    from collections import Counter
    all_preds = list(pred)
    overall_counter = Counter(all_preds)
    most_common_overall_pred, overall_max_count = overall_counter.most_common(1)[0] if overall_counter else (None, 0)
    result = []
    for label in unique_labels:
        label_mask = (y_tensor == label)
        label_preds = pred[label_mask]
        label_counter = Counter(label_preds.tolist())
        most_common_pred, count = label_counter.most_common(1)[0] if label_counter else (None, 0)
        if count < overall_max_count:
            most_common_pred = most_common_overall_pred
            count = overall_max_count
        result.append((label.item(), most_common_pred, count))
    return result
    

def perturb_targets_norm_count(targets, target_labels, ncount, nviews, epsilon = 1, offset = 0):
    
    views = []
    ix = torch.randperm(targets.shape[0])
    if ix.shape[0] < ncount:
        rep_count = math.ceil(ncount/ix.shape[0])            
        ix = ix.repeat(rep_count)[:ncount]
        ix = ix[torch.randperm(ix.shape[0])]
    else:
        ix = ix[:ncount]
    for v in range(nviews):
        rand = ((torch.rand(ncount, targets.shape[1]) - offset) * epsilon).to(targets.device)
        views.append(normalize(targets[ix] + rand))
    
    target_labels = target_labels[ix]
    return views, target_labels

def simplex_loss(feat, labels, assigned_targets, assigned_targets_label, unassigned_targets):
    
    unique_labels, inverse_indices, counts = torch.unique(labels, return_inverse=True, return_counts = True)
    averaged = torch.zeros(len(unique_labels), feat.shape[1]).cuda()
    
    for i, l in enumerate(unique_labels):
        label_indices = torch.where(labels == l)[0]
        averaged_row = torch.mean(feat[label_indices], dim=0)
        averaged[i] = averaged_row
    averaged = normalize(averaged)

  
    mask = ~torch.isin(assigned_targets_label.cuda(), unique_labels)
    assigned_targets_not_in_batch = assigned_targets[mask]
    all_targets = normalize(torch.cat((averaged, assigned_targets_not_in_batch, unassigned_targets))) 

    sim = F.cosine_similarity(all_targets[None,:,:], all_targets[:,None,:], dim=-1)
    loss = torch.log(torch.exp(sim/1).sum(axis = 1)).sum() / all_targets.shape[0]

    return loss

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():      
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        if x is None:  
            return
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implementation')
    
    parser.add_argument('--LOG_DIR', default='./logs', help='Path to log folder')
    parser.add_argument('--dataset', default='cub', help='Training dataset') 
    parser.add_argument('--embedding-size', default=512, type=int, dest='sz_embedding', help='Size of embedding')
    parser.add_argument('--batch-size', default=120, type=int, dest='sz_batch', help='batch.') 
    parser.add_argument('--epochs', default=60, type=int, dest='nb_epochs', help='training epochs.')

    parser.add_argument('--gpu-id', default=0, type=int, help='gpu')

    parser.add_argument('--workers', default=0, type=int, dest='nb_workers', help='Number of workers for dataloader.')
    parser.add_argument('--model', default='resnet18', help='Model') 
    parser.add_argument('--loss', default='Proxy_Anchor', help='Criterion for training') 
    parser.add_argument('--optimizer', default='adamw', help='Optimizer setting')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate setting') 
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay setting')
    parser.add_argument('--lr-decay-step', default=5, type=int, help='Learning decay step setting')  
    parser.add_argument('--lr-decay-gamma', default=0.5, type=float, help='Learning decay gamma setting')
    parser.add_argument('--alpha', default=32, type=float, help='Scaling Parameter setting')
    parser.add_argument('--mrg', default=0.1, type=float, help='Margin parameter setting')
    parser.add_argument('--warm', default=5, type=int, help='Warmup training epochs') 
    parser.add_argument('--bn-freeze', default=True, type=bool, help='Batch normalization parameter freeze')
    parser.add_argument('--l2-norm', default=True, type=bool, help='L2 normlization')
    parser.add_argument('--remark', default='', help='Any reamrk')

    parser.add_argument('--use_split_modlue', type=bool, default=True)
    parser.add_argument('--use_GM_clustering', type=bool, default=True) 

    parser.add_argument('--exp', type=str, default='0')

    ####
    args = parser.parse_args()
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # TEST_LIST = ['cub']
    # args.resume = True # False # True
    # args.only_2step = False
    
    ####
    pth_rst = './result/' + args.dataset
    os.makedirs(pth_rst, exist_ok=True)
    pth_rst_exp = pth_rst + '/' + args.model + '_sp_' + str(args.use_split_modlue) + '_gm_' + str(args.use_GM_clustering) + '_' + args.exp
    os.makedirs(pth_rst_exp, exist_ok=True)

    ####
    pth_dataset = '../datasets'
    if args.dataset == 'cub':
        pth_dataset += '/CUB200'
    elif args.dataset == 'mit':
        pth_dataset += '/MIT67'
    elif args.dataset == 'dog':
        pth_dataset += '/DOG120'
    elif args.dataset == 'air':
        pth_dataset += '/AIR100'

    
    dset_tr_0 = dataset.load(name=args.dataset, root=pth_dataset, mode='train_0', transform=dataset.utils.make_transform(is_train=True))
    dlod_tr_0 = torch.utils.data.DataLoader(dset_tr_0, batch_size=args.sz_batch, shuffle=True, num_workers=args.nb_workers)
    nb_classes = dset_tr_0.nb_classes()

    
    if args.model.find('resnet18') > -1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=False, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
    else:
        print('?')
        sys.exit()

    model = model.cuda()
    criterion_pa = losses.Proxy_Anchor(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).cuda()

    
    param_groups = [
        {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(), 'lr': float(args.lr) * 1},]
    param_groups.append({'params': criterion_pa.parameters(), 'lr': float(args.lr) * 100})
    
   
    opt_pa = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    print('Training parameters: {}'.format(vars(args)))
    print('Training for {} epochs'.format(args.nb_epochs))
    losses_list = []
    best_recall = [0]
    best_epoch = 0

    
  
    dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', transform=dataset.utils.make_transform(is_train=False))
    dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    for epoch in range(0, args.nb_epochs):
        model.train()

        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []

        
        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion_pa.parameters())
            else:
                unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion_pa.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        total, correct = 0, 0
        pbar = tqdm(enumerate(dlod_tr_0))
        for batch_idx, (x, y, z) in pbar:
        
            feats = model(x.squeeze().cuda())
            loss_pa = criterion_pa(feats, y.squeeze().cuda())
            
            cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
            new_loss = 0
            for i, feats in enumerate(feats):
                most_similar_idx = torch.argmax(cos_sim[i])
                similarity = cos_sim[i][most_similar_idx]
               
                if similarity > 0.1:
                   new_loss += similarity - 0.1
            print(new_loss)

            loss_pa = loss_pa + new_loss * 10
            
            opt_pa.zero_grad()
            loss_pa.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == 'Proxy_Anchor':
                torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

            losses_per_epoch.append(loss_pa.data.cpu().numpy())
            opt_pa.step()

            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f}/{:.4f} Acc: {:.4f}'.format(
                epoch, batch_idx + 1, len(dlod_tr_0), 100. * batch_idx / len(dlod_tr_0), loss_pa.item(), 0, 0))

        losses_list.append(np.mean(losses_per_epoch))
        scheduler_pa.step()

        if (epoch >= 0):
            with torch.no_grad():
                print('Evaluating..')
                Recalls = utils.evaluate_cos(model, dlod_ev, epoch)

            #### Best model save
            if best_recall[0] < Recalls[0]:
                best_recall = Recalls
                best_epoch = epoch
                torch.save({'model_pa_state_dict': model.state_dict(), 'proxies_param': criterion_pa.proxies}, '{}/{}_{}_best_step_0.pth'.format(pth_rst_exp, args.dataset, args.model))
                with open('{}/{}_{}_best_results.txt'.format(pth_rst_exp, args.dataset, args.model), 'w') as f:
                    f.write('Best Epoch: {}\tBest Recall@{}: {:.4f}\n'.format(best_epoch, 1, best_recall[0] * 100))





    print('==> Resuming from checkpoint..')
    pth_pth = './result/cub/resnet1/cub_resnet18_best_step_0.pth'
    print(pth_pth)


    checkpoint = torch.load(pth_pth)
    model.load_state_dict(checkpoint['model_pa_state_dict'])
    criterion_pa.proxies = checkpoint['proxies_param']

    model = model.cuda()
    model.eval()

   
    with torch.no_grad():
        all_k_feats, _ = utils.evaluate_cos_(model, dlod_tr_0)
    k_af = AffinityPropagation().fit(all_k_feats.cpu().numpy())
    cluster_labels = k_af.labels_
    cluster_means = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_means:
           cluster_means[label] = all_k_feats[i]
        else:
           cluster_means[label] += all_k_feats[i]
    num_clusters = len(cluster_means)
    for label in cluster_means:
        cluster_means[label] /= np.sum(cluster_labels == label)
    closest_threshold = 0.4
    min_distance_from_80 = float('inf')
    distance_threshold = 0.4
    while distance_threshold <= 0.6:
          new_cluster_means = []
          merged_labels = []
          for i, mean1 in enumerate(cluster_means.values()):
              if isinstance(mean1, torch.Tensor):
                 mean1 = mean1.cpu().numpy()
              for j, mean2 in enumerate(cluster_means.values()):
                  if isinstance(mean2, torch.Tensor):
                     mean2 = mean2.cpu().numpy()
                  if i == j or i in merged_labels or j in merged_labels:
                     continue
                  dist = np.linalg.norm(mean1 - mean2)
                  if dist < distance_threshold:
                     merged_labels.append(j)
          final_num_clusters = len(merged_labels)
          final_num_clusters = num_clusters - final_num_clusters
          distance_from_80 = abs(final_num_clusters - 140)
          if distance_from_80 < min_distance_from_80:
             min_distance_from_80 = distance_from_80
             closest_threshold = distance_threshold
          distance_threshold += 0.01
    

    dset_ev = dataset.load(name=args.dataset, root=pth_dataset, mode='eval_0', transform=dataset.utils.make_transform(is_train=False))
    dlod_ev = torch.utils.data.DataLoader(dset_ev, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)


    projection_layer = ProjectionLayer(encoder_outdim=512, proj_hidden_dim=2048, proj_output_dim=256).cuda()
    rv = find_reseverve_vectors_all()
    

    all_labels = torch.unique(torch.cat([torch.tensor([y]) for _, y, _ in dset_tr_0]))
    prototypes_labels = all_labels.tolist()
    
    base_prototypes = [torch.zeros(256).cuda() for _ in range(len(all_labels))]
    
    total_counts = [0] * len(all_labels)
    for batch_idx, (x, y, z) in tqdm(enumerate(dlod_tr_0)):
        with torch.no_grad():
            feats = model(x.squeeze().cuda())
            proj_feats = projection_layer(feats)

            for label_idx, label in enumerate(all_labels):
                label_mask = y == label
                base_prototypes[label_idx] += proj_feats[label_mask].sum(dim=0)
                total_counts[label_idx] += torch.sum(y == label)
            del feats
            del proj_feats
    
    
    for i in range(len(base_prototypes)):
        base_prototypes[i] /= total_counts[i]
    

    
    base_prototypes_tensor = torch.stack(base_prototypes)
    base_prototypes = normalize(base_prototypes_tensor)
    assigned_vectors, unassigned_rv = assign_base_classifier(base_prototypes, rv)
    print(len(assigned_vectors))

    TTT_class_features, TTT_class_labels = update_projection_layer(projection_layer, dlod_tr_0, dlod_ev, base_prototypes, assigned_vectors)


    args.nb_epochs = 1
    args.warm = 10
    args.steps = 1 

    dlod_tr_prv = dlod_tr_0
    dset_tr_now_md = 'train_1'
    dset_ev_now_md = 'eval_1' 
    nb_classes_prv = nb_classes
    nb_classes_evn = nb_classes 


    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
    dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md, transform=dataset.utils.make_transform(is_train=False))
    dlod_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    dlod_ev_now = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    print('==> Calc. proxy mean and sigma for exemplar..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_prv)
        feats = losses.l2_norm(feats)
        expler_s = feats.std(dim=0).cuda()
    ####
    print('==> Init. Split old and new..')
    thres = 0.
    with torch.no_grad():
        feats, labels = utils.evaluate_cos_(model, dlod_tr_now)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
        preds_cs, _ = torch.max(cos_sim, dim=1)
        utils.show_OnN(feats, labels, preds_cs, nb_classes_prv, pth_rst_exp, thres, True)

    ####
    print('==> Fine. Split old and new..')
    if args.use_split_modlue:
        from splitNet import SplitModlue

        ev_dataset = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
        ev_dataset_train = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=True))

        split_module = SplitModlue(save_path=pth_rst_exp)
        idx_n, idx_o = split_module.split_old_and_new(main_model=model, proxy=criterion_pa,
                                                      old_new_dataset_eval=ev_dataset, old_new_dataset_train=ev_dataset_train, last_old_num=nb_classes, thres_cos=thres) # , step=i)
        dset_tr_o = generate_dataset(dset_tr_now, idx_o)
        dset_tr_n = generate_dataset(dset_tr_now, idx_n)
        dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    else:
        idx = torch.where(preds_cs >= thres, 0, 1)
        idx_o = torch.nonzero(idx).squeeze()
        dset_tr_o = generate_dataset(dset_tr_now, idx_o)
        idx_n = torch.nonzero(1 - idx).squeeze()
        dset_tr_n = generate_dataset(dset_tr_now, idx_n)
        dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    ####
    print('==> Replace old labels..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_o)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
        _, preds_lb = torch.max(cos_sim, dim=1)
        preds_lb_o = preds_lb.detach().cpu().numpy()

    selected_feats_old = []
    selected_labels_old = []

    for label in np.unique(preds_lb_o):
        label_indices = np.where(preds_lb_o == label)[0]
        label_feats = feats[label_indices]
        mean_feat = label_feats.mean(dim=0)
        distances = torch.norm(label_feats - mean_feat, dim=1)
        sorted_indices = torch.argsort(distances)
        if len(sorted_indices) > 5:
           selected_indices = sorted_indices[:5]
        else:
           selected_indices = sorted_indices
        selected_label_feats = label_feats[selected_indices]
        selected_labels = [label] * len(selected_label_feats)
        selected_feats_old.extend(selected_label_feats)
        selected_labels_old.extend(selected_labels)


    print('==> Clustering splitted new and replace new labels..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_n)
    all_feats = feats
    clst_a = AffinityPropagation().fit(feats.cpu().numpy()) # 0.75
    p, c = np.unique(clst_a.labels_, return_counts=True)
    nb_classes_k = len(p)
    print(p, c)
    preds_lb_n = clst_a.labels_

    ####
    if args.use_GM_clustering:
        gm = GaussianMixture(n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params='kmeans').fit(feats.cpu().numpy()) 
        preds_lb_n = gm.predict(feats.cpu().numpy())


    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=True))

    dset_tr_n = generate_dataset(dset_tr_now, idx_n)

    dset_tr_n.ys = (preds_lb_n + nb_classes_prv).tolist()



    print('==> Training splitted new..')
    nb_classes_now = nb_classes_prv + nb_classes_k


    bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0., 0., 0., 0., 0.
    bst_epoch_a, bst_epoch_o, bst_epoch_n = 0., 0., 0.



    unique_labels = np.unique(preds_lb_n)
    class_means = {}
    selected_features = []
    for label in unique_labels:
        indices = np.where(preds_lb_n == label)[0]
        class_features = feats.cpu().numpy()[indices]
        mean_feat = np.mean(class_features, axis=0)
        distances = np.linalg.norm(class_features - mean_feat, axis=1)
        closest_indices = np.argsort(distances)[:10]
        for index in closest_indices:
            selected_features.append((class_features[index], label))
        class_means[label] = torch.from_numpy(mean_feat)


    for label, _ in class_means.items():
        selected_feats = [feat for feat, lbl in selected_features if lbl == label]
        mean_feat = np.mean(selected_feats, axis=0)
        class_means[label] = torch.from_numpy(mean_feat)


    merged_classes = {}
    for label1 in unique_labels:
        if label1 not in merged_classes:
           merged_label = label1
           for label2 in unique_labels:
               if label2 not in merged_classes:
                  mean1 = class_means[label1]
                  mean2 = class_means[label2]
                  distance = np.linalg.norm(mean1 - mean2)
                  if distance < 0.5:
                     if merged_label < label2:
                        merged_classes[label2] = merged_label
                     else:
                        merged_classes[merged_label] = label2
                        merged_label = label2


    new_selected_features = []
    new_label_counter = 140
    label_mapping = {}
    for feat, label in selected_features:
        if label in merged_classes:
           if merged_classes[label] not in label_mapping:
              label_mapping[merged_classes[label]] = new_label_counter
              new_label_counter += 1
           new_label = label_mapping[merged_classes[label]]
        else:
           if label not in label_mapping:
              label_mapping[label] = new_label_counter
              new_label_counter += 1
           new_label = label_mapping[label]
        new_selected_features.append((feat, new_label))
    selected_features = new_selected_features


    new_class_means = {}
    for feat, label in selected_features:
        if label not in new_class_means:
           new_class_means[label] = [feat]
        else:
           new_class_means[label].append(feat)

    new_mean_tensors = []
    for label, feats_list in new_class_means.items():
        feats_array = np.array(feats_list)
        mean_feat = np.mean(feats_array, axis=0)
        mean_tensor = torch.from_numpy(mean_feat).cuda()
        new_mean_tensors.append(mean_tensor)

    print(len(new_class_means))
    projected_class_means_tensor = projection_layer(torch.stack(new_mean_tensors))
    print(len(unassigned_rv))
    assigned_vectors_new, unassigned_rv_new = assign_base_classifier(projected_class_means_tensor, unassigned_rv)
    print(len(assigned_vectors_new))

    update_projection_layer_new(projection_layer, selected_features, dlod_ev_now, selected_feats_old, selected_labels_old, assigned_vectors_new, assigned_vectors, TTT_class_features, TTT_class_labels)


    
    args.nb_epochs = 1
    args.warm = 10
    args.steps = 1 


    dlod_tr_prv = dlod_tr_0
    dset_tr_now_md = 'train_2' 
    dset_ev_now_md = 'eval_2' 
    nb_classes_prv = nb_classes
    nb_classes_evn = nb_classes 


    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
    dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md, transform=dataset.utils.make_transform(is_train=False))
    dlod_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    dlod_ev_now = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    ####
    print('==> Calc. proxy mean and sigma for exemplar..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_prv)
        feats = losses.l2_norm(feats)
        expler_s = feats.std(dim=0).cuda()
    ####
    print('==> Init. Split old and new..')
    thres = 0.
    with torch.no_grad():
        feats, labels = utils.evaluate_cos_(model, dlod_tr_now)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
        preds_cs, _ = torch.max(cos_sim, dim=1)
        utils.show_OnN(feats, labels, preds_cs, nb_classes_prv, pth_rst_exp, thres, True)

    ####
    print('==> Fine. Split old and new..')
    if args.use_split_modlue:
        from splitNet import SplitModlue

        ev_dataset = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
        ev_dataset_train = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=True))

        split_module = SplitModlue(save_path=pth_rst_exp)
        idx_n, idx_o = split_module.split_old_and_new(main_model=model, proxy=criterion_pa,
                                                      old_new_dataset_eval=ev_dataset, old_new_dataset_train=ev_dataset_train, last_old_num=nb_classes, thres_cos=thres) # , step=i)
        dset_tr_o = generate_dataset(dset_tr_now, idx_o)
        dset_tr_n = generate_dataset(dset_tr_now, idx_n)
        dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    else:
        idx = torch.where(preds_cs >= thres, 0, 1)
        idx_o = torch.nonzero(idx).squeeze()
        dset_tr_o = generate_dataset(dset_tr_now, idx_o)
        idx_n = torch.nonzero(1 - idx).squeeze()
        dset_tr_n = generate_dataset(dset_tr_now, idx_n)
        dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    ####
    print('==> Replace old labels..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_o)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
        _, preds_lb = torch.max(cos_sim, dim=1)
        preds_lb_o = preds_lb.detach().cpu().numpy()


    selected_feats_old_2 = []
    selected_labels_old_2 = []

    for label in np.unique(preds_lb_o):
        label_indices = np.where(preds_lb_o == label)[0]
        label_feats = feats[label_indices]
        mean_feat = label_feats.mean(dim=0)
        distances = torch.norm(label_feats - mean_feat, dim=1)
        sorted_indices = torch.argsort(distances)
        if len(sorted_indices) > 5:
           selected_indices = sorted_indices[:5]
        else:
           selected_indices = sorted_indices
        selected_label_feats = label_feats[selected_indices]
        selected_labels = [label] * len(selected_label_feats)
        selected_feats_old_2.extend(selected_label_feats)
        selected_labels_old_2.extend(selected_labels)

    selected_feats_old.extend(selected_feats_old_2)
    selected_labels_old.extend(selected_labels_old_2)

    print('==> Clustering splitted new and replace new labels..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_n)
    feats = torch.cat((feats, all_feats), dim=0)
    all_feats = feats
    clst_a = AffinityPropagation().fit(feats.cpu().numpy()) # 0.75
    p, c = np.unique(clst_a.labels_, return_counts=True)
    nb_classes_k = len(p)
    print(p, c)
    preds_lb_n = clst_a.labels_

    ####
    if args.use_GM_clustering:
        gm = GaussianMixture(n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params='kmeans').fit(feats.cpu().numpy()) 
        preds_lb_n = gm.predict(feats.cpu().numpy())

    
    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=True))
    
    dset_tr_n = generate_dataset(dset_tr_now, idx_n)
    
    dset_tr_n.ys = (preds_lb_n + nb_classes_prv).tolist()

    print('==> Training splitted new..')
    nb_classes_now = nb_classes_prv + nb_classes_k


    bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0., 0., 0., 0., 0.
    bst_epoch_a, bst_epoch_o, bst_epoch_n = 0., 0., 0.

 
    unique_labels = np.unique(preds_lb_n)
    class_means = {}
    selected_features = []
    for label in unique_labels:
        indices = np.where(preds_lb_n == label)[0]
        class_features = feats.cpu().numpy()[indices]
        mean_feat = np.mean(class_features, axis=0)
        distances = np.linalg.norm(class_features - mean_feat, axis=1)
        closest_indices = np.argsort(distances)[:10]
        for index in closest_indices:
            selected_features.append((class_features[index], label))
        class_means[label] = torch.from_numpy(mean_feat)


    for label, _ in class_means.items():
        selected_feats = [feat for feat, lbl in selected_features if lbl == label]
        mean_feat = np.mean(selected_feats, axis=0)
        class_means[label] = torch.from_numpy(mean_feat)


    merged_classes = {}
    for label1 in unique_labels:
        if label1 not in merged_classes:
           merged_label = label1
           for label2 in unique_labels:
               if label2 not in merged_classes:
                  mean1 = class_means[label1]
                  mean2 = class_means[label2]
                  distance = np.linalg.norm(mean1 - mean2)
                  if distance < 0.5:
                     if merged_label < label2:
                        merged_classes[label2] = merged_label
                     else:
                        merged_classes[merged_label] = label2
                        merged_label = label2


    new_selected_features = []
    new_label_counter = 140
    label_mapping = {}
    for feat, label in selected_features:
        if label in merged_classes:
           if merged_classes[label] not in label_mapping:
              label_mapping[merged_classes[label]] = new_label_counter
              new_label_counter += 1
           new_label = label_mapping[merged_classes[label]]
        else:
           if label not in label_mapping:
              label_mapping[label] = new_label_counter
              new_label_counter += 1
           new_label = label_mapping[label]
        new_selected_features.append((feat, new_label))


    selected_features = new_selected_features
    


    new_class_means = {}
    for feat, label in selected_features:
        if label not in new_class_means:
           new_class_means[label] = [feat]
        else:
           new_class_means[label].append(feat)

    new_mean_tensors = []
    for label, feats_list in new_class_means.items():
        feats_array = np.array(feats_list)
        mean_feat = np.mean(feats_array, axis=0)
        mean_tensor = torch.from_numpy(mean_feat).cuda()
        new_mean_tensors.append(mean_tensor)

    projected_class_means_tensor = projection_layer(torch.stack(new_mean_tensors))
    assigned_vectors_new, unassigned_rv_new = assign_base_classifier(projected_class_means_tensor, unassigned_rv)
    update_projection_layer_new(projection_layer, selected_features, dlod_ev_now, selected_feats_old, selected_labels_old, assigned_vectors_new, assigned_vectors, TTT_class_features, TTT_class_labels)

    ####
    args.nb_epochs = 1
    args.warm = 10
    args.steps = 1 # 2

    dlod_tr_prv = dlod_tr_0
    dset_tr_now_md = 'train_3' 
    dset_ev_now_md = 'eval_3' 
    nb_classes_prv = nb_classes
    nb_classes_evn = nb_classes 


    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
    dset_ev_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_ev_now_md, transform=dataset.utils.make_transform(is_train=False))
    dlod_tr_now = torch.utils.data.DataLoader(dset_tr_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    dlod_ev_now = torch.utils.data.DataLoader(dset_ev_now, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    ####
    print('==> Calc. proxy mean and sigma for exemplar..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_prv)
        feats = losses.l2_norm(feats)
        expler_s = feats.std(dim=0).cuda()
    ####
    print('==> Init. Split old and new..')
    thres = 0.
    with torch.no_grad():
        feats, labels = utils.evaluate_cos_(model, dlod_tr_now)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
        preds_cs, _ = torch.max(cos_sim, dim=1)
        utils.show_OnN(feats, labels, preds_cs, nb_classes_prv, pth_rst_exp, thres, True)

    ####
    print('==> Fine. Split old and new..')
    if args.use_split_modlue:
        from splitNet import SplitModlue

        ev_dataset = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=False))
        ev_dataset_train = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=True))

        split_module = SplitModlue(save_path=pth_rst_exp)
        idx_n, idx_o = split_module.split_old_and_new(main_model=model, proxy=criterion_pa,
                                                      old_new_dataset_eval=ev_dataset, old_new_dataset_train=ev_dataset_train, last_old_num=nb_classes, thres_cos=thres) # , step=i)
        dset_tr_o = generate_dataset(dset_tr_now, idx_o)
        dset_tr_n = generate_dataset(dset_tr_now, idx_n)
        dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
    else:
        idx = torch.where(preds_cs >= thres, 0, 1)
        idx_o = torch.nonzero(idx).squeeze()
        dset_tr_o = generate_dataset(dset_tr_now, idx_o)
        idx_n = torch.nonzero(1 - idx).squeeze()
        dset_tr_n = generate_dataset(dset_tr_now, idx_n)
        dlod_tr_o = torch.utils.data.DataLoader(dset_tr_o, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)
        dlod_tr_n = torch.utils.data.DataLoader(dset_tr_n, batch_size=args.sz_batch, shuffle=False, num_workers=args.nb_workers)

    ####
    print('==> Replace old labels..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_o)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
        _, preds_lb = torch.max(cos_sim, dim=1)
        preds_lb_o = preds_lb.detach().cpu().numpy()

    selected_feats_old_3 = []
    selected_labels_old_3 = []

    for label in np.unique(preds_lb_o):
        label_indices = np.where(preds_lb_o == label)[0]
        label_feats = feats[label_indices]
        mean_feat = label_feats.mean(dim=0)
        distances = torch.norm(label_feats - mean_feat, dim=1)
        sorted_indices = torch.argsort(distances)
        if len(sorted_indices) > 5:
           selected_indices = sorted_indices[:5]
        else:
           selected_indices = sorted_indices
        selected_label_feats = label_feats[selected_indices]
        selected_labels = [label] * len(selected_label_feats)
        selected_feats_old_3.extend(selected_label_feats)
        selected_labels_old_3.extend(selected_labels)

    selected_feats_old.extend(selected_feats_old_3)
    selected_labels_old.extend(selected_labels_old_3)

    print('==> Clustering splitted new and replace new labels..')
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dlod_tr_n)
    feats = torch.cat((feats, all_feats), dim=0)
    clst_a = AffinityPropagation().fit(feats.cpu().numpy()) # 0.75
    p, c = np.unique(clst_a.labels_, return_counts=True)
    nb_classes_k = len(p)
    print(p, c)
    preds_lb_n = clst_a.labels_

    ####
    if args.use_GM_clustering:
        gm = GaussianMixture(n_components=nb_classes_k, max_iter=1000, tol=1e-4, init_params='kmeans').fit(feats.cpu().numpy()) 
        preds_lb_n = gm.predict(feats.cpu().numpy())


    dset_tr_now = dataset.load(name=args.dataset, root=pth_dataset, mode=dset_tr_now_md, transform=dataset.utils.make_transform(is_train=True))

    dset_tr_n = generate_dataset(dset_tr_now, idx_n)

    dset_tr_n.ys = (preds_lb_n + nb_classes_prv).tolist()

    print('==> Training splitted new..')
    nb_classes_now = nb_classes_prv + nb_classes_k


    bst_acc_a, bst_acc_oo, bst_acc_on, bst_acc_no, bst_acc_nn = 0., 0., 0., 0., 0.
    bst_epoch_a, bst_epoch_o, bst_epoch_n = 0., 0., 0.



    unique_labels = np.unique(preds_lb_n)
    class_means = {}
    selected_features = []
    for label in unique_labels:
        indices = np.where(preds_lb_n == label)[0]
        class_features = feats.cpu().numpy()[indices]
        mean_feat = np.mean(class_features, axis=0)
        distances = np.linalg.norm(class_features - mean_feat, axis=1)
        closest_indices = np.argsort(distances)[:10]
        for index in closest_indices:
            selected_features.append((class_features[index], label))
        class_means[label] = torch.from_numpy(mean_feat)


    for label, _ in class_means.items():
        selected_feats = [feat for feat, lbl in selected_features if lbl == label]
        mean_feat = np.mean(selected_feats, axis=0)
        class_means[label] = torch.from_numpy(mean_feat)


    merged_classes = {}
    for label1 in unique_labels:
        if label1 not in merged_classes:
           merged_label = label1
           for label2 in unique_labels:
               if label2 not in merged_classes:
                  mean1 = class_means[label1]
                  mean2 = class_means[label2]
                  distance = np.linalg.norm(mean1 - mean2)
                  if distance < 0.5:
                     if merged_label < label2:
                        merged_classes[label2] = merged_label
                     else:
                        merged_classes[merged_label] = label2
                        merged_label = label2


    new_selected_features = []
    new_label_counter = 140
    label_mapping = {}
    for feat, label in selected_features:
        if label in merged_classes:
           if merged_classes[label] not in label_mapping:
              label_mapping[merged_classes[label]] = new_label_counter
              new_label_counter += 1
           new_label = label_mapping[merged_classes[label]]
        else:
           if label not in label_mapping:
              label_mapping[label] = new_label_counter
              new_label_counter += 1
           new_label = label_mapping[label]
        new_selected_features.append((feat, new_label))


    selected_features = new_selected_features
    


    new_class_means = {}
    for feat, label in selected_features:
        if label not in new_class_means:
           new_class_means[label] = [feat]
        else:
           new_class_means[label].append(feat)

    new_mean_tensors = []
    for label, feats_list in new_class_means.items():
        feats_array = np.array(feats_list)
        mean_feat = np.mean(feats_array, axis=0)
        mean_tensor = torch.from_numpy(mean_feat).cuda()
        new_mean_tensors.append(mean_tensor)


    projected_class_means_tensor = projection_layer(torch.stack(new_mean_tensors))

    assigned_vectors_new, unassigned_rv_new = assign_base_classifier(projected_class_means_tensor, unassigned_rv)

    update_projection_layer_new(projection_layer, selected_features, dlod_ev_now, selected_feats_old, selected_labels_old, assigned_vectors_new, assigned_vectors, TTT_class_features, TTT_class_labels)