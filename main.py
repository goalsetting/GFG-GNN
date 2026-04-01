from __future__ import print_function, division
import argparse
import random
import numpy as np
import torch.nn.functional as F
from Models.MainModel import GNN_OUR, target_distribution
from Models.focal_loss import FocalLoss
from Models.kmeans import kmeans

from torch.optim import Adam

from Utils.Create_Graph import get_user_groups
from abcore_data import get_abcore_data, get_ori_data
import os
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, \
    average_precision_score
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

import warnings


# 屏蔽 Pandas 的版本兼容性警告
warnings.filterwarnings("ignore", message="Pandas requires version")


# 或者屏蔽所有 UserWarning（不推荐，可能掩盖其他重要警告）
warnings.filterwarnings("ignore", category=UserWarning)

import torch



def init_data(adjs, n_id, train=True):
    u_id_mask = (n_id < dataset.max_u)
    u_id = n_id[u_id_mask]

    edge_u_id = list(set(adjs.edge_index[0].numpy()))
    edge_u_id.sort()
    edge_u_id = torch.LongTensor(edge_u_id)

    i_id_mask = (n_id >= dataset.max_u)
    i_id = n_id[i_id_mask] - dataset.max_u

    u_x = dataset.u_x[u_id]
    loss_u_x = dataset.u_x[n_id[edge_u_id]]
    i_x = dataset.i_x[i_id]

    if (train):
        u_l = dataset.train_u_l[n_id[edge_u_id]]
        u_l_mask = (u_l != -1)
        u_l_y = u_l[u_l_mask]

        edge_y = dataset.train_edge_y[adjs.e_id]
        edge_y_mask = (edge_y != -1)
        edge_y = edge_y[edge_y_mask]
        edge_x = dataset.train_edge_x[adjs.e_id][edge_y_mask]

    else:
        u_l = dataset.test_u_l[n_id[edge_u_id]]
        u_l_mask = (u_l != -1)
        u_l_y = u_l[u_l_mask]

        edge_y = dataset.test_edge_y[adjs.e_id]
        edge_y_mask = (edge_y != -1)
        edge_y = edge_y[edge_y_mask]
        edge_x = dataset.test_edge_x[adjs.e_id][edge_y_mask]

    return [u_x[:, :dataset.max_b], u_x[:, dataset.max_b:], i_x[:, :dataset.max_b], i_x[:, dataset.max_b:], u_id_mask,
            i_id_mask, edge_u_id, u_l_mask, edge_y_mask, u_l_y, edge_x[:, :dataset.max_b],
            edge_x[:, dataset.max_b:dataset.max_b * 2], edge_x[:, dataset.max_b * 2:], edge_y, loss_u_x]


def train(adjs, n_id, train=True, test_init_data=None, final_epoch = False, k=0):
    global model
    global optimizer
    global dataset
    global focal_loss
    global head_users
    global tail_users
    global cold_users


    if(train):
        model.train()
        optimizer.zero_grad()
        u_x_core, u_x, i_x_core, i_x, u_id_mask, i_id_mask, edge_u_id, u_l_mask, edge_y_mask, u_l_y, edge_x_u_core, edge_x_i_core, edge_x_out_core, edge_y, loss_u_x = init_data(adjs, n_id, train)
        adjs = adjs.to(device)
    else:
        torch.cuda.empty_cache()
        model.eval()
        u_x_core, u_x, i_x_core, i_x, u_id_mask, i_id_mask, edge_u_id, u_l_mask, edge_y_mask, u_l_y, edge_x_u_core, edge_x_i_core, edge_x_out_core, edge_y, loss_u_x = test_init_data

    x = torch.zeros([len(n_id), u_x.shape[1]+dataset.max_b]).to(device)

    # x[u_id_mask] = u_x
    # x[i_id_mask] = model.i_emb(i_x)
    x[u_id_mask] = torch.cat((model.core_u(u_x_core), u_x),dim=1)
    x[i_id_mask] = torch.cat((model.core_i(i_x_core),model.i_emb(i_x)),dim=1)
    edge_x = torch.cat((model.core_u(edge_x_u_core), model.core_i(edge_x_i_core), edge_x_out_core),dim=1)
    edge_u_x = x[edge_u_id].clone()


    x2, x_bar, q = model(x, edge_u_x, edge_u_id, adjs.edge_index,u_l_y = u_l_y, u_l_mask= u_l_mask, train = train)
    p = 0
    if args.type == 'A':
        if(train):
            p = target_distribution(q)

    pre_l = torch.zeros([len(n_id), 2]).to(device)
    tmp_pre_l = torch.zeros([len(x2), 2]).to(device)
    loss_pre_l = model.tl(x2[u_l_mask])
    tmp_pre_l[u_l_mask] = loss_pre_l
    pre_l[edge_u_id] = tmp_pre_l

    total_x = torch.zeros([len(n_id), x2.shape[1]]).to(device)
    total_x[edge_u_id] = x2
    
    edge_us = adjs.edge_index[0][edge_y_mask]
    pre_e_l = model.tel(torch.cat((pre_l[:,1][edge_us].unsqueeze(-1), total_x[edge_us] ,edge_x),dim=1))


    
    if(train):
        label_loss = focal_loss(loss_pre_l, u_l_y)
        edge_loss = focal_loss(pre_e_l, edge_y)

        game_loss = model.game_head.game_loss_value.clone()
        if args.type == 'A':
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            loss = args.ll * label_loss + args.el * edge_loss + args.kl * kl_loss + args.gl * game_loss
        else:
            ae_loss = model.c_gnn.self_loss.clone()
            loss = args.ll * label_loss + args.el * edge_loss + args.al * ae_loss + args.gl * game_loss
        #



        loss.backward(retain_graph=True)
        optimizer.step()
        # model.c_gnn.self_loss = None
        # model.game_head.game_loss_value = None
    edge_y = edge_y.cpu()
    pre_e_l = pre_e_l[:,1].cpu().detach()
    
    max_th = args.th
    # auc = roc_auc_score(edge_y, pre_e_l)
    if train:
        auc = 0
    else:
        # auc = average_precision_score(edge_y, pre_e_l)
        auc=roc_auc_score(edge_y, pre_e_l)

    
    pre_result = (pre_e_l > args.th)
    f1 = f1_score(edge_y, pre_result)


    def metrics_for_mask(mask, train):
        if mask.sum() == 0:
            return None

        if train:
            return {
                'auc': average_precision_score(y_true[mask], y_score[mask]),
                'f1': f1_score(y_true[mask], y_pred[mask])
            }
        else:
            return {
                'auc': average_precision_score(y_true[mask], y_score[mask]),
                'f1': f1_score(y_true[mask], y_pred[mask]),
                'precision': precision_score(y_true[mask], y_pred[mask]),
                'recall': recall_score(y_true[mask], y_pred[mask]),
                'acc': accuracy_score(y_true[mask], y_pred[mask])
            }





    if(train):
        return loss.item(), f1, auc
    else:
        # ====== 新增: 计算不同用户组性能 ======
        with torch.no_grad():
            global_user_ids = n_id[edge_us.cpu()].cpu().numpy()  # 全局用户id
            y_true = edge_y.cpu().numpy()
            y_pred = pre_result.cpu().numpy()
            y_score = pre_e_l.cpu().numpy()
        # 构造 mask
        head_mask = np.array([uid in head_users for uid in global_user_ids])
        tail_mask = np.array([uid in tail_users for uid in global_user_ids])
        cold_mask = np.array([uid in cold_users for uid in global_user_ids])

        tail_cold_mask = np.logical_or(tail_mask, cold_mask)
        head_metrics = metrics_for_mask(head_mask, train)
        tail_metrics = metrics_for_mask(tail_cold_mask, train)

        pre = precision_score(edge_y, pre_result)
        acc = accuracy_score(edge_y, pre_result)
        recall = recall_score(edge_y, pre_result)

        # torch.cuda.empty_cache()
        return auc, f1, acc, pre, recall, max_th, head_metrics, tail_metrics

def train_exp(dataset,name=None):
    global model
    global optimizer
    global head_users
    global tail_users
    global cold_users


    model = GNN_OUR(
                n_input=dataset.u_x.shape[1],
                n_i=dataset.i_x.shape[1],
                n_clusters=args.n_clusters,
                n_enc = args.hidden_dim, 
                hidden = args.hidden,
                n_z=args.n_z,
                pre_ae_epoch = args.pre_ae_epoch,
                num_item = dataset.max_i,
                max_b=dataset.max_b,
                name = name,
                B_name= B_name,
                args = args,
                dataset = dataset
                ).to(device)
    print(model)
    print(args)

    optimizer = Adam(model.parameters(), lr=args.lr)
    # KNN Graph
    head_users, tail_users, cold_users = get_user_groups(dataset.train_edge,dataset.test_edge)
    train_loader = NeighborSampler(dataset.train_edge, sizes=[-1], batch_size=1024, num_workers=6, shuffle=True)
    test_loader = NeighborSampler(dataset.test_edge, sizes=[-1], batch_size=99999999, num_workers=6, shuffle=False)
    assert len(test_loader) == 1

    batch_num = len(train_loader)
    if args.type == 'A':
        with torch.no_grad():
            _, _, z = model.ae(dataset.u_x)
        model.cluster_layer.data = kmeans(z, args.n_clusters).to(device)
    else:
        model.cluster_layer.data = kmeans(dataset.u_x, args.n_clusters).to(device)
    torch.cuda.empty_cache()
    max_train_f1 = 0
    max_test_f1 = 0
    max_test_acc = 0
    max_test_pre = 0
    max_test_recall = 0
    max_test_auc = 0
    max_epoch = 0

    for batch_size, n_id, adjs in test_loader:
        test_init_data = init_data(adjs, n_id, train=False)
        test_adjs = adjs.to(device)
        test_n_id = n_id



    torch.autograd.set_detect_anomaly(True)

    u_edge_indexs = []
    u_edge_index_globals = []



    for epoch in range(1, args.epoch+1):
        total_loss = total_f1 = total_auc = 0

        round = 0
        for batch_size, n_id, adjs in tqdm(train_loader):

            loss, f1, auc = train(adjs, n_id,k=round)
            total_loss = total_loss + loss
            total_f1 = total_f1 + f1
            total_auc = total_auc + auc
            round = round + 1
        loss = total_loss / batch_num
        f1 = total_f1 / batch_num
        auc = total_auc / batch_num
        
        test_auc, test_f1, test_acc, test_pre, test_recall, max_th, head_metrics, tail_metrics = train(test_adjs, test_n_id, train=False, test_init_data=test_init_data, final_epoch=(epoch==args.epoch))
        print('{} loss:{:.5f} f1:{:.4f}  tauc:{:.4f} h_tauc:{:.4f} t_tauc:{:.4f} tf1:{:.4f} h_tf1:{:.4f} t_tf1:{:.4f} tacc:{:.4f} h_tacc:{:.4f} t_tacc:{:.4f} tpre:{:.4f} h_tpre:{:.4f} t_tpre:{:.4f} trecall:{:.4f} h_trecall:{:.4f} t_trecall:{:.4f} th:{}'.format(epoch, loss, f1, test_auc, head_metrics['auc'], tail_metrics['auc'], test_f1, head_metrics['f1'], tail_metrics['f1'], test_acc, head_metrics['acc'], tail_metrics['acc'], test_pre, head_metrics['precision'], tail_metrics['precision'], test_recall, head_metrics['recall'], tail_metrics['recall'], max_th))

        # torch.cuda.empty_cache()
        if(test_f1 > max_train_f1):
            max_train_f1 = test_f1
            max_test_f1 = test_f1
            max_test_f1_h = head_metrics['f1']
            max_test_f1_t = tail_metrics['f1']
            max_test_auc = test_auc
            max_test_auc_h = head_metrics['auc']
            max_test_auc_t = tail_metrics['auc']
            max_test_acc = test_acc
            max_test_acc_h = head_metrics['acc']
            max_test_acc_t = tail_metrics['acc']
            max_test_pre = test_pre
            max_test_pre_h = head_metrics['precision']
            max_test_pre_t = tail_metrics['precision']
            max_test_recall = test_recall
            max_test_recall_h = head_metrics['recall']
            max_test_recall_t = tail_metrics['recall']
            max_epoch = epoch

    print('max epoch:{} auc:{:.4f} h_tauc:{:.4f} t_tauc:{:.4f} f1:{:.4f} h_tf1:{:.4f} t_tf1:{:.4f} acc:{:.4f} h_tacc:{:.4f} t_tacc:{:.4f} pre:{:.4f} h_tpre:{:.4f} t_tpre:{:.4f} recall:{:.4f} h_trecall:{:.4f} t_trecall:{:.4f}'.format(max_epoch,max_test_auc,max_test_auc_h,max_test_auc_t, max_test_f1, max_test_f1_h, max_test_f1_t, max_test_acc, max_test_acc_h, max_test_acc_t, max_test_pre, max_test_pre_h, max_test_pre_t, max_test_recall, max_test_recall_h, max_test_recall_t))

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # 忽略UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    # # 强制CPU模式（覆盖所有隐式CUDA调用）
    # torch.set_default_device('cuda')
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--n_z', default=64, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--pre_ae_epoch', default=150, type=int)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--hidden', default=1, type=int)

    parser.add_argument('--ll', type=float, default=0.07)
    parser.add_argument('--el', type=float, default=0.5)
    parser.add_argument('--al', type=float, default=0.16)
    parser.add_argument('--kl', type=float, default=0.07)
    parser.add_argument('--gl', type=float, default=0.2)

    parser.add_argument('--th', type=float, default=0.45)
    parser.add_argument('--gnn', type=str, default='sage')
    parser.add_argument('--topk', type=int, default= 5)
    parser.add_argument('--gt', type=float, default= 0.5)
    parser.add_argument('--type', type=str, default= 'A')
    parser.add_argument('--merge', type=str, default= 'fuse')

    root_path, _ = os.path.split(os.path.abspath(__file__))

    parser.add_argument('--root_path', type=str, default=root_path)
    args = parser.parse_args()
    setup_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    focal_loss = FocalLoss(2)

    name = 'alpha'
    B_name = 'TC_10W'
    dataset = get_abcore_data(device,name,B_name)


    if(os.path.isdir(root_path + '/model') == False):
        os.mkdir(root_path + '/model')

    train_exp(dataset, name)
