import torch.nn as nn
from torch.nn import Linear, Parameter
import torch

from Models.CoreMoudle import CORE, Cross_GNN, TL, TEL, PairGameHead, AE
from Models.GNN import SAGE_NET, GCN_NET
from Models.pre_train import pre_train
from Utils.Create_Graph import FraudAwareAugmentor_core, build_user_knn_edges_pyg, merge_user_edges_into_bipartite
import os

from Utils.Game_Utils import total_game_loss

#gt topk
class GNN_OUR(nn.Module):
    def __init__(self, n_input, n_i, n_clusters, n_enc, hidden, n_z, pre_ae_epoch, num_item, max_b, args, dataset, name =None, B_name =None):
        super(GNN_OUR, self).__init__()
        if (args.gnn == 'sage'):
            GNN_NET = SAGE_NET
        else:
            GNN_NET = GCN_NET
        self.args = args
        self.n_input = n_input
        self.core_u = CORE(max_b)
        self.core_i = CORE(max_b)
        # self.ae = AE(n_enc, hidden,
        #              n_input, n_z)
        self.c_gnn = Cross_GNN(n_input, args )
        self.tl = TL(n_clusters=n_clusters)
        self.tel = TEL(1 + n_clusters + n_input + n_i)
        self.i_emb = Linear(n_i-max_b, n_input-max_b)

        self.trans_emb = Linear((n_input//2)*2, n_input)

        self.trans_emb_z = Linear(n_input, n_enc)

        self.gnn_in = GNN_NET(n_input, n_enc)
        self.hidden_gnn = nn.ModuleList([GNN_NET(n_enc, n_enc) for i in range(hidden)])
        self.gnn_nz = GNN_NET(n_enc, n_z)
        self.gnn_cluster = GNN_NET(n_z, n_clusters)
        self.game_head = PairGameHead(in_feats=n_input, pair_mode="concat", threshold= args.gt)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        self.aug_struct =  FraudAwareAugmentor_core(n_input*2, topk=args.topk)
        # self.u_edge_index = None

        self.gnn_fuse = GNN_NET(n_input, n_input)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = 1

        # degree
        if args.type == 'A':
            self.ae = AE(n_enc, hidden,
                         n_input, n_z)
            if name is not None:
                if (not os.path.exists(args.root_path + f'/model/ae_pre_train_{name}.pkl')):
                    pre_train(dataset, n_clusters, n_input, n_z, n_enc, hidden, pre_ae_epoch, name= name,root_path=args.root_path)
                self.ae.load_state_dict(torch.load(args.root_path + f'/model/ae_pre_train_{name}.pkl', map_location='cuda'))
            else:
                if (not os.path.exists(args.root_path + f'/model/ae_pre_train_{B_name}.pkl')):
                    pre_train(dataset, n_clusters, n_input, n_z, n_enc, hidden, pre_ae_epoch, name= B_name,root_path=args.root_path)
                self.ae.load_state_dict(torch.load(args.root_path + f'/model/ae_pre_train_{B_name}.pkl', map_location='cuda'))

    def forward(self, x, edge_u_x, edge_u_id, edge_index, train=True, u_l_y = None, u_l_mask = None):
        q = 0

        #GNN融合模块
        edge_index_strut_g, edge_index_strut_l, _, d_u = self.aug_struct(edge_index, edge_u_id, x.shape[0], x)
        u_edge_index, _, u_edge_index_global, _ = build_user_knn_edges_pyg(k=self.args.topk, u_x=edge_u_x, u_ids=edge_u_id)

        x_fuse = self.trans_emb(self.c_gnn(x[edge_u_id], u_edge_index, edge_index_strut_l, train, Type = self.args.type))

        x[edge_u_id] = x_fuse
        # x1 = self.gnn_fuse(x[edge_u_id],u_edge_index)
        # x2 = self.gnn_fuse(x[edge_u_id],edge_index_strut_l)
        # x_fuse = self.att(x1, x2, d_u)

        u_edge_index, _, u_edge_index_global, _ = build_user_knn_edges_pyg(k=3, u_x=x_fuse, u_ids=edge_u_id)

        output_game = self.game_head(x_fuse, u_edge_index, u_edge_index_global, fraud_label=u_l_y, fraud_label_mask=u_l_mask, train=train)
        new_edge_index = output_game['new_edge_index']
        raw_edge_index = u_edge_index
        edge_probs = output_game['edge_probs']
        P = output_game['P']
        num_items = x.shape[0] - edge_u_id.shape[0]
        edge_index_all = edge_index.clone()

        if new_edge_index is not None:
            edge_index, _, _ = merge_user_edges_into_bipartite(new_edge_index, edge_index,device='cuda')
        # self.u_edge_index = raw_edge_index
        h = None
        if self.args.type == 'A':
            x_bar, h, z = self.ae(x_fuse)
        else:
            x_bar = x_fuse

        # xs.append(x)
        x = self.gnn_in(x, edge_index)
        xs = []
        xs.append(x.clone())
        for i, layer in enumerate(self.hidden_gnn):
            # 修正的梯度友好操作
            if h is not None:
                residual = torch.zeros_like(x)
                residual[edge_u_id] = h[i]
                x = x + residual
                x[edge_u_id] = x[edge_u_id] + h[i]
            x = layer(x, edge_index)
            xs.append(x.clone())

        if h is not None:
            residual = torch.zeros_like(x)
            residual[edge_u_id] = h[-1]
            x = x + residual
            x[edge_u_id] = x[edge_u_id] + h[-1]
        x = self.gnn_nz(x, edge_index)
        xs.append(x.clone())
        if self.args.merge == 'fuse':
            x = torch.stack(xs,dim=0).mean(0)
        else:
            x = x
        if h is not None:
            # residual = torch.zeros_like(x)
            # residual[edge_u_id] = self.trans_emb_z(x_fuse)
            # x = x + residual
            x[edge_u_id] = x[edge_u_id] + z

        x = self.gnn_cluster(x, edge_index, active=False)

        if train:
            if h is not None:
                q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
                q = q.pow((self.v + 1.0) / 2.0)
                q = (q.t() / torch.sum(q, 1)).t()

            loss_pre_l = self.tl(torch.sigmoid(x[edge_u_id]))

            self.game_head.game_loss_value = total_game_loss(raw_edge_index, edge_probs, map_edge(edge_index_all), P,
                       x[edge_u_id], u_l_y, u_l_mask, num_items, y_pre=loss_pre_l[:,1])[0]


        x = torch.sigmoid(x[edge_u_id])


        return x, x_bar, q


def map_edge(edge_index_all):
    # 假设 edge_index_all = [2, E]
    user_idx_all = edge_index_all[0]
    item_idx_all = edge_index_all[1]

    # 1. 获取唯一用户和物品，并按顺序排序（可以不排序也行，只要唯一映射即可）
    unique_users = torch.unique(user_idx_all)
    unique_items = torch.unique(item_idx_all)

    # 2. 构建原始索引到连续索引的映射
    # 注意 scatter 方式生成映射
    user_map = torch.zeros(unique_users.max().item() + 1, dtype=torch.long, device=user_idx_all.device)
    user_map[unique_users] = torch.arange(unique_users.size(0), device=user_idx_all.device)

    item_map = torch.zeros(unique_items.max().item() + 1, dtype=torch.long, device=item_idx_all.device)
    item_map[unique_items] = torch.arange(unique_items.size(0), device=item_idx_all.device)

    # 3. 映射边索引
    user_idx_new = user_map[user_idx_all]
    item_idx_new = item_map[item_idx_all]

    # 4. 构造新的 edge_index
    edge_index_new = torch.stack([user_idx_new, item_idx_new], dim=0)
    return edge_index_new

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()