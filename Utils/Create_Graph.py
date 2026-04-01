import torch
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

def normalize_rows(x, eps=1e-8):
    """按行 l2 归一化（用于 cosine similarity）"""
    norm = x.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return x / norm


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from torch_sparse import coalesce

import torch
from torch_geometric.utils import degree

def get_user_groups(train_edge_index, test_edge_index, top_ratio=0.2):
    # 获取全局用户 id
    train_users = train_edge_index[0]
    test_users = test_edge_index[0]

    # 统计训练集用户交互次数
    max_uid = max(train_users.max(), test_users.max()).item() + 1
    user_counts = torch.bincount(train_users, minlength=max_uid)
    user_counts = user_counts + torch.bincount(test_users, minlength=max_uid)

    # 头部 / 长尾划分（基于训练集）
    user_ids = torch.arange(max_uid).to(train_edge_index.device)
    nonzero_mask = user_counts > 0
    active_users = user_ids[nonzero_mask]
    counts_active = user_counts[nonzero_mask]
    k = max(1, int(len(active_users) * top_ratio))
    sorted_idx = torch.argsort(counts_active, descending=True)
    head_users = set(active_users[sorted_idx[:k]].tolist())
    tail_users = set(active_users[sorted_idx[k:]].tolist())
    print(counts_active[active_users[sorted_idx[:k]].tolist()].sum()/(counts_active[active_users[sorted_idx[k:]].tolist()].sum()+counts_active[active_users[sorted_idx[:k]].tolist()].sum()))
    # 冷启动用户：测试集中出现但训练集中未出现
    # cold_users = set(test_users.tolist()) - set(train_users.tolist())
    cold_users = set(test_users.tolist()) - head_users - tail_users

    return head_users, tail_users, cold_users

def sparse_topk_sym_chunked(C: torch.sparse_coo_tensor, topk: int, chunk_size = 40960):
    """
    对稀疏矩阵 C 按行 Top-k 并对称化，分块实现
    C: [N, N] 稀疏矩阵
    topk: 每行保留的最大邻居数量
    chunk_size: 每次处理的行数
    return:
        row_sym, col_sym, val_sym: 对称 Top-k 边的稀疏 COO 索引和值
    """
    device = C.device
    row, col = C.coalesce().indices()  # [E], [E]
    val = C.coalesce().values()        # [E]
    N = C.size(0)

    all_row_topk = []
    all_col_topk = []
    all_val_topk = []

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)

        # 取当前 chunk 的行
        mask_chunk = (row >= start) & (row < end)
        row_chunk = row[mask_chunk]
        col_chunk = col[mask_chunk]
        val_chunk = val[mask_chunk]

        if row_chunk.numel() == 0:
            continue

        # 行局部编号
        row_chunk_local = row_chunk - start
        num_rows_chunk = end - start

        # 按行分组 Top-k
        row_counts = torch.bincount(row_chunk, minlength=end)
        row_ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(row_counts[start:end], dim=0)])

        row_topk_chunk = []
        col_topk_chunk = []
        val_topk_chunk = []

        for r_local in range(num_rows_chunk):
            start_idx, end_idx = row_ptr[r_local].item(), row_ptr[r_local+1].item()
            if end_idx - start_idx == 0:
                continue
            k = min(topk, end_idx - start_idx)
            vals_r = val_chunk[start_idx:end_idx]
            cols_r = col_chunk[start_idx:end_idx]
            topk_vals, topk_idx = torch.topk(vals_r, k=k, largest=True)
            row_topk_chunk.append(torch.full((k,), r_local + start, device=device, dtype=torch.long))
            col_topk_chunk.append(cols_r[topk_idx])
            val_topk_chunk.append(topk_vals)

        if len(row_topk_chunk) > 0:
            all_row_topk.append(torch.cat(row_topk_chunk))
            all_col_topk.append(torch.cat(col_topk_chunk))
            all_val_topk.append(torch.cat(val_topk_chunk))

    # 合并所有 chunk 的 Top-k
    row_topk = torch.cat(all_row_topk)
    col_topk = torch.cat(all_col_topk)
    val_topk = torch.cat(all_val_topk)

    # 对称化
    idx_sym = torch.cat([
        torch.stack([row_topk, col_topk]),
        torch.stack([col_topk, row_topk])
    ], dim=1)

    val_sym = torch.cat([val_topk, val_topk], dim=0) / 2

    # 合并重复边
    idx_flat = idx_sym[0] * N + idx_sym[1]
    unique_idx, inverse_idx = torch.unique(idx_flat, return_inverse=True)

    row_sym = unique_idx // N
    col_sym = unique_idx % N
    val_sym_agg = torch.zeros_like(unique_idx, dtype=val_sym.dtype, device=device).scatter_add_(0, inverse_idx, val_sym)

    return row_sym, col_sym, val_sym_agg


def sparse_topk_sym(C: torch.sparse_coo_tensor, topk: int):
    """
    对稀疏矩阵 C 做 row-wise top-k 采样并对称化
    C: [N, N] 稀疏矩阵
    topk: 每行保留的最大邻居数量
    return:
        row_sym, col_sym, val_sym: 对称 Top-k 边的稀疏 COO 索引和值
    """
    device = C.device
    row, col = C.indices()  # [E], [E]
    val = C.values()        # [E]

    N = C.size(0)

    # 1️⃣ 按行分组
    row_unique = torch.arange(N, device=device)
    mask = row.unsqueeze(1) == row_unique.unsqueeze(0)  # [E, N]

    # 2️⃣ 对每行进行 Top-k 选择
    row_topk_idx = []
    col_topk_idx = []
    val_topk = []

    for r in range(N):
        mask_r = row == r
        vals_r = val[mask_r]
        cols_r = col[mask_r]

        if vals_r.numel() == 0:
            continue
        k = min(topk, vals_r.numel())
        topk_vals, topk_indices = torch.topk(vals_r, k=k, largest=True)
        row_topk_idx.append(torch.full((k,), r, device=device, dtype=torch.long))
        col_topk_idx.append(cols_r[topk_indices])
        val_topk.append(topk_vals)

    row_topk_idx = torch.cat(row_topk_idx, dim=0)
    col_topk_idx = torch.cat(col_topk_idx, dim=0)
    val_topk = torch.cat(val_topk, dim=0)

    # 3️⃣ 对称化: C_sym = (C + C^T) / 2
    idx_sym = torch.cat([torch.stack([row_topk_idx, col_topk_idx]),
                         torch.stack([col_topk_idx, row_topk_idx])], dim=1)
    val_sym = torch.cat([val_topk, val_topk], dim=0) / 2

    # 合并重复边
    idx_flat = idx_sym[0] * N + idx_sym[1]
    unique_idx, inverse_idx = torch.unique(idx_flat, return_inverse=True)
    row_sym = unique_idx // N
    col_sym = unique_idx % N
    val_sym = torch.zeros_like(unique_idx, dtype=val_sym.dtype, device=device).scatter_add_(0, inverse_idx, val_sym)

    return row_sym, col_sym, val_sym


class FraudAwareAugmentor_core(torch.nn.Module):
    def __init__(self, input, hidden_dim=32, topk=3):
        super().__init__()
        self.topk = topk
        self.item_mlp = torch.nn.Sequential(
            torch.nn.Linear(2+input, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_index, user_idx, num_nodes, x, fraud_label_i=None):
        """
        edge_index: [2, E] 混合索引 (采样子图)
        user_idx: Tensor([u1, u2, ...]) 当前 batch 的用户节点索引 (原始全局索引)
        num_nodes: 当前子图节点总数 (用户+物品)
        fraud_label_i: [num_items] 欺诈标签（若无则传 None）
        return:
            new_edge_index: [2, E_new] 用户-用户增广边 (原始索引)
            new_edge_weight: [E_new] 对应的边权重
        """
        device = edge_index.device
        src, dst = edge_index
        user_idx = user_idx.to(device)
        # === 1️⃣ 区分用户和物品节点 ===
        user_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        user_mask[user_idx] = True

        # 只保留 user->item 边
        mask = user_mask[src] & (~user_mask[dst])
        src_u, dst_i = src[mask], dst[mask]

        # === 2️⃣ 建立局部映射 ===
        user_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        item_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        user_map[user_idx] = torch.arange(len(user_idx), device=device)
        item_nodes = torch.arange(num_nodes, device=device)[~user_mask]
        item_map[item_nodes] = torch.arange(len(item_nodes), device=device)

        u_local = user_map[src_u]
        i_local = item_map[dst_i]
        num_users = len(user_idx)
        num_items = len(item_nodes)

        u_x_list =  x[src_u]
        i_x_list = x[dst_i]

        # === 3️⃣ 度计算 ===
        d_u = torch.log1p(degree(u_local, num_users, dtype=torch.float))
        d_i = torch.log1p(degree(i_local, num_items, dtype=torch.float))

        # === 4️⃣ 欺诈标签与可学习权重 ===
        if fraud_label_i is None:
            fraud_label_i = torch.zeros(num_items, device=device)
        x_i = torch.cat([u_x_list,d_u[u_local].view(-1,1),i_x_list,d_i[i_local].view(-1,1)], dim=1)
        w_i = torch.sigmoid(self.item_mlp(x_i)).squeeze(-1)  # [num_items]

        # # === 5️⃣ 边权重 ===
        # e_weight = torch.sqrt(d_u[u_local]) * torch.sqrt(d_i[i_local]) * w_i[i_local]
        e_weight = w_i

        # === 6️⃣ 构造 A' 并计算 C = A' A'^T ===
        A_prime = torch.sparse_coo_tensor(
            indices=torch.stack([u_local, i_local]),
            values=e_weight,
            size=(num_users, num_items)
        )
        C = torch.sparse.mm(A_prime, A_prime.t())  # [num_users, num_users]

        # === 7️⃣ 左右归一化 D_u^{1/2} ===
        D_half = torch.sqrt(d_u)
        row, col = C.indices()
        val = C.values() * D_half[row] * D_half[col]

        # === 8️⃣ Top-k 采样 + 对称化 ===
        C = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]), values=val, size=(num_users, num_users)
        )

        row_sym, col_sym, val_sym = sparse_topk_sym_chunked(C, topk=min(self.topk, num_users))
        #
        # # 取每个用户的 top-k 邻居
        # topk_val, topk_idx = torch.topk(dense_C, k=min(self.topk, num_users), dim=1)
        # row_idx = torch.arange(num_users, device=device).unsqueeze(1).repeat(1, topk_idx.size(1))
        # row_idx = row_idx.reshape(-1)
        # col_idx = topk_idx.reshape(-1)
        # val_idx = topk_val.reshape(-1)

        # # 保证对称性 (C = (C + C^T) / 2)
        # dense_C_sym = (dense_C + dense_C.t()) / 2
        # nonzero_mask = dense_C_sym > 0
        # row_sym, col_sym = nonzero_mask.nonzero(as_tuple=True)
        # val_sym = dense_C_sym[row_sym, col_sym]

        # === 9️⃣ 转换为 edge_index ===
        new_edge_index_global = torch.stack([
            user_idx[row_sym],
            user_idx[col_sym]
        ], dim=0)  # 原始全局索引

        new_edge_index_local = torch.stack([
            row_sym,
            col_sym
        ], dim=0)  # 局部重新排序索引

        new_edge_weight = val_sym

        return new_edge_index_global,new_edge_index_local, new_edge_weight, d_u

class FraudAwareAugmentor(torch.nn.Module):
    def __init__(self, hidden_dim=16, topk=5):
        super().__init__()
        self.topk = topk
        self.item_mlp = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_index, user_idx, num_nodes, fraud_label_i=None):
        """
        edge_index: [2, E] 混合索引 (采样子图)
        user_idx: Tensor([u1, u2, ...]) 当前 batch 的用户节点索引 (原始全局索引)
        num_nodes: 当前子图节点总数 (用户+物品)
        fraud_label_i: [num_items] 欺诈标签（若无则传 None）
        return:
            new_edge_index: [2, E_new] 用户-用户增广边 (原始索引)
            new_edge_weight: [E_new] 对应的边权重
        """
        device = edge_index.device
        src, dst = edge_index
        user_idx = user_idx.to(device)
        # === 1️⃣ 区分用户和物品节点 ===
        user_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        user_mask[user_idx] = True

        # 只保留 user->item 边
        mask = user_mask[src] & (~user_mask[dst])
        src_u, dst_i = src[mask], dst[mask]

        # === 2️⃣ 建立局部映射 ===
        user_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        item_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        user_map[user_idx] = torch.arange(len(user_idx), device=device)
        item_nodes = torch.arange(num_nodes, device=device)[~user_mask]
        item_map[item_nodes] = torch.arange(len(item_nodes), device=device)

        u_local = user_map[src_u]
        i_local = item_map[dst_i]
        num_users = len(user_idx)
        num_items = len(item_nodes)

        # === 3️⃣ 度计算 ===
        d_u = torch.log1p(degree(u_local, num_users, dtype=torch.float))
        d_i = torch.log1p(degree(i_local, num_items, dtype=torch.float))

        # === 4️⃣ 欺诈标签与可学习权重 ===
        if fraud_label_i is None:
            fraud_label_i = torch.zeros(num_items, device=device)
        x_i = torch.stack([d_i, fraud_label_i], dim=-1)
        w_i = torch.sigmoid(self.item_mlp(x_i)).squeeze(-1)  # [num_items]

        # === 5️⃣ 边权重 ===
        e_weight = torch.sqrt(d_u[u_local]) * torch.sqrt(d_i[i_local]) * w_i[i_local]

        # === 6️⃣ 构造 A' 并计算 C = A' A'^T ===
        A_prime = torch.sparse_coo_tensor(
            indices=torch.stack([u_local, i_local]),
            values=e_weight,
            size=(num_users, num_items)
        )
        C = torch.sparse.mm(A_prime, A_prime.t())  # [num_users, num_users]

        # === 7️⃣ 左右归一化 D_u^{1/2} ===
        D_half = torch.sqrt(d_u)
        row, col = C.indices()
        val = C.values() * D_half[row] * D_half[col]

        # === 8️⃣ Top-k 采样 + 对称化 ===
        C = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]), values=val, size=(num_users, num_users)
        )

        row_sym, col_sym, val_sym = sparse_topk_sym_chunked(C, topk=min(self.topk, num_users))
        #
        # # 取每个用户的 top-k 邻居
        # topk_val, topk_idx = torch.topk(dense_C, k=min(self.topk, num_users), dim=1)
        # row_idx = torch.arange(num_users, device=device).unsqueeze(1).repeat(1, topk_idx.size(1))
        # row_idx = row_idx.reshape(-1)
        # col_idx = topk_idx.reshape(-1)
        # val_idx = topk_val.reshape(-1)

        # # 保证对称性 (C = (C + C^T) / 2)
        # dense_C_sym = (dense_C + dense_C.t()) / 2
        # nonzero_mask = dense_C_sym > 0
        # row_sym, col_sym = nonzero_mask.nonzero(as_tuple=True)
        # val_sym = dense_C_sym[row_sym, col_sym]

        # === 9️⃣ 转换为 edge_index ===
        new_edge_index_global = torch.stack([
            user_idx[row_sym],
            user_idx[col_sym]
        ], dim=0)  # 原始全局索引

        new_edge_index_local = torch.stack([
            row_sym,
            col_sym
        ], dim=0)  # 局部重新排序索引

        new_edge_weight = val_sym

        return new_edge_index_global,new_edge_index_local, new_edge_weight, d_u



class LearnableProjector(nn.Module):
    """
    根据 C = D_u^{1/2} A D_i^{1/2} W D_i^{1/2} A^T D_u^{1/2}
    计算用户-用户稀疏共现（edge_index, edge_weight），
    参数可训练：gamma_u, gamma_i (scalars) 与 per-item theta (vector -> w_i = softplus(theta_i)).
    """
    def __init__(self, num_items, use_item_gamma=False, device='cuda'):
        super().__init__()
        self.device = device
        # per-item latent param -> positive weight
        self.theta = nn.Parameter(torch.zeros(num_items, dtype=torch.float32))  # init 0 -> softplus(0)=~0.693
        # optional gamma scalars for fraud embedding
        self.gamma_u = nn.Parameter(torch.tensor(0.0))  # scalar (can make vector if desired)
        self.gamma_i = nn.Parameter(torch.tensor(0.0)) if use_item_gamma else None

        # hyperparams
        self.eps = 1e-8

    def forward(self, edge_index_ui, num_users, num_items, fraud_u=None, fraud_i=None,
                topk=5, max_users_per_item=200, min_cooccur=0):
        """
        edge_index_ui: LongTensor(2, M) (user_idx, item_idx)
        fraud_u: None or tensor [num_users] containing 0/1 or probabilities
        fraud_i: None or tensor [num_items]
        返回: edge_index_uu (2, E), edge_weight (E,)
        """
        device = self.device
        u_idx = edge_index_ui[0].to(device)
        i_idx = edge_index_ui[1].to(device)

        M = u_idx.size(0)
        # 1) degrees
        deg_u = torch.zeros(num_users, device=device).float().scatter_add_(0, u_idx, torch.ones(M, device=device))
        deg_i = torch.zeros(num_items, device=device).float().scatter_add_(0, i_idx, torch.ones(M, device=device))

        # 2) base log(1+d)
        tilde_d_u = torch.log1p(deg_u)  # [num_users]
        tilde_d_i = torch.log1p(deg_i)  # [num_items]

        # 3) embed fraud into scaling (exponential to keep positive)
        if fraud_u is not None:
            f_u = fraud_u.to(device).float()
        else:
            f_u = torch.zeros(num_users, device=device)

        if fraud_i is not None:
            f_i = fraud_i.to(device).float()
        else:
            f_i = torch.zeros(num_items, device=device)

        # s_u, s_i (positive)
        s_u = tilde_d_u * torch.exp(self.gamma_u * f_u)            # [num_users]
        if self.gamma_i is not None:
            s_i = tilde_d_i * torch.exp(self.gamma_i * f_i)        # [num_items]
        else:
            s_i = tilde_d_i

        # 4) item weight w_j = softplus(theta_j) (positive)
        w_i = F.softplus(self.theta) + self.eps  # ensure >0

        # 5) compute per-edge value val_e = sqrt(s_u[u]) * sqrt(s_i[i]) * sqrt(w_i)
        sqrt_su = torch.sqrt(s_u + self.eps)
        sqrt_si = torch.sqrt(s_i + self.eps)
        # get sqrt_si for each edge and sqrt_w for each edge
        sqrt_si_e = sqrt_si[i_idx]             # [M]
        sqrt_w_e  = torch.sqrt(w_i[i_idx])     # [M]
        val_e = sqrt_su[u_idx] * sqrt_si_e * sqrt_w_e  # [M]

        # 6) group edges by item to produce user-user contributions
        # Build item->list of edge indices
        item2edges = defaultdict(list)
        for e, j in enumerate(i_idx.tolist()):
            item2edges[j].append(e)

        # accumulate pair contributions in dict (p<q as key)
        pair2val = defaultdict(float)
        for j, edges in item2edges.items():
            if len(edges) <= 1:
                continue
            # optional cap to avoid explosion for very popular items
            if max_users_per_item and len(edges) > max_users_per_item:
                # sample subset of edges (可以改为topn按时间/权重)
                edges = edges[:max_users_per_item]

            # get users and vals for this item
            users = [ u_idx[e].item() for e in edges ]
            vals  = val_e[torch.tensor(edges, device=device)]  # [k]
            # compute outer products vals * vals^T and accumulate
            k = len(users)
            # vectorized pairwise (k x k) -> we take only upper triangular
            V = vals.unsqueeze(1) * vals.unsqueeze(0)  # k x k
            users_tensor = torch.tensor(users, device=device)
            for a in range(k):
                ua = users[a]
                for b in range(a+1, k):
                    ub = users[b]
                    contrib = V[a,b].item()
                    if contrib == 0.0:
                        continue
                    key = (ua, ub) if ua < ub else (ub, ua)
                    pair2val[key] += contrib

        # 7) convert pair2val -> edge_index / weights
        pairs = list(pair2val.items())  # [ ((u,v), val), ... ]
        if len(pairs) == 0:
            return torch.empty(2,0,dtype=torch.long, device=device), torch.empty(0, device=device)

        u_list = [k[0] for k,_ in pairs]
        v_list = [k[1] for k,_ in pairs]
        vals   = torch.tensor([v for _,v in pairs], dtype=torch.float32, device=device)

        # make symmetric edges (u->v and v->u) for PyG usage
        edge_index_uu = torch.tensor([u_list + v_list, v_list + u_list], dtype=torch.long, device=device)
        edge_weight = torch.cat([vals, vals], dim=0)

        # 8) optional threshold / min_cooccur filter and top-k per-user
        if min_cooccur > 0:
            mask = edge_weight >= float(min_cooccur)
            edge_index_uu = edge_index_uu[:, mask]
            edge_weight = edge_weight[mask]

        # top-k neighbors per source node
        if topk is not None and topk > 0:
            # for each source keep topk by weight
            src = edge_index_uu[0].cpu().tolist()
            dst = edge_index_uu[1].cpu().tolist()
            wt  = edge_weight.cpu().tolist()
            nbrs = defaultdict(list)
            for s,d,w in zip(src,dst,wt):
                nbrs[s].append((d,w))
            new_src=[]
            new_dst=[]
            new_w=[]
            for s, lst in nbrs.items():
                lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)[:topk]
                for d,w in lst_sorted:
                    new_src.append(s); new_dst.append(d); new_w.append(w)
            edge_index_uu = torch.tensor([new_src, new_dst], dtype=torch.long, device=device)
            edge_weight = torch.tensor(new_w, dtype=torch.float32, device=device)

        # final normalization (可选) —— 将权重归一到 [0,1]
        if edge_weight.numel()>0:
            edge_weight = edge_weight / (edge_weight.max() + self.eps)

        return edge_index_uu, edge_weight



def build_user_knn_edges(
    u_x,
    u_ids,
    k=10,
    metric="cosine",
    chunk_size=4096,
    device=None,
    exclude_self=False,
    directed=False,
    return_edge_weights=False,
    topk_by="k"  # "k" 或 "threshold"
):
    """
    在 sampled users (u_x, u_ids) 内部构造 user-user 边 (稀疏)。
    - u_x: [M, F] (torch.tensor)
    - u_ids: [M] 全局节点 id (torch.long). u_ids[i] 对应 u_x[i].
    - k: top-k 每个用户保留的邻居个数（若 topk_by="threshold"，则 k 会被忽略）
    - chunk_size: 分块计算每次处理多少行以节省内存
    - metric: "cosine" 或 "dot"（若 dot，可视为 un-normalized similarity）
    - exclude_self: 是否排除自环
    - directed: 若 False，会把每条边做成无向（即加入反向并去重），若 True 则保留有向 top-k
    - return_edge_weights: 是否返回相似度作为边属性
    返回:
      edge_index: [2, E] (torch.long)
      edge_weights: [E] (torch.float) or None
    """
    assert u_x.dim() == 2
    M, F = u_x.shape
    device = device or u_x.device
    u_x = u_x.to(device)
    u_ids = u_ids.to(device)

    if metric == "cosine":
        u_x = normalize_rows(u_x)  # in-place not necessary

    all_src = []
    all_dst = []
    all_w = []

    # 用分块计算：每次计算 chunk_rows x M 的相似度，然后取 top-k
    with torch.no_grad():
        for start in range(0, M, chunk_size):
            end = min(M, start + chunk_size)
            q = u_x[start:end]  # [chunk, F]
            # sim = q @ u_x.T  -> shape [chunk, M]
            sim = torch.matmul(q, u_x.t())  # on device

            if exclude_self:
                # 对每一行对应的全局 self idx (local index) 设为 -inf
                # self local index = start + row_idx
                local_self_indices = torch.arange(start, end, device=device)
                sim[torch.arange(end-start, device=device).unsqueeze(1), local_self_indices.unsqueeze(0)] = -1e9

            # 处理 topk 或 threshold
            if topk_by == "k":
                topk = min(k, M - (1 if exclude_self else 0))
                if topk <= 0:
                    continue
                vals, idx = torch.topk(sim, k=topk, dim=1)  # [chunk, topk]
            else:
                # topk_by == "threshold" (未实现 threshold arg here), fallback to k
                topk = min(k, M - (1 if exclude_self else 0))
                vals, idx = torch.topk(sim, k=topk, dim=1)

            # flatten并映射到全局 node id
            rows = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, topk)  # local src idx
            src_global = u_ids[rows.reshape(-1)]   # [chunk*topk]
            dst_global = u_ids[idx.reshape(-1)]   # [chunk*topk]
            w_flat = vals.reshape(-1)

            all_src.append(src_global)
            all_dst.append(dst_global)
            all_w.append(w_flat)

    if len(all_src) == 0:
        # 没有边
        return torch.empty((2,0), dtype=torch.long, device=device), None if return_edge_weights else None

    src_all = torch.cat(all_src, dim=0)
    dst_all = torch.cat(all_dst, dim=0)
    w_all = torch.cat(all_w, dim=0)

    # 如果需要无向且去重：把 (u,v) 与 (v,u) 视作同一条边 -> 只保留一侧
    if not directed:
        # 把每条边转成有序对 (min,max) 用于去重
        a = torch.minimum(src_all, dst_all)
        b = torch.maximum(src_all, dst_all)
        keys = a * (u_ids.max().item() + 1) + b  # 简单哈希（注意节点数较大时可能溢出）
        # 更安全的办法使用 torch.unique on stacked pairs:
        pairs = torch.stack([a, b], dim=1)  # [E,2]
        # 使用 unique 行
        pairs_unique, inv = torch.unique(pairs, dim=0, return_inverse=True)
        # 为每个 unique pair 取权重的 max（也可以用 mean 等）
        # compute max weight per group
        num_groups = pairs_unique.shape[0]
        max_w = torch.zeros(num_groups, device=device) - 1e9
        max_w.scatter_reduce_(0, inv, w_all, reduce="amax", include_self=True)
        # 消除可能的 -1e9
        max_w[max_w < -1e8] = 0.0
        # pairs_unique 是最终的 edge list
        edge_index = pairs_unique.t().contiguous()  # [2, E_u]
        edge_weights = max_w
    else:
        edge_index = torch.stack([src_all, dst_all], dim=0)
        edge_weights = w_all

    if return_edge_weights:
        return edge_index.long(), edge_weights.float()
    else:
        return edge_index.long(), None

import torch
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

def normalize_rows(x, eps=1e-8):
    norm = x.norm(p=2, dim=1, keepdim=True).clamp(min=eps)
    return x / norm

def _coalesce_safe(edge_index, edge_weight, num_nodes):
    """
    兼容不同版本 coalesce 签名：尝试 new-signature (num_nodes=...) 否则 fallback to positional.
    coalesce(..., reduce='sum') 返回 (edge_index, edge_weight)
    """
    try:
        ei, ew = coalesce(edge_index, edge_weight, num_nodes=num_nodes, reduce='sum')
    except TypeError:
        ei, ew = coalesce(edge_index, edge_weight, num_nodes, 'sum')
    return ei, ew

def _process_raw_edges(src_all, dst_all, w_all, num_nodes, directed=False, exclude_self=True, agg="max", device=None):
    """
    把原始三列 (src_all, dst_all, w_all) 处理为 coalesced edge_index/edge_weight。
    - src_all/dst_all: 1D LongTensor (local indices)
    - w_all: 1D FloatTensor
    - num_nodes: int (对应节点数量，用于 coalesce/to_undirected)
    返回 (edge_index, edge_weight)
    """
    device = device or src_all.device
    edge_index = torch.stack([src_all, dst_all], dim=0).to(device)
    edge_weight = w_all.to(device)

    if not directed:
        # to_undirected may accept named arg or positional
        try:
            edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=num_nodes)
        except TypeError:
            edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes)
        if exclude_self:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        # coalesce (sum)
        edge_index, edge_weight = _coalesce_safe(edge_index, edge_weight, num_nodes)
        # 如果需要 max/mean 做二次聚合（基于原始 pairs）
        if agg in ("max", "mean"):
            a = torch.minimum(src_all, dst_all)
            b = torch.maximum(src_all, dst_all)
            pairs = torch.stack([a, b], dim=1)
            pairs_unique, inv = torch.unique(pairs, dim=0, return_inverse=True)
            num_groups = pairs_unique.size(0)
            if agg == "max":
                max_w = torch.full((num_groups,), -1e9, device=device)
                # scatter_reduce_ 可能在旧torch里不支持 include_self 参数，但通常 reduce="amax" 可用
                try:
                    max_w.scatter_reduce_(0, inv, w_all, reduce="amax", include_self=True)
                except TypeError:
                    # fallback: iterative scatter (slower but robust)
                    for i in range(num_groups):
                        max_w[i] = w_all[inv == i].max() if (inv == i).any() else 0.0
                max_w[max_w < -1e8] = 0.0
                edge_index = pairs_unique.t().contiguous()
                edge_weight = max_w
            else:  # mean
                sum_w = torch.zeros(num_groups, device=device)
                cnt = torch.zeros(num_groups, device=device)
                sum_w.scatter_add_(0, inv, w_all)
                cnt.scatter_add_(0, inv, torch.ones_like(w_all))
                mean_w = sum_w / cnt.clamp(min=1.0)
                edge_index = pairs_unique.t().contiguous()
                edge_weight = mean_w
    else:
        # directed: optionally remove self loops, then coalesce on directed pairs
        if exclude_self:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = _coalesce_safe(edge_index, edge_weight, num_nodes)
        if agg in ("max", "mean"):
            pairs = torch.stack([src_all, dst_all], dim=1)
            pairs_unique, inv = torch.unique(pairs, dim=0, return_inverse=True)
            num_groups = pairs_unique.size(0)
            if agg == "max":
                max_w = torch.full((num_groups,), -1e9, device=device)
                try:
                    max_w.scatter_reduce_(0, inv, w_all, reduce="amax", include_self=True)
                except TypeError:
                    for i in range(num_groups):
                        max_w[i] = w_all[inv == i].max() if (inv == i).any() else 0.0
                max_w[max_w < -1e8] = 0.0
                edge_index = pairs_unique.t().contiguous()
                edge_weight = max_w
            else:
                sum_w = torch.zeros(num_groups, device=device)
                cnt = torch.zeros(num_groups, device=device)
                sum_w.scatter_add_(0, inv, w_all)
                cnt.scatter_add_(0, inv, torch.ones_like(w_all))
                mean_w = sum_w / cnt.clamp(min=1.0)
                edge_index = pairs_unique.t().contiguous()
                edge_weight = mean_w

    return edge_index.long(), edge_weight.float()

def build_user_knn_edges_pyg(
    u_x,            # [M, F] 用户特征（采样后）
    u_ids,          # [M] 全局节点 id (long) — 用于生成 global edge_index
    k=8,
    metric="cosine",
    chunk_size=4096,
    device=None,
    exclude_self=False,
    directed=False,
    return_edge_weights=False,
    topk_by="k",
    agg="max",      # 'max'|'mean'|'sum'
):
    """
    返回 (edge_index_local, edge_weight_local, edge_index_global, edge_weight_global)
    - edge_index_local: indices in [0..M-1] (shape [2, E_local])
    - edge_index_global: indices mapped by u_ids (shape [2, E_global])
    其余行为与之前描述一致。
    """
    assert u_x.dim() == 2
    M, F = u_x.shape
    device = device or u_x.device
    u_x = u_x.to(device)
    u_ids = u_ids.to(device)
    assert u_ids.numel() == M, "u_ids length must equal number of rows in u_x"

    if metric == "cosine":
        u_x = normalize_rows(u_x)

    all_src_local = []
    all_dst_local = []
    all_w_local = []

    with torch.no_grad():
        for start in range(0, M, chunk_size):
            end = min(M, start + chunk_size)
            q = u_x[start:end]                       # [chunk, F]
            sim = torch.matmul(q, u_x.t())           # [chunk, M]

            if exclude_self:
                local_self = torch.arange(start, end, device=device)
                sim[torch.arange(end - start, device=device).unsqueeze(1), local_self.unsqueeze(0)] = -1e9

            if topk_by == "k":
                topk = min(k, M - (1 if exclude_self else 0))
                if topk <= 0:
                    continue
                vals, idx = torch.topk(sim, k=topk, dim=1)   # [chunk, topk]
            else:
                topk = min(k, M - (1 if exclude_self else 0))
                vals, idx = torch.topk(sim, k=topk, dim=1)

            rows = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, topk)
            src_local = rows.reshape(-1)                 # local indices 0..M-1
            dst_local = idx.reshape(-1)                  # local indices 0..M-1
            w_flat = vals.reshape(-1)

            all_src_local.append(src_local)
            all_dst_local.append(dst_local)
            all_w_local.append(w_flat)

    if len(all_src_local) == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=device)
        return (empty, None, empty, None) if return_edge_weights else (empty, None, empty, None)

    src_all_local = torch.cat(all_src_local, dim=0)
    dst_all_local = torch.cat(all_dst_local, dim=0)
    w_all_local = torch.cat(all_w_local, dim=0)

    # 处理 local (num_nodes=M)
    edge_index_local, edge_weight_local = _process_raw_edges(
        src_all_local, dst_all_local, w_all_local, num_nodes=M,
        directed=directed, exclude_self=exclude_self, agg=agg, device=device
    )

    # 构建 global 源/目的（把 local idx 映射为 u_ids）
    src_all_global = u_ids[src_all_local]   # [E_raw]
    dst_all_global = u_ids[dst_all_local]   # [E_raw]
    w_all_global = w_all_local

    # num_nodes_global：尽量使用全局图所需的节点数（至少 max(u_ids)+1）
    num_nodes_global = int(u_ids.max().item() + 1)

    edge_index_global, edge_weight_global = _process_raw_edges(
        src_all_global, dst_all_global, w_all_global, num_nodes=num_nodes_global,
        directed=directed, exclude_self=exclude_self, agg=agg, device=device
    )

    if not return_edge_weights:
        edge_weight_local = None
        edge_weight_global = None

    return edge_index_local, edge_weight_local, edge_index_global, edge_weight_global

def merge_user_edges_into_bipartite(
    bip_edge_index,
    user_edge_index,
    bip_edge_attr=None,
    user_edge_attr=None,
    bip_edge_type=None,
    user_edge_type=None,
    device=None
):
    """
    把 user_user 边拼接到采样后的二分图的 edge_index 中
    - bip_edge_index: [2, E_b]
    - user_edge_index: [2, E_u]
    - bip_edge_attr: optional [E_b, D]
    - user_edge_attr: optional [E_u, D_u] (若 None 可创建 e.g. sim weight 或 zeros)
    - bip_edge_type: optional [E_b] (整数)
    - user_edge_type: optional [E_u] (整数)
    返回:
      new_edge_index, new_edge_attr, new_edge_type
    """
    device = device or bip_edge_index.device
    bip_edge_index = bip_edge_index.to(device)
    user_edge_index = user_edge_index.to(device)

    new_edge_index = torch.cat([bip_edge_index, user_edge_index], dim=1)

    new_edge_attr = None
    if bip_edge_attr is not None or user_edge_attr is not None:
        bip_ea = bip_edge_attr if bip_edge_attr is not None else torch.zeros((bip_edge_index.shape[1], 0), device=device)
        user_ea = user_edge_attr if user_edge_attr is not None else torch.zeros((user_edge_index.shape[1], bip_ea.shape[1]), device=device)
        # 若维度不同，可 pad 或裁剪；这里简化处理成 concat 后统一成 float
        if bip_ea.shape[1] != user_ea.shape[1]:
            # pad smaller to match
            maxd = max(bip_ea.shape[1], user_ea.shape[1])
            def _pad(ea, d):
                if ea.shape[1] < d:
                    pad = torch.zeros((ea.shape[0], d - ea.shape[1]), device=device, dtype=ea.dtype)
                    return torch.cat([ea, pad], dim=1)
                return ea
            bip_ea = _pad(bip_ea, maxd)
            user_ea = _pad(user_ea, maxd)
        new_edge_attr = torch.cat([bip_ea, user_ea], dim=0)

    new_edge_type = None
    if bip_edge_type is not None or user_edge_type is not None:
        bip_et = bip_edge_type if bip_edge_type is not None else torch.zeros(bip_edge_index.shape[1], dtype=torch.long, device=device)
        user_et = user_edge_type if user_edge_type is not None else torch.zeros(user_edge_index.shape[1], dtype=torch.long, device=device)
        new_edge_type = torch.cat([bip_et.to(device), user_et.to(device)], dim=0)

    return new_edge_index, new_edge_attr, new_edge_type