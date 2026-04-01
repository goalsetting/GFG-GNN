import torch
import torch.nn.functional as F

# ---------- 辅助函数 ----------
def safe_mean(tensor, dim=None, eps=1e-9):
    """对可能为空的张量做安全平均"""
    if tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device)
    if dim is None:
        return tensor.mean()
    else:
        mask = ~torch.isnan(tensor)
        # 如果全部是nan，返回0
        if mask.sum(dim=dim).eq(0).all():
            return torch.zeros_like(tensor.mean(dim=dim))
        return torch.nanmean(tensor, dim=dim)


def node_soft_label_from_edge_probs(num_nodes, edge_index, edge_probs):
    """
    对于无标签节点，使用其入/出边的合作概率平均作为“软合作度”代理。
    返回 node_coop_prob (越大表示越倾向合作)，以及 fraud_prob = 1 - coop_prob
    edge_index: (2, E) long tensor
    edge_probs: (E,) float tensor, in [0,1], 越大越合作
    """
    device = edge_probs.device
    deg = torch.zeros(num_nodes, device=device)
    coop_sum = torch.zeros(num_nodes, device=device)

    src = edge_index[0]
    dst = edge_index[1]
    # treat edges as undirected for aggregation: add to both ends
    coop_sum = coop_sum.index_add(0, src, edge_probs)
    deg = deg.index_add(0, src, torch.ones_like(edge_probs))
    coop_sum = coop_sum.index_add(0, dst, edge_probs)
    deg = deg.index_add(0, dst, torch.ones_like(edge_probs))

    deg = deg.clamp(min=1.0)  # 防0除
    node_coop = coop_sum / deg
    node_fraud = 1.0 - node_coop
    return node_coop, node_fraud


def edge_game_loss(edge_index, edge_probs, y, y_mask, alpha=2.0, eps=1e-9):
    """
    细粒度层面的博弈效能函数（只考虑有标签用户）
    edge_index: (2, E)
    edge_probs: (E,) 越大表示合作概率
    y: (N,) 0=normal, 1=fraud
    y_mask: (N,) bool, True表示该节点有标签
    alpha: 欺诈用户间合作边权重 (欺诈合作效能更大)
    返回: loss, dict(各类边平均效能)
    """
    device = edge_probs.device
    src, dst = edge_index
    N = y_mask.size(0)
    # ----------------------------------
    # 1️⃣ 构建原节点 -> 有标签节点的映射
    # ----------------------------------
    idx_map = torch.full((N,), -1, dtype=torch.long, device=device)
    idx_map[y_mask] = torch.arange(y.size(0), device=device)

    # ----------------------------------
    # 2️⃣ 过滤仅保留有标签节点之间的边
    # ----------------------------------
    src_new = idx_map[src]
    dst_new = idx_map[dst]
    edge_mask = (src_new >= 0) & (dst_new >= 0)

    src_new = src_new[edge_mask]
    dst_new = dst_new[edge_mask]
    probs = edge_probs[edge_mask]

    if src_new.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    y_src = y[src_new]
    y_dst = y[dst_new]

    # 各种情况mask
    normal_mask = (y_src == 0) & (y_dst == 0)  # 正常-正常
    fraud_mask = (y_src == 1) & (y_dst == 1)  # 欺诈-欺诈
    mixed_mask = y_src != y_dst  # 混合


    # # 筛选：两端节点都为有标签
    # mask_valid = y_mask[src] & y_mask[dst]
    # src, dst = src[mask_valid], dst[mask_valid]
    # probs = edge_probs[mask_valid]
    # if src.numel() == 0:
    #     return torch.tensor(0.0, device=device), {}
    #
    # # 边类型
    # y_src, y_dst = y[src], y[dst]
    # normal_mask = (y_src == 0) & (y_dst == 0)
    # fraud_mask = (y_src == 1) & (y_dst == 1)
    # mixed_mask = ((y_src == 1) & (y_dst == 0)) | ((y_src == 0) & (y_dst == 1))

    # 计算效能
    def safe_mean(x):
        return x.mean() if x.numel() > 0 else torch.tensor(0.0, device=device)

    E_nn = safe_mean(probs[normal_mask])        # normal-normal 合作效能
    E_ff = safe_mean(probs[fraud_mask])         # fraud-fraud 合作效能
    E_fn = safe_mean(1 - probs[mixed_mask])     # fraud-normal 欺诈效能 (1-合作概率)

    # 合作效能（欺诈合作加权更高）
    E_coop = E_nn + alpha * E_ff
    E_fraud = E_fn

    # 对比损失：最大化合作效能 / 欺诈效能
    loss = -torch.log((E_coop + eps) / (E_fraud + eps))

    info = {
        "E_nn": E_nn.detach(),
        "E_ff": E_ff.detach(),
        "E_fn": E_fn.detach(),
        "E_coop": E_coop.detach(),
        "E_fraud": E_fraud.detach(),
        "ratio": (E_coop / (E_fraud + eps)).detach(),
    }
    return loss, info


def compute_T_from_edges(C, edge_index_all, num_items):
    # edge_index_all: [2, num_edges]
    user_idx, item_idx = edge_index_all  # 用户、商品索引
    num_comms = C.size(1)

    # 初始化商品-社区权重
    T = torch.zeros(num_items, num_comms, device=C.device)

    # 对每条边累加 C[u]
    T.index_add_(0, item_idx, C[user_idx])  # 按商品维度聚合社区权重

    # 归一化（每个商品归一化为分布）
    T = T / (T.sum(dim=1, keepdim=True) + 1e-8)
    return T  # [num_items, num_comms]

# ---------------------------
# compute_obs：严格区分 y 与 y_mask
# ---------------------------
def compute_obs(C,                 # [num_users, K]
                edge_index_all,    # [2, E_bip]  user,item
                T,                 # [num_items, K]  (商品被社区使用权重)
                y,                 # [num_labeled_users] 仅包含有标签用户的真实标签 (0/1)
                y_mask,            # [num_users] bool，全体用户的掩码（标识哪些用户有标签）
                y_pre=None,        # [num_users] 软标签预测（用于 mode='soft' 的无标签用户）
                mode='hard'        # 'hard' 或 'soft'
               ):
    """
    输出:
      Obs_norm: [K, K] 归一化后的 Obs (sum = 1)
      diagnostics: dict
    说明:
      - 使用 idx_map 将全体用户映射到 y 的索引：idx_map[u] = pos in y if labeled else -1
      - mode='hard': 仅统计来自有标签用户的边 (严格使用真实标签 y[idx_map[u]])
      - mode='soft': 对每条边, 若 user 有标签则用真实 y，否则用 y_pre[user]
    """
    device = C.device
    user_idx_all = edge_index_all[0]
    item_idx_all = edge_index_all[1]
    num_users = C.size(0)
    K = C.size(1)
    EPS = 1e-9
    # idx_map: 全体节点 -> 有标签 y 的索引
    idx_map = torch.full((num_users,), -1, dtype=torch.long, device=device)
    # labeled_positions = y_mask.nonzero(as_tuple=True)[0]
    # y 的顺序应该和 y_mask True 的顺序一致： idx_map[y_mask==True] = torch.arange(len(y))
    idx_map[y_mask] = torch.arange(y.size(0), device=device)

    # 根据 mode 构造每条边对应的 label value (y_u_edge)
    if y_pre is None:
        # 只保留由有标签用户发出的边
        edge_mask = idx_map[user_idx_all] >= 0
        if edge_mask.sum() == 0:
            Obs = torch.zeros((K, K), device=device)
            return Obs, {'Obs': Obs}
        u_idx = user_idx_all[edge_mask]
        i_idx = item_idx_all[edge_mask]
        # 对应 y 的下标
        y_pos = idx_map[u_idx]   # index into y
        y_u = y[y_pos].float()   # 真正的标签值 (0/1)
    else:
        # 对所有边都考虑：有标签->用真实标签，无标签->用 y_pre
        u_idx = user_idx_all
        i_idx = item_idx_all
        # prepare a full-length vector y_full for all users:
        # if labeled: use y[idx_map[u]]; else: use y_pre[u]
        if y_pre is None:
            raise ValueError("y_pre must be provided when mode='soft'")
        # build y_full:
        # where idx_map >=0 -> y[idx_map], else y_pre[u]
        y_full = y_pre.clone().to(device)
        labeled_users = (idx_map >= 0).nonzero(as_tuple=True)[0]
        if labeled_users.numel() > 0:
            # map labels back
            # positions in y correspond to idx_map == 0..len(y)-1
            # for each labeled user u: y_full[u] = y[idx_map[u]]
            y_full[labeled_users] = y[idx_map[labeled_users]].float()
        y_u = y_full[u_idx].float()

    # Cu: [E_sel, K], Ti: [E_sel, K]
    Cu = C[u_idx]       # 用户社区向量
    Ti = T[i_idx]       # 商品社区向量

    # Obs_{k,l} = sum_e C[u,k] * y_u * T[i,l]
    # => (K,K) = (Cu.T * y_u) @ Ti
    Obs = torch.matmul((Cu.T * y_u), Ti)   # [K, K]
    Obs_norm = torch.sigmoid(Obs)
    # Obs_norm = Obs
    # # 归一化：全局归一化（sum -> 1）
    # total = Obs.sum()
    # if total.item() == 0:
    #     Obs_norm = Obs  # 全零的情况保持全零
    # else:
    #     Obs_norm = Obs / (total + EPS)

    diagnostics = {
        'Obs': Obs.detach(),
        'Obs_norm': Obs_norm.detach(),
        'used_edges_count': u_idx.numel()
    }
    return Obs_norm, diagnostics


def community_game_loss(
        C,  # [num_users, K] 用户社区归属概率
        Obs_norm,  # [K, K] 观测欺诈流强度 (来自 compute_obs)
        P,  # [num_users, num_users] 用户间合作概率矩阵
        y=None,  # [num_labeled_users] 欺诈标签
        y_mask=None,  # [num_users] bool，有标签用户掩码
        fraud_threshold=0.5  # 划分欺诈社区阈值
):
    """
    在所有用户范围内计算社区层面的博弈效能（不依赖用户间边索引）。

    计算流程：
    1. 根据用户间合作概率矩阵 edge_probs_all，计算社区间期望合作矩阵 E_comm；
       E_comm[k,l] = sum_{i,j} p_ij * C[i,k] * C[j,l]
    2. 根据真实标签（y, y_mask）计算社区欺诈率；
    3. 区分正常/欺诈社区；
    4. 计算：
       - coop_eff: normal-normal 社区间平均合作强度；
       - fraud_signal: sum(Obs_norm * E_comm_norm)
    5. 形成博弈损失：L = -log((coop_eff + eps) / (fraud_signal + eps))
    """
    device = C.device
    N, K = C.size()
    EPS = 1e-9
    # 1️⃣ 计算社区间期望合作矩阵 E_comm
    # E_comm = C^T * P * C
    # E_comm = torch.sigmoid(E_comm)
    E_comm = torch.matmul(C.T, torch.matmul(P, C))  # [K, K]
    E_comm = (E_comm + E_comm.T) / 2.0  # 对称化
    # E_comm_norm = E_comm
    # total = E_comm.sum()
    # E_comm_norm = E_comm / (total + EPS)
    E_comm_norm = torch.sigmoid(E_comm)

    # # 2️⃣ 基于有标签用户计算社区欺诈率
    # if y is not None and y_mask is not None and y_mask.sum() > 0:
    #     C_labeled = C[y_mask]  # [n_labeled, K]
    #     y_labeled = y.float().to(device)
    #     num = (C_labeled * y_labeled.unsqueeze(1)).sum(dim=0)
    #     denom = C_labeled.sum(dim=0).clamp(min=EPS)
    #     fraud_rate = num / denom
    # else:
    #     fraud_rate = torch.zeros(K, device=device)
    #
    # # 3️⃣ 区分欺诈 / 正常社区
    # fraud_mask = (fraud_rate > fraud_threshold).float()
    # normal_mask = 1.0 - fraud_mask
    # normal_pairs = (normal_mask.unsqueeze(1) * normal_mask.unsqueeze(0)).sum()

    # 4️⃣ 计算效能指标
    # coop_eff = (E_comm_norm * (normal_mask.unsqueeze(1) * normal_mask.unsqueeze(0))).sum() / (normal_pairs + EPS)
    coop_eff = E_comm_norm.diag().sum()/E_comm_norm.sum()
    fraud_signal = (Obs_norm * E_comm_norm).sum()/E_comm_norm.sum()

    # 5️⃣ 对比式博弈损失
    loss_game = - torch.log(coop_eff / (fraud_signal+EPS))

    diagnostics = {
        'E_comm': E_comm.detach(),
        'E_comm_norm': E_comm_norm.detach(),
        'coop_eff': coop_eff.detach(),
        'fraud_signal': fraud_signal.detach(),
        'Obs_norm_sum': Obs_norm.sum().detach()
    }
    #        'fraud_rate': fraud_rate.detach(), 'fraud_mask': fraud_mask.detach(),
    return loss_game, diagnostics

# ---------- 最终整合函数 ----------
def total_game_loss(edge_index, edge_probs, edge_index_all, P,
                    C, y, y_mask, num_items,
                    lambda_comm=1.0, y_pre=None):
    """
    edge_index, edge_probs: 用于边级博弈（通常为用户-用户边）
    edge_index_all, edge_probs_all: 用于社区级（可以是完整二分图）
    C: 社区分布矩阵 (n, K)
    y, y_mask: 节点标签和掩码
    返回: 总损失 scalar, 以及可选的诊断信息 dict
    """
    device = C.device
    edge_loss,_ = edge_game_loss(edge_index, edge_probs, y, y_mask)

    T = compute_T_from_edges(C, edge_index_all, num_items)

    obs_norm, _ = compute_obs(C,edge_index_all, T, y, y_mask,y_pre=y_pre)

    comm_loss, comm_infos = community_game_loss(C, obs_norm, P)

    total = edge_loss + lambda_comm * comm_loss
    info = {
        'edge_loss': edge_loss.detach(),
        'comm_loss': comm_loss.detach(),
        **comm_infos
    }
    return total, info


