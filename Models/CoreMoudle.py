from __future__ import print_function, division

import torch
import torch.nn as nn

from torch.nn import Linear, Parameter

import torch.nn.functional as F

from Models.GNN import GCN_NET, SAGE_NET


class PairGameHead(nn.Module):
    """
    PairGameHead — 对边（edge）进行成对特征预测（0/1 背叛/合作）。
    只对 edge_index 上的边计算 (稀疏计算)，节省内存与计算。

    输入:
        x: [N, F] 节点特征（可包含 fraud_prob 或不包含）
        edge_index: [2, E] (src, dst)
        edge_attr: optional [E, D] 边属性
        edge_type: optional [E] 整数类型（0..T-1），用于索引不同 payoffs
        fraud_prob: optional [N] (0..1)，若没有拼进 x，则会在成对特征中追加
    超参数:
        pair_mode: 'concat'|'abs_diff'|'both' 等，成对特征构造方式
        threshold: 二分类阈值 (默认 0.5)
    输出 (dict):
        edge_logits: [E]
        edge_probs: [E]
        edge_actions: [E] (0=合作,1=背叛)  —— **确定性**，无采样
        edge_payoff_src: [E]
        edge_payoff_dst: [E]
        edge_payoff_total: [E]
    """
    def __init__(
        self,
        in_feats,
        hidden=32,
        pair_mode="both",
        threshold=0.4,
        num_types=1
    ):
        super().__init__()
        self.pair_mode = pair_mode
        self.threshold = threshold
        self.num_types = max(1, num_types)

        # pair MLP: 输入维度按照 pair_mode 自动计算
        # 可能的输入项: x_src, x_dst, |x_src-x_dst|, x_src * x_dst, edge_attr, fraud probs
        # 用户需确保 in_feats 对应单个节点特征维度
        # 我构建一个灵活的 MLP：先用 concat(src,dst,absdiff,prod) 再降维
        pair_in_dim = in_feats * 2
        if pair_mode in ("both",):
            pair_in_dim += in_feats  # abs diff
            pair_in_dim += in_feats  # elementwise product
        elif pair_mode in ("abs_diff",):
            pair_in_dim = in_feats
        # 如果 user 会传 edge_attr，我们会在 forward 时拼接（动态）
        self.mlp = nn.Sequential(
            nn.Linear(pair_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        # payoff scalars per type: 为简单起见，这里用标量向量按 type 索引
        T = self.num_types
        def _init_param(arr, default):
            if arr is None:
                return torch.tensor([default] * T, dtype=torch.float32)
            a = torch.as_tensor(arr, dtype=torch.float32)
            if a.numel() == 1:
                a = a.repeat(T)
            assert a.numel() == T, "payoff arrays must match num_types or be scalar"
            return a

        self.game_loss_value = None

    def _build_pair_feature(self, x_src, x_dst, pair_mode):
        # x_src/x_dst: [E, F]
        if pair_mode == "concat":
            return torch.cat([x_src, x_dst], dim=1)
        elif pair_mode == "abs_diff":
            return torch.abs(x_src - x_dst)
        elif pair_mode == "both":
            # concat(src, dst, absdiff, prod)
            return torch.cat([x_src, x_dst, torch.abs(x_src - x_dst), x_src * x_dst], dim=1)
        else:
            raise ValueError("Unknown pair_mode: " + str(pair_mode))


    def forward(self, x, edge_index, edge_index_global, edge_attr=None, edge_type=None, C=None, fraud_label=None,fraud_label_mask=None, edge_index_all = None, num_items = 0, train = True):
        """
        x: [N, F]
        edge_index: [2, E]
        edge_attr: optional [E, D]
        edge_type: optional [E] int
        fraud_prob: optional [N]
        """
        device = x.device
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        # 1) 从节点特征中选取边端点特征（仅对边的索引）
        x_src = x[src_idx]   # [E, F]
        x_dst = x[dst_idx]   # [E, F]

        # 如果 fraud_prob 给出但未拼到 x 中，则把两个端点的 fraud 拼到 pair feature 末尾
        extra_cols = []

        # 若 edge_attr 存在，也拼接进来（按边）
        if edge_attr is not None:
            extra_cols.append(edge_attr.to(device))

        # 构建 pair feature（只在 E 条边上）
        pair_feat = self._build_pair_feature(x_src, x_dst, self.pair_mode)  # [E, P]
        if len(extra_cols) > 0:
            pair_feat = torch.cat([pair_feat] + extra_cols, dim=1)

        # 2) 通过 MLP 得到 edge logits -> prob -> 硬判定 (无采样)
        edge_logits = self.mlp(pair_feat).squeeze(-1)  # [E]
        edge_probs = torch.sigmoid(edge_logits)        # [E]
        # 确定性动作：prob > threshold => 背叛(0)，否则合作(1)
        edge_actions = (edge_probs > self.threshold).float()  # [E] 0/1
        mask = (edge_actions == 1)  # boolean tensor shape [E]

        # 直接按 mask 过滤 edge_index（稀疏操作）
        new_edge_index = edge_index_global[:, mask]
        raw_edge_index = edge_index[:, mask]
        P = None
        if train:
            #求所有用户博弈关系
            N = x.size(0)

            # 1️⃣ 构建所有用户对索引
            u_idx = torch.arange(N, device=device)
            src_idx, dst_idx = torch.meshgrid(u_idx, u_idx, indexing='ij')
            src_idx, dst_idx = src_idx.reshape(-1), dst_idx.reshape(-1)

            # 2️⃣ 构建 pair 特征
            pair_feat = self._build_pair_feature(x[src_idx], x[dst_idx], self.pair_mode)
            if extra_cols is not None and len(extra_cols) > 0:
                pair_feat = torch.cat([pair_feat] + extra_cols, dim=1)

            # 3️⃣ 预测 edge 概率
            edge_logits_all = self.mlp(pair_feat).squeeze(-1)
            edge_probs_all = torch.sigmoid(edge_logits_all)

            # 4️⃣ 转化为 [N, N]
            P = edge_probs_all.view(N, N)


        return {
            "new_edge_index": new_edge_index,
            "raw_edge_index": raw_edge_index,
            "edge_probs": edge_probs,
            "edge_actions": edge_actions,           # 0=双方合作情形, 1=背叛情形（single-value 表示）
            "P": P,
        }


class AE(nn.Module):

    def __init__(self, n_enc, hidden,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_in = Linear(n_input, n_enc)
        self.hidden_enc = nn.ModuleList([Linear(n_enc, n_enc) for i in range(hidden)])
        self.z_layer = Linear(n_enc, n_z)

        self.dec_in = Linear(n_z, n_enc)
        self.hidden_dec = nn.ModuleList([Linear(n_enc, n_enc) for i in range(hidden)])
        self.x_bar_layer = Linear(n_enc, n_input)

    def forward(self, x):
        enc_result = []
        enc_result.append(F.relu(self.enc_in(x)))
        for layer in self.hidden_enc:
            enc_result.append(F.relu(layer(enc_result[-1])))
        z = self.z_layer(enc_result[-1])

        dec = F.relu(self.dec_in(z))
        for layer in self.hidden_dec:
            dec = F.relu(layer(dec))
        x_bar = self.x_bar_layer(dec)

        return x_bar, enc_result, z


class CORE(nn.Module):
    def __init__(self, max_b):
        super(CORE, self).__init__()
        self.weight = Parameter(torch.ones([max_b]))

    def forward(self, x):
        return F.softmax(self.weight * x, dim=1)


class TL(nn.Module):
    def __init__(self, n_clusters):
        super(TL, self).__init__()
        self.to_label1 = Linear(n_clusters, n_clusters // 2)
        self.to_label2 = Linear(n_clusters // 2, 2)

    def forward(self, x):
        x = F.relu(self.to_label1(x))
        x = F.softmax(self.to_label2(x), dim=1)
        return x


class TEL(nn.Module):
    def __init__(self, n):
        super(TEL, self).__init__()
        self.to_edge_label1 = Linear(n, 64)
        self.to_edge_label2 = Linear(64, 2)

    def forward(self, edge_x):
        x = F.relu(self.to_edge_label1(edge_x))
        x = F.softmax(self.to_edge_label2(x), dim=1)
        return x

class DegreeFeatureGuidedAttention(nn.Module):
    def __init__(self, feat_dim, hidden_dim=32):
        super().__init__()
        self.att_mlp = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),  # +1 是度特征
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, x2, degrees):
        """
        x: [N, F] 第一个 view 输出
        x2: [N, F] 第二个 view 输出
        degrees: [N] 节点度
        features: [N, feat_dim] 节点特征（可为原始或某层特征）

        return:
            fused: [N, F] 融合后的特征
            alpha: [N] 注意力权重
        """
        # 1️⃣ 度特征
        deg_feat = degrees.unsqueeze(-1)

        # 2️⃣ 拼接度 + 特征
        att_input = torch.cat([deg_feat, x], dim=-1)  # [N, feat_dim + 1]
        att_input2 = torch.cat([deg_feat, x2], dim=-1)  # [N, feat_dim + 1]

        # 3️⃣ 学习注意力权重
        alpha = torch.sigmoid(self.att_mlp(att_input))  # [N, 1]
        alpha2 = torch.sigmoid(self.att_mlp(att_input2))  # [N, 1]
        alpha_norm = alpha / (alpha + alpha2)
        alpha2_norm = alpha2 / (alpha + alpha2)

        # 4️⃣ 融合两个视图
        fused = alpha_norm * x + alpha2_norm * x2  # [N, F]

        return fused

class Cross_GNN(nn.Module):
    def __init__(self, n_input, args):
        super(Cross_GNN, self).__init__()
        if (args.gnn == 'sage'):
            GNN_NET = SAGE_NET
        else:
            GNN_NET = GCN_NET
        self.gnn_in = GNN_NET(n_input, 32)
        self.gnn_hid = GNN_NET(32, 32)
        self.gnn_out = GNN_NET(32, n_input//2)
        self.fc1 = torch.nn.Linear(n_input//2, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.tau = 0.5
        self.self_loss = None

    def forward(self, x , edge_index_u, edge_index_u2, Train = True, Type = 'A'):
        x1 = self.gnn_in(x, edge_index_u)
        x2 = self.gnn_in(x, edge_index_u2)
        X = F.normalize(x1)
        Y = F.normalize(x2)

        similarities = (X * Y).sum(dim=1)
        # similarities[similarities <= threshold]=0
        mian = x1 + x2 * similarities.view(-1, 1)
        sup = x2 + x1 * similarities.view(-1, 1)
        # x_cat = torch.cat([x1, x2], dim=1)

        # x_cat = torch.cat([x1, x2], dim=1)
        x1 = self.gnn_hid(mian, edge_index_u)
        x2 = self.gnn_hid(sup , edge_index_u2)

        X = F.normalize(x1)
        Y = F.normalize(x2)
        #
        similarities = (X * Y).sum(dim=1)
        # similarities[similarities <= threshold]=0
        mian = x1 + x2 * similarities.view(-1, 1)
        sup = x2 + x1 * similarities.view(-1, 1)
        # x_cat = torch.cat([x1, x2], dim=1)

        x1 = self.gnn_out(mian, edge_index_u)
        x2 = self.gnn_out(sup, edge_index_u)

        x_cat = torch.cat([x1, x2], dim=1)

        if Train and Type == 'C':
            self.self_loss = self.loss(x1,x2)

        return x_cat

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret