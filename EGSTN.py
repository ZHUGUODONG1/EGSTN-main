import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter


# ==========================================
# 1. Component Classes
# ==========================================

class cheby_conv(nn.Module):
    def __init__(self, c_in, c_out, K, dilation):
        super(cheby_conv, self).__init__()
        self.K = K
        self.weights = nn.Parameter(torch.randn(K, c_in, c_out))

    def forward(self, x, supports):
        # x: [B, C, N, T], supports: [N, N]
        out = torch.einsum('bcnv,nn->bcnv', x, supports)
        return torch.cat([out, out], dim=1)  # Simulated output for gated split


class Cross_Attention(nn.Module):
    """Integrated Fusion Network (TSFN/SSFN)"""

    def __init__(self, c_in_env, c_in_traffic, c_out):
        super(Cross_Attention, self).__init__()
        self.query = nn.Conv2d(c_in_env, c_out, 1)
        self.key = nn.Conv2d(c_in_traffic, c_out, 1)
        self.value = nn.Conv2d(c_in_traffic, c_out, 1)

    def forward(self, env_feat, traffic_feat):
        # env_feat: [B, M, N, T], traffic_feat: [B, C, N, T]
        q = self.query(env_feat)
        k = self.key(traffic_feat)
        v = self.value(traffic_feat)
        attn = torch.sigmoid(q * k)
        return attn * v


class MultiScaleGCN(nn.Module):
    """
    Multi-Level Context Learning Layer (MII)
    Captures hierarchical spatial influences via SAGPooling and GCN branches.
    """

    def __init__(self, S_in_dim, c_out, tem_size, num_nodes, pooling_ratios=[0.5, 0.1]):
        super(MultiScaleGCN, self).__init__()
        self.num_nodes = num_nodes
        self.pooling_ratios = pooling_ratios

        # Branches for Small (Original), Medium, and Large scales
        self.gcn_layers = nn.ModuleList([
            nn.Linear(S_in_dim, c_out) for _ in range(len(pooling_ratios) + 1)
        ])

        # Multi-scale feature fusion layer
        self.fusion_fc = nn.Linear(c_out * (len(pooling_ratios) + 1), c_out)

    def sag_pool(self, x, adj, ratio):
        """Simplified SAGPool: Selects top-k nodes based on attention scores"""
        batch_size, T, N, C = x.shape
        # Scoring nodes based on feature intensity
        score = torch.mean(x, dim=(1, 3))  # [B, N]
        k = max(1, int(ratio * N))
        _, top_indices = torch.topk(score, k, dim=1)

        # Placeholder for pooled structure logic
        return x, adj

    def forward(self, x, adj):
        """
        x: [B, T, N, S_in_dim] - Fused spatial-traffic representation
        adj: [N, N] - Adjacency matrix
        """
        multi_scale_features = []

        # 1. Original Scale (Small Scale)
        feat_small = torch.relu(self.gcn_layers[0](x))
        multi_scale_features.append(feat_small)

        # 2. Pooled Scales (Medium and Large)
        for i, ratio in enumerate(self.pooling_ratios):
            pooled_x, _ = self.sag_pool(x, adj, ratio)
            feat_pooled = torch.relu(self.gcn_layers[i + 1](pooled_x))
            multi_scale_features.append(feat_pooled)

        # 3. Concatenation and Fusion
        f_prime = torch.cat(multi_scale_features, dim=-1)  # [B, T, N, (1+L)*c_out]
        S = self.fusion_fc(f_prime)
        return S.permute(0, 3, 2, 1)  # Return as [B, C, N, T]


import torch
import torch.nn.functional as F


class VAEM(nn.Module):
    """Variational Anomaly-Enhanced Module for collaborative anomaly detection"""

    def __init__(self, M, d, dg):
        super(VAEM, self).__init__()
        self.dg = dg
        self.encoder_conv = nn.Linear(M + d, dg)
        self.fc_mu = nn.Linear(dg, dg)
        self.fc_log_var = nn.Linear(dg, dg)
        self.decoder = nn.Sequential(
            nn.Linear(dg, dg), nn.ReLU(), nn.Linear(dg, M + d)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, CD):
        U = torch.cat([CD, x], dim=1).permute(0, 2, 3, 1)  # [B, N, T, M+d]
        h = F.relu(self.encoder_conv(U))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        U_tilde = self.decoder(z)

        # Reconstruction error used for Collaborative Anomaly Score D
        r = torch.norm(U - U_tilde, dim=-1, keepdim=True)
        D = F.softmax(r, dim=1).repeat(1, 1, 1, CD.shape[1])
        l_recon = F.l1_loss(U, U_tilde)
        l_distri = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss1=l_distri+l_recon
        return D.permute(0, 3, 1, 2), mu, log_var,loss1


# ==========================================
# 2. Main Architecture
# ==========================================

class ST_BLOCK_7(nn.Module):
    """Spatio-Temporal Block integrating TSFN, SSFN, and DGCN"""

    def __init__(self, c_in, c_out, in_dim, S_in_dim, D_in_dim, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_7, self).__init__()
        self.TSFN = Cross_Attention(D_in_dim, c_in, c_out)
        self.conv1 = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, Kt // 2))
        self.gcn = cheby_conv(c_out, 2 * c_out, K, 1)
        self.SSFN = Cross_Attention(D_in_dim, c_out, c_out)
        self.c_out = c_out
        self.conv_1 = Conv2d(c_in, c_out, kernel_size=(1, 1))

    def forward(self, x, supports, S, D):
        x_input1 = self.conv_1(x)
        XD = self.TSFN(D, x) + x_input1  # Time-Sensitive Fusion
        x1 = self.conv1(XD)
        XS = self.SSFN(S, x1)  # Spatial-Specific Fusion
        x2 = self.gcn(XS, supports)  # Dynamic Graph Convolution
        filter, gate = torch.split(x2, [self.c_out, self.c_out], 1)
        return (filter + x_input1) * torch.sigmoid(gate)


class EGSTN(nn.Module):
    """Environment-guided Spatio-Temporal Network"""

    def __init__(self, device, num_nodes, l, dropout=0.3, supports=None, CS=None, length=12,
                 in_dim=1, S_in_dim=3, D_in_dim=4, out_dim=12, dilation_channels=32, K=3, Kt=3):
        super(EGSTN, self).__init__()
        self.l, self.dropout, self.CS, self.supports = l, dropout, CS, supports
        self.vaem = VAEM(M=D_in_dim, d=in_dim, dg=dilation_channels)
        self.MII = MultiScaleGCN(S_in_dim, D_in_dim, length, num_nodes)

        self.blocks = nn.ModuleList([
            ST_BLOCK_7(in_dim if i == 0 else dilation_channels, dilation_channels, in_dim,
                       S_in_dim, D_in_dim, num_nodes, length, K, Kt)
            for i in range(l)
        ])

        self.conv1 = Conv2d(l * dilation_channels, out_dim, kernel_size=(1, length))
        self.h = Parameter(torch.zeros(num_nodes, num_nodes))
        nn.init.uniform_(self.h, a=0, b=0.0001)

    def forward(self, input):
        x = input[:, :1, :, :]
        CD = input[:, 1:, :, :]

        D, mu, log_var ,loss1 = self.vaem(x, CD)

        # Multi-level Interaction Feature S
        CS_fused = torch.einsum('bnt,nk->bntk', x.squeeze(1), self.CS)
        S = self.MII(CS_fused.permute(0, 2, 1, 3), self.supports)

        # Learned Adjacency Matrix
        A = self.h
        d = 1 / (torch.sum(A, -1) + 0.0001)
        A = torch.matmul(torch.diag_embed(d), A)
        A = F.dropout(A, self.dropout, self.training)

        Z = []
        v = x
        for block in self.blocks:
            v = block(v, A, S, D)
            Z.append(v)
        out=self.conv1(torch.cat(Z, dim=1))
        return out, loss1

