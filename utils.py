# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 13:20:23 2018

@author: gk
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, Conv3d, ModuleList, Parameter, LayerNorm, BatchNorm1d, BatchNorm3d

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""


##gwnet
# class nconv(nn.Module):
#     def __init__(self):
#         super(nconv,self).__init__()

#     def forward(self,x, A):
#         A=A.transpose(-1,-2)
#         x = torch.einsum('ncvl,vw->ncwl',(x,A))


class DelayEstimators(nn.Module):
    def __init__(self, tem_size=12,c_in=32, in_dim=1, S_in_dim=4, dilation_channels=32):
        super(DelayEstimators, self).__init__()
        self.out_channels=dilation_channels

        # Initialize convolution layers
        self.conv_trend_Q = nn.Conv2d(c_in, dilation_channels, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1),
                                    bias=True)
        self.conv_layers_K = nn.ModuleList(
            [nn.Conv2d(in_dim, dilation_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True) for _
             in range(S_in_dim)])

        self.conv_layers_V = nn.ModuleList(
            [nn.Conv2d(in_dim, dilation_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True) for _
             in range(S_in_dim)])
        self.scale= S_in_dim ** -0.5
    def forward(self, Trend, CD):
        """
        Forward pass of the Delay Estimators model.
        Args:
            Trned (Tensor): Tensor of shape [32, 1, 100, 13] - current traffic.
            C (Tensor): Tensor of shape [32, 4, 100, 13] - factors.
        Returns:
            Tensor: The final aggregated output after applying convolutions, attention, and averaging.
        """
        # Apply convolution to the trend input (current traffic)
        B, C, N, T = CD.shape
        Q = self.conv_trend_Q(Trend).reshape(B, self.out_channels, -1).permute(0, 2, 1)
        # Split C along the 2nd dimension (dim=1) into 4 parts
        CD_split = torch.split(CD, 1, dim=1)
        agg = None
        # Apply convolution and attention mechanism on each split of C
        for i in range(C):
            K = self.conv_layers_K[i](CD_split[i]).reshape(B, self.out_channels, -1)  # Apply convolution to the ith split
            V = self.conv_layers_V[i](CD_split[i]).reshape(B, self.out_channels, -1).permute(0, 2, 1)
            atten_wights =  F.softmax(torch.bmm(Q, K) * self.scale,dim=-1)
            # Compute attention weights (softmax)
            prod = torch.bmm(atten_wights, V).permute(0, 2, 1).reshape(B, self.out_channels, N, T)
            # Accumulate the product in agg
            if agg is None:
                agg = prod
            else:
                agg += prod
        # Take the average of the accumulated product
        agg = agg / C
        return agg


# 通道注意力机制模块 (CA)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.global_avg_pool(x))))
        return self.sigmoid(avg_out) * x


class Coss_Attention(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Coss_Attention, self).__init__()
        self.query = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.scale = in_channels ** -0.5
        self.out_channels=out_channels

    def forward(self, x ,y):
        B, C, N, T = x.shape
        Q = self.query(x).reshape(B, self.out_channels, -1).permute(0, 2, 1)
        K = self.key(y).reshape(B, self.out_channels, -1)
        V = self.value(y).reshape(B, self.out_channels, -1).permute(0, 2, 1)
        attn = torch.bmm(Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V).permute(0, 2, 1).reshape(B, self.out_channels, N, T)
        return out  # 残差连接

# 自注意力机制 (SAT)
class SelfAttention(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.scale = in_channels ** -0.5
        self.out_channels=out_channels

    def forward(self, x):
        B, C, N, T = x.shape  # Batch size, Channels, Nodes, Time steps
        Q = self.query(x).reshape(B, self.out_channels, -1).permute(0, 2, 1)
        K = self.key(x).reshape(B, self.out_channels, -1)
        V = self.value(x).reshape(B, self.out_channels, -1).permute(0, 2, 1)
        attn = torch.bmm(Q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, V).permute(0, 2, 1).reshape(B, self.out_channels, N, T)
        return out  # 残差连接


# 图卷积层 (GCN)
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = out.permute(0, 2, 3, 1)
        out = torch.matmul(out, self.weight)
        out = out.permute(0, 3, 1, 2)
        return out

class GCN1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN1, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        return out

# SAG Pooling 模块
class SAGPool(nn.Module):
    def __init__(self, in_channels, length, pool_ratio=0.5):
        super(SAGPool, self).__init__()
        self.gcn = GCN1(in_channels*length,1)  # 用于计算节点的重要性分数
        self.sigmoid = nn.Sigmoid()
        self.pool_ratio = pool_ratio

    def forward(self, x, adj):
        x=x.permute(0,2,3,1)
        B,N,T,F=x.shape
        x=x.reshape(B,N,T*F)
        score = self.gcn(x, adj).squeeze()
        score = self.sigmoid(score)

        num_nodes = N
        num_pool = int(self.pool_ratio * num_nodes)

        _, top_idx = torch.topk(score, num_pool, dim=-1)
        x_new = torch.gather(x, 1, top_idx.unsqueeze(-1).expand(-1, -1, T*F)).reshape(B,num_pool,T,F).permute(0,3,1,2)
        return x_new


# 多尺度图卷积网络 (使用 SAG Pooling)
class MultiScaleGCN(nn.Module):
    def __init__(self, in_channels, out_channels, length, num_nodes, pool_ratio1=0.5,pool_ratio2=0.1,eb_dim=10):
        super(MultiScaleGCN, self).__init__()
        self.gcn1 = GCN(in_channels, out_channels)
        self.gcn2 = GCN(in_channels, out_channels)
        self.gcn3 = GCN(in_channels, out_channels)
        self.channel_attention1 = ChannelAttention(out_channels)
        self.channel_attention2 = ChannelAttention(out_channels)
        self.channel_attention3 = ChannelAttention(out_channels)
        self.pool1 = SAGPool(in_channels,length, pool_ratio=pool_ratio1)  # SAG Pooling
        self.pool2 = SAGPool(in_channels,length, pool_ratio=pool_ratio2)  # SAG Pooling
        self.self_attention = SelfAttention(3*out_channels,out_channels)
        self.E1 = nn.Parameter(torch.randn(num_nodes, eb_dim))  # 动态矩阵 E1
        self.E2 = nn.Parameter(torch.randn(eb_dim, num_nodes))  # 动态矩阵 E2
        self.E11 = nn.Parameter(torch.randn(int(num_nodes*pool_ratio1), eb_dim))  # 动态矩阵 E1
        self.E21 = nn.Parameter(torch.randn(eb_dim, int(num_nodes*pool_ratio1)))  # 动态矩阵 E2
        self.E12 = nn.Parameter(torch.randn(int(num_nodes*pool_ratio2), eb_dim))  # 动态矩阵 E1
        self.E22 = nn.Parameter(torch.randn(eb_dim, int(num_nodes*pool_ratio2)))  # 动态矩阵 E2

    def forward(self, x,adj):
        B, T, N, C = x.shape

        adj1 = torch.matmul(self.E1, self.E2)
        adj2 = torch.matmul(self.E11, self.E21)
        adj3 = torch.matmul(self.E12, self.E22)

        out1 = self.gcn1(x, adj1)
        out1 = self.channel_attention1(out1)

        out2 = self.pool1(x, adj)
        out2 = self.gcn2(out2, adj2)
        out2 = self.channel_attention2(out2)

        out3 = self.pool2(x, adj)
        out3 = self.gcn3(out3, adj3)
        out3 = self.channel_attention3(out3)

        # 上采样到原始大小
        out2_up = F.interpolate(out2, size=(out1.shape[2], out1.shape[3]), mode='bilinear', align_corners=True)
        out3_up = F.interpolate(out3, size=(out1.shape[2], out1.shape[3]), mode='bilinear', align_corners=True)

        # 拼接多尺度结果
        multi_scale_out = torch.cat([out1, out2_up, out3_up], dim=1)

        # 自注意力机制
        final_out = self.self_attention(multi_scale_out)

        return final_out


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        dims = A.dim()
        A = A.transpose(-1, -2)

        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            raise NotImplementedError('PGCN not implemented for A of dimension ' + str(dims))

        return x.contiguous()



class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(cheby_conv, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out


