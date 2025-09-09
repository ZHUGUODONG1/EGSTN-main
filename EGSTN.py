import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  Conv2d,  Parameter
from utils import cheby_conv,Coss_Attention
from utils import DelayEstimators,MultiScaleGCN

class ST_BLOCK_7(nn.Module):
    def __init__(self,c_in,c_out,in_dim,S_in_dim,D_in_dim,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_7,self).__init__()
        self.DE = DelayEstimators(tem_size, c_in,in_dim,  D_in_dim, c_out)
        self.MII = MultiScaleGCN(S_in_dim, c_out, tem_size, num_nodes)
        self.conv_pro=Conv2d(c_in, in_dim, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

        self.TSFN=Coss_Attention(c_in,c_out)

        self.conv1=Conv2d(c_out, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out,2*c_out,K,1)
        self.SSFN = Coss_Attention(c_out, c_out)

        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)

    def forward(self,x, supports, CD, CS, adj):
        Trend=x
        D = self.DE(Trend, CD)
        u=self.conv_pro(x)
        CS = torch.einsum('ikl,ke->ikel', u.squeeze(1), CS).permute(0, 2, 1, 3)
        S = self.MII(CS, adj)
        x_input1=self.conv_1(x)
        XD=self.TSFN(D,x)+x_input1
        x1=self.conv1(XD)
        XS=self.SSFN(S,x1)
        x2=self.gcn(XS,supports)
        filter,gate=torch.split(x2,[self.c_out,self.c_out],1)
        x=(filter+x_input1)*torch.sigmoid(gate)
        return x

class EGSTN(nn.Module):
    def __init__(self, device, num_nodes, l, dropout=0.3, supports=None,CS=None, length=12,
                 in_dim=1, S_in_dim=3 ,D_in_dim=4, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3):
        super(EGSTN, self).__init__()
        tem_size = length
        self.num_nodes = num_nodes
        self.l = l
        self.dropout=dropout
        self.block = nn.ModuleList()
        self.CS=CS

        self.block1 = ST_BLOCK_7(in_dim, dilation_channels, in_dim, S_in_dim,D_in_dim, num_nodes, tem_size, K, Kt)
        for i in range(l - 1):
            self.block.append(ST_BLOCK_7(dilation_channels, dilation_channels,in_dim, S_in_dim ,D_in_dim, num_nodes, tem_size, K, Kt))

        self.conv1 = Conv2d(dilation_channels, 12, kernel_size=(1, tem_size), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.afm=Conv2d(l*dilation_channels, l*dilation_channels, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=True)
        self.supports = supports
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self, input):
        x=input[:,:1,:,:]
        CD=input[:,1:,:,:]
        A = self.h
        d = 1 / (torch.sum(A, -1) + 0.0001)
        D = torch.diag_embed(d)
        A = torch.matmul(D, A)
        A = F.dropout(A, self.dropout, self.training)
        Z=[]
        v = self.block1(x,A , CD, self.CS, self.supports)
        Z.append(v)
        for i in range(self.l - 1):
            v = self.block[i](v, A, CD, self.CS, self.supports)
            Z.append(v)
        Z=torch.cat(Z, dim=1)
        a=self.afm(Z)
        Z_split = torch.split(Z, Z.shape[1] // self.l, dim=1)
        a_split = torch.split(a, a.shape[1] // self.l, dim=1)
        dot_products = []
        for a_part, z_part in zip(a_split, Z_split):
            dot_products.append(a_part * z_part)  # 对 dim=1 做点积
        f = torch.sum(torch.stack(dot_products), dim=1)/self.l
        x = self.conv1(f)
        return x
#
# D_in_dim=4
# seq_length=12
# num_nodes=524
# batch_size=4
# in_dim=1
# l=3
# dropout=0.5
# S_in_dim=3
# nhid=32
# device='cpu'
#
# x=torch.randn(batch_size,in_dim+D_in_dim,num_nodes,seq_length)
# supports=torch.randn(num_nodes,num_nodes)
# CS=torch.randn(num_nodes,3)
# model=EGSTN(device, num_nodes,l, dropout, supports=supports, batch_size=batch_size,
#                            in_dim=in_dim,  S_in_dim=S_in_dim, D_in_dim=D_in_dim, out_dim=seq_length,
#                            residual_channels=nhid, dilation_channels=nhid,
#                            skip_channels=nhid * 8, end_channels=nhid * 16)
# out=model(x,CS)
# print(out.shape)