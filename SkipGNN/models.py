import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ResGCN, GraphConvolution
from torch.nn.parameter import Parameter
import math

    
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class ResidualGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(ResidualGCN, self).__init__()
        
        
        # original graph
        self.gcn1 = ResGCN( nfeat, nhid1,is_first=True)
        self.gcn2 = ResGCN(nhid1, nhid2)
        
        self.dropout = dropout
        
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def forward(self, x, adj, idx):
        x1,m1=self.gcn1(x,adj,None)
        
        x1=F.relu(x1)
        # m1=F.relu(m1)
        # x1=F.dropout(x1, self.dropout, training=self.training)
        # m1=F.dropout(m1)
        
        x2,m2=self.gcn2(x1,adj,m1)
        x2=F.relu(x2)
        
        x = F.dropout(x2, self.dropout, training = self.training)
        
        feat_p1 = x[idx[0]] 
        feat_p2 = x[idx[1]] 
        feat = torch.cat((feat_p1, feat_p2), dim = 1)
        o = self.decoder1(feat)
        o = self.decoder2(o)
        return o, x    
    
    def embed(self,x,adj,idx):
        x1,m1=self.gcn1(x,adj,None)
        
        x1=F.relu(x1)
        # m1=F.relu(m1)
        # x1=F.dropout(x1, self.dropout, training=self.training)
        # m1=F.dropout(m1)
        
        x2,m2=self.gcn2(x1,adj,m1)
        x2=F.relu(x2)
        
        return x2

class SkipGNN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid_decode1, dropout):
        super(SkipGNN, self).__init__()
        
        # original graph
        self.o_gc1 = GraphConvolution(nfeat, nhid1)
        self.o_gc2 = GraphConvolution(nhid1, nhid2)
        
        # original graph for skip update
        self.o_gc1_s = GraphConvolution(nhid1, nhid1)
        
        #skip graph
        self.s_gc1 = GraphConvolution(nfeat, nhid1)
        
        #skip graph for original update
        self.s_gc1_o = GraphConvolution(nfeat, nhid1)
        self.s_gc2_o = GraphConvolution(nhid1, nhid2)
       
        self.dropout = dropout
        
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def forward(self, x, o_adj, s_adj, idx):
        
        o_x = F.relu(self.o_gc1(x, o_adj) + self.s_gc1_o(x, s_adj))       
        s_x = F.relu(self.s_gc1(x, s_adj) + self.o_gc1_s(o_x, o_adj))
        
        o_x = F.dropout(o_x, self.dropout, training = self.training)
        s_x = F.dropout(s_x, self.dropout, training = self.training)
        
        x = self.o_gc2(o_x, o_adj) + self.s_gc2_o(s_x, s_adj)
        
        feat_p1 = x[idx[0]] # the first biomedical entity embedding retrieved
        feat_p2 = x[idx[1]] # the second biomedical entity embedding retrieved
        feat = torch.cat((feat_p1, feat_p2), dim = 1)
        o = self.decoder1(feat)
        o = self.decoder2(o)
        return o, x