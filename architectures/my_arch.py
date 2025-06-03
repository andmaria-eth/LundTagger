import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, BatchNorm
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import SAGEConv, ChebConv, GatedGraphConv,GATConv

    
class LundNetTagger(nn.Module):

    def __init__(self, dim,n_out):
        super().__init__()
        self.dim = dim
        rate = 0.1

        nns1 = nn.Sequential(
            nn.Linear(2*dim,128),
            GraphNorm(128),
            nn.ReLU(),
            nn.Linear(128,128),
            GraphNorm(128),
            nn.ReLU(),
            nn.Linear(128,128),
            GraphNorm(128),
            nn.ReLU(),
        )
        self.conv1 = EdgeConv(nns1,aggr='mean')

        nns2 = nn.Sequential(
            nn.Linear(2*128,256),
            GraphNorm(256),
            nn.ReLU(),
            nn.Linear(256,256),
            GraphNorm(256),
            nn.ReLU(),
        )
        self.conv2 = EdgeConv(nns2,aggr='mean')

        nns3 = nn.Sequential(
            nn.Linear(256*2,256),
            GraphNorm(256),
            nn.ReLU()
        )
        self.conv3 = EdgeConv(nns3,aggr='mean')
        self.lin = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(rate),
            nn.Linear(256,n_out),
        )
        self.drop = nn.Dropout(rate)
        self.relu = nn.ReLU()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x,edge_index)
        x = self.drop(x)
        x = self.conv2(x,edge_index)
        x = self.drop(x)
        x = self.conv3(x,edge_index)
        x = self.drop(x)
        x = global_mean_pool(x,batch=data.batch)
        x = self.lin(x)
        return x