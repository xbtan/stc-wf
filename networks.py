import torch
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv,TransformerConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool


class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        


        # self.conv1 = GCNConv(self.num_features, self.nhid)
        # self.conv2 = GCNConv(self.nhid, self.nhid)
        # self.conv3 = GCNConv(self.nhid, self.nhid)
        self.lin=torch.nn.Linear(self.num_features, self.nhid//2)

        self.conv1 = GATConv(self.nhid//2, self.nhid, heads=3,edge_dim=2)
        # self.conv2 = GATv2Conv(self.nhid, self.nhid, heads=3,concat=False,edge_dim=2)
        # self.conv3 = GATv2Conv(self.nhid, self.nhid, heads=3,concat=False,edge_dim=2)
        # self.conv1 = TransformerConv(self.num_features, self.nhid, heads=3,edge_dim=2)
        # self.conv1 = GATv2Conv(self.num_features, self.nhid, heads=3,edge_dim=2)
        # self.conv2 = GATv2Conv(self.nhid, self.nhid, heads=1,edge_dim=2)
        # self.conv3 = GATv2Conv(self.nhid, self.nhid, heads=1,edge_dim=2)
        # self.conv2 = GATConv(self.nhid, self.nhid, heads=3, edge_dim=1)
        # self.conv3 = GATConv(self.nhid, self.nhid, heads=3, edge_dim=1)


        # self.pool = SAGPool(self.nhid*3, ratio=self.pooling_ratio)
        self.pool = SAGPool(self.nhid*3, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*3*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr,data.batch
        x=F.elu(self.lin(x))
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        # x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        # x3 = F.relu(self.conv3(x2, edge_index, edge_attr))


        # x = torch.cat((x1), dim=1)
        # x = torch.cat((x1, x2, x3), dim=1)
        x=x1

        x, edge_index, _, batch, _ = self.pool(x, edge_index, edge_attr, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

