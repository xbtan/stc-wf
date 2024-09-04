import torch
from torch_geometric.data import InMemoryDataset,Data
from tqdm import tqdm
from utils import *

class trainTrafficDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(trainTrafficDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return 'train_data.pt'
 
    def download(self):
        pass
    
    def process(self):
        data_list = []
        # process by session_id
        # graphs, num_classes = load_data('data.csv')
        # graphs, _ = load_data('train_enc_data.csv')
        graphs, _ = load_data('train_data.csv')
        for graph in tqdm(graphs):
            # node_features =  torch.FloatTensor(graph.node_features)
            # graph.node_features=np.array(graph.node_features,dtype=np.float32)
            # node_features = torch.FloatTensor(node_features).unsqueeze(1)
            # x = torch.FloatTensor(np.array(graph.node_features))
            x = torch.FloatTensor(graph.node_features)
            y = torch.LongTensor([graph.label-1])
            
            # edges = [list(pair) for pair in graph.g.edges()]
            # edges.extend([[i, j] for j, i in edges])
            # m=[]
            # n=[]
            # for edge in edges:
            #     m.append(int(edge[0]))
            #     n.append(int(edge[1]))
            # edge_index=[m,n]
            # # print(edge_index)
            # edge_index = torch.tensor(edge_index, dtype=torch.long)

            edge_index = torch.LongTensor(graph.edge_mat)

            edge_attr = torch.FloatTensor(graph.edge_attr)

            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=y)
            # print(data.edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class validTrafficDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(validTrafficDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return 'valid_data.pt'
 
    def download(self):
        pass
    
    def process(self):
        data_list = []
        # process by session_id
        # graphs, num_classes = load_data('data.csv')
        # graphs, _ = load_data('valid_enc_data.csv')
        graphs, _ = load_data('valid_data.csv')
        for graph in tqdm(graphs):
            # node_features =  torch.FloatTensor(graph.node_features)
            # graph.node_features=np.array(graph.node_features,dtype=np.float32)
            # node_features = torch.FloatTensor(node_features).unsqueeze(1)
            # x = torch.FloatTensor(np.array(graph.node_features))
            x = torch.FloatTensor(graph.node_features)
            y = torch.LongTensor([graph.label-1])
            
            # edges = [list(pair) for pair in graph.g.edges()]
            # edges.extend([[i, j] for j, i in edges])
            # m=[]
            # n=[]
            # for edge in edges:
            #     m.append(int(edge[0]))
            #     n.append(int(edge[1]))
            # edge_index=[m,n]
            # # print(edge_index)
            # edge_index = torch.tensor(edge_index, dtype=torch.long)

            edge_index = torch.LongTensor(graph.edge_mat)

            edge_attr = torch.FloatTensor(graph.edge_attr)

            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=y)
            # print(data.edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class testTrafficDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(testTrafficDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return 'test_data.pt'
 
    def download(self):
        pass
    
    def process(self):
        data_list = []
        # process by session_id
        # graphs, num_classes = load_data('data.csv')
        # graphs, _ = load_data('test_enc_data.csv')
        graphs, _ = load_data('test_data.csv')
        for graph in tqdm(graphs):
            # node_features =  torch.FloatTensor(graph.node_features)
            # graph.node_features=np.array(graph.node_features,dtype=np.float32)
            # node_features = torch.FloatTensor(node_features).unsqueeze(1)
            # x = torch.FloatTensor(np.array(graph.node_features))
            x = torch.FloatTensor(graph.node_features)
            y = torch.LongTensor([graph.label-1])
            
            # edges = [list(pair) for pair in graph.g.edges()]
            # edges.extend([[i, j] for j, i in edges])
            # m=[]
            # n=[]
            # for edge in edges:
            #     m.append(int(edge[0]))
            #     n.append(int(edge[1]))
            # edge_index=[m,n]
            # # print(edge_index)
            # edge_index = torch.tensor(edge_index, dtype=torch.long)

            edge_index = torch.LongTensor(graph.edge_mat)

            edge_attr = torch.FloatTensor(graph.edge_attr)

            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr,y=y)
            # print(data.edge_attr)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])