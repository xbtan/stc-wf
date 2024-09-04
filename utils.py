from turtle import begin_fill
from attr import attr
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedKFold
import scipy.sparse as sp
import math

class AssociateGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = []
        self.edge_mat = 0
        self.edge_attr=[]
        self.max_neighbor = 0
        self.graph_indicator=[]

def split_by_item(num_list):
    temp = []
    split_list=[]
    for i in range(len(num_list)):
        if i < len(num_list)-1:
            if num_list[i] != num_list[i+1]:
                temp.append(i+1)
    temp.append(len(num_list))
    for j in range(len(temp)):
        if j == 0:
            split_list.append(num_list[:temp[j]])
        else:
            split_list.append(num_list[temp[j-1]:temp[j]])
    return split_list

def create_graph(ip_locate,features,graph_indicator,begin_time):
    ips=split_by_item(ip_locate)
    G = nx.Graph()
    temp=0
    len_list=[]
    nums_list=[]
    num_list=[]
    edge_attr=[]
    for i in range(len(ips)):
        len_list.append(len(ips[i]))
        for j in range(len(ips[i])):
            G.add_node(temp+j,feature=features[temp+j],graph_indicator=graph_indicator[temp+j])
            num_list.append(temp+j)
            for t in range(j+1,len(ips[i])):
                # G.add_edge(temp+j,temp+t,weight=[1,1])
                G.add_edge(temp+j,temp+t,weight=[math.exp(-abs(begin_time[temp+j]-begin_time[temp+t])),1])
        temp=temp+len(ips[i])
        nums_list.append(num_list)
        num_list=[]
    # print(nums_list)
    for i in range(len(nums_list)):
        if i>0:
            # for x in nums_list[i-1]:
            #     for y in nums_list[i]:
            #         G.add_edge(x,y, weight=1/abs(begin_time[x]-begin_time[y]))
            G.add_weighted_edges_from([(x,y,[math.exp(-abs(begin_time[x]-begin_time[y])),0]) for x in nums_list[i-1] for y in nums_list[i]])
    # nx.draw_networkx(G)
    # plt.show()
    # print(len(G.nodes))
    return G


# df=pd.read_csv('data.csv')
# for m in range(min(df['graph indicator']), max(df['graph indicator']) + 1):
#     data_all=df[df['graph indicator'] == m]
#     data=data_all
#     # data=data_all.loc[list(data_all.index)[0]:list(data_all.index)[0]+M-1,:]
#     graph_indicator=data['graph indicator'].values.tolist()
#     ip_locate = data['IP归属'].values.tolist()
    
#     labels=data['label'].values.tolist()
#     features=data[['up_pkts','down_pkts','up_bytes','down_bytes','up_b_std','down_b_std','inter_arrival_time_std']].values.tolist()
#     label = labels[0]
#     begin_time = data['begin_time'].values.tolist()
#     g = create_graph(ip_locate,features,graph_indicator,begin_time)
# fea_names=['amin','amax','amean','amad','astd','avar','askew','akurt','aper1','aper2','aper3','aper4','aper5','aper6','aper7','aper8','aper9','alen',
#     'umin','umax','umean','umad','ustd','uvar','uskew','ukurt','uper1','uper2','uper3','uper4','uper5','uper6','uper7','uper8','uper9','ulen',
#     'dmin','dmax','dmean','dmad','dstd','dvar','dskew','dkurt','dper1','dper2','dper3','dper4','dper5','dper6','dper7','dper8','dper9','dlen',
#     'inter_arrival_time_std']
# M=15
def load_data(dataset):
# def load_data(dataset,degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    print('Loading data...')
    g_list = []
    # label_dict = {'b站':1,'CSDN':2,'博客园':3,'京东':4,'淘宝':5,'网易云音乐':6,'优酷':7,'知乎':8}
    # label_dict = {'b站首页':1,'CSDN首页':2,'博客园':3,'京东首页':4,'淘宝首页':5,'网易云音乐':6,'优酷视频首页':7,'知乎发现页':8}
    # label_dict = {'b站视频60s':1,'b站视频20s':1,'b站视频5s':1,'b站首页60s':2,'b站首页20s':2,'b站首页5s':2}
    label_dict = {'bilibili':1,'CSDN':2,'QQ音乐':3,'爱奇艺':4,'东方财富':5,'斗鱼':6,
    '豆瓣':7,'凤凰网':8,'虎扑':9,'简书':10,'今日头条':11,'京东':12,'汽车之家':13,
    '搜狐':14,'淘宝':15,'腾讯门户':16,'腾讯视频':17,'网易云音乐':18,'喜马拉雅':19,'新华网':20,'新浪':21}

    stcol=[]
    for i in range(20):
        stcol.append('len'+str(i))

    df=pd.read_csv(dataset, encoding='gbk')
    # df=df.iloc[:5,:]
    for m in range(min(df['graph_indicator']), max(df['graph_indicator']) + 1):

        data_all=df[df['graph_indicator'] == m]
        data=data_all
        # data=data_all.loc[list(data_all.index)[0]:list(data_all.index)[0]+M-1,:]
        graph_indicator=data['graph_indicator'].values.tolist()
        ip_locate = data['IP归属'].values.tolist()
        labels=data['label'].values.tolist()
        if len(labels)==0:
            continue
        # features=data[['up_pkts','down_pkts','up_bytes','down_bytes','up_b_std','down_b_std','inter_arrival_time_std']].values.tolist()
        # features=data[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24',
        # '25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50',
        # '51','52','53','54','55','56','57','58','59','60','61','62','63']].values.tolist()
        # features=data[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']].values.tolist()

        ## 前100数据包
        # (0.0271, 'ulen'), (0.0282, 'dskew'), (0.0289, 'dper6'), (0.0305, 'dlen'), (0.0319, 'dmax'), (0.038, 'uper8'), (0.0446, 'uper9'), (0.1424, 'umax')
        # features=data[stcol+['umax','uper9','uper8','dmax','dlen','dper6','dskew','ulen']].values.tolist()

        # (0.0259, 'dmax'), (0.0289, 'uper8'), (0.0358, 'uper9'), (0.1169, 'umax'), (0.2023, 'inter_arrival_time_std')
        # features=data[stcol+['inter_arrival_time_std','umax','uper9','uper8','dmax']].values.tolist()

        # features=data[stcol].values.tolist()

        ## 全部数据包
        # (0.0258, 'aper7'), (0.0274, 'dkurt'), (0.0285, 'uper8'), (0.0292, 'uvar'), (0.0333, 'dmax'), (0.0387, 'alen'), (0.0388, 'dlen'), (0.0402, 'uper9'), (0.1464, 'umax') 
        # features=data[stcol+['umax','alen','uper9','dlen','uper8','dmean'] ].values.tolist()
        features=data[stcol+['umax','alen','uper9','dlen','uper8','dmean']].values.tolist()

        # features=data[['umax','alen','uper9','uper8','dlen'] ].values.tolist()
        # features=data[stcol].values.tolist()





        # features=np.array(features,dtype=int)
        label = labels[0]
        begin_time = data['begin_time'].values.tolist()
        g = create_graph(ip_locate,features,graph_indicator,begin_time)

        g_list.append(AssociateGraph(g, label)) 
    
    for g in g_list:
        # g.neighbors = [[] for i in range(len(g.g))]
        # for i, j in g.g.edges():
        #     g.neighbors[i].append(j)
        #     g.neighbors[j].append(i)
        # degree_list = []
        # for i in range(len(g.g)):
        #     g.neighbors[i] = g.neighbors[i]
        #     degree_list.append(len(g.neighbors[i]))

        # g.max_neighbor = max(degree_list)
        # # print(g.max_neighbor)

        g.label = label_dict[g.label]
        # print(g.label)

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])
        # print(edges)
        g.edge_mat = np.transpose(np.array(edges, dtype=np.int32), (1,0))
        # print(g.edge_mat)
        for (u, v, wt) in g.g.edges.data('weight'):
            # print(f'{u}',f'{v}',f'{wt}')
            g.edge_attr.append(wt)
        g.edge_attr.extend([i for i in g.edge_attr])
        # print(g.edge_attr)
        # g.edge_attr= np.transpose(np.array(g.edge_attr), (1,0))
        
        for i in range(len(g.g.nodes)):
            g.node_features.append(g.g.nodes[i]['feature'])
            g.graph_indicator.append(g.g.nodes[0]['graph_indicator'])
        g.graph_indicator=np.array(g.graph_indicator,dtype=np.int64)



        # print("node_features:")  
        # print(g.node_features)  
        # print("edge_attr:")
        # print(g.edge_attr)
        # print("edge_mat:")
        # print(g.edge_mat)

        # print(g.edge_attr)

        # deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        # print(deg_list)


    # if degree_as_tag:
    #     for g in g_list:
    #         g.node_tags = list(dict(g.g.degree).values())
    #     #Extracting unique tag labels   
    #     tagset = set([])
    #     for g in g_list:
    #         tagset = tagset.union(set(g.node_tags))
    #     tagset = list(tagset)
    #     # print(tagset)
    #     tag2index = {tagset[i]:i for i in range(len(tagset))}
        # print(tag2index)


    # for g in g_list:
    #     for i in range(len(g.g.nodes)):
    #         g.node_features.append(g.g.nodes[i]['feature'])
    #         g.graph_indicator.append(g.g.nodes[0]['graph_indicator'])
    #     g.graph_indicator=np.array(g.graph_indicator,dtype=np.int64)
        # g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
        # g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        # g.node_features=np.array(g.node_features,dtype=np.float32)
        # print(g.node_features)
    

    # for g in g_list:
    #     for i in range(len(g.g.nodes)):
    #         g.graph_indicator.append(g.g.nodes[0]['graph_indicator'])
    #         # g.graph_indicator.append(0)
    # # g.node_features = np.zeros((len(g.node_tags), len(tagset)), dtype=np.float32)
    # # g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
    #     g.graph_indicator=np.array(g.graph_indicator,dtype=np.int64)
        # print(g.graph_indicator)

    print('# classes: %d' % len(label_dict))
    # print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))
    
    return g_list, len(label_dict)

# load_data('data.csv')
# 画图
# import matplotlib.pyplot as plt

# use_degree_as_tag = False
# graphs, num_classes = load_data('traffic/traffic.csv', use_degree_as_tag)

# G=graphs[0].g

# options = {'font_family': 'serif', 'font_weight': 'semibold', 'font_size': '12', 'font_color': '#ffffff'} 
# plt.figure(figsize=(10,10))
# nx.draw(G,node_size = 600,with_labels=True) 
# plt.savefig('original_network.png', format='png', dpi=500)
# plt.show()


def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def separate_data(graph_list, fold_idx, seed=0):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list