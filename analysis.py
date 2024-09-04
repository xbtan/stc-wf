import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
import numpy as np


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
    for i in range(len(ips)):
        len_list.append(len(ips[i]))
        for j in range(len(ips[i])):
            G.add_node(temp+j,feature=features[temp+j],graph_indicator=graph_indicator[temp+j])
            num_list.append(temp+j)
            for t in range(j+1,len(ips[i])):
                # G.add_edge(temp+j,temp+t,weight=[1,1])
                G.add_edge(temp+j,temp+t)
        temp=temp+len(ips[i])
        nums_list.append(num_list)
        num_list=[]
    # print(nums_list)
    for i in range(len(nums_list)):
        if i>0:
            # for x in nums_list[i-1]:
            #     for y in nums_list[i]:
            #         G.add_edge(x,y, weight=1/abs(begin_time[x]-begin_time[y]))
            G.add_edges_from([(x,y) for x in nums_list[i-1] for y in nums_list[i]])
            # G.add_weighted_edges_from([(x,y,[math.exp(-abs(begin_time[x]-begin_time[y])),0]) for x in nums_list[i-1] for y in nums_list[i]])
    nx.draw_networkx(G)
    
    # plt.savefig(path,dpi=1080)
    plt.show()
    
    print(len(G.nodes))
    return G


data=pd.read_csv('analysis.csv')
tag=data['tag'].values.tolist()
# print(tag)


file_list=data['file_name'].values.tolist()
indicator_list=data['graph_indicator'].values.tolist()
# print(len(file_list))
# print(len(set(file_list)))
# num=0
# duf=list()
# for file in file_list:
#     if file_list.count(file)>1:
#         duf.append(file)
# print(len(set(duf)))

wrong_id,wrong_file=list(),list()
for i in range(len(tag)):
    if tag[i]==False:
        wrong_file.append(file_list[i])
        wrong_id.append(indicator_list[i])
# print(wrong_id)
# print(wrong_file)
# print(len(wrong_id))
# print(len(wrong_file))


test_data=pd.read_csv('test_data.csv')

stcol=[]
for i in range(20):
    stcol.append('len'+str(i))
# for id in wrong_id:
#     data=test_data[test_data['graph_indicator'] == id]
#     data=data[['file_name','src_ip','src_port','protol','dst_ip','IP归属','dst_port','begin_time']+
#     stcol+['umax','alen','uper9','uper8','dlen','label','graph_indicator']]
#     data.to_csv('wrong_data.csv',mode='a',index= False,encoding='utf_8_sig')
#     graph_indicator=data['graph_indicator'].values.tolist()
#     # print(graph_indicator)
#     ip_locate = data['IP归属'].values.tolist()
#     labels=data['label'].values.tolist()
#     if len(labels)==0:
#         continue
#     features=data[stcol+['umax','alen','uper9','uper8','dlen'] ].values.tolist()
#     label = labels[0]
#     path='../analysis/graph/'+data['file_name'].values.tolist()[0]+'.jpg'
#     begin_time = data['begin_time'].values.tolist()
#     g = create_graph(ip_locate,features,graph_indicator,begin_time,path)



def get_target_ids(label_name):
    target_ids=[]
    train_data=pd.read_csv('train_data.csv')
    for m in range(min(train_data['graph_indicator']), max(train_data['graph_indicator']) + 1):
        data=train_data[train_data['graph_indicator'] == m]
        labels=data['label'].values.tolist()
        label=labels[0]
        if(label==label_name):
            target_ids.append(data['graph_indicator'].values.tolist()[0])
    return target_ids

def get_target_sta(label_name):
    target_ids=get_target_ids(label_name)
    len_seq=[]
    train_data=pd.read_csv('train_data.csv')
    for id in target_ids:
        data=train_data[train_data['graph_indicator'] == id]
        graph_indicator=data['graph_indicator'].values.tolist()
        len_seq.append(len(graph_indicator))
    mean=np.mean(np.array(len_seq))
    minimum=min(len_seq) 
    maximum=max(len_seq) 
    num=len(len_seq)
    return [mean,minimum,maximum,num]

def get_target_data(label_name,amount,file_name):
    target_ids=get_target_ids(label_name)
    if amount>len(target_ids):
        amount=len(target_ids)
    ids=random.sample(target_ids,amount)
    train_data=pd.read_csv('train_data.csv')
    for id in ids:
        data=train_data[train_data['graph_indicator'] == id]
        data=data[['file_name','src_ip','src_port','protol','dst_ip','IP归属','dst_port','begin_time']+
        stcol+['umax','alen','uper9','uper8','dlen','label','graph_indicator']]
        data.to_csv(file_name, mode='a',index= False,encoding='utf_8_sig')
        graph_indicator=data['graph_indicator'].values.tolist()
        # print(graph_indicator)
        ip_locate = data['IP归属'].values.tolist()
        labels=data['label'].values.tolist()
        if len(labels)==0:
            continue
        features=data[stcol+['umax','alen','uper9','uper8','dlen'] ].values.tolist()
        label = labels[0]
        # path='../analysis/graph/'+data['file_name'].values.tolist()[0]+'.jpg'
        begin_time = data['begin_time'].values.tolist()
        g = create_graph(ip_locate,features,graph_indicator,begin_time)


sta=get_target_sta('腾讯视频')
print(sta)
get_target_data('腾讯视频',20,'腾讯视频.csv')
    # g = create_graph(ip_locate,features,graph_indicator,begin_time,path)

