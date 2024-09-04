from cProfile import label
from dataclasses import dataclass
from operator import ne
import pandas as pd

def removeDuplicate(old_list):
    new_list=list(set(old_list))
    new_list.sort(key=old_list.index)
    return new_list


def get_file_label(file):
    data=pd.read_csv(file)
    old_indicator_list=data['graph_indicator'].values.tolist()
    indicator_list=removeDuplicate(old_indicator_list)
    # print(len(new_indicator_list))
    file_name_list,label_list=list(),list()
    for indicator in indicator_list:
        data_all=data[data['graph_indicator'] == indicator]
        file_name_list.append(data_all['file_name'].values.tolist()[0])
        label_list.append(data_all['label'].values.tolist()[0])
    # print(len(file_name_list))
    # print(len(label_list))
    # file_label=pd.DataFrame({'graph_indicator':indicator_list,'file_name':file_name_list,'true_label':label_list})
    # file_label.to_csv('1.csv',index= False,encoding='utf_8_sig')
    return indicator_list,file_name_list,label_list
# print(file_label)

# data=pd.read_csv('analysis.csv')
# true=data['true_label'].values.tolist()
# pre=data['predict'].values.tolist()
# # print(len(true))
# tag=[None]*len(true)
# for i in range(len(true)):
#     if true[i]!=pre[i]:
#         tag[i]='false'
# # print(tag)
# tag=pd.DataFrame({'tag':tag})
# new_data=pd.concat([data,tag],axis=1)
# new_data.to_csv('analysis1.csv',index= False,encoding='utf_8_sig')