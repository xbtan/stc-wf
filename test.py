from re import A
from numpy import NAN, dtype
import torch
# from torch_geometric.datasets import TUDataset
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
# from torch_geometric import utils
from TrafficDataset import trainTrafficDataset,validTrafficDataset,testTrafficDataset
from networks import  Net
import torch.nn.functional as F
import argparse
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import plot_utils
import matplotlib.image as mpimgs
import pandas as pd
from file_label import get_file_label


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=52,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.1,
                    help='dropout ratio')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=500,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
# dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
train_dataset=trainTrafficDataset('traffic')
valid_dataset=validTrafficDataset('traffic')
test_dataset=testTrafficDataset('traffic')

# dataset=TrafficDataset('train_Traffic')
# testdataset=testTrafficDataset('test_Traffic')
# for data in dataset:
#     print(data)

# print(dataset.data.x)
# print(dataset.data.x.size())
# print(dataset.data.y)
# print(dataset.data.y.size())
# print(dataset.data.edge_index)
# print(dataset.data.edge_index.size())
# exit(0)
args.num_classes = train_dataset.num_classes
args.num_features = train_dataset.num_features
# print(dataset.num_classes)
# print(dataset.num_features)


num_training = len(train_dataset)
num_val = len(valid_dataset)
num_test = len(test_dataset)
training_set,validation_set,test_set = train_dataset,valid_dataset,test_dataset

# num_training = int(len(dataset)*0.9)
# num_val = len(dataset) - num_training
# training_set,validation_set = random_split(dataset,[num_training,num_val])
# test_set=testdataset
# print(len(training_set))
# print(len(validation_set))
# for data in test_set:
#     print(data.y)
# exit(0)


train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False)

# model = Net(args).to(args.device)
# # print(model)
# # exit(0)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model,loader):
    y_test=torch.LongTensor([]).to(args.device)
    y_recognize=torch.LongTensor([]).to(args.device)
    y_test1=torch.LongTensor([]).to(args.device)
    y_recognize1=torch.LongTensor([]).to(args.device)
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)

        ptr=data.ptr.cpu().detach().numpy().tolist()
        # print(data.ptr)

        for i in range(len(ptr)-1):
            ptr[i]=ptr[i+1]-ptr[i]
        ptr=ptr[:-1]

        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()

        num_re=torch.LongTensor([]).to(args.device)
        num_pre=torch.LongTensor([]).to(args.device)
        for i in range(len(data.y)):
            # print(data.y[i])
            temp_re=torch.full((1,ptr[i]),data.y[i]).to(args.device).reshape(-1)
            num_re=torch.cat((num_re,temp_re),dim=-1)
            temp_pre=torch.full((1,ptr[i]),pred[i]).to(args.device).reshape(-1)
            num_pre=torch.cat((num_pre,temp_pre),dim=-1)
        # print(num_re)

        y_test=torch.cat((y_test,num_re),dim=-1)
        # print(y_test)
        y_recognize=torch.cat((y_recognize,num_pre),dim=-1)
        # print(data.y)
        # print(pred)
        # y_test.append(data.y.cpu())
        # y_recognize.append(pred.cpu())
        y_test1=torch.cat((y_test1,data.y),dim=-1)
        y_recognize1=torch.cat((y_recognize1,pred),dim=-1)
    # print(classification_report(y_test.cpu().detach().numpy().tolist(), y_recognize.cpu().detach().numpy().tolist(), digits=4))
    # print(y_test)
    # print(y_recognize)
    return correct / len(loader.dataset), loss / len(loader.dataset), y_test, y_recognize, y_test1, y_recognize1

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc,test_loss, y_test, y_recognize, y_test1, y_recognize1 = test(model,test_loader)

print("Test accuarcy:{}".format(test_acc))

label_dict = {'bilibili':1,'CSDN':2,'QQ音乐':3,'爱奇艺':4,'东方财富':5,'斗鱼':6,
    '豆瓣':7,'凤凰网':8,'虎扑':9,'简书':10,'今日头条':11,'京东':12,'汽车之家':13,
    '搜狐':14,'淘宝':15,'腾讯门户':16,'腾讯视频':17,'网易云音乐':18,'喜马拉雅':19,'新华网':20,'新浪':21}
# label_dict = {'b站':1,'CSDN':2,'博客园':3,'京东':4,'淘宝':5,'网易云':6,'优酷':7,'知乎':8}
a_inv = {value: key for key, value in label_dict.items()}


y_test=y_test.cpu().detach().numpy().tolist()
y_recognize=y_recognize.cpu().detach().numpy().tolist()
y_test=[a_inv[int(item)+1] for item in y_test]
y_recognize=[a_inv[int(item)+1] for item in y_recognize]

# target_names=list(label_dict.keys())
# print("单流指标：")
# print(classification_report(y_test, y_recognize, digits=4))
# scr=classification_report(y_test, y_recognize, digits=4, output_dict=True)
# scr = pd.DataFrame(scr).transpose()
# scr.to_csv("result.csv", index= True, encoding='utf_8_sig')

y_test1=y_test1.cpu().detach().numpy().tolist()
y_recognize1=y_recognize1.cpu().detach().numpy().tolist()
y_test1=[a_inv[int(item)+1] for item in y_test1]
y_recognize1=[a_inv[int(item)+1] for item in y_recognize1]
# print(y_test1)

indicator_list,file_name_list,label_list=get_file_label('test_data.csv')
# import operator
# print(operator.eq(y_test1,label_list))
tag=[NAN]*len(label_list)
for i in range(len(label_list)):
    if label_list[i]!=y_recognize1[i]:
        tag[i]=False
true_pre=pd.DataFrame({'graph_indicator':indicator_list,'file_name':file_name_list,'true_label':label_list,
'predict':y_recognize1,'tag':tag})
true_pre.to_csv('analysis.csv',index= False,encoding='utf_8_sig')

print("多流指标：")
print(classification_report(y_test1, y_recognize1, digits=4))
mcr=classification_report(y_test1, y_recognize1, digits=4, output_dict=True)
mcr = pd.DataFrame(mcr).transpose()
mcr.to_csv("result.csv", index= True,encoding='utf_8_sig')



# 绘制单流混淆矩阵
labels = list(label_dict.keys())
# print(labels)

# pre_ = []
# tar_ = []
# # for label in y_test:
# #     if a_inv[label+1] not in labels:
# #         labels.append(a_inv[label+1])
# # lables=labels.sort()
# for x in y_recognize:
#     pre_.append(x)
# for y in y_test:
#     tar_.append(y)
# 混淆矩阵
# confusion_mat = confusion_matrix(y_test, y_recognize,labels=labels)
# plot_utils.plot_confusion_matrix(confusion_mat, classes=labels, normalize=False, save_dir='confusion_matrix',
#                                  save_name='confusion_matrix-single', colorbar=True)

# 绘制多流混淆矩阵
# labels = []
# pre_ = []
# tar_ = []
# # for label in y_test1:
# #     if a_inv[label+1] not in labels:
# #         labels.append(a_inv[label+1])
# for x in y_recognize1:
#     pre_.append(x)
# for y in y_test1:
#     tar_.append(y)
# 混淆矩阵
confusion_mat = confusion_matrix(y_test1, y_recognize1,labels=labels)

plot_utils.plot_confusion_matrix(confusion_mat, classes=labels, normalize=False, save_dir='confusion_matrix',
                                 save_name='confusion_matrix-multiple', colorbar=True)