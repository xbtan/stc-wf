from re import A
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
from sklearn.metrics import classification_report
import time
import pandas as pd

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
# exit(0)
# dataset=TrafficDataset('train_Traffic')
# testdataset=testTrafficDataset('test_Traffic')
# for data in dataset:
#     print(data)

# print(dataset.data.x)
print('特征维度：',train_dataset.data.x.size())
# exit(0)
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


train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=True)
test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=True)

model = Net(args).to(args.device)
# print(model)
# exit(0)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


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

        num_re=torch.tensor([]).to(args.device)
        num_pre=torch.tensor([]).to(args.device)
        for i in range(len(data.y)):
            # print(data.y[i])
            temp_re=torch.full((1,ptr[i]),data.y[i]).to(args.device).reshape(-1)
            num_re=torch.cat((num_re,temp_re),dim=-1)
            temp_pre=torch.full((1,ptr[i]),pred[i]).to(args.device).reshape(-1)
            num_pre=torch.cat((num_pre,temp_pre),dim=-1)
        # print(num_re)
        y_test = y_test.float()
        num_re = num_re.float()
        print(y_test)
        print(num_re)
        y_test=torch.cat((y_test,num_re),dim=-1)
        # print(y_test)
        y_recognize = y_recognize.float()
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
    
    

min_loss = 1e10
patience = 0

Loss_list = []
Accuracy_list = []

t1=time.time()
for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc,val_loss, _, _, _, _ = test(model,val_loader)

    Loss_list.append(val_loss)
    Accuracy_list.append(val_acc)

    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break 

t2=time.time()
print("train time:",t2-t1)

model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
t3=time.time()
test_acc,test_loss, y_test, y_recognize, y_test1, y_recognize1 = test(model,test_loader)
t4=time.time()
print("test time:",t4-t3)
print("Test accuarcy:{}".format(test_acc))

label_dict = {'bilibili':1,'CSDN':2,'QQ音乐':3,'爱奇艺':4,'东方财富':5,'斗鱼':6,
    '豆瓣':7,'凤凰网':8,'虎扑':9,'简书':10,'今日头条':11,'京东':12,'汽车之家':13,
    '搜狐':14,'淘宝':15,'腾讯门户':16,'腾讯视频':17,'网易云音乐':18,'喜马拉雅':19,'新华网':20,'新浪':21}
a_inv = {value: key for key, value in label_dict.items()}
y_test=y_test.cpu().detach().numpy().tolist()
y_recognize=y_recognize.cpu().detach().numpy().tolist()
# y_test=[a_inv[int(item)+1] for item in y_test]
# y_recognize=[a_inv[int(item)+1] for item in y_recognize]
target_names=list(label_dict.keys())
print("单流指标：")
print(classification_report(y_test, y_recognize, digits=4))

y_test1=y_test1.cpu().detach().numpy().tolist()
y_recognize1=y_recognize1.cpu().detach().numpy().tolist()
print("多流指标：")
print(classification_report(y_test1, y_recognize1, digits=4))

from pylab import *
import matplotlib
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman 

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

x = range(0, 100)
ep=[i+1 for i in x]
y1 = Accuracy_list
y2 = Loss_list
valid_result=pd.DataFrame({'epoch':ep,'accuracy':y1})
valid_result.to_csv('valid_result.csv',index=False)
### 双坐标
# fig,ax1 = plt.subplots(figsize=(14,6))
# ax1.set_ylim(0,1,0.1)
# plt.plot(x,y1,'r',label="Validation Accuracy")
# #显示网格
# # plt.grid(True)
# plt.xlabel("Number of Epoch",font1)
# plt.ylabel('Validation Accuracy',font1)
# plt.title("Dynamic of the training process")
# #设置线标的位置
# plt.legend(loc='upper left')
# # ax1.tick_params(labelsize=17)

# ax2=ax1.twinx()
# plt.plot(x,y2,'g',label='Validation Loss')
# plt.legend(loc='upper right')
# # ax2.tick_params(labelsize=17)
# ax2.set_ylabel("Validation Loss",font1)
# ax2.set_ylim(0,0.5,0.05)
# plt.savefig('Dynamic of Training Process.jpg')
# plt.show()

### 单坐标
plt.figure(figsize=(14,6))
plt.plot(x, y1, label = 'Validation Accuracy',linewidth=2)
plt.plot(x, y2, label = 'Validation Loss',linewidth=2)
plt.tick_params(labelsize=18)  #设置坐标值字体大小
plt.legend(loc = 0,fontsize=18) #图例位置自动
plt.axis('tight')
plt.xlabel("Number of Epoch",font1)
plt.ylabel("Accuracy / Loss",font1)
plt.ylim(0,1)
plt.title('Dynamic of Training Process',fontsize=20)

plt.savefig('Dynamic of Training Process.jpg')
plt.show()

