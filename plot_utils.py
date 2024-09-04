## Orbis liulei ##
import os
import itertools
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import xlrd,xlwt

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
defaultencoding = 'utf-8'

def excel2numpy(file_name='',sheet_name='',rows=[],cols=[],save_name=''):
    data=xlrd.open_workbook(file_name)
    if len(rows)==1 and isinstance(rows,int):
        rows=[0,rows]
    if len(cols)==1 and isinstance(cols,int):
        cols=[0,cols]
    mat_all=[]
    if len(sheet_name)==0:
        for sheet in data.sheets():
            table =data.sheet_by_name(sheet.name)
            if len(rows)<2 or rows[1]>table.nrows:
                rows=[0,table.nrows]
            if len(cols)<2 or cols[1]>table.ncols:
                cols=[0,table.ncols]
            mat=np.zeros((rows[1]-rows[0],cols[1]-cols[0]))
            for i in np.arange(rows[0],rows[1]):
                for j in np.arange(cols[0],cols[1]):
                    text = table.cell_value(i, j)   #.encode('utf-8')
                    mat[i-rows[0],j-cols[0]]=float(text)
            mat_all.append(mat)
        mat_all=np.array(mat_all)
        mat_all=np.concatenate(mat_all, axis=0)
    else:
        table = data.sheet_by_name(sheet_name)
        if len(rows) < 2 or rows[1] > table.nrows:
            rows = [0, table.nrows]
        if len(cols) < 2 or cols[1] > table.ncols:
            cols = [0, table.ncols]
        mat = np.zeros((rows[1] - rows[0], cols[1] - cols[0]))
        for i in np.arange(rows[0], rows[1]):
            for j in np.arange(cols[0], cols[1]):
                text = table.cell_value(i, j)  # .encode('utf-8')
                mat[i - rows[0], j - cols[0]] = float(text)
        mat_all=mat
    if len(save_name)>0:
        save_name = os.path.splitext(file_name)[0]
        np.save(save_name,mat_all)
    return mat_all

def numpy2excel(data,save_filename='',save_dir='results'):
    if not isinstance(data,np.ndarray):
        data=np.array(data)
    if data.ndim==1:
        data= data[:, np.newaxis]
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('result', cell_overwrite_ok=True)
    rows,cols = data.shape
    for j in range(cols):
        for i in range(rows):
            #print(i, j, data[i,j])
            sheet.write(i, j, data[i,j])
    if len(save_filename)==0:
        print("Please set save_filename,default name is numpy2excel")
        save_filename='numpy2excel'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    book.save(os.path.join(save_dir,save_filename+'.xlsx'))

def plot_error_curve(error_train,error_test,xlabel='Epoch',ylabel='loss',title='',legend_label=['train', 'Raw_data'],save_dir='results/figures',fontsize=13):
    n=len(error_train)
    x = np.arange(0, n, 1)
    plt.plot(x,error_train)
    plt.plot(x,error_test)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if len(title)>0:
        plt.title(title, fontsize=fontsize+5)
    label =legend_label
    plt.legend(label, loc=0, ncol=2)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'Error_curve.jpg'),dpi=200)
    plt.savefig(os.path.join(save_dir,'Error_curve.png'),dpi=200)
    plt.savefig(os.path.join(save_dir,'Error_curve.pdf'))
    plt.savefig(os.path.join(save_dir,'Error_curve.eps'))
    plt.close()
    print("Plot error curve finished")

def plot_acc_curve(error_train,error_test,xlabel='Epoch',ylabel='Accuary(%)',title='',legend_label=['train', 'Raw_data'],save_dir='results/figures',fontsize=13):
    n=len(error_train)
    x = np.arange(0, n, 1)
    plt.plot(x,error_train)
    plt.plot(x,error_test)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if len(title)>0:
        plt.title(title, fontsize=fontsize+5)
    label =legend_label
    plt.legend(label, loc=0, ncol=2)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'Acc_curve.jpg'),dpi=200)
    plt.savefig(os.path.join(save_dir,'Acc_curve.png'),dpi=200)
    plt.savefig(os.path.join(save_dir,'Acc_curve.pdf'))
    plt.savefig(os.path.join(save_dir,'Acc_curve.eps'))
    plt.close()
    print("Plot acc curve finished")

def plot_kappa_curve(kappa,kappa_2,xlabel='Epoch',ylabel='Kappa',title='',legend_label=['kappa', 'kappa_2'],save_dir='results/figures',fontsize=13):
    n=len(kappa)
    x = np.arange(0, n, 1)
    plt.plot(x,kappa)
    plt.plot(x,kappa_2)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if len(title)>0:
        plt.title(title, fontsize=fontsize+5)
    label =legend_label
    plt.legend(label, loc=0, ncol=2)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'kappa_curve.jpg'),dpi=200)
    plt.savefig(os.path.join(save_dir,'kappa_curve.png'),dpi=200)
    plt.savefig(os.path.join(save_dir,'kappa_curve.pdf'))
    plt.savefig(os.path.join(save_dir,'kappa_curve.eps'))
    plt.close()
    print("Plot kappa curve finished")

def plot_confusion_matrix(confusion_mat, classes,normalize, title='',save_dir='results_plot_confusion_matrix',save_name='confusion_matrix', confusion_matap=plt.cm.get_cmap('Blues'),colorbar=True,fontsize=14): # plt.cm.Blues   plt.cm.get_cmap('RdYlBu')
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    ### usage1
    mat=excel2numpy('a.xlsx')
    y_test=mat[:,0]
    y_pred=mat[:,1]
    confusion_mat = confusion_matrix(y_test, y_pred)
    class_names=[0,1,2,3,4,5,6,7,8,9]
    plot_confusion_matrix(confusion_mat, classes=class_names, normalize=True)
    """
    fig = plt.figure()
    if normalize:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #print(confusion_mat)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=confusion_matap)
    plt.title(title,fontsize=fontsize+4)
    if colorbar:
        cbar=plt.colorbar()
        #cbar.set_ticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    
    plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签 
    plt.rcParams['axes.unicode_minus']=False

    fmt = '.3f' if normalize else 'd'
    thresh = confusion_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
            plt.text(j, i, "%.2f%%"  %(confusion_mat[i, j]*100),horizontalalignment="center", color="white" if confusion_mat[i, j] > thresh else "black",fontsize=fontsize-1)
    else:
        for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
            plt.text(j, i, "%d"  %(confusion_mat[i, j]),horizontalalignment="center", color="white" if confusion_mat[i, j] > thresh else "black",fontsize=fontsize-1)
    plt.ylabel('True label',fontsize=fontsize)
    plt.xlabel('Predicted label',fontsize=fontsize)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    plt.tight_layout()
    fig.set_size_inches(10.2, 8.5)
    #plt.savefig(os.path.join(save_dir,save_name+'.eps'))
    plt.savefig(os.path.join(save_dir,save_name+'.png'), dpi=200)
    #plt.savefig(os.path.join(save_dir,save_name+'.jpg'), dpi=200)
    plt.savefig(os.path.join(save_dir,save_name+'.pdf'))
    np.save(os.path.join(save_dir,save_name),confusion_mat)
    plt.close()
    print("Plot confusion matrix finished")


def AUC_ROC(true_vessel_arr, pred_vessel_arr, save_fname):
    """
    Area under the ROC curve with x axis flipped
    true_vessel_arr  20*512*512*1
    """
    fpr, tpr, _ = roc_curve(true_vessel_arr, pred_vessel_arr)
    save_obj({"fpr": fpr, "tpr": tpr}, save_fname)
    AUC_ROC = roc_auc_score(true_vessel_arr.flatten(), pred_vessel_arr.flatten())
    return AUC_ROC


def plot_AUC_ROC(fprs, tprs, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k', '#cd919e'] if len(fprs) == 9 else ['r', 'y', 'm', 'g', 'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 8, 0] if len(fprs) == 9 else [4, 1, 2, 3, 0]

    # print auc
    print
    "****** ROC AUC ******"
    print
    "CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_roc*.npy)"
    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print
            "{} : {:04}".format(method_names[index], auc(fprs[index], tprs[index]))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(fprs[index], tprs[index], colors[index] + '*', label='Human')
        else:
            plt.step(fprs[index], tprs[index], colors[index], where='post', label=method_names[index].replace("_", " "),
                     linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('ROC Curve')
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.xlim(0, 0.3)
    plt.ylim(0.7, 1.0)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(fig_dir, "ROC.png"))
    plt.close()


def plot_AUC_PR(precisions, recalls, method_names, fig_dir, op_pts):
    # set font style
    font = {'family': 'serif'}
    matplotlib.rc('font', **font)

    # sort the order of plots manually for eye-pleasing plots
    colors = ['r', 'b', 'y', 'g', '#7e7e7e', 'm', 'c', 'k', '#cd919e'] if len(precisions) == 9 else ['r', 'y', 'm', 'g',
                                                                                                     'k']
    indices = [7, 2, 5, 3, 4, 6, 1, 8, 0] if len(precisions) == 9 else [4, 1, 2, 3, 0]

    # print auc
    print
    "****** Precision Recall AUC ******"
    print
    "CAVEAT : AUC of V-GAN with 8bit images might be lower than the floating point array (check <home>/pretrained/auc_pr*.npy)"
    for index in indices:
        if method_names[index] != 'CRFs' and method_names[index] != '2nd_manual':
            print
            "{} : {:04}".format(method_names[index], auc(recalls[index], precisions[index]))

    # plot results
    for index in indices:
        if method_names[index] == 'CRFs':
            plt.plot(recalls[index], precisions[index], colors[index] + '*',
                     label=method_names[index].replace("_", " "))
        elif method_names[index] == '2nd_manual':
            plt.plot(recalls[index], precisions[index], colors[index] + '*', label='Human')
        else:
            plt.step(recalls[index], precisions[index], colors[index], where='post',
                     label=method_names[index].replace("_", " "), linewidth=1.5)

    # plot individual operation points
    for op_pt in op_pts:
        plt.plot(op_pt[0], op_pt[1], 'r.')

    plt.title('Precision Recall Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(fig_dir, "Precision_recall.png"))
    plt.close()


def AUC_PR(true_vessel_img, pred_vessel_img, save_fname):
    """
    Precision-recall curve
    """
    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(), pos_label=1)
    save_obj({"precision": precision, "recall": recall}, save_fname)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)



def accuracy(x,y1,y2,path):
    a = plt.subplot(1, 1, 1)

    # 这里b表示blue，g表示green，r表示red，-表示连接线，--表示虚线链接
    a1 = a.plot(x, y1, 'bx-', label='train')
    a2 = a.plot(x, y2, 'g^-', label='Raw_data')

    # 标记图的题目，x和y轴
    plt.title("My matplotlib learning")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 显示图例
    handles, labels = a.get_legend_handles_labels()
    a.legend(handles[::-1], labels[::-1])
    plt.savefig(path)
    plt.close('all')
    # plt.show()
