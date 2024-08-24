import copy
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve,auc
import os
import random
import copy
import warnings
warnings.filterwarnings("ignore")
import numpy as np
 
type_n = 2

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
#模型的训练过程
def train_model(model, optimizer, data_o, train_loader, val_loader, test_loader, args):
    m = torch.nn.Sigmoid()
    #设置损失函数
    loss_fct=torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    #记录损失历史
    loss_history = []
    #设置初始最大auc和f1值
    max_auc = 0
    max_f1=0

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')

    # Train model
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    stoping = 0
    #循环训练
    for epoch in range(args.epochs):#
        #stoping=0
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (inp) in enumerate(train_loader):
            #取出输入数据和标签，将标签转换为torch tensor，并移至GPU（如果可用）
            label=inp[2]
            label=np.array(label,dtype=np.int64)
            label=torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            model.train() #变为训练模式
            optimizer.zero_grad() #清空梯度

            output, _ = model(data_o, inp)
            #前向传播计算输出
            log = torch.squeeze(output)
            #计算三种损失：分类损失、节点分类损失和边分类损失，
            loss1 = loss_fct(log, label.long())
            #并根据权重加权求和得到总损失 
            loss_train = args.loss_ratio1 * loss1

            loss_history.append(loss_train.cpu().detach().numpy())
            #反向传播计算梯度并更新参数
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            #记录预测值和标签，用于后续计算评估指标
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))


        #计算训练集上的准确率、F1值、召回率和精度
        y_pred_train1 = []
        y_label_train = np.array(y_label_train)
        y_pred_train = np.array(y_pred_train).reshape((-1, type_n))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    #print(y_pred_train[i][j])
                    y_pred_train1.append(j)
                    break

        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
        precision1 = precision_score(y_label_train, y_pred_train1, average='macro')

        #验证过程
        #如果不是快速模式，则在验证集上测试模型，计算验证集上的准确率、F1值、召回率和精度
        if not args.fastmode:
            # data_s, data_a,
            acc_val, f1_val, recall_val,precision_val, loss_val = test(model, val_loader, data_o, args,0)
            #if acc_val >= max_auc and f1_val>=max_f1
            #如果当前验证集上的准确率和F1值均超过历史最大值，则保存当前模型
            if acc_val >= max_auc and f1_val>=max_f1:
                model_max = copy.deepcopy(model)
                max_auc = acc_val
                max_f1=f1_val
                stoping=0
            else:
                stoping=stoping+1
            #输出各项指标
            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(acc),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'recall_val: {:.4f}'.format(recall_val),
                  'precision_val: {:.4f}'.format(precision_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test, f1_test, recall_test,precision_test, loss_test= test(model_max, test_loader, data_o, args,1)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test),'recall_test: {:.4f}'.format(recall_test))
def test(model, loader, data_o, args,printfou):

    #设置损失函数和模型为评估模式
    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    #lbl = data_a.y
    zhongzi=args.zhongzi
    with torch.no_grad():
        #对于每个batch，进行前向传播计算输出
        for i, (inp) in enumerate(loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            output, _ = model(data_o, inp)
            log = torch.squeeze(m(output))

            #计算三种损失：分类损失、节点分类损失和边分类损失，并根据权重加权求和得到总损失
            loss1 = loss_fct(log, label.long())
            loss = args.loss_ratio1 * loss1

            #记录预测值和标签，用于后续计算评估指标
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
    #计算准确率、F1值、召回率和精度
    y_pred_train1=[]
    y_label_train = np.array(y_label)
    y_pred_train = np.array(y_pred).reshape((-1, type_n))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    y_label_train1 = np.zeros((y_label_train.shape[0], type_n))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    #计算AUC和AUPR等指标
    auc_hong=0
    aupr_hong=0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):

        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:

            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)

    if printfou==1:
        with open(args.out_file, 'a') as f:


            f.write(str(zhongzi)+'  '+str(acc)+'   '+str(f1_score1)+'   '+str(recall1)+'   '+str(precision1)+'   '+str(auc1)+'   '+str(aupr)+'   '+str(auc_macro)+'   '+str(aupr_macro)+'\n')

    return acc,f1_score1,recall1,precision1,loss