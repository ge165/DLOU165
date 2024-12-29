import copy
import csv
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, precision_recall_curve, auc
import random
import warnings

warnings.filterwarnings("ignore")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1)

def calculate_metrics(y_true, y_pred):
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels, average='macro')
    recall = recall_score(y_true, y_pred_labels, average='macro')
    precision = precision_score(y_true, y_pred_labels, average='macro')
    return acc, f1, recall, precision

def calculate_auc_aupr(y_true, y_pred, num_classes):
    y_true_one_hot = np.zeros((y_true.shape[0], num_classes))
    y_true_one_hot[np.arange(y_true.shape[0]), y_true] = 1
    
    auc_macro, aupr_macro = 0, 0
    for i in range(num_classes):
        if np.sum(y_true_one_hot[:, i].reshape((-1))) == 0:
            continue
        auc_macro += roc_auc_score(y_true_one_hot[:, i].reshape((-1)), y_pred[:, i].reshape((-1)))
        precision, recall, _ = precision_recall_curve(y_true_one_hot[:, i].reshape((-1)), y_pred[:, i].reshape((-1)))
        aupr_macro += auc(recall, precision)

    auc_macro /= num_classes
    aupr_macro /= num_classes

    auc1 = roc_auc_score(y_true_one_hot.reshape((-1)), y_pred.reshape((-1)), average='micro')
    precision, recall, _ = precision_recall_curve(y_true_one_hot.reshape((-1)), y_pred.reshape((-1)))
    aupr = auc(recall, precision)
    return auc1, aupr, auc_macro, aupr_macro

def train_model(model, optimizer, data_o, train_loader, val_loader, test_loader, args):
    t_total = time.time()
    loss_fct = nn.CrossEntropyLoss()
    loss_history = []
    max_acc, max_f1 = 0, 0
    stoping = 0
    type_n = args.type_number

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')

    print('Start Training...')
    model_max = copy.deepcopy(model)
    
    for epoch in range(args.epochs): #
        t = time.time()
        print(f'-------- Epoch {epoch + 1} --------')
        model.train()
        y_pred_train, y_label_train = [], []

        for inp in train_loader:
            label = torch.tensor(np.array(inp[2], dtype=np.int64))
            if args.cuda:
                label = label.cuda()

            optimizer.zero_grad()
            output, _ ,cl_loss= model(data_o, inp)
            loss_train = args.loss_ratio1 * loss_fct(output.squeeze(), label.long()) + args.loss_ratio2 * cl_loss
            loss_history.append(loss_train.item())
            loss_train.backward()
            optimizer.step()

            y_label_train.extend(label.cpu().numpy().flatten())
            y_pred_train.extend(output.cpu().detach().numpy())

        acc, f1, recall, precision= calculate_metrics(np.array(y_label_train), np.array(y_pred_train).reshape(-1, type_n))

        if not args.fastmode:
            acc_val, f1_val, recall_val, precision_val, loss_val = test(model, val_loader, data_o, args, False)
            if acc_val >= max_acc and f1_val >= max_f1:
                model_max = copy.deepcopy(model)
                max_acc, max_f1 = acc_val, f1_val
                stoping = 0
            else:
                stoping += 1
            print(f'epoch: {epoch + 1}, loss_train: {loss_train.item():.4f}, acc_train: {acc:.4f}, acc_val: {acc_val:.4f}, f1_val: {f1_val:.4f}, time:{(time.time() - t):.4f}')

        torch.cuda.empty_cache()

    print("Optimization Finished!")
    print(f'Total time elapsed: {(time.time() - t_total):.4f}s')
    acc_test, f1_test, recall_test, precision_test, loss_test = test(model_max, test_loader, data_o, args, True)
    print(f'Test Results - Loss: {loss_test.item():.4f}, Acc: {acc_test:.4f}, F1: {f1_test:.4f}')

def test(model, loader, data_o, args, is_final_test):
    loss_fct = nn.CrossEntropyLoss()
    model.eval()
    y_pred, y_label = [], []
    type_n = args.type_number
    
    with torch.no_grad():
        for inp in loader:
            label = torch.tensor(np.array(inp[2], dtype=np.int64))
            if args.cuda:
                label = label.cuda()
            
            output, _, cl_loss = model(data_o, inp)
            loss_test = args.loss_ratio1 * loss_fct(output.squeeze(), label.long()) + args.loss_ratio2 * cl_loss
            y_label.extend(label.cpu().numpy().flatten())
            y_pred.extend(output.cpu().numpy())

    acc, f1, recall, precision = calculate_metrics(np.array(y_label), np.array(y_pred).reshape(-1, type_n))
    auc1, aupr, auc_macro, aupr_macro = calculate_auc_aupr(np.array(y_label), np.array(y_pred).reshape(-1, type_n), type_n)
    
    if is_final_test:
        with open(args.out_file, 'a') as f:
            f.write(f'{args.zhongzi}  {acc:.16f}   {f1:.16f}   {recall:.16f}   {precision:.16f}   {auc1:.16f}   {aupr:.16f}   {auc_macro:.16f}   {aupr_macro:.16f}\n')

    return acc, f1, recall, precision, loss_test