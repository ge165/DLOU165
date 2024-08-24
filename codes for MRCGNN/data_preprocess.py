import networkx as nx

from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import *
import pandas as pd
import csv
import random
from tqdm import tqdm
import copy
import numpy as np

type_n = 2
gene_n = 11284

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

#用于处理三元组数据
class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relationtype=triple[:,2]
        #self.label = triple[:, 3]

    def __len__(self):
        return len(self.relationtype)

    def __getitem__(self, index):


        return  (self.entity1[index], self.entity2[index], self.relationtype[index])

#用于加载训练、验证和测试数据集，并将其转换为 PyTorch 的 DataLoader 对象。
def load_data(args, val_ratio=0.1, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and symmetric adjacency"""
    # read data

    #从 data/drug_listxiao.csv 文件中读取药物列表，并将药物名存储在 drug_list 中
    drug_list = []
    gene_list = []
    with open('data' + str(type_n) +'/train.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] not in drug_list:
                drug_list.append(row[0])
    with open('data' + str(type_n) +'/train.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] not in gene_list:
                gene_list.append(row[1])
    with open('data' + str(type_n) +'/test.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] not in gene_list:
                gene_list.append(row[1])
                
    f.close
    print(len(drug_list))
    print(len(gene_list))

    zhongzi=args.zhongzi

    #该函数用于加载训练集、验证集和测试集
    def loadtrainvaltest():
        #train dataset
        #读取训练数据
        train=pd.read_csv('data' + str(type_n) +'/train.csv')
        #生成三元组
        train_pos=[(h, t, r) for h, t, r in zip(train['d1'], train['d2'], train['type'])]
        #np.random.seed(args.seed)
        #并随机打乱顺序
        np.random.shuffle(train_pos)
        train_pos = np.array(train_pos)
        #将药物名称转换为索引
        for i in range(train_pos.shape[0]):
            train_pos[i][0] = int(drug_list.index(train_pos[i][0]))
            train_pos[i][1] = int(gene_list.index(train_pos[i][1]))
            train_pos[i][2] = int(train_pos[i][2])
        label_list=[]
        #将关系类型转换为 one-hot 编码
        for i in range(train_pos.shape[0]):
            label=np.zeros((type_n))
            label[int(train_pos[i][2])]=1
            label_list.append(label)
        label_list=np.array(label_list)
        #将三元组和one-hot编码的关系类型合并
        train_data= np.concatenate([train_pos, label_list],axis=1)

        #val dataset，读取验证数据集，和训练进行同样的处理
        train_data,val_data = train_test_split(train_data,test_size=0.2,random_state=42)

        #test dataset，测试集也是如此
        test = pd.read_csv('data' + str(type_n) +'/test.csv')
        test_pos = [(h, t, r) for h, t, r in zip(test['d1'],test['d2'], test['type'])]
        #np.random.seed(args.seed)
        np.random.shuffle(test_pos)
        test_pos= np.array(test_pos)

        for i in range(len(test_pos)):
            test_pos[i][0] = int(drug_list.index(test_pos[i][0]))
            test_pos[i][1] = int(gene_list.index(test_pos[i][1]))
            test_pos[i][2] = int(test_pos[i][2])
        label_list = []
        for i in range(len(test_pos)):
            label = np.zeros((type_n))
            label[int(test_pos[i][2])] = 1
            label_list.append(label)
        label_list = np.array(label_list)
        test_data = np.concatenate([test_pos, label_list], axis=1)

        print(train_data.shape)
        print(val_data.shape)
        print(test_data.shape)
        return train_data,val_data,test_data
    #调用这个函数
    train_data,val_data,test_data=loadtrainvaltest()
    #设置DataLoader参数，包括批次大小、是否打乱数据、工作线程数和是否丢弃最后一个不完整的批次。
    params = {'batch_size': args.batch, 'shuffle': False, 'num_workers': args.workers, 'drop_last': False}

    #创建 DataLoader
    training_set = Data_class(train_data)

    train_loader = DataLoader(training_set, **params)


    validation_set = Data_class(val_data)

    val_loader = DataLoader(validation_set, **params)


    test_set = Data_class(test_data)

    test_loader = DataLoader(test_set, **params)

    print('Extracting features...')


    #随机生成特征
    num_gene = gene_n
    features_dim = 512 #特征维度
    features1 = np.random.rand(num_gene,features_dim).astype(np.float32)
    #药物特征
    data = np.load('drug_feat.npz')
    features = data['feats']
    ids = data['drug_id']
    ids = ids.tolist()

    features2 = []
    feature_dim = features.shape[1]
    for i in range(len(drug_list)):
        if drug_list[i] in ids:
            # 如果drug_list[i]在ids中，找到对应的特征
            features2.append(features[ids.index(drug_list[i])])
        else:
            # 如果drug_list[i]不在ids中，随机生成一个特征向量
            random_feature = np.random.rand(feature_dim)
            features2.append(random_feature)
    features2 = np.array(features2)

    features_o = np.concatenate((features1,features2),axis=0)
    #print(features_o.shape)
    args.dimensions = features_o.shape[1]

    # adversarial nodes
    id = np.arange(features_o.shape[0])
    #随机打乱节点的生成
    id = np.random.permutation(id)

    x_o = torch.tensor(features_o, dtype=torch.float)
    #print(x_o.shape)
    #深拷贝训练数据
    positive1=copy.deepcopy(train_data)
    #print(positive1.shape)

    edge_index_o = []
    label_list = []
    label_list11 = []
    for i in range(positive1.shape[0]):

    #for h, t, r ,label in positive1:
    #生成图的边索引 edge_index_o 和标签列表 label_list 以及 label_list11
        a = []
        a.append(int(positive1[i][0]))
        a.append(int(positive1[i][1]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        a = []
        a.append(int(positive1[i][1]))
        a.append(int(positive1[i][0]))
        edge_index_o.append(a)
        label_list.append(int(positive1[i][2]))
        b = []
        b.append(int(positive1[i][2]))
        b.append(int(positive1[i][2]))
        label_list11.append(b)
    
    edge_index_o = torch.tensor(edge_index_o, dtype=torch.long)
    #print(edge_index_o.shape)

    #创建图数据对象
    #原始节点特征和边索引
    data_o = Data(x=x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)

    random.shuffle(label_list11)
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

    label_list11 = flatten(label_list11)
    #print(len(label_list11))
    print('Loading finished!')
    return data_o, train_loader, val_loader, test_loader