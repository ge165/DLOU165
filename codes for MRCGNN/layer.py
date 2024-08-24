import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,RGCNConv
import numpy as np
import csv
import os
import random

type_n =2
gene_n = 11284

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

#鉴别器，用于对比学习中的对比损失计算
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

#平均读取器，用于从节点表示中读取全局图表示
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)
#多层感知机，定义了一个简单的两层全连接神经网络
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

#定义了主要的图神经网络模型
class MRCGNN(nn.Module):
    #
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout,zhongzi):
        super(MRCGNN, self).__init__()

        #两个RGCN层，用于对图数据进行编码
        self.encoder_o1 = RGCNConv(feature, hidden1,num_relations=type_n)
        self.encoder_o2 = RGCNConv(hidden1 , hidden2,num_relations=type_n)
        #两层GCN层
        self.encoder_o3 = GCNConv(feature,hidden1)
        self.encoder_o4 = GCNConv(hidden1,hidden2)
        #attt是一个参数，用于加权组合不同层的输出
        self.attt = torch.zeros(2)
        self.attt[0] = 0.5
        self.attt[1] = 0.5
        self.attt = nn.Parameter(self.attt)
        #一个鉴别器，用于对比学习
        #self.disc = Discriminator(hidden2 * 2)

        self.dropout = dropout
        #self.sigm = nn.Sigmoid()
        #一个平均读取器，用于读取全局图表示
        #self.read = AvgReadout()
        #一个多层感知机，用于最终的分类任务
        self.mlp = nn.ModuleList([nn.Linear(1408, 256),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(256, 128),
                                  nn.ELU(),
                                  nn.Dropout(p=0.1),
                                  nn.Linear(128, 65),
                                  nn.Linear(65,type_n)
                                  ])
                    
        # 基因特征
        num_drugs = gene_n
        features_dim = 512
        # 随机生成特征
        features1 = np.random.rand(num_drugs, features_dim).astype(np.float32)

        self.features1 = torch.from_numpy(features1).cuda()

        # 药物特征
        drug_list = []
        with open('data' + str(type_n) +'/train.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] not in drug_list:
                    drug_list.append(row[0])
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
        self.features2 = torch.from_numpy(features2).float().cuda()
        #print(features2.shape)
        features3 = np.concatenate((features1,features2),axis=0)
        self.features3 = torch.from_numpy(features3).float().cuda()
        #print(features3.shape)
    def MLP(self, vectors, layer):
            for i in range(layer):
                vectors = self.mlp[i](vectors)

            return vectors
    # data_s, data_a,
    def forward(self, data_o, idx):
        #print(max(idx[1]))

        #RGCN for DDI event graph and two corrupted graph
        #对这三个输入数据进行编码
        x_o, adj ,e_type= data_o.x, data_o.edge_index,data_o.edge_type
        e_type=torch.tensor(e_type,dtype=torch.int64)

        x1_o = F.relu(self.encoder_o1(x_o, adj,e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, adj,e_type)
        #GCN
        x1_o1 = F.relu(self.encoder_o3(x_o,adj))
        x1_o1 = F.dropout(x1_o1,self.dropout,training=self.training)
        x2_o1 = self.encoder_o4(x1_o1,adj)

        #print(max(idx[0]))
        a = [int(i) for i in list(idx[0])]
        
        b = [int(i) for i in list(idx[1])]
        #print(len(a),len(b))

        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        #print(aa.shape,bb.shape)
        #layer attnetion
        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)
        final1 = torch.cat((self.attt[0] * x1_o1, self.attt[1] * x2_o1), dim=1)

        entity1 = final[aa]
        entity2 = final[bb]
        entity3 = final1[aa]
        entity4 = final1[bb]
        
        #skip connection
        entity1_res = self.features3[aa].to('cuda')
        entity2_res = self.features3[bb].to('cuda')
        #print(entity1_res.shape,entity2_res.shape)
        #print("shape:",entity1.shape,entity2.shape,entity3.shape,entity4.shape,entity1_res.shape,entity2_res.shape)
        
        entity1 = torch.cat((entity3, entity1), dim=1)
        entity3 = torch.cat((entity1, entity1_res), dim=1)
        entity2 = torch.cat((entity4, entity2), dim=1)
        entity4 = torch.cat((entity2, entity2_res), dim=1)

        #concatenate1 = self.concatenate(entity1,entity2)
        concatenate = torch.cat((entity3, entity4), dim=1)

        # 打印传入 MLP 的向量形状
        #print("concatenate shape:", concatenate.shape)

        #通过 MLP 将节点对的表示进行进一步处理，得到最终的分类结果 log
        feature = self.MLP(concatenate, 8)
        log = feature

        return log, x2_o
