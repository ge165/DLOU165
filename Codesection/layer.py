import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv,TransformerConv
import numpy as np
import csv
import os
import random
from losses import SupConLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1)

class AttentionFusion_auto(nn.Module):
    def __init__(self, n_dim_input1, n_dim_input2):
        super(AttentionFusion_auto, self).__init__()
        self.linear = nn.Linear(n_dim_input1, n_dim_input2)

    def forward(self, input1, input2):
        mid_emb = torch.cat((input1, input2), 1).float()
        return F.relu(self.linear(mid_emb))

class MRCGNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2, hidden3,dropout, type_n):
        super(MRCGNN, self).__init__()
        self.encoder_o1 = RGCNConv(feature, hidden1, num_relations=type_n)
        self.encoder_o2 = RGCNConv(hidden1, hidden2*2, num_relations=type_n)
        self.encoder_o4 = TransformerConv(feature, hidden2, heads=4)
        self.encoder_o5 = TransformerConv(hidden1, hidden3, heads=4)
        
        self.attt = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.mlp = nn.ModuleList([
            nn.Linear(448,256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, type_n)
            ])
        self.attention_fusion1 = AttentionFusion_auto(384, 128)
        self.attention_fusion2 = AttentionFusion_auto(640, 320)
        self.attention_fusion3 = AttentionFusion_auto(640, 448)

        self.dropout = dropout
        self.cl_loss = SupConLoss(0.07, 0.07)

        drug_list = []
        with open('data/drug_smiles.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                drug_list.append(row[0])

        data = np.load('data/feat_1.npz')
        features = data['feats']
        ids = data['drug_id'].tolist()

        features1 = []
        for i in range(len(drug_list)):
            features1.append(features[ids.index(drug_list[i])])
        features1 = np.array(features1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.features1 = torch.from_numpy(features1).to(device)

    def forward(self, data_o, idx):
        x_o, adj, e_type = data_o.x, data_o.edge_index, torch.tensor(data_o.edge_type, dtype=torch.int64)
        
        x1_o = F.relu(self.encoder_o1(x_o, adj, e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o,adj,e_type)

        x1_o1 = F.relu(self.encoder_o4(x_o, adj))
        x1_o1 = F.dropout(x1_o1, self.dropout, training=self.training)
        x2_o1 = self.encoder_o5(x1_o1, adj)

        loss = self.cl_loss(x2_o, x2_o1)

        final = torch.cat((self.attt[0] * x1_o, self.attt[1] * x2_o), dim=1)
        final1 = torch.cat((self.attt[0] * x1_o1, self.attt[1] * x2_o1), dim=1)    
        
        a, b = [int(i) for i in list(idx[0])], [int(i) for i in list(idx[1])]
        aa, bb = torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)
        entity1, entity2 = final[aa], final[bb]
        entity3, entity4 = final1[aa], final1[bb]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        entity1_res = self.features1[aa].to(device)
        entity2_res = self.features1[bb].to(device)

        entity1 = self.attention_fusion1(entity1, entity3)
        entity3 = self.attention_fusion2(entity1, entity1_res)
        entity2 = self.attention_fusion1(entity2, entity4)
        entity4 = self.attention_fusion2(entity2, entity2_res)
        concatenate = self.attention_fusion3(entity3, entity4)

        for layer in self.mlp:
            concatenate = layer(concatenate)

        return concatenate, x2_o, loss