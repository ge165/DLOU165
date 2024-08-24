import numpy as np
import torch
import csv

drug_list = []
with open('rcgn\MRCGNN-gai\codes for MRCGNN\data1\\train.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] not in drug_list:
            drug_list.append(row[0])
data = np.load('rcgn\MRCGNN-gai\codes for MRCGNN\drug_feat.npz')
features = data['feats']
ids = data['drug_id']
ids = ids.tolist()

features1 = []
feature_dim = features.shape[1]
for i in range(1,len(drug_list)):
    if drug_list[i] in ids:
        # 如果drug_list[i]在ids中，找到对应的特征
        features1.append(features[ids.index(drug_list[i])])
    else:
        # 如果drug_list[i]不在ids中，随机生成一个特征向量
        random_feature = np.random.rand(feature_dim)
        features1.append(random_feature)
features1 = np.array(features1)

num_drugs = 11285
features_dim = 512 #特征维度
features_o = np.random.rand(num_drugs,features_dim).astype(np.float32)

print(features1[1].shape,type(features1))
print(features_o.shape,type(features_o))
print(features1[1:10])
print(features_o[1:10])