from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *
import pandas as pd
import csv
import random
import numpy as np

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1)

class Dataclass(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.relation_type=triple[:,2]

    def __len__(self):
        return len(self.relation_type)

    def __getitem__(self, index):

        return  (self.entity1[index], self.entity2[index], self.relation_type[index])

def load_data(args):
    drug_list = []
    with open('data/drug_smiles.csv', 'r') as f:
        reader = csv.reader(f)
        drug_list = [row[0] for row in reader]

    type_n = args.type_number
    def load_train_val_test():
        def process_data(file_path):
            data = pd.read_csv(file_path)
            data_pos = [(h, t, r) for h, t, r in zip(data['d1'], data['d2'], data['type'])]
            np.random.shuffle(data_pos)
            data_pos = np.array(data_pos)

            for i in range(data_pos.shape[0]):
                data_pos[i][0] = int(drug_list.index(data_pos[i][0]))
                data_pos[i][1] = int(drug_list.index(data_pos[i][1]))
                data_pos[i][2] = int(data_pos[i][2])

            label_list = np.zeros((data_pos.shape[0], type_n))
            for i in range(data_pos.shape[0]):
                label_list[i][int(data_pos[i][2])] = 1

            return np.concatenate([data_pos, label_list], axis=1)

        zhongzi = args.zhongzi
        train_data = process_data(f'data/{zhongzi}/ddi_training1.csv')
        val_data = process_data(f'data/{zhongzi}/ddi_validation1.csv')
        test_data = process_data(f'data/{zhongzi}/ddi_test1.csv')
        
        return train_data, val_data, test_data
    
    train_data,val_data,test_data=load_train_val_test()
    params = {'batch_size': args.batch, 'shuffle': False, 'num_workers': args.workers, 'drop_last': False}

    train_loader = DataLoader(Dataclass(train_data), **params)
    val_loader = DataLoader(Dataclass(val_data), **params)
    test_loader = DataLoader(Dataclass(test_data), **params)
    
    print('Extracting features...')

    data = np.load('data/feat_1.npz')
    features = data['feats']
    ids = data['drug_id'].tolist()
    features_o = np.array([features[ids.index(drug)] for drug in drug_list])

    args.dimensions = features_o.shape[1]
    
    x_o = torch.tensor(features_o, dtype=torch.float)

    edge_index_o, label_list = [], []
    for h, t, r in train_data[:, :3]:
        edge_index_o.append([int(h), int(t)])
        edge_index_o.append([int(t), int(h)])
        label_list.append(int(r))
        label_list.append(int(r))
    
    edge_index_o = torch.tensor(edge_index_o, dtype=torch.long)

    data_o = Data(x=x_o, edge_index=edge_index_o.t().contiguous(), edge_type=label_list)

    print('Loading finished!')
    return data_o, train_loader, val_loader, test_loader