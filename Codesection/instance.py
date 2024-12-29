import os
import random
import torch
import numpy as np
from torch.optim import Adam
from layer import MRCGNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1)

def Create_model(args):
    model = MRCGNN(
        feature=args.dimensions,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        hidden3=args.hidden3,
        dropout=args.dropout,
        type_n=args.type_number
        )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer