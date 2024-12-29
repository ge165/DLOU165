import torch
import numpy as np
import os
import random
from parms_setting import settings
from data_preprocess import load_data
from instance import Create_model
from train import train_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(1)
args = settings()
args.cuda = not args.no_cuda and torch.cuda.is_available()

data_o, train_loader, val_loader, test_loader = load_data(args)

model, optimizer = Create_model(args)

train_model(model, optimizer, data_o, train_loader, val_loader, test_loader, args)