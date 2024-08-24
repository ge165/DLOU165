import torch
import numpy as np

from parms_setting import settings
from data_preprocess import load_data
from instantiation import Create_model
from train import train_model
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #一些随机种子，确保每次运行的结果是准确的
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False #设置了cudnn后端，以控制确定性和性能
    torch.backends.cudnn.deterministic = True

set_random_seed(1, deterministic=False)

# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()

#不同的数据集和用于训练，验证和测试的数据加载器
data_o, train_loader, val_loader, test_loader = load_data(args)

# train and test model
#创建模型和优化器
model, optimizer = Create_model(args)
train_model(model, optimizer, data_o, train_loader, val_loader, test_loader, args)