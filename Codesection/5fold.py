import os
import torch
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for k in range(5):

    os.system('python main.py --out_file test.txt --zhongzi '+str(k))