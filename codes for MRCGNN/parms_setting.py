import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os
import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import torch
import random
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
#定义和解析命令行参数，以便在执行脚本时可以动态设置各种参数
def settings():
    #创建一个参数解析器对象，并添加多个命令行参数
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')

    parser.add_argument('--out_file', required=True,default='result.txt',
                        help='Path to data result file. e.g., result.txt')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate. Default is 5e-4.')
#
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')

    parser.add_argument('--weight_decay', default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')
#
    parser.add_argument('--batch', type=int, default=512,
                        help='Batch size. Default is 512.')
#
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train. Default is 30.')

    parser.add_argument('--network_ratio', type=float, default=1,
                        help='Remain links in network. Default is 1')

    parser.add_argument('--loss_ratio1', type=float, default=1,
                        help='Ratio of task1. Default is 1')
###
    parser.add_argument('--loss_ratio2', type=float, default=0.05,
                        help='Ratio of task2. Default is 0.1')
##
    parser.add_argument('--loss_ratio3', type=float, default=0.1,
                        help='Ratio of task3. Default is 0.1')
#
    # GCN parameters#
    parser.add_argument('--dimensions', type=int, default=512,
                        help='dimensions of feature. Default is 128.')

    parser.add_argument('--hidden1', default=64,
                        help='Number of hidden units for encoding layer 1 for CSGNN. Default is 64.')
#
    parser.add_argument('--hidden2', default=32,
                        help='Number of hidden units for encoding layer 2 for CSGNN. Default is 32.')

    parser.add_argument('--decoder1', default=512,
                        help='Number of hidden units for decoding layer 1 for CSGNN. Default is 512.')
    parser.add_argument('--zhongzi', default=0,
                        help='Number of zhongzi.')
    #解析命令行输入的参数，最终返回解析后的参数对象
    args = parser.parse_args()

    return args
