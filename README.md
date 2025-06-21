# MRCGN
(代码数据中的初始特征是借用一种蒸馏的方法提取的，详见仓库https://github.com/HongxinXiang/IEM.git)
# Abstract
暂无
# Environment
## GPU environment
CUDA 11.0

## create a new conda environment
- conda create -n rgcn python=3.7.10
- conda activate rgcn
  
## Requirements
- numpy==1.18.5
- torch==1.7.1+cu110
- torchvision==0.8.2+cu110
- torchaudio==0.7.2
- torch-geometric==2.0.0
- torch-scatter==2.0.7
- torch-sparse==0.6.9

## install environment
This repositories is built based on python == 3.8.19. You could simply run

`pip install -r requirements.txt`

to install other packages.

# Datasets
| #名称 | #药物数量 | #DDI种类数量 |
| :---: | :---: | :---: |
| Deng  | 572 | 65 |
| Ryu | 1700 | 86 |

# 🌟Quick Run
在代码目录下运行下面这个命令。
```
python 5fold.py
```
结果在文件夹中的result.txt中。

## 更换数据集
将data中的数据换成想要的数据集，再修改parms_setting.py中的type_number即可。
