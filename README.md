# MRCGN
(The initial features in the code data were extracted using a distillation-based methodology, as detailed in the repository: https://github.com/HongxinXiang/IEM.git)
# Abstract
......
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
This repositories is built based on python == 3.7.10. You could simply run

`pip install -r requirements.txt`

to install other packages.

# Datasets
| #name | #totality| #DDItype |
| :---: | :---: | :---: |
| Deng  | 572 | 65 |
| Ryu | 1700 | 86 |

# Quick Run
Run the following command in the code directory.
```
python 5fold.py
```
The result is in the "result.txt" file within the folder.

## Change the dataset
Replace the data in "data" with the desired dataset, and then modify "type_number" in "parms_setting.py".
