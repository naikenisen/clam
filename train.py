import os
from random import seed
from utils.file_utils import save_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_MIL_Dataset
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from src.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes=2
dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                        data_dir= os.path.join(data_root_dir, 'tumor_vs_normal_resnet_features'),
                        shuffle = False, 
                        seed = seed, 
                        print_info = True,
                        label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                        patient_strat=False,
                        ignore=[])

if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

results_dir = os.path.join(results_dir, str(exp_code) + '_s{}'.format(seed))
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
if split_dir is None:
    split_dir = os.path.join('splits', task+'_{}'.format(int(label_frac*100)))
else:
    split_dir = os.path.join('splits', split_dir)
print('split_dir: ', split_dir)
assert os.path.isdir(split_dir)

if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

if k_start == -1:
    start = 0
else:
    start = k_start
if k_end == -1:
    end = k
else:
    end = k_end

all_test_auc = []
all_val_auc = []
all_test_acc = []
all_val_acc = []
folds = np.arange(start, end)
for i in folds:
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
            csv_path='{}/splits_{}.csv'.format(split_dir, i))
    
    datasets = (train_dataset, val_dataset, test_dataset)
    results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, n_classes)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    filename = os.path.join(results_dir, 'split_{}_results.pkl'.format(i))
    save_pkl(filename, results)

final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
    'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})

if len(folds) != k:
    save_name = 'summary_partial_{}_{}.csv'.format(start, end)
else:
    save_name = 'summary.csv'
final_df.to_csv(os.path.join(results_dir, save_name))

print("finished!")
print("end script")
