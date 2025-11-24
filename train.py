import pdb
import os
import math
from random import seed
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from src.config import *

def main():
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
        seed_torch(seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(results_dir, save_name))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': k, 
            'k_start': k_start,
            'k_end': k_end,
            'task': task,
            'max_epochs': max_epochs, 
            'results_dir': results_dir, 
            'lr': lr,
            'experiment': exp_code,
            'reg': reg,
            'label_frac': label_frac,
            'bag_loss': bag_loss,
            'seed': seed,
            'model_type': model_type,
            'model_size': model_size,
            "use_drop_out": drop_out,
            'weighted_sample': weighted_sample,
            'opt': opt}

if model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': bag_weight,
                    'inst_loss': inst_loss,
                    'B': B})

print('\nLoad Dataset')

if task == 'task_1_tumor_vs_normal':
    n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            seed = seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif task == 'task_2_tumor_subtyping':
    n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            seed = seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])

    if model_type in ['clam_sb', 'clam_mb']:
        assert subtyping 
        
else:
    raise NotImplementedError
    
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

settings.update({'split_dir': split_dir})


with open(results_dir + '/experiment_{}.txt'.format(exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main()
    print("finished!")
    print("end script")


