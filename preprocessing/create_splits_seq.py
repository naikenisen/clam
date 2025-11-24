import pdb
import os
import pandas as pd
from src.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import src.config as config
import numpy as np

n_classes = 2
dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                        shuffle = False, 
                        seed = config.seed, 
                        print_info = True,
                        label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                        patient_strat=True,
                        ignore=[])

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * config.val_frac).astype(int)
test_num = np.round(num_slides_cls * config.test_frac).astype(int)

if config.label_frac > 0:
    label_fracs = [config.label_frac]
else:
    label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

for lf in label_fracs:
    split_dir = 'splits/'+ str(config.task) + '_{}'.format(int(lf * 100))
    os.makedirs(split_dir, exist_ok=True)
    dataset.create_splits(k = config.k, val_num = val_num, test_num = test_num, label_frac=lf)
    for i in range(config.k):
        dataset.set_splits()
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        splits = dataset.return_splits(from_id=True)
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
        save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
        descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
