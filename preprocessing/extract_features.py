import time
import os
import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import numpy as np
from src.file_utils import save_hdf5
from src.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag, get_eval_transforms
import src.config as config

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from src.transform_utils import get_eval_transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
    "conch_v1_5":
    {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	}
}
def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50_trunc':
        model = TimmCNNEncoder()
    elif model_name == 'uni_v1':
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'conch_v1_5':
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install huggingface transformers (e.g. 'pip install transformers') to use CONCH v1.5")
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 448, 'TITAN is used with 448x448 CONCH v1.5 features'
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms


def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


if __name__ == '__main__':

	print('initializing dataset')
	bags_dataset = Dataset_All_Bags(config.csv_path)
	
	os.makedirs(config.feat_dir, exist_ok=True)
	dest_files = os.listdir(config.feat_dir)

	model, img_transforms = get_encoder(config.model_name, target_img_size=config.target_patch_size)		
	model = model.to(device)
	_ = model.eval()

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
	
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(config.slide_ext)[0]
		bag_name = slide_id + '.h5'
		bag_candidate = os.path.join(config.data_dir, 'patches', bag_name)

		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(bag_name)
		if not config.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(config.feat_dir, 'h5_files', bag_name)
		file_path = bag_candidate
		time_start = time.time()

		dataset = Whole_Slide_Bag(file_path=file_path, img_transforms=img_transforms)
		loader = DataLoader(dataset=dataset, batch_size=config.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)

		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(config.feat_dir, 'pt_files', bag_base+'.pt'))
