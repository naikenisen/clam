# Configuration parameters for training
data_root_dir = None
embed_dim = 1024
max_epochs = 200
lr = 1e-4
label_frac = 1.0
reg = 1e-5
seed = 1
k = 10
k_start = -1
k_end = -1
results_dir = './results'
split_dir = None
log_data = False
testing = False
early_stopping = False
opt = 'adam'
drop_out = 0.25
bag_loss = 'ce'
model_type = 'clam_sb'
exp_code = None
weighted_sample = False
model_size = 'small'
task = None
no_inst_cluster = False
inst_loss = None
subtyping = False
bag_weight = 0.7
B = 8

# Patch creation parameters
source = None  # path to folder containing raw wsi image files
step_size = 256
patch_size = 256
patch = False
seg = False
stitch = False
auto_skip = True
save_dir = None  # directory to save processed data
preset = None  # predefined profile of default segmentation and filter parameters (.csv)
patch_level = 0  # downsample level at which to patch
custom_downsample = 1  # custom downscale when native downsample is not available
process_list = None  # name of list of images to process with parameters (.csv)

# Split creation parameters
val_frac = 0.1  # fraction of labels for validation
test_frac = 0.1  # fraction of labels for test

# Feature extraction parameters
data_dir = None  # directory containing patch data
csv_path = None  # path to CSV file with slide information
feat_dir = None  # directory to save extracted features
model_name = 'resnet50_trunc'  # model for feature extraction: resnet50_trunc, uni_v1, conch_v1
batch_size = 256  # batch size for feature extraction
slide_ext = '.svs'  # slide file extension
no_auto_skip = False  # don't skip slides that already have features
target_patch_size = 224  # desired size of patches for scaling before feature embedding