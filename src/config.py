# Training parameters
# Répertoire racine contenant les données (features extraites)
data_root_dir = None
# Dimension des embeddings de features
embed_dim = 1024
# Nombre maximum d'époques d'entraînement
max_epochs = 200
# Taux d'apprentissage (learning rate)
lr = 1e-4
# Fraction des labels à utiliser pour l'entraînement (1.0 = 100%, 0.1 = 10%)
label_frac = 1.0
# Régularisation L2 (weight decay)
reg = 1e-5
# Seed aléatoire pour la reproductibilité
seed = 1
# Nombre de folds pour la validation croisée
k = 10
# Index de départ pour les folds (défaut: -1 = démarrer à 0)
k_start = -1
# Index de fin pour les folds (défaut: -1 = aller jusqu'à k)
k_end = -1
# Répertoire où sauvegarder les résultats d'entraînement
results_dir = './results'
# Répertoire contenant les fichiers de splits (train/val/test)
split_dir = None
# Activer le logging des données (ex: avec wandb)
log_data = False
# Mode test uniquement (sans entraînement)
testing = False
# Activer l'arrêt anticipé (early stopping) basé sur la validation
early_stopping = False
# Optimiseur à utiliser ('adam', 'sgd', etc.)
opt = 'adam'
# Taux de dropout pour la régularisation
drop_out = 0.25
# Fonction de perte pour les bags ('ce' = cross-entropy, 'svm' = hinge loss)
bag_loss = 'ce'
# Type de modèle CLAM ('clam_sb' = single branch, 'clam_mb' = multi branch)
model_type = 'clam_sb'
# Code d'expérience pour identifier les runs
exp_code = None
# Utiliser l'échantillonnage pondéré pour équilibrer les classes
weighted_sample = False
# Taille du modèle ('small' ou 'big')
model_size = 'small'
# Tâche à effectuer (ex: 'task_1_tumor_vs_normal', 'task_2_tumor_subtyping')
task = None
# Désactiver le clustering des instances
no_inst_cluster = False
# Fonction de perte pour les instances (None, 'svm', 'ce')
inst_loss = None
# Mode sous-typage (classification multi-classes)
subtyping = False
# Poids de la perte des bags dans la perte totale (entre 0 et 1)
bag_weight = 0.7
# Nombre d'instances positives et négatives à échantillonner par bag
B = 8

# Patch creation parameters
# Répertoire source contenant les fichiers d'images WSI brutes
source = None
# Pas de déplacement entre patches adjacents (en pixels)
step_size = 256
# Taille des patches à extraire (en pixels)
patch_size = 256
# Activer l'extraction de patches
patch = False
# Activer la segmentation du tissu
seg = False
# Activer la création de visualisations cousues (stitched)
stitch = False
# Sauter automatiquement les slides déjà traités
auto_skip = True
# Répertoire où sauvegarder les données traitées (patches, masks, stitches)
save_dir = None
# Nom du fichier preset CSV contenant les paramètres prédéfinis (optionnel)
preset = None
# Niveau de résolution pour l'extraction de patches (0 = résolution maximale)
patch_level = 0
# Facteur de downsampling personnalisé quand non disponible nativement (1 ou 2)
custom_downsample = 1
# Nom du fichier CSV listant les images à traiter avec leurs paramètres (optionnel)
process_list = None

# Split creation parameters
# Fraction des données à utiliser pour la validation (0.1 = 10%)
val_frac = 0.1
# Fraction des données à utiliser pour le test (0.1 = 10%)
test_frac = 0.1


# Feature extraction parameters
# Répertoire contenant les données de patches (dossier 'patches')
data_dir = None
# Chemin vers le fichier CSV listant les slides à traiter
csv_path = None
# Répertoire où sauvegarder les features extraites
feat_dir = None
# Modèle à utiliser pour l'extraction de features
# Options: 'resnet50_trunc', 'uni_v1', 'conch_v1'
model_name = 'resnet50_trunc'
# Taille du batch pour l'extraction de features
batch_size = 256
# Extension des fichiers de slides (ex: '.svs', '.tif', '.ndpi')
slide_ext = '.svs'
# Ne pas sauter les slides qui ont déjà des features extraites
no_auto_skip = False
# Taille cible des patches pour le redimensionnement avant extraction
# (doit correspondre à la taille attendue par le modèle)
target_patch_size = 224