
# ======================== PARAMÈTRES D'ENTRAÎNEMENT ========================
data_root_dir = None           # Répertoire racine contenant les données (features extraites)
embed_dim = 1024               # Dimension des embeddings de features
max_epochs = 200               # Nombre maximum d'époques d'entraînement
lr = 1e-4                      # Taux d'apprentissage (learning rate)
label_frac = 1.0               # Fraction des labels à utiliser pour l'entraînement (1.0 = 100%, 0.1 = 10%)
reg = 1e-5                     # Régularisation L2 (weight decay)
seed = 1                       # Graine aléatoire pour la reproductibilité
k = 10                         # Nombre de folds pour la validation croisée
k_start = -1                   # Index de départ pour les folds (défaut: -1 = démarrer à 0)
k_end = -1                     # Index de fin pour les folds (défaut: -1 = aller jusqu'à k)
results_dir = './results'      # Répertoire où sauvegarder les résultats d'entraînement
split_dir = None               # Répertoire contenant les fichiers de splits (train/val/test)
log_data = False               # Activer le logging des données (ex: avec wandb)
testing = False                # Mode test uniquement (sans entraînement)
early_stopping = False         # Activer l'arrêt anticipé (early stopping) basé sur la validation
opt = 'adam'                   # Optimiseur à utiliser ('adam', 'sgd', etc.)
drop_out = 0.25                # Taux de dropout pour la régularisation
bag_loss = 'ce'                # Fonction de perte pour les bags ('ce' = cross-entropy, 'svm' = hinge loss)
model_type = 'clam_sb'         # Type de modèle CLAM ('clam_sb' = single branch, 'clam_mb' = multi branch)
exp_code = None                # Code d'expérience pour identifier les runs
weighted_sample = False        # Utiliser l'échantillonnage pondéré pour équilibrer les classes
model_size = 'small'           # Taille du modèle ('small' ou 'big')
task = None                    # Tâche à effectuer (ex: 'task_1_tumor_vs_normal', 'task_2_tumor_subtyping')
no_inst_cluster = False        # Désactiver le clustering des instances
inst_loss = None               # Fonction de perte pour les instances (None, 'svm', 'ce')
subtyping = False              # Mode sous-typage (classification multi-classes)
bag_weight = 0.7               # Poids de la perte des bags dans la perte totale (entre 0 et 1)
B = 8                          # Nombre d'instances positives et négatives à échantillonner par bag

# ======================== PARAMÈTRES DE CRÉATION DE PATCHES ========================
source = None                  # Répertoire source contenant les fichiers d'images WSI brutes
step_size = 256                # Pas de déplacement entre patches adjacents (en pixels)
patch_size = 256               # Taille des patches à extraire (en pixels)
patch = False                  # Activer l'extraction de patches
seg = False                    # Activer la segmentation du tissu
stitch = False                 # Activer la création de visualisations cousues (stitched)
auto_skip = True               # Sauter automatiquement les slides déjà traités
save_dir = None                # Répertoire où sauvegarder les données traitées (patches, masks, stitches)
preset = None                  # Nom du fichier preset CSV contenant les paramètres prédéfinis (optionnel)
patch_level = 0                # Niveau de résolution pour l'extraction de patches (0 = résolution maximale)
custom_downsample = 1          # Facteur de downsampling personnalisé quand non disponible nativement (1 ou 2)
process_list = None            # Nom du fichier CSV listant les images à traiter avec leurs paramètres (optionnel)

# ======================== PARAMÈTRES DE SPLIT ========================
val_frac = 0.1                 # Fraction des données à utiliser pour la validation (0.1 = 10%)
test_frac = 0.1                # Fraction des données à utiliser pour le test (0.1 = 10%)

# ======================== PARAMÈTRES D'EXTRACTION DE FEATURES ========================
data_dir = None                # Répertoire contenant les données de patches (dossier 'patches')
csv_path = None                # Chemin vers le fichier CSV listant les slides à traiter
feat_dir = None                # Répertoire où sauvegarder les features extraites
model_name = 'resnet50_trunc'  # Modèle à utiliser pour l'extraction de features ('resnet50_trunc', 'uni_v1', 'conch_v1')
batch_size = 256               # Taille du batch pour l'extraction de features
slide_ext = '.svs'             # Extension des fichiers de slides (ex: '.svs', '.tif', '.ndpi')
no_auto_skip = False           # Ne pas sauter les slides qui ont déjà des features extraites
target_patch_size = 224        # Taille cible des patches pour le redimensionnement avant extraction (doit correspondre à la taille attendue par le modèle)