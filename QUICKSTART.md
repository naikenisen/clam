# 🚀 Guide de Démarrage Rapide

## Installation (2 minutes)

```bash
# 1. Créer l'environnement
conda env create -f env.yml
conda activate clam

# 2. Ou installation manuelle
pip install torch torchvision pandas numpy scikit-learn h5py tqdm
```

## Premier Entraînement (3 étapes)

### 1️⃣ Configurer votre expérience

Ouvrez `config.py` et modifiez :

```python
class Config:
    # Chemin vers vos features extraites
    DATA_ROOT = '/path/to/your/features'
    
    # Tâche (tumor_vs_normal ou tumor_subtyping)
    TASK = 'tumor_vs_normal'
    
    # Où sauvegarder les résultats
    RESULTS_DIR = './results/exp_001'
    
    # Le reste peut rester par défaut !
```

### 2️⃣ Lancer l'entraînement

```bash
python train.py
```

C'est tout ! Le script va :
- Charger automatiquement vos données
- Entraîner sur tous les folds (cross-validation)
- Sauvegarder les meilleurs modèles
- Générer un résumé des performances

### 3️⃣ Évaluer le modèle

```bash
python evaluate.py
```

Résultats disponibles dans `eval_results/`

## Inférence sur Nouveaux Slides

```bash
python infer.py \
    --checkpoint results/exp_001/fold_0_best.pt \
    --model_type clam_sb \
    --n_classes 2 \
    --input /path/to/slide.h5 \
    --output_dir predictions/
```

## 📝 Exemples de Configurations

### Tumor vs Normal (Binaire)

```python
class MyExperiment(Config):
    DATA_ROOT = '/data/features'
    TASK = 'tumor_vs_normal'
    SPLIT_DIR = './splits/task_1_tumor_vs_normal_75'
    RESULTS_DIR = './results/tumor_normal_exp1'
    
    MODEL_TYPE = 'clam_sb'
    MAX_EPOCHS = 200
    LEARNING_RATE = 1e-4

ACTIVE_CONFIG = MyExperiment
```

### Tumor Subtyping (Multi-classe)

```python
class SubtypingExperiment(Config):
    DATA_ROOT = '/data/features'
    TASK = 'tumor_subtyping'
    SPLIT_DIR = './splits/task_2_tumor_subtyping_50'
    RESULTS_DIR = './results/subtyping_exp1'
    
    MODEL_TYPE = 'clam_mb'  # Multi-branch pour multi-classe
    MAX_EPOCHS = 150
    LEARNING_RATE = 2e-4

ACTIVE_CONFIG = SubtypingExperiment
```

### Expérience Rapide (Debug)

```python
class QuickTest(Config):
    DATA_ROOT = '/data/features'
    TASK = 'tumor_vs_normal'
    RESULTS_DIR = './results/quick_test'
    
    MAX_EPOCHS = 10  # Peu d'epochs
    N_FOLDS = 3      # Seulement 3 folds
    FOLD_START = 0
    FOLD_END = 1     # Un seul fold pour tester

ACTIVE_CONFIG = QuickTest
```

## 🎯 Workflow Typique

```bash
# 1. Préparer une nouvelle expérience
vim config.py  # Créer une nouvelle classe de config

# 2. Vérifier la config
python config.py  # Affiche tous les paramètres

# 3. Lancer l'entraînement
python train.py  # Lance le training

# 4. Évaluer
python evaluate.py  # Évalue tous les folds

# 5. Analyser les résultats
cat results/exp_001/summary.csv
cat eval_results/exp_001/evaluation_summary.csv

# 6. Inférence sur nouveaux slides
python infer.py --checkpoint results/exp_001/fold_0_best.pt \
                --input new_slides/ \
                --n_classes 2
```

## 📊 Comprendre les Résultats

### Pendant l'entraînement

```
Epoch 1/200 - Train Loss: 0.6234 - Val Loss: 0.5123 - Val Acc: 0.7500 - Val AUC: 0.8234
Epoch 2/200 - Train Loss: 0.5234 - Val Loss: 0.4823 - Val Acc: 0.7800 - Val AUC: 0.8456
...
```

### Résumé final

```
=== Final Results ===
Test Acc: 0.8234 ± 0.0312
Test AUC: 0.8756 ± 0.0234
```

### Fichiers générés

```
results/exp_001/
├── fold_0_best.pt          # Meilleur modèle fold 0
├── fold_1_best.pt          # Meilleur modèle fold 1
├── ...
└── summary.csv             # Résultats tous folds

eval_results/exp_001/
├── fold_0_test_predictions.csv  # Prédictions détaillées
├── fold_1_test_predictions.csv
├── ...
└── evaluation_summary.csv       # Résumé évaluation
```

## ⚙️ Paramètres Principaux

| Paramètre | Valeur par défaut | Description |
|-----------|------------------|-------------|
| `MAX_EPOCHS` | 200 | Nombre max d'epochs |
| `LEARNING_RATE` | 1e-4 | Taux d'apprentissage |
| `DROPOUT` | 0.25 | Dropout rate |
| `BAG_WEIGHT` | 0.7 | Poids de la bag loss |
| `PATIENCE` | 20 | Epochs avant early stop |
| `MODEL_TYPE` | 'clam_sb' | clam_sb ou clam_mb |

## 🐛 Debugging

### Voir la configuration active
```bash
python config.py
```

### Tester sur un seul fold
```python
class DebugConfig(Config):
    FOLD_START = 0
    FOLD_END = 1  # Un seul fold
    MAX_EPOCHS = 5  # Peu d'epochs
```

### Vérifier les données
```python
from src.dataset import create_datasets
dataset, n_classes, label_dict = create_datasets('tumor_vs_normal', '/data/features')
print(f"Dataset size: {len(dataset)}")
print(f"Classes: {label_dict}")
```

## 💡 Astuces

1. **Commencez petit** : Testez d'abord sur 1 fold avec peu d'epochs
2. **Utilisez des noms descriptifs** : `RESULTS_DIR = './results/exp_clamsb_lr0001'`
3. **Sauvegardez vos configs** : Créez une classe par expérience
4. **Versionnez** : Commitez `config.py` après chaque expérience réussie
5. **Logs** : Les résultats sont auto-sauvegardés, pas besoin de tout noter

## ❓ Besoin d'Aide ?

- Voir `README.md` pour plus de détails
- Voir `CLEANUP.md` pour comprendre les changements
- Les erreurs sont généralement liées aux chemins dans `config.py`

Bon entraînement ! 🎉
