# CLAM - Version Simplifiée et Épurée

Structure ultra-simplifiée du projet CLAM avec fichier de configuration centralisé.

**858 lignes de code** au total pour toutes les fonctionnalités essentielles ! 🎯

## 📁 Structure du Projet

```
clam/
├── config.py          # ⭐ Configuration centralisée
├── train.py           # Script d'entraînement
├── evaluate.py        # Script d'évaluation  
├── infer.py           # Script d'inférence
├── src/               # Modules principaux
│   ├── models.py      # Modèles CLAM (SB & MB)
│   ├── dataset.py     # Gestion des données
│   └── utils.py       # Utilitaires
├── dataset_csv/       # Fichiers CSV des datasets
├── splits/            # Splits train/val/test
├── env.yml            # Environnement conda
└── README.md          # Ce fichier
```

## 🚀 Utilisation Ultra-Simple

### 1. Configurer votre expérience

Éditez le fichier `config.py` :

```python
# config.py
class Config:
    # Chemins
    DATA_ROOT = '/path/to/your/features'
    RESULTS_DIR = './results'
    SPLIT_DIR = './splits/task_1_tumor_vs_normal_75'
    
    # Tâche
    TASK = 'tumor_vs_normal'
    
    # Modèle
    MODEL_TYPE = 'clam_sb'
    
    # Entraînement
    MAX_EPOCHS = 200
    LEARNING_RATE = 1e-4
    # ... etc
```

### 2. Lancer l'entraînement

```bash
python train.py
```

C'est tout ! Plus besoin de passer 20 arguments dans le terminal 🎉

### 3. Évaluer

```bash
python evaluate.py
```

### 4. Inférence

```bash
python infer.py --input slide.h5
```

## 📁 Structure

```
├── config.py              # ⭐ Configuration centralisée
├── train.py              # Entraînement (utilise config.py)
├── evaluate.py           # Évaluation (utilise config.py)
├── infer.py              # Inférence
└── src/
    ├── models.py         # Modèles CLAM
    ├── dataset.py        # Gestion des données
    └── utils.py          # Utilitaires
```

## 🎯 Configurations Prédéfinies

Le fichier `config.py` inclut plusieurs configurations prêtes à l'emploi :

```python
# Pour tumor vs normal
from config import TumorVsNormalConfig
ACTIVE_CONFIG = TumorVsNormalConfig

# Pour tumor subtyping
from config import TumorSubtypingConfig
ACTIVE_CONFIG = TumorSubtypingConfig

# Personnalisée
from config import ExperimentConfig
ACTIVE_CONFIG = ExperimentConfig
```

## ✏️ Créer Votre Propre Configuration

```python
# Dans config.py
class MyExperiment(Config):
    DATA_ROOT = '/my/data/path'
    RESULTS_DIR = './results/my_exp'
    MAX_EPOCHS = 100
    LEARNING_RATE = 2e-4
    # Override uniquement ce dont vous avez besoin

# Puis l'activer
ACTIVE_CONFIG = MyExperiment
```

## 🔍 Voir la Configuration Active

```bash
python config.py
```

Affiche tous les paramètres actifs.

## 🆚 Avantages vs Arguments Terminal

**Avant** (avec arguments terminal) :
```bash
python train.py \
    --data_root /path/to/features \
    --task tumor_vs_normal \
    --split_dir splits/task_1 \
    --model_type clam_sb \
    --embed_dim 1024 \
    --dropout 0.25 \
    --max_epochs 200 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --bag_weight 0.7 \
    --patience 20 \
    --k 10 \
    --results_dir results/exp1
```

**Maintenant** (avec config.py) :
```bash
python train.py
```

✅ Plus simple  
✅ Réutilisable  
✅ Versionnable avec Git  
✅ Moins d'erreurs de frappe  
✅ Configuration visible et documentée  
✅ Partage facile entre expériences  

## 📊 Workflow Typique

1. **Créer une config** pour votre expérience dans `config.py`
2. **Lancer** avec `python train.py`
3. **Évaluer** avec `python evaluate.py`
4. **Répéter** en modifiant juste la config

## ⚙️ Options Avancées

Si vous avez vraiment besoin d'arguments terminal pour un cas spécifique, vous pouvez toujours utiliser les anciens scripts `train_new.py` et `eval_new.py`.
