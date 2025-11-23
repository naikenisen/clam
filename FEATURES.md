# 📦 Fonctionnalités Disponibles

## 🎯 Vue d'Ensemble

Cette version simplifiée conserve **100% des fonctionnalités essentielles** de CLAM dans seulement **858 lignes de code**.

---

## 🧠 Modèles

### CLAM-SB (Single Branch)
- ✅ Attention mechanism avec gating
- ✅ Instance-level clustering sur une branche
- ✅ Bag-level et instance-level loss
- ✅ Classification multi-classe
- ✅ Features pooling avec attention

**Utilisation** : Problèmes de classification standard

### CLAM-MB (Multi Branch)
- ✅ Attention mechanism multi-branches
- ✅ Une branche d'attention par classe
- ✅ Instance-level clustering par classe
- ✅ Idéal pour subtyping
- ✅ Meilleure interprétabilité

**Utilisation** : Problèmes de subtyping complexes

---

## 📊 Entraînement

### Cross-Validation
- ✅ K-fold cross-validation
- ✅ Configuration flexible du nombre de folds
- ✅ Possibilité de n'entraîner que certains folds
- ✅ Résultats agrégés automatiquement

### Optimisation
- ✅ Optimiseur Adam
- ✅ Learning rate configurable
- ✅ Weight decay (régularisation L2)
- ✅ Early stopping avec patience configurable
- ✅ Sauvegarde automatique du meilleur modèle

### Loss Functions
- ✅ Bag-level loss (CrossEntropy)
- ✅ Instance-level loss (CrossEntropy)
- ✅ Pondération configurable bag/instance
- ✅ Support multi-classe

### Monitoring
- ✅ Affichage des métriques à chaque epoch
- ✅ Train loss, validation loss
- ✅ Accuracy et AUC
- ✅ Early stopping automatique

---

## 📈 Évaluation

### Métriques
- ✅ Accuracy (par classe et globale)
- ✅ AUC-ROC (binaire et multi-classe)
- ✅ Prédictions détaillées par slide
- ✅ Probabilités de classe

### Modes d'Évaluation
- ✅ Évaluation sur test set
- ✅ Évaluation sur validation set
- ✅ Évaluation sur train set
- ✅ Évaluation fold par fold
- ✅ Évaluation agrégée tous folds

### Outputs
- ✅ Fichiers CSV avec prédictions détaillées
- ✅ Résumé statistique (moyenne ± std)
- ✅ Sauvegarde automatique des résultats

---

## 🔮 Inférence

### Capacités
- ✅ Prédiction sur slides individuels
- ✅ Prédiction par batch (dossier de slides)
- ✅ Extraction des probabilités par classe
- ✅ Sauvegarde des scores d'attention (optionnel)

### Formats Supportés
- ✅ Fichiers H5 (features pré-extraites)
- ✅ Traitement de multiples slides
- ✅ Gestion des erreurs par slide

### Outputs
- ✅ CSV avec prédictions et probabilités
- ✅ Scores d'attention (format numpy)
- ✅ Résumé des prédictions

---

## 💾 Gestion des Données

### Datasets
- ✅ Chargement depuis CSV
- ✅ Support fichiers H5
- ✅ Mapping automatique des labels
- ✅ Splits train/val/test

### Preprocessing
- ✅ Chargement automatique des features
- ✅ Gestion des batches
- ✅ Shuffle configurablr
- ✅ Multi-threading (num_workers)

### Tâches Supportées
- ✅ Classification binaire (tumor vs normal)
- ✅ Classification multi-classe (subtyping)
- ✅ Configuration facile via `config.py`

---

## ⚙️ Configuration

### Système de Config Centralisé
- ✅ Un seul fichier (`config.py`)
- ✅ Configs prédéfinies (TumorVsNormal, TumorSubtyping)
- ✅ Création facile de configs personnalisées
- ✅ Héritage de configurations
- ✅ Affichage de la config active

### Paramètres Configurables

#### Chemins
- `DATA_ROOT` : Dossier des features
- `RESULTS_DIR` : Dossier de sauvegarde
- `SPLIT_DIR` : Dossier des splits

#### Modèle
- `MODEL_TYPE` : clam_sb ou clam_mb
- `EMBED_DIM` : Dimension des embeddings
- `DROPOUT` : Taux de dropout

#### Entraînement
- `MAX_EPOCHS` : Nombre max d'epochs
- `LEARNING_RATE` : Learning rate
- `WEIGHT_DECAY` : Régularisation
- `BAG_WEIGHT` : Pondération loss
- `PATIENCE` : Early stopping

#### Cross-Validation
- `N_FOLDS` : Nombre de folds
- `FOLD_START` : Fold de départ
- `FOLD_END` : Fold de fin

#### Autres
- `SEED` : Random seed
- `NUM_WORKERS` : Threads chargement
- `BATCH_SIZE` : Taille batch (=1 pour MIL)

---

## 🛠️ Utilitaires

### Reproductibilité
- ✅ Random seed pour tous les générateurs
- ✅ Déterminisme CUDA
- ✅ Seeds configurables

### Métriques
- ✅ Calcul automatique accuracy
- ✅ Calcul automatique AUC
- ✅ Support binaire et multi-classe
- ✅ Gestion des cas edge

### Early Stopping
- ✅ Monitoring de la validation loss
- ✅ Patience configurable
- ✅ Epoch minimum avant stop
- ✅ Sauvegarde automatique meilleur modèle

### Average Meter
- ✅ Calcul moyennes en temps réel
- ✅ Tracking des métriques
- ✅ Mise à jour incrémentale

---

## 📦 Outputs Générés

### Pendant l'Entraînement
```
results/
└── experiment_name/
    ├── fold_0_best.pt
    ├── fold_1_best.pt
    ├── ...
    └── summary.csv
```

### Pendant l'Évaluation
```
eval_results/
└── experiment_name/
    ├── fold_0_test_predictions.csv
    ├── fold_1_test_predictions.csv
    ├── ...
    └── evaluation_summary.csv
```

### Pendant l'Inférence
```
predictions/
├── predictions.csv
├── slide1_attention.npy
├── slide2_attention.npy
└── ...
```

---

## 🚫 Fonctionnalités NON Incluses

Ces fonctionnalités ont été retirées car non essentielles :

### Extraction de Features
- ❌ Extraction depuis images brutes (utilisez CLAM original)
- ❌ Patching des WSI
- ❌ Preprocessing des slides

### Visualisation
- ❌ Génération de heatmaps
- ❌ Visualisation d'attention
- ❌ Plots avancés

### Features Avancées
- ❌ Presets de configuration
- ❌ Multi-GPU training
- ❌ Mixed precision training
- ❌ Logging tensorboard

### Pourquoi ?
Ces fonctionnalités ajoutent de la complexité sans être essentielles à l'entraînement et l'évaluation de base. Si vous en avez besoin, utilisez la version originale de CLAM.

---

## 🎯 En Résumé

### Ce qui est inclus
✅ Tout ce dont vous avez besoin pour **entraîner**, **évaluer** et **déployer** des modèles CLAM

### Ce qui ne l'est pas
❌ Preprocessing, visualisation avancée, et features expérimentales

### Philosophie
**Minimalisme fonctionnel** : Garder uniquement ce qui est nécessaire et éliminer tout le superflu.

---

## 📚 Documentation

- `README.md` : Documentation générale
- `QUICKSTART.md` : Guide de démarrage rapide
- `CLEANUP.md` : Détails du nettoyage effectué
- `FEATURES.md` : Ce fichier

## 🔗 Liens

- **CLAM Original** : https://github.com/mahmoodlab/CLAM
- **Paper** : https://www.nature.com/articles/s41551-020-00682-w
