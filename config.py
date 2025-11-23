"""Configuration centralisée pour CLAM"""
import os


class Config:
    """Configuration de base"""
    
    # === Chemins ===
    DATA_ROOT = '/path/to/your/features'  # À modifier selon votre configuration
    RESULTS_DIR = './results'
    SPLIT_DIR = './splits/task_1_tumor_vs_normal_75'
    
    # === Tâche ===
    TASK = 'tumor_vs_normal'  # 'tumor_vs_normal' ou 'tumor_subtyping'
    
    # === Modèle ===
    MODEL_TYPE = 'clam_sb'  # 'clam_sb' ou 'clam_mb'
    EMBED_DIM = 1024
    DROPOUT = 0.25
    
    # === Entraînement ===
    MAX_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    BAG_WEIGHT = 0.7  # Pour pondération loss bag vs instance
    PATIENCE = 20  # Pour early stopping
    
    # === Cross-validation ===
    N_FOLDS = 10
    FOLD_START = 0
    FOLD_END = 10  # -1 pour tous les folds
    
    # === Autres ===
    SEED = 42
    NUM_WORKERS = 4
    BATCH_SIZE = 1  # Pour MIL, toujours 1


class TumorVsNormalConfig(Config):
    """Configuration pour tumor vs normal"""
    TASK = 'tumor_vs_normal'
    SPLIT_DIR = './splits/task_1_tumor_vs_normal_75'
    N_CLASSES = 2
    LABEL_DICT = {'normal_tissue': 0, 'tumor_tissue': 1}


class TumorSubtypingConfig(Config):
    """Configuration pour tumor subtyping"""
    TASK = 'tumor_subtyping'
    SPLIT_DIR = './splits/task_2_tumor_subtyping_50'
    N_CLASSES = 3
    LABEL_DICT = {'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2}


class ExperimentConfig(Config):
    """Configuration pour une expérience spécifique"""
    # Hérite de Config et override ce dont vous avez besoin
    
    # Exemple d'expérience personnalisée
    DATA_ROOT = '/home/user/data/features'
    RESULTS_DIR = './results/exp_001'
    
    MODEL_TYPE = 'clam_mb'
    MAX_EPOCHS = 100
    LEARNING_RATE = 2e-4
    
    # Evaluation sur un fold spécifique
    EVAL_FOLD = 0  # -1 pour tous
    EVAL_SPLIT = 'test'  # 'train', 'val', ou 'test'


# === Configuration active ===
# Changez cette ligne pour utiliser une configuration différente
ACTIVE_CONFIG = TumorVsNormalConfig


def get_config():
    """Retourne la configuration active"""
    return ACTIVE_CONFIG()


# === Helper functions ===
def print_config(config=None):
    """Affiche la configuration"""
    if config is None:
        config = get_config()
    
    print("\n" + "="*50)
    print("CONFIGURATION")
    print("="*50)
    
    for attr in dir(config):
        if not attr.startswith('_') and attr.isupper():
            value = getattr(config, attr)
            print(f"{attr:20s}: {value}")
    
    print("="*50 + "\n")


if __name__ == '__main__':
    # Test de la configuration
    print_config()
