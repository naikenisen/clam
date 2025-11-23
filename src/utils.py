"""Utility functions"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_metrics(Y_true, Y_pred, Y_prob, n_classes):
    """Calculate accuracy and AUC"""
    acc = accuracy_score(Y_true, Y_pred)
    
    if n_classes == 2:
        auc = roc_auc_score(Y_true, Y_prob[:, 1])
    else:
        auc = roc_auc_score(Y_true, Y_prob, multi_class='ovr')
    
    return acc, auc


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=20, min_epochs=50):
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, save_path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
