"""Script d'entraînement simplifié avec config.py"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, print_config
from src.models import build_model
from src.dataset import create_datasets
from src.utils import set_seed, calculate_metrics, EarlyStopping, AverageMeter


def train_epoch(model, loader, optimizer, device, bag_weight=0.7):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    
    for features, labels, _ in tqdm(loader, desc='Training'):
        features = features.to(device)
        labels = labels.to(device)
        
        logits, Y_prob, Y_hat, A, results = model(features, label=labels, instance_eval=True)
        
        bag_loss = nn.CrossEntropyLoss()(logits, labels)
        inst_loss = results.get('instance_loss', 0)
        loss = bag_weight * bag_loss + (1 - bag_weight) * inst_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item())
    
    return losses.avg


def validate(model, loader, device):
    """Validate the model"""
    model.eval()
    losses = AverageMeter()
    
    all_probs = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for features, labels, _ in tqdm(loader, desc='Validation'):
            features = features.to(device)
            labels = labels.to(device)
            
            logits, Y_prob, Y_hat, A, _ = model(features)
            loss = nn.CrossEntropyLoss()(logits, labels)
            
            losses.update(loss.item())
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(Y_hat.cpu().numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    return losses.avg, all_labels, all_preds, all_probs


def train_fold(config, fold):
    """Train a single fold"""
    print(f'\n=== Training Fold {fold} ===')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.SEED)
    
    # Load data
    dataset, n_classes, label_dict = create_datasets(config.TASK, config.DATA_ROOT)
    
    # Load split
    split_file = f'{config.SPLIT_DIR}/splits_{fold}.csv'
    split_df = pd.read_csv(split_file)
    
    train_data = dataset[split_df[split_df['train'] == True].index]
    val_data = dataset[split_df[split_df['val'] == True].index]
    test_data = dataset[split_df[split_df['test'] == True].index]
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=config.NUM_WORKERS)
    
    # Build model
    model = build_model(
        model_type=config.MODEL_TYPE,
        n_classes=n_classes,
        embed_dim=config.EMBED_DIM,
        dropout=config.DROPOUT
    ).to(device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.LEARNING_RATE, 
                                weight_decay=config.WEIGHT_DECAY)
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    # Training loop
    save_path = f'{config.RESULTS_DIR}/fold_{fold}_best.pt'
    
    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device, config.BAG_WEIGHT)
        val_loss, val_labels, val_preds, val_probs = validate(model, val_loader, device)
        
        val_acc, val_auc = calculate_metrics(val_labels, val_preds, val_probs, n_classes)
        
        print(f'Epoch {epoch+1}/{config.MAX_EPOCHS} - '
              f'Train Loss: {train_loss:.4f} - '
              f'Val Loss: {val_loss:.4f} - '
              f'Val Acc: {val_acc:.4f} - '
              f'Val AUC: {val_auc:.4f}')
        
        early_stopping(epoch, val_loss, model, save_path)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    # Test evaluation
    model.load_state_dict(torch.load(save_path))
    test_loss, test_labels, test_preds, test_probs = validate(model, test_loader, device)
    test_acc, test_auc = calculate_metrics(test_labels, test_preds, test_probs, n_classes)
    
    print(f'\nFold {fold} Results - Test Acc: {test_acc:.4f} - Test AUC: {test_auc:.4f}')
    
    return {
        'fold': fold,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'val_acc': val_acc,
        'val_auc': val_auc
    }


def main():
    # Charger la configuration
    config = get_config()
    print_config(config)
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Determine folds to run
    fold_end = config.FOLD_END if config.FOLD_END != -1 else config.N_FOLDS
    
    # Train each fold
    results = []
    for fold in range(config.FOLD_START, fold_end):
        fold_results = train_fold(config, fold)
        results.append(fold_results)
    
    # Save summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{config.RESULTS_DIR}/summary.csv', index=False)
    
    print('\n=== Final Results ===')
    print(f"Test Acc: {results_df['test_acc'].mean():.4f} ± {results_df['test_acc'].std():.4f}")
    print(f"Test AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}")


if __name__ == '__main__':
    main()
