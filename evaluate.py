"""Script d'évaluation simplifié avec config.py"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config, print_config
from src.models import build_model
from src.dataset import create_datasets
from src.utils import set_seed, calculate_metrics


def evaluate(model, loader, device):
    """Evaluate the model"""
    model.eval()
    
    all_probs = []
    all_labels = []
    all_preds = []
    all_slide_ids = []
    
    with torch.no_grad():
        for features, labels, slide_ids in tqdm(loader, desc='Evaluating'):
            features = features.to(device)
            labels = labels.to(device)
            
            logits, Y_prob, Y_hat, A, _ = model(features)
            
            all_probs.append(Y_prob.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(Y_hat.cpu().numpy())
            all_slide_ids.extend(slide_ids)
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    return all_labels, all_preds, all_probs, all_slide_ids


def main():
    # Charger la configuration
    config = get_config()
    
    # Pour l'évaluation, on peut override certains paramètres
    eval_fold = getattr(config, 'EVAL_FOLD', -1)
    eval_split = getattr(config, 'EVAL_SPLIT', 'test')
    
    print_config(config)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.SEED)
    
    # Créer le dossier de résultats
    eval_dir = config.RESULTS_DIR.replace('results', 'eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load dataset
    dataset, n_classes, label_dict = create_datasets(config.TASK, config.DATA_ROOT)
    
    # Build model
    model = build_model(
        model_type=config.MODEL_TYPE,
        n_classes=n_classes,
        embed_dim=config.EMBED_DIM,
        dropout=config.DROPOUT
    ).to(device)
    
    # Determine which folds to evaluate
    if eval_fold >= 0:
        folds = [eval_fold]
    else:
        fold_end = config.FOLD_END if config.FOLD_END != -1 else config.N_FOLDS
        folds = range(config.FOLD_START, fold_end)
    
    # Evaluate each fold
    all_results = []
    
    for fold in folds:
        print(f'\n=== Evaluating Fold {fold} ===')
        
        # Load checkpoint
        checkpoint_path = f'{config.RESULTS_DIR}/fold_{fold}_best.pt'
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            continue
        
        model.load_state_dict(torch.load(checkpoint_path))
        
        # Load split
        split_file = f'{config.SPLIT_DIR}/splits_{fold}.csv'
        split_df = pd.read_csv(split_file)
        
        # Get indices for requested split
        if eval_split == 'train':
            indices = split_df[split_df['train'] == True].index
        elif eval_split == 'val':
            indices = split_df[split_df['val'] == True].index
        else:
            indices = split_df[split_df['test'] == True].index
        
        eval_data = torch.utils.data.Subset(dataset, indices)
        eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False, 
                                num_workers=config.NUM_WORKERS)
        
        # Evaluate
        labels, preds, probs, slide_ids = evaluate(model, eval_loader, device)
        acc, auc = calculate_metrics(labels, preds, probs, n_classes)
        
        print(f'Fold {fold} - {eval_split.upper()} - Acc: {acc:.4f}, AUC: {auc:.4f}')
        
        # Save results
        results_df = pd.DataFrame({
            'slide_id': slide_ids,
            'true_label': labels,
            'pred_label': preds,
            'prob_class_0': probs[:, 0],
            'prob_class_1': probs[:, 1] if n_classes > 1 else 0
        })
        if n_classes > 2:
            results_df['prob_class_2'] = probs[:, 2]
        
        results_df.to_csv(f'{eval_dir}/fold_{fold}_{eval_split}_predictions.csv', 
                         index=False)
        
        all_results.append({
            'fold': fold,
            'split': eval_split,
            'acc': acc,
            'auc': auc
        })
    
    # Save summary
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(f'{eval_dir}/evaluation_summary.csv', index=False)
        
        print('\n=== Summary ===')
        print(f"Mean Acc: {summary_df['acc'].mean():.4f} ± {summary_df['acc'].std():.4f}")
        print(f"Mean AUC: {summary_df['auc'].mean():.4f} ± {summary_df['auc'].std():.4f}")


if __name__ == '__main__':
    main()
