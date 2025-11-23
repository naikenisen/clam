"""Simple inference script for CLAM"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import h5py

from src.models import build_model
from src.utils import set_seed


def load_slide_features(h5_path):
    """Load features from h5 file"""
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:])
    return features


def infer_slide(model, features, device):
    """Run inference on a single slide"""
    model.eval()
    
    with torch.no_grad():
        features = features.to(device)
        logits, Y_prob, Y_hat, A, _ = model(features)
        
        # Get attention scores
        attention = A.cpu().numpy().squeeze()
        
    return {
        'pred_class': Y_hat.item(),
        'probs': Y_prob.cpu().numpy().squeeze(),
        'attention': attention
    }


def main():
    parser = argparse.ArgumentParser(description='CLAM Inference')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='clam_sb', 
                        choices=['clam_sb', 'clam_mb'])
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.25)
    
    # Data
    parser.add_argument('--input', type=str, required=True,
                        help='Path to h5 feature file or directory of h5 files')
    parser.add_argument('--output_dir', type=str, default='./predictions')
    
    # Other
    parser.add_argument('--save_attention', action='store_true', 
                        help='Save attention scores')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print('Loading model...')
    model = build_model(
        model_type=args.model_type,
        n_classes=args.n_classes,
        embed_dim=args.embed_dim,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print('Model loaded successfully')
    
    # Get list of h5 files to process
    if os.path.isfile(args.input):
        h5_files = [args.input]
    elif os.path.isdir(args.input):
        h5_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                    if f.endswith('.h5')]
    else:
        raise ValueError(f"Input path not found: {args.input}")
    
    print(f'Found {len(h5_files)} slides to process')
    
    # Run inference on each slide
    results = []
    
    for h5_path in tqdm(h5_files, desc='Processing slides'):
        slide_id = os.path.basename(h5_path).replace('.h5', '')
        
        try:
            # Load features
            features = load_slide_features(h5_path)
            
            # Run inference
            output = infer_slide(model, features, device)
            
            # Store results
            result = {
                'slide_id': slide_id,
                'predicted_class': output['pred_class']
            }
            
            # Add probabilities for each class
            for i, prob in enumerate(output['probs']):
                result[f'prob_class_{i}'] = prob
            
            results.append(result)
            
            # Save attention if requested
            if args.save_attention:
                attention_path = os.path.join(args.output_dir, f'{slide_id}_attention.npy')
                np.save(attention_path, output['attention'])
            
        except Exception as e:
            print(f'Error processing {slide_id}: {str(e)}')
            continue
    
    # Save predictions
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        output_path = os.path.join(args.output_dir, 'predictions.csv')
        results_df.to_csv(output_path, index=False)
        print(f'\nPredictions saved to: {output_path}')
        
        # Print summary
        print('\n=== Prediction Summary ===')
        print(results_df['predicted_class'].value_counts().sort_index())
    else:
        print('No successful predictions')


if __name__ == '__main__':
    main()
