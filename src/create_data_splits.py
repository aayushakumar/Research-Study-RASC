"""
Create proper train/val/test splits for RASC reproduction.
Splits are done at the QUESTION level to prevent data leakage.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Evaluation_CoTs', 'Final_results')

def create_splits(full_data_path, output_dir, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, 
                 use_subset_pct=0.30, random_state=42):
    """
    Create train/val/test splits from full dataset.
    
    Args:
        full_data_path: Path to full dataset JSON
        output_dir: Directory to save split files
        train_ratio: Proportion for training (of selected subset)
        val_ratio: Proportion for validation (of selected subset)
        test_ratio: Proportion for test (of selected subset)
        use_subset_pct: Use this percentage of full data (0.30 = 30%)
        random_state: Random seed for reproducibility
    """
    print(f"Loading full dataset from: {full_data_path}")
    with open(full_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df_full = pd.DataFrame(data)
    print(f"Full dataset size: {len(df_full)} questions")
    
    # First, sample the percentage we want to use (30%)
    np.random.seed(random_state)
    n_to_use = int(len(df_full) * use_subset_pct)
    
    # Stratify by 'dataset' if available
    if 'dataset' in df_full.columns:
        df_subset = df_full.groupby('dataset', group_keys=False).apply(
            lambda x: x.sample(frac=use_subset_pct, random_state=random_state)
        ).reset_index(drop=True)
    else:
        df_subset = df_full.sample(n=n_to_use, random_state=random_state).reset_index(drop=True)
    
    print(f"Using {len(df_subset)} questions ({use_subset_pct*100}% of full dataset)")
    
    # Split into train and temp (val + test)
    train_size = train_ratio / (train_ratio + val_ratio + test_ratio)
    temp_size = 1 - train_size
    
    stratify_col = df_subset['dataset'] if 'dataset' in df_subset.columns else None
    
    df_train, df_temp = train_test_split(
        df_subset, 
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Split temp into val and test
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    
    stratify_temp = df_temp['dataset'] if 'dataset' in df_temp.columns else None
    
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1 - val_ratio_of_temp),
        random_state=random_state,
        stratify=stratify_temp
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(df_train)} questions ({len(df_train)/len(df_subset)*100:.1f}%)")
    print(f"  Val:   {len(df_val)} questions ({len(df_val)/len(df_subset)*100:.1f}%)")
    print(f"  Test:  {len(df_test)} questions ({len(df_test)/len(df_subset)*100:.1f}%)")
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_split.json')
    val_path = os.path.join(output_dir, 'val_split.json')
    test_path = os.path.join(output_dir, 'test_split.json')
    
    df_train.to_json(train_path, orient='records', indent=2)
    df_val.to_json(val_path, orient='records', indent=2)
    df_test.to_json(test_path, orient='records', indent=2)
    
    print(f"\nSaved splits to:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    
    # Save split info
    split_info = {
        'full_dataset_size': len(df_full),
        'subset_percentage': use_subset_pct,
        'subset_size': len(df_subset),
        'train_size': len(df_train),
        'val_size': len(df_val),
        'test_size': len(df_test),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_state': random_state
    }
    
    info_path = os.path.join(output_dir, 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"  Info:  {info_path}")
    
    return df_train, df_val, df_test


if __name__ == '__main__':
    full_data_path = os.path.join(DATA_DIR, 'final_with_feature.json')
    output_dir = DATA_DIR
    
    print("="*70)
    print("Creating Train/Val/Test Splits for RASC Reproduction")
    print("="*70)
    
    df_train, df_val, df_test = create_splits(
        full_data_path=full_data_path,
        output_dir=output_dir,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        use_subset_pct=0.30,  # Use 30% of full data
        random_state=42
    )
    
    print("\n" + "="*70)
    print("Split creation complete!")
    print("="*70)
