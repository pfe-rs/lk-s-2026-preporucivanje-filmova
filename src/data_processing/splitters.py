import pandas as pd
import numpy as np
from typing import Dict, List

class DataSplitter:
    def __init__(self, ratings_df: pd.DataFrame, relevance_threshold: float = 4.0):
        self.ratings_df = ratings_df.copy()
        self.relevance_threshold = relevance_threshold

    def leave_one_out(self) -> Dict[str, pd.DataFrame]:
        """Leave-One-Out (LOO). Последний interaction -> Test, предпоследний -> Val, остальное -> Train."""
        df = self.ratings_df.sort_values(by=['userId', 'timestamp'])
        grouped = df.groupby('userId')
        
        train_list, val_list, test_list = [], [], []
        
        for uid, group in grouped:
            if len(group) < 2:
                train_list.append(group)
            elif len(group) == 2:
                train_list.append(group.iloc[:1])
                test_list.append(group.iloc[1:])
            else:
                train_list.append(group.iloc[:-2])
                val_list.append(group.iloc[-2:-1])
                test_list.append(group.iloc[-1:])
                
        return {
            'train': pd.concat(train_list),
            'val': pd.concat(val_list) if val_list else pd.DataFrame(),
            'test': pd.concat(test_list) if test_list else pd.DataFrame()
        }

    def global_temporal_split(self, train_ratio: float = 0.8) -> Dict[str, pd.DataFrame]:
        """Глобальный временной сплит (Индустриальный стандарт)."""
        df = self.ratings_df.sort_values(by='timestamp')
        split_idx = int(len(df) * train_ratio)
        
        return {
            'train': df.iloc[:split_idx],
            'val': pd.DataFrame(),
            'test': df.iloc[split_idx:]
        }

    def chronological_kfold(self, k: int = 5) -> List[Dict[str, pd.DataFrame]]:
        """Хронологический K-Fold для каждого пользователя."""
        df = self.ratings_df.sort_values(by=['userId', 'timestamp'])
        folds = []
        
        for fold_idx in range(k):
            train_list, test_list = [], []
            grouped = df.groupby('userId')
            
            for uid, group in grouped:
                n = len(group)
                if n < 2:
                    train_list.append(group)
                    continue
                    
                fold_size = n // k
                if fold_size == 0:
                    train_list.append(group)
                    continue
                    
                test_start = fold_idx * fold_size
                test_end = test_start + fold_size if fold_idx < k - 1 else n
                
                test_list.append(group.iloc[test_start:test_end])
                train_list.append(pd.concat([group.iloc[:test_start], group.iloc[test_end:]]))
                
            folds.append({
                'train': pd.concat(train_list),
                'val': pd.DataFrame(),
                'test': pd.concat(test_list)
            })
            
        return folds