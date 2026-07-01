import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, ratings_df: pd.DataFrame, relevance_threshold: float = 3.0):
        self.relevance_threshold = relevance_threshold
        self.item_col = 'movieId' if 'movieId' in ratings_df.columns else 'itemId'
        self.df_user_time = ratings_df.sort_values(by=['userId', 'timestamp']).reset_index(drop=True)
        self.df_time = ratings_df.sort_values(by='timestamp').reset_index(drop=True)

    def _apply_cold_start_filters(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        trained_items = set(train_df[self.item_col].unique())
        trained_users = set(train_df['userId'].unique())

        if not val_df.empty:
            val_df = val_df[
                val_df[self.item_col].isin(trained_items) &
                val_df['userId'].isin(trained_users)
            ].reset_index(drop=True)

        if not test_df.empty:
            test_df = test_df[
                test_df[self.item_col].isin(trained_items) &
                test_df['userId'].isin(trained_users)
            ].reset_index(drop=True)

        return {'train': train_df, 'val': val_df, 'test': test_df}

    def hybrid_stratified_split(self, random_state: int = 42) -> Dict[str, pd.DataFrame]:
        df_shuffled = self.df_user_time.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df = df_shuffled.sort_values(by='userId', kind='stable').reset_index(drop=True)

        user_ids = df['userId'].values
        n_rows = len(user_ids)
        if n_rows == 0:
            return {'train': df.iloc[0:0], 'val': df.iloc[0:0], 'test': df.iloc[0:0]}

        group_sizes, cumcount = self._group_boundaries(user_ids)
        
        train_counts = np.zeros_like(group_sizes)
        val_counts = np.zeros_like(group_sizes)
        
        mask_ge_6 = group_sizes >= 6
        mask_3_5 = (group_sizes >= 3) & (group_sizes <= 5)
        mask_2 = group_sizes == 2
        mask_1 = group_sizes == 1
        
        test_val_counts = np.round(group_sizes * 0.30).astype(np.int64)
        
        ge6_test_val = test_val_counts[mask_ge_6]
        val_counts_ge_6 = ge6_test_val // 2
        test_counts_ge_6 = ge6_test_val - val_counts_ge_6
        
        train_counts[mask_ge_6] = group_sizes[mask_ge_6] - val_counts_ge_6 - test_counts_ge_6
        val_counts[mask_ge_6] = val_counts_ge_6
        
        train_counts[mask_3_5] = group_sizes[mask_3_5] - 2
        val_counts[mask_3_5] = 1
        
        train_counts[mask_2] = 1
        val_counts[mask_2] = 0
        
        train_counts[mask_1] = 1
        val_counts[mask_1] = 0
        
        train_ends = np.repeat(train_counts, group_sizes)
        val_ends = train_ends + np.repeat(val_counts, group_sizes)

        train_mask = cumcount < train_ends
        val_mask = (cumcount >= train_ends) & (cumcount < val_ends)
        test_mask = cumcount >= val_ends

        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True) if val_mask.any() else pd.DataFrame(columns=df.columns)
        test_df = df[test_mask].copy().reset_index(drop=True)

        return self._apply_cold_start_filters(train_df, val_df, test_df)

    def _group_boundaries(self, ids: np.ndarray):
        n_rows = len(ids)
        diff = ids[:-1] != ids[1:]
        change_indices = np.flatnonzero(diff) + 1
        start_indices = np.zeros(len(change_indices) + 1, dtype=np.int64)
        start_indices[1:] = change_indices
        end_indices = np.empty_like(start_indices)
        end_indices[:-1] = start_indices[1:]
        end_indices[-1] = n_rows
        group_sizes = end_indices - start_indices
        cumcount = np.arange(n_rows, dtype=np.int64) - np.repeat(start_indices, group_sizes)
        return group_sizes, cumcount

    def _proportional_train_val_counts(self, group_sizes: np.ndarray, train_ratio: float, val_ratio: float):
        train_counts = np.floor(group_sizes * train_ratio).astype(np.int64)
        val_counts = np.floor(group_sizes * val_ratio).astype(np.int64)

        train_counts = np.maximum(1, train_counts)
        np.copyto(train_counts, group_sizes, where=train_counts > group_sizes)
        remaining = group_sizes - train_counts

        np.copyto(val_counts, remaining, where=val_counts > remaining)
        test_counts = remaining - val_counts

        force_test_mask = (test_counts == 0) & (group_sizes >= 2)
        train_counts = np.where(force_test_mask & (val_counts == 0), train_counts - 1, train_counts)
        val_counts = np.where(force_test_mask & (val_counts > 0), val_counts - 1, val_counts)

        return train_counts, val_counts

    def temporal_user_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        df = self.df_user_time
        user_ids = df['userId'].values
        n_rows = len(user_ids)
        if n_rows == 0:
            return {'train': df.iloc[0:0], 'val': df.iloc[0:0], 'test': df.iloc[0:0]}

        group_sizes, cumcount = self._group_boundaries(user_ids)
        train_counts, val_counts = self._proportional_train_val_counts(group_sizes, train_ratio, val_ratio)

        train_ends = np.repeat(train_counts, group_sizes)
        val_ends = train_ends + np.repeat(val_counts, group_sizes)

        train_mask = cumcount < train_ends
        val_mask = (cumcount >= train_ends) & (cumcount < val_ends)
        test_mask = cumcount >= val_ends

        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True) if val_mask.any() else pd.DataFrame(columns=df.columns)
        test_df = df[test_mask].copy().reset_index(drop=True)

        return self._apply_cold_start_filters(train_df, val_df, test_df)

    def stratified_random_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, random_state: int = 42) -> Dict[str, pd.DataFrame]:
        df_shuffled = self.df_user_time.sample(frac=1, random_state=random_state).reset_index(drop=True)
        df = df_shuffled.sort_values(by='userId', kind='stable').reset_index(drop=True)

        user_ids = df['userId'].values
        n_rows = len(user_ids)
        if n_rows == 0:
            return {'train': df.iloc[0:0], 'val': df.iloc[0:0], 'test': df.iloc[0:0]}

        group_sizes, cumcount = self._group_boundaries(user_ids)
        train_counts, val_counts = self._proportional_train_val_counts(group_sizes, train_ratio, val_ratio)

        train_ends = np.repeat(train_counts, group_sizes)
        val_ends = train_ends + np.repeat(val_counts, group_sizes)

        train_mask = cumcount < train_ends
        val_mask = (cumcount >= train_ends) & (cumcount < val_ends)
        test_mask = cumcount >= val_ends

        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True) if val_mask.any() else pd.DataFrame(columns=df.columns)
        test_df = df[test_mask].copy().reset_index(drop=True)

        return self._apply_cold_start_filters(train_df, val_df, test_df)

    def leave_one_out(self, test_size: int = 5, val_size: int = 0) -> Dict[str, pd.DataFrame]:
        df = self.df_user_time
        user_ids = df['userId'].values
        n_rows = len(user_ids)
        if n_rows == 0:
            return {'train': df.iloc[0:0], 'val': df.iloc[0:0], 'test': df.iloc[0:0]}

        group_sizes, cumcount = self._group_boundaries(user_ids)
        group_size_all = np.repeat(group_sizes, group_sizes)
        inverse_cumcount = group_size_all - 1 - cumcount

        allocated_test = np.where(
            group_size_all >= 2,
            np.minimum(test_size, np.maximum(1, group_size_all // 2)),
            0
        )
        allocated_val = np.where(
            group_size_all - allocated_test > 1,
            np.minimum(val_size, group_size_all - allocated_test - 1),
            0
        )

        test_mask = inverse_cumcount < allocated_test
        val_mask = (~test_mask) & (inverse_cumcount < (allocated_test + allocated_val))
        train_mask = (~test_mask) & (~val_mask)

        train_df = df[train_mask].copy().reset_index(drop=True)
        val_df = df[val_mask].copy().reset_index(drop=True) if val_mask.any() else pd.DataFrame(columns=df.columns)
        test_df = df[test_mask].copy().reset_index(drop=True)

        return self._apply_cold_start_filters(train_df, val_df, test_df)

    def global_temporal_split(self, train_ratio: float = 0.8) -> Dict[str, pd.DataFrame]:
        df = self.df_time
        split_idx = int(len(df) * train_ratio)

        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        val_df = pd.DataFrame(columns=df.columns)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        return self._apply_cold_start_filters(train_df, val_df, test_df)

    def chronological_kfold(self, k: int = 5) -> List[Dict[str, pd.DataFrame]]:
        df = self.df_user_time
        user_ids = df['userId'].values
        n_rows = len(user_ids)
        if n_rows == 0:
            return [{'train': df.iloc[0:0], 'val': df.iloc[0:0], 'test': df.iloc[0:0]} for _ in range(k)]

        group_sizes, cumcount = self._group_boundaries(user_ids)
        group_size_all = np.repeat(group_sizes, group_sizes)
        fold_size_all = np.maximum(1, group_size_all // k)

        fold_assignment = np.minimum(cumcount // fold_size_all, k - 1).astype(np.int32)

        folds = []
        for fold_idx in range(1, k):
            train_mask = fold_assignment < fold_idx
            test_mask = fold_assignment == fold_idx

            train_df = df[train_mask].copy().reset_index(drop=True)
            val_df = pd.DataFrame(columns=df.columns)
            test_df = df[test_mask].copy().reset_index(drop=True)

            folds.append(self._apply_cold_start_filters(train_df, val_df, test_df))
        return folds