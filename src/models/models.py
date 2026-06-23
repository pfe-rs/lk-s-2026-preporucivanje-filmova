import numpy as np
import pandas as pd
from typing import List, Dict, Any, Protocol

class RecommenderProtocol(Protocol):
    def fit(self, train_df: pd.DataFrame) -> Any: ...
    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]: ...
    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]: ...

class CFWrapper:
    def __init__(self, cf_model):
        self.cf_model = cf_model
        self.all_item_ids = set()
        
    def fit(self, train_df: pd.DataFrame):
        self.all_item_ids = set(train_df['movieId'].unique())
        self.cf_model.fit(train_df)
        return self
        
    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        return [self.cf_model.predict_score(user_id, mid) for mid in item_ids]
        
    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        if hasattr(self.cf_model, 'recommend_for_user'):
            recs = self.cf_model.recommend_for_user(user_id, list(watched_items), top_n=k)
            return [mid for mid, score in recs]
        return []

# class ContentBasedRecommender:
#     def __init__(self, genre_matrix: np.ndarray, movie_id_to_idx: Dict[int, int]):
#         self.genre_matrix = genre_matrix
#         self.movie_id_to_idx = movie_id_to_idx
#         self.user_profiles = {}
        
#     def fit(self, train_df: pd.DataFrame):
#         for user_id, group in train_df.groupby('userId'):
#             profile = np.zeros(self.genre_matrix.shape[1])
#             for _, row in group.iterrows():
#                 mid = row['movieId']
#                 if mid in self.movie_id_to_idx:
#                     idx = self.movie_id_to_idx[mid]
#                     profile += row['rating'] * self.genre_matrix[idx]
#             self.user_profiles[user_id] = profile
            
#     def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
#         if user_id not in self.user_profiles:
#             return [0.0] * len(item_ids)
            
#         profile = self.user_profiles[user_id]
#         scores = []
#         for mid in item_ids:
#             if mid in self.movie_id_to_idx:
#                 idx = self.movie_id_to_idx[mid]
#                 item_vec = self.genre_matrix[idx]
#                 dot = np.dot(profile, item_vec)
#                 norm = np.linalg.norm(profile) * np.linalg.norm(item_vec)
#                 scores.append(dot / norm if norm > 0 else 0.0)
#             else:
#                 scores.append(0.0)
#         return scores

#     def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
#         all_items = list(set(self.movie_id_to_idx.keys()) - watched_items)
#         scores = self.predict_scores(user_id, all_items)
#         sorted_items = [x for _, x in sorted(zip(scores, all_items), reverse=True)]
#         return sorted_items[:k]

class HybridRecommender:
    def __init__(self, cf_model: RecommenderProtocol, cb_model: RecommenderProtocol, alpha: float = 0.5):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha 
        
    def fit(self, train_df: pd.DataFrame):
        self.cf_model.fit(train_df)
        self.cb_model.fit(train_df)
        return self
        
    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        cf_scores = np.array(self.cf_model.predict_scores(user_id, item_ids))
        cb_scores = np.array(self.cb_model.predict_scores(user_id, item_ids))
        
        def normalize(arr):
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val == 0: return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
            
        return (self.alpha * normalize(cf_scores) + (1 - self.alpha) * normalize(cb_scores)).tolist()

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        return self.cf_model.get_top_k_recommendations(user_id, watched_items, k)