import numpy as np
import pandas as pd
from typing import List, Dict, Any, Protocol

class RecommenderProtocol(Protocol):
    def fit(self, *args, **kwargs) -> Any: ...
    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]: ...
    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]: ...
    def explain_recommendation(self, movie_id: int, liked_items: set, top_n_reasons: int = 3) -> List[Dict[str, Any]]: ...

class HybridRecommender:
    def __init__(self, cf_model: RecommenderProtocol, cb_model: RecommenderProtocol, alpha: float = 0.5):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha 
        
    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "HybridRecommender":
        self.cf_model.fit(ratings_df)
        self.cb_model.fit(movies_df, ratings_df)
        return self
        
    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        cf_scores = np.array(self.cf_model.predict_scores(user_id, item_ids))
        cb_scores = np.array(self.cb_model.predict_scores(user_id, item_ids))
        
        def normalize(arr: np.ndarray) -> np.ndarray:
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val == 0: 
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
            
        return (self.alpha * normalize(cf_scores) + (1 - self.alpha) * normalize(cb_scores)).tolist()

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        if not hasattr(self.cb_model, 'all_movie_ids'):
            return self.cf_model.get_top_k_recommendations(user_id, watched_items, k)
            
        candidate_ids = list(self.cb_model.all_movie_ids - set(watched_items))
        if not candidate_ids:
            return []
            
        scores = self.predict_scores(user_id, candidate_ids)
        scored_items = list(zip(scores, candidate_ids))
        scored_items.sort(reverse=True, key=lambda x: x[0])
        
        return [mid for score, mid in scored_items[:k]]

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        cb_reasons = self.cb_model.explain_recommendation(movie_id, liked_items, top_n_reasons)
        cf_reasons = self.cf_model.explain_recommendation(movie_id, liked_items, top_n_reasons)
        
        combined = {}
        for reason in cb_reasons + cf_reasons:
            mid = reason['movie_id']
            if mid not in combined:
                combined[mid] = []
            combined[mid].append(reason['similarity'])
            
        final_reasons = []
        for mid, sims in combined.items():
            final_reasons.append({
                'movie_id': mid,
                'similarity': sum(sims) / len(sims)
            })
            
        final_reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return final_reasons[:top_n_reasons]