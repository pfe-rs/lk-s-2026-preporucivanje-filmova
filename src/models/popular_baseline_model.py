import pandas as pd
import numpy as np
from typing import List, Dict, Any

class PopularityBaseline:
    def __init__(self):
        self.item_popularity = {}
        self.all_items = []
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame) -> "PopularityBaseline":
        popularity_counts = ratings_df.groupby('movieId').size()
        self.item_popularity = popularity_counts.to_dict()
        self.all_items = list(self.item_popularity.keys())
        self.is_fitted = True
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return [float(self.item_popularity.get(mid, 0)) for mid in item_ids]

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        candidates = [mid for mid in self.all_items if mid not in watched_items]
        
        candidates_with_scores = [
            (mid, self.item_popularity.get(mid, 0)) 
            for mid in candidates
        ]
        
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [mid for mid, _ in candidates_with_scores[:k]]

    def explain_recommendation(self, movie_id: int, liked_items: set, top_n_reasons: int = 3) -> List[Dict[str, Any]]:
        return []