import numpy as np
import pandas as pd
from typing import List, Dict, Any

class CascadingHybridRecommender:
    def __init__(self, primary_model, secondary_model, primary_k: int = 50):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.primary_k = primary_k

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.primary_model.fit(ratings_df)
        self.secondary_model.fit(movies_df, ratings_df)
        return self

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        broad_candidates = self.primary_model.get_top_k_recommendations(
            user_id=user_id, 
            watched_items=watched_items, 
            k=self.primary_k
        )
        
        if not broad_candidates:
            return []
            
        candidate_scores = self.secondary_model.predict_scores(user_id, broad_candidates)
        
        scored_candidates = list(zip(candidate_scores, broad_candidates))
        scored_candidates.sort(reverse=True, key=lambda x: x[0])
        
        return [mid for score, mid in scored_candidates[:k]]

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        return self.secondary_model.predict_scores(user_id, item_ids)

    def explain_recommendation(self, movie_id: int, liked_items: set, top_n_reasons: int = 3) -> List[Dict[str, Any]]:
        return self.secondary_model.explain_recommendation(movie_id, liked_items, top_n_reasons)