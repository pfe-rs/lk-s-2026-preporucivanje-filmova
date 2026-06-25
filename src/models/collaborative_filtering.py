import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity

MIN_RATING = 1.0
MAX_RATING = 5.0


class CollaborativeFiltering:
    def __init__(self, k_components: int = 50, random_state: int = 42) -> None:
        self.n_components = k_components
        self.random_state = random_state

        self.svd_model: Optional[SVD] = None
        self.trainset = None
        self._all_raw_movie_ids: List[int] = []

    def fit(self, df_ratings: pd.DataFrame) -> "CollaborativeFiltering":
        reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))
        data = Dataset.load_from_df(
            df_ratings[["userId", "movieId", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()

        self.svd_model = SVD(
            n_factors=self.n_components, random_state=self.random_state
        )
        self.svd_model.fit(self.trainset)

        self._all_raw_movie_ids = [
            self.trainset.to_raw_iid(m_id) for m_id in self.trainset.all_items()
        ]
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if self.svd_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        prediction = self.svd_model.predict(user_id, movie_id)
        return float(prediction.est)

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        if self.svd_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        watched_set = set(watched_movie_ids)
        unwatched_movies = [
            m_id for m_id in self._all_raw_movie_ids if m_id not in watched_set
        ]

        predictions = [
            (movie_id, self.predict_score(user_id, movie_id))
            for movie_id in unwatched_movies
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if self.svd_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        return [self.predict_score(user_id, mid) for mid in item_ids]

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10
    ) -> List[int]:
        recs = self.recommend_for_user(user_id, list(watched_items), top_n=k)
        return [mid for mid, _ in recs]

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        if self.svd_model is None or self.trainset is None:
            return []

        try:
            inner_target = self.trainset.to_inner_iid(movie_id)
        except ValueError:
            return []

        target_vector = self.svd_model.qi[inner_target].reshape(1, -1)
        reasons = []

        for liked_id in liked_items:
            try:
                inner_liked = self.trainset.to_inner_iid(liked_id)
                liked_vector = self.svd_model.qi[inner_liked].reshape(1, -1)
                
                sim = cosine_similarity(target_vector, liked_vector)[0][0]
                if sim > 0:
                    reasons.append({
                        'movie_id': liked_id,
                        'similarity': float(sim)
                    })
            except ValueError:
                continue

        reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return reasons[:top_n_reasons]