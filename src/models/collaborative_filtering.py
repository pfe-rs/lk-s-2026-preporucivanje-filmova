import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from typing import List, Optional, Tuple, Dict, Any

MIN_RATING = 0.5
MAX_RATING = 5.0

class CollaborativeFiltering:
    def __init__(
        self, 
        k_components: int = 50, 
        reg_all: float = 0.02, 
        lr_all: float = 0.005,
        n_epochs: int = 20,
        alpha: float = 0.2, 
        min_ratings: int = 15, 
        random_state: int = 42
    ) -> None:
        self.n_components = k_components
        self.reg_all = reg_all
        self.lr_all = lr_all
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.min_ratings = min_ratings
        self.random_state = random_state
        
        self._raw_to_inner_user: Dict[int, int] = {}
        self._raw_to_inner_item: Dict[int, int] = {}
        self._inner_to_raw_user: Dict[int, int] = {}
        self._inner_to_raw_item: Dict[int, int] = {}
        
        self._pu: np.ndarray = np.array([])
        self._qi: np.ndarray = np.array([])
        self._bu: np.ndarray = np.array([])
        self._bi: np.ndarray = np.array([])
        
        self._item_popularity: np.ndarray = np.array([])
        self._valid_item_mask: np.ndarray = np.array([])
        self._global_mean: float = 0.0
        self._popular_movies: List[int] = []
        self._is_fitted: bool = False

    def fit(self, df_ratings: pd.DataFrame) -> "CollaborativeFiltering":
        popularity_series = df_ratings.groupby("movieId").size()
        filtered_popularity = popularity_series[popularity_series >= self.min_ratings]
        self._popular_movies = filtered_popularity.sort_values(ascending=False).index.tolist()
        
        unique_users = df_ratings["userId"].unique()
        unique_movies = df_ratings["movieId"].unique()
        
        self._raw_to_inner_user = {raw: inner for inner, raw in enumerate(unique_users)}
        self._raw_to_inner_item = {raw: inner for inner, raw in enumerate(unique_movies)}
        self._inner_to_raw_user = {inner: raw for raw, inner in self._raw_to_inner_user.items()}
        self._inner_to_raw_item = {inner: raw for raw, inner in self._raw_to_inner_item.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_movies)
        
        u_indices = df_ratings["userId"].map(self._raw_to_inner_user).values
        i_indices = df_ratings["movieId"].map(self._raw_to_inner_item).values
        ratings = df_ratings["rating"].values.astype(np.float32)
        
        sparse_coo = coo_matrix((ratings, (u_indices, i_indices)), shape=(n_users, n_items))
        
        self._global_mean = float(sparse_coo.data.mean())
        
        np.random.seed(self.random_state)
        self._pu = np.random.normal(0, 0.1, (n_users, self.n_components))
        self._qi = np.random.normal(0, 0.1, (n_items, self.n_components))
        self._bu = np.zeros(n_users, dtype=np.float64)
        self._bi = np.zeros(n_items, dtype=np.float64)
        
        self._item_popularity = np.ones(n_items, dtype=np.float64)
        self._valid_item_mask = np.ones(n_items, dtype=bool)
        
        for r_id, i_id in self._raw_to_inner_item.items():
            pop = float(popularity_series.get(r_id, 0))
            self._item_popularity[i_id] = max(pop, 1.0)
            if pop < self.min_ratings:
                self._valid_item_mask[i_id] = False

        rows = sparse_coo.row
        cols = sparse_coo.col
        data = sparse_coo.data
        
        for epoch in range(self.n_epochs):
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            
            for idx in indices:
                u = rows[idx]
                i = cols[idx]
                r = data[idx]
                
                pred = self._global_mean + self._bu[u] + self._bi[i] + np.dot(self._pu[u], self._qi[i])
                err = r - pred
                
                self._bu[u] += self.lr_all * (err - self.reg_all * self._bu[u])
                self._bi[i] += self.lr_all * (err - self.reg_all * self._bi[i])
                
                p_old = self._pu[u].copy()
                self._pu[u] += self.lr_all * (err * self._qi[i] - self.reg_all * self._pu[u])
                self._qi[i] += self.lr_all * (err * p_old - self.reg_all * self._qi[i])
                
        self._is_fitted = True
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        u_inner = self._raw_to_inner_user.get(user_id)
        i_inner = self._raw_to_inner_item.get(movie_id)
        
        if u_inner is not None and i_inner is not None:
            est = self._global_mean + self._bu[u_inner] + self._bi[i_inner] + np.dot(self._pu[u_inner], self._qi[i_inner])
            return float(np.clip(est, MIN_RATING, MAX_RATING))
            
        if i_inner is not None:
            est = self._global_mean + self._bi[i_inner]
            return float(np.clip(est, MIN_RATING, MAX_RATING))
            
        return self._global_mean

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        watched_set = set(watched_movie_ids)
        u_inner = self._raw_to_inner_user.get(user_id)

        if u_inner is None:
            unwatched_popular = [m_id for m_id in self._popular_movies if m_id not in watched_set]
            return [(m_id, self._global_mean) for m_id in unwatched_popular[:top_n]]

        all_scores = self._global_mean + self._bu[u_inner] + self._bi + np.dot(self._qi, self._pu[u_inner])
        all_scores = np.clip(all_scores, MIN_RATING, MAX_RATING)

        penalized_scores = all_scores / (self._item_popularity ** self.alpha)

        watched_inners = [self._raw_to_inner_item[m] for m in watched_movie_ids if m in self._raw_to_inner_item]
        
        mask = np.ones(len(all_scores), dtype=bool)
        if watched_inners:
            mask[watched_inners] = False

        mask = mask & self._valid_item_mask

        remaining_inners = np.where(mask)[0]
        if len(remaining_inners) == 0:
            return []
            
        remaining_penalized = penalized_scores[remaining_inners]
        
        top_k = min(top_n, len(remaining_penalized))
        top_indices = np.argsort(-remaining_penalized)[:top_k]
        
        return [
            (self._inner_to_raw_item[remaining_inners[idx]], float(all_scores[remaining_inners[idx]]))
            for idx in top_indices
        ]

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        u_inner = self._raw_to_inner_user.get(user_id)
        scores = []
        
        for mid in item_ids:
            i_inner = self._raw_to_inner_item.get(mid)
            if u_inner is not None and i_inner is not None:
                est = self._global_mean + self._bu[u_inner] + self._bi[i_inner] + np.dot(self._pu[u_inner], self._qi[i_inner])
                scores.append(float(np.clip(est, MIN_RATING, MAX_RATING)))
            elif i_inner is not None:
                est = self._global_mean + self._bi[i_inner]
                scores.append(float(np.clip(est, MIN_RATING, MAX_RATING)))
            else:
                scores.append(self._global_mean)
        return scores

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10
    ) -> List[int]:
        recs = self.recommend_for_user(user_id, list(watched_items), top_n=k)
        return [mid for mid, _ in recs]

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        if not self._is_fitted:
            return []

        target_inner = self._raw_to_inner_item.get(movie_id)
        if target_inner is None:
            return []

        liked_inners = [self._raw_to_inner_item[mid] for mid in liked_items if mid in self._raw_to_inner_item]
        if not liked_inners:
            return []

        target_vector = self._qi[target_inner]
        liked_vectors = self._qi[liked_inners]

        norm_target = np.linalg.norm(target_vector)
        norm_liked = np.linalg.norm(liked_vectors, axis=1)
        
        if norm_target == 0:
            return []
            
        denom = norm_liked * norm_target
        denom[denom == 0] = 1e-9
        
        sims = np.dot(liked_vectors, target_vector) / denom
        
        reasons = []
        for idx, inner_id in enumerate(liked_inners):
            sim = sims[idx]
            if sim > 0:
                reasons.append({
                    'movie_id': self._inner_to_raw_item[inner_id],
                    'similarity': float(sim)
                })

        reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return reasons[:top_n_reasons]