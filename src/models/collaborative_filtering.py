import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Tuple, Dict, Any, Set
import time
import os
import psutil
from loguru import logger
from src.utils.logger import LoggingConfig, StepLogger
from tqdm import tqdm
from numba import njit, prange

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

MIN_RATING = np.float32(0.5)
MAX_RATING = np.float32(5.0)


@njit(parallel=True, fastmath=True, cache=True)
def _als_solve(indptr, indices, data, other_factors, other_bias, mu, reg, out_factors, out_bias):
    n = indptr.shape[0] - 1
    k = other_factors.shape[1]
    dim = k + 1
    
    for row in prange(n):
        start = indptr[row]
        end = indptr[row + 1]
        cnt = end - start
        
        if cnt == 0:
            out_bias[row] = 0.0
            for a in range(k):
                out_factors[row, a] = 0.0
            continue
            
        A = np.zeros((dim, dim))
        bvec = np.zeros(dim)
        
        for pos in range(start, end):
            j = indices[pos]
            r = data[pos]
            target = r - mu - other_bias[j]
            
            A[0, 0] += 1.0
            bvec[0] += target
            
            for a in range(k):
                fa = other_factors[j, a]
                A[0, a + 1] += fa
                A[a + 1, 0] += fa
                bvec[a + 1] += fa * target
                
                for c in range(k):
                    A[a + 1, c + 1] += fa * other_factors[j, c]
                    
        for a in range(dim):
            A[a, a] += reg
            
        w = np.linalg.solve(A, bvec)
        out_bias[row] = w[0]
        for a in range(k):
            out_factors[row, a] = w[a + 1]


class CollaborativeFiltering:
    def __init__(
        self,
        k_components: int = 100,
        reg_all: float = 0.02,
        lr_all: float = 0.005,
        n_epochs: int = 20,
        alpha: float = 0.5,
        min_ratings: int = 5,
        random_state: int = 42,
        use_gpu: bool = True,
        logging_config: Optional[LoggingConfig] = None,
        progress_bar: bool = True
    ) -> None:
        self.k_components = k_components
        self.reg_all = np.float32(reg_all)
        self.lr_all = np.float32(lr_all)
        self.n_epochs = n_epochs
        self.alpha = np.float32(alpha)
        self.min_ratings = min_ratings
        self.random_state = random_state
        self.use_gpu = use_gpu and HAS_GPU
        self.progress_bar = progress_bar

        self._raw_to_inner_user: Dict[int, int] = {}
        self._raw_to_inner_item: Dict[int, int] = {}
        self._inner_to_raw_user: np.ndarray = np.array([])
        self._inner_to_raw_item: np.ndarray = np.array([])

        self._global_mean: np.float32 = np.float32(3.0)
        self._item_popularity: np.ndarray = np.array([])
        self._popular_movies: List[int] = []
        self._popular_inner: np.ndarray = np.array([], dtype=np.int32)

        self._is_fitted: bool = False
        self._device: str = "cpu"

        self.config = logging_config or LoggingConfig()
        self.step_logger = StepLogger(self.config)

        logger.info("CollaborativeFiltering initialised. Explicit-feedback biased ALS model ready for training.")

    def fit(self, df_ratings: pd.DataFrame) -> "CollaborativeFiltering":
        if df_ratings.empty:
            raise ValueError("Ratings DataFrame is empty. Cannot fit model.")

        total_start = time.perf_counter()
        total_cpu = time.process_time()

        required_cols = {"userId", "movieId", "rating"}
        if not required_cols.issubset(df_ratings.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        df_clean = df_ratings.dropna(subset=["rating"]).copy()
        df_clean = df_clean[np.isfinite(df_clean["rating"])]

        popularity_series = df_clean["movieId"].value_counts()
        valid_movies = popularity_series[popularity_series >= self.min_ratings].index
        df_filtered = df_clean[df_clean["movieId"].isin(valid_movies)]

        unique_users = df_filtered["userId"].unique()
        unique_items = df_filtered["movieId"].unique()
        n_users = len(unique_users)
        n_items = len(unique_items)

        self._raw_to_inner_user = dict(zip(unique_users, range(n_users)))
        self._raw_to_inner_item = dict(zip(unique_items, range(n_items)))
        self._inner_to_raw_user = unique_users.astype(np.int32)
        self._inner_to_raw_item = unique_items.astype(np.int32)

        user_idx = df_filtered["userId"].map(self._raw_to_inner_user).values.astype(np.int32)
        item_idx = df_filtered["movieId"].map(self._raw_to_inner_item).values.astype(np.int32)
        ratings = df_filtered["rating"].values.astype(np.float64)

        self._global_mean = np.float32(ratings.mean())

        popularity_mapped = popularity_series[valid_movies].rename(self._raw_to_inner_item)
        self._item_popularity = np.zeros(n_items, dtype=np.float32)
        self._item_popularity[popularity_mapped.index] = popularity_mapped.values.astype(np.float32)

        self._popular_movies = valid_movies.tolist()
        self._popular_inner = np.array(
            [self._raw_to_inner_item[m] for m in self._popular_movies if m in self._raw_to_inner_item],
            dtype=np.int32
        )

        user_csr = csr_matrix((ratings, (user_idx, item_idx)), shape=(n_users, n_items))
        item_csr = csr_matrix((ratings, (item_idx, user_idx)), shape=(n_items, n_users))

        user_indptr = user_csr.indptr.astype(np.int64)
        user_indices = user_csr.indices.astype(np.int64)
        user_data = user_csr.data.astype(np.float64)

        item_indptr = item_csr.indptr.astype(np.int64)
        item_indices = item_csr.indices.astype(np.int64)
        item_data = item_csr.data.astype(np.float64)

        rng = np.random.default_rng(self.random_state)
        init_scale = 1.0 / np.sqrt(self.k_components)
        p = rng.normal(0.0, init_scale, size=(n_users, self.k_components)).astype(np.float64)
        q = rng.normal(0.0, init_scale, size=(n_items, self.k_components)).astype(np.float64)
        bu = np.zeros(n_users, dtype=np.float64)
        bi = np.zeros(n_items, dtype=np.float64)
        mu = float(self._global_mean)
        reg = float(self.reg_all)

        epochs_iter = tqdm(range(self.n_epochs), desc="ALS epochs") if self.progress_bar else range(self.n_epochs)
        for _ in epochs_iter:
            _als_solve(user_indptr, user_indices, user_data, q, bi, mu, reg, p, bu)
            _als_solve(item_indptr, item_indices, item_data, p, bu, mu, reg, q, bi)

        self._user_factors = p.astype(np.float32)
        self._item_factors = q.astype(np.float32)
        self._user_biases = bu.astype(np.float32)
        self._item_biases = bi.astype(np.float32)

        if self.use_gpu and HAS_GPU:
            self._user_factors = cp.asarray(self._user_factors)
            self._item_factors = cp.asarray(self._item_factors)
            self._user_biases = cp.asarray(self._user_biases)
            self._item_biases = cp.asarray(self._item_biases)
            self._global_mean_gpu = cp.asarray(self._global_mean)
            self._device = "gpu"
        else:
            self._device = "cpu"

        self._is_fitted = True

        self.step_logger.log_step("Training complete", total_start, total_cpu)
        total_wall = time.perf_counter() - total_start
        total_cpu_elapsed = time.process_time() - total_cpu
        mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        logger.info(f"Fitting complete | wall={total_wall:.2f}s cpu={total_cpu_elapsed:.2f}s mem={mem_final:.1f}MB")
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        u_inner = self._raw_to_inner_user.get(user_id)
        i_inner = self._raw_to_inner_item.get(movie_id)

        if u_inner is None or i_inner is None:
            return float(np.clip(self._global_mean, MIN_RATING, MAX_RATING))

        if self._device == "gpu":
            dot = (self._user_factors[u_inner] @ self._item_factors[i_inner].T).item()
            mu = self._global_mean_gpu.item()
            bu = self._user_biases[u_inner].item()
            bi = self._item_biases[i_inner].item()
        else:
            dot = np.dot(self._user_factors[u_inner], self._item_factors[i_inner])
            mu = self._global_mean
            bu = self._user_biases[u_inner]
            bi = self._item_biases[i_inner]

        pred = mu + bu + bi + dot
        return float(np.clip(pred, MIN_RATING, MAX_RATING))

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        u_inner = self._raw_to_inner_user.get(user_id)
        if u_inner is None:
            return [float(np.clip(self._global_mean, MIN_RATING, MAX_RATING)) for _ in item_ids]

        item_idx = np.array([self._raw_to_inner_item.get(mid, -1) for mid in item_ids], dtype=np.int32)
        valid_mask = item_idx >= 0
        scores = np.full(len(item_ids), self._global_mean, dtype=np.float32)

        if valid_mask.any():
            valid_idx = item_idx[valid_mask]
            if self._device == "gpu":
                P_u = self._user_factors[u_inner]
                Q_valid = self._item_factors[valid_idx]
                bi_valid = self._item_biases[valid_idx]
                bu = self._user_biases[u_inner]
                dots = P_u @ Q_valid.T
                preds = self._global_mean_gpu + bu + bi_valid + dots
                preds = cp.clip(preds, MIN_RATING, MAX_RATING)
                scores[valid_mask] = cp.asnumpy(preds)
            else:
                P_u = self._user_factors[u_inner]
                Q_valid = self._item_factors[valid_idx]
                bi_valid = self._item_biases[valid_idx]
                bu = self._user_biases[u_inner]
                dots = np.dot(P_u, Q_valid.T)
                preds = self._global_mean + bu + bi_valid + dots
                preds = np.clip(preds, MIN_RATING, MAX_RATING)
                scores[valid_mask] = preds

        return scores.tolist()

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        u_inner = self._raw_to_inner_user.get(user_id)
        if u_inner is None:
            watched_set = set(watched_movie_ids)
            fallback = []
            for raw_id in self._popular_movies:
                if raw_id not in watched_set:
                    fallback.append((raw_id, float(np.clip(self._global_mean, MIN_RATING, MAX_RATING))))
                if len(fallback) >= top_n:
                    break
            return fallback

        watched_inner = set()
        for mid in watched_movie_ids:
            idx = self._raw_to_inner_item.get(mid)
            if idx is not None:
                watched_inner.add(idx)

        if self._device == "gpu":
            P_u = self._user_factors[u_inner]
            bu = self._user_biases[u_inner]
            scores = self._item_factors @ P_u
            scores = scores + self._global_mean_gpu + bu + self._item_biases
            scores = cp.asnumpy(scores)
        else:
            P_u = self._user_factors[u_inner]
            bu = self._user_biases[u_inner]
            scores = self._item_factors @ P_u
            scores = scores + self._global_mean + bu + self._item_biases
            scores = np.asarray(scores, dtype=np.float32)

        for idx in watched_inner:
            scores[idx] = -np.inf

        n_items = len(self._item_factors)
        top_k = min(top_n, n_items)
        if top_k <= 0:
            return []
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        results = []
        for idx in top_indices:
            score = np.clip(scores[idx], MIN_RATING, MAX_RATING)
            raw_id = int(self._inner_to_raw_item[idx])
            results.append((raw_id, float(score)))
        return results

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10
    ) -> List[int]:
        recs = self.recommend_for_user(user_id, list(watched_items), top_n=k)
        return [mid for mid, _ in recs]

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[Set[int]],
        k: int
    ) -> List[List[int]]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        n_users_batch = len(user_ids)
        if n_users_batch == 0 or k <= 0:
            return [[] for _ in range(n_users_batch)]

        chunk_size = 256
        all_results = [None] * n_users_batch

        for start in range(0, n_users_batch, chunk_size):
            end = min(start + chunk_size, n_users_batch)
            chunk_user_ids = user_ids[start:end]
            chunk_watched = watched_items_list[start:end]
            b = len(chunk_user_ids)

            u_inner_list = [self._raw_to_inner_user.get(uid) for uid in chunk_user_ids]
            known_mask = [uid is not None for uid in u_inner_list]

            P_batch = np.zeros((b, self.k_components), dtype=np.float32)
            bu_batch = np.zeros(b, dtype=np.float32)
            for i, known in enumerate(known_mask):
                if known:
                    if self._device == "gpu":
                        P_batch[i] = cp.asnumpy(self._user_factors[u_inner_list[i]])
                        bu_batch[i] = cp.asnumpy(self._user_biases[u_inner_list[i]])
                    else:
                        P_batch[i] = self._user_factors[u_inner_list[i]]
                        bu_batch[i] = self._user_biases[u_inner_list[i]]

            if self._device == "gpu":
                Q = cp.asnumpy(self._item_factors)
                Bi = cp.asnumpy(self._item_biases)
                mu = cp.asnumpy(self._global_mean_gpu)
            else:
                Q = self._item_factors
                Bi = self._item_biases
                mu = self._global_mean

            scores_matrix = P_batch @ Q.T + mu + bu_batch[:, None] + Bi[None, :]
            scores_matrix = np.clip(scores_matrix, MIN_RATING, MAX_RATING)

            for i in range(b):
                watched = chunk_watched[i]
                watched_inner = []
                for mid in watched:
                    idx = self._raw_to_inner_item.get(mid)
                    if idx is not None:
                        watched_inner.append(idx)
                if watched_inner:
                    scores_matrix[i, watched_inner] = -np.inf
                top_k = min(k, Q.shape[0])
                if top_k == 0:
                    all_results[start + i] = []
                    continue
                top_indices = np.argpartition(-scores_matrix[i], top_k - 1)[:top_k]
                top_indices = top_indices[np.argsort(-scores_matrix[i, top_indices])]
                raw_ids = [int(self._inner_to_raw_item[idx]) for idx in top_indices]
                all_results[start + i] = raw_ids

        return all_results

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        if not self._is_fitted:
            return []

        inner_movie = self._raw_to_inner_item.get(movie_id)
        if inner_movie is None:
            return []

        liked_inner = []
        for mid in liked_items:
            idx = self._raw_to_inner_item.get(mid)
            if idx is not None:
                liked_inner.append(idx)
        if not liked_inner:
            return []

        if self._device == "gpu":
            Q_movie = cp.asnumpy(self._item_factors[inner_movie])
            Q_liked = cp.asnumpy(self._item_factors[liked_inner])
        else:
            Q_movie = self._item_factors[inner_movie]
            Q_liked = self._item_factors[liked_inner]

        movie_norm = np.linalg.norm(Q_movie) + 1e-8
        liked_norms = np.linalg.norm(Q_liked, axis=1) + 1e-8
        similarities = (Q_liked @ Q_movie.T) / (liked_norms * movie_norm)

        top_n = min(top_n_reasons, len(liked_inner))
        top_indices = np.argpartition(-similarities, top_n - 1)[:top_n]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        explanations = []
        for idx in top_indices:
            raw_id = int(self._inner_to_raw_item[liked_inner[idx]])
            sim = float(similarities[idx])
            explanations.append({"movie_id": raw_id, "similarity": sim})
        return explanations