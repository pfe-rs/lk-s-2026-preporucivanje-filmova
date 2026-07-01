import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Optional, Tuple, Dict, Any, Set
import time
import os
import psutil
from loguru import logger
from tqdm import tqdm
from numba import njit, prange
from src.utils.logger import LoggingConfig, StepLogger

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
            out_factors[row] = 0.0
            continue
        F = np.empty((cnt, k), dtype=other_factors.dtype)
        target = np.empty(cnt, dtype=data.dtype)
        for i in range(cnt):
            j = indices[start + i]
            F[i] = other_factors[j]
            target[i] = data[start + i] - mu - other_bias[j]
        A = np.empty((dim, dim), dtype=F.dtype)
        bvec = np.empty(dim, dtype=F.dtype)
        A[0, 0] = cnt
        bvec[0] = np.sum(target)
        sum_F = np.sum(F, axis=0)
        A[0, 1:] = sum_F
        A[1:, 0] = sum_F
        A[1:, 1:] = F.T @ F
        bvec[1:] = F.T @ target
        for a in range(dim):
            A[a, a] += reg
        w = np.linalg.solve(A, bvec)
        out_bias[row] = w[0]
        out_factors[row] = w[1:]


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
        logging_config: Optional[LoggingConfig] = None,
        progress_bar: bool = True,
    ) -> None:
        self.k_components = k_components
        self.reg_all = np.float32(reg_all)
        self.lr_all = np.float32(lr_all)
        self.n_epochs = n_epochs
        self.alpha = np.float32(alpha)
        self.min_ratings = min_ratings
        self.random_state = random_state
        self.progress_bar = progress_bar

        self._raw_to_inner_user: Dict[int, int] = {}
        self._raw_to_inner_item: Dict[int, int] = {}
        self._inner_to_raw_user: np.ndarray = np.array([])
        self._inner_to_raw_item: np.ndarray = np.array([])

        self._global_mean: np.float32 = np.float32(3.0)
        self._item_popularity: np.ndarray = np.array([])
        self._popular_movies: List[int] = []
        self._popular_inner: np.ndarray = np.array([], dtype=np.int32)

        self._pu: np.ndarray = np.array([])
        self._qi: np.ndarray = np.array([])
        self._bu: np.ndarray = np.array([])
        self._bi: np.ndarray = np.array([])

        self._is_fitted: bool = False

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
        ratings = df_filtered["rating"].values.astype(np.float32)

        self._global_mean = np.float32(ratings.mean())

        popularity_mapped = popularity_series[valid_movies].rename(self._raw_to_inner_item)
        self._item_popularity = np.zeros(n_items, dtype=np.float32)
        self._item_popularity[popularity_mapped.index] = popularity_mapped.values.astype(np.float32)

        self._popular_movies = valid_movies.tolist()
        self._popular_inner = np.array(
            [self._raw_to_inner_item[m] for m in self._popular_movies if m in self._raw_to_inner_item],
            dtype=np.int32,
        )

        user_csr = csr_matrix((ratings, (user_idx, item_idx)), shape=(n_users, n_items))
        item_csr = csr_matrix((ratings, (item_idx, user_idx)), shape=(n_items, n_users))

        user_indptr = user_csr.indptr.astype(np.int32)
        user_indices = user_csr.indices.astype(np.int32)
        user_data = user_csr.data.astype(np.float32)

        item_indptr = item_csr.indptr.astype(np.int32)
        item_indices = item_csr.indices.astype(np.int32)
        item_data = item_csr.data.astype(np.float32)

        rng = np.random.default_rng(self.random_state)
        init_scale = np.float32(1.0 / np.sqrt(self.k_components))
        p = rng.normal(0.0, init_scale, size=(n_users, self.k_components)).astype(np.float32)
        q = rng.normal(0.0, init_scale, size=(n_items, self.k_components)).astype(np.float32)
        bu = np.zeros(n_users, dtype=np.float32)
        bi = np.zeros(n_items, dtype=np.float32)
        mu = np.float32(self._global_mean)
        reg = np.float32(self.reg_all)

        epochs_iter = tqdm(range(self.n_epochs), desc="ALS epochs") if self.progress_bar else range(self.n_epochs)
        for _ in epochs_iter:
            _als_solve(user_indptr, user_indices, user_data, q, bi, mu, reg, p, bu)
            _als_solve(item_indptr, item_indices, item_data, p, bu, mu, reg, q, bi)

        self._pu = np.ascontiguousarray(p, dtype=np.float32)
        self._qi = np.ascontiguousarray(q, dtype=np.float32)
        self._bu = np.ascontiguousarray(bu, dtype=np.float32)
        self._bi = np.ascontiguousarray(bi, dtype=np.float32)
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

        pred = (
            self._global_mean
            + self._bu[u_inner]
            + self._bi[i_inner]
            + np.dot(self._pu[u_inner], self._qi[i_inner])
        )
        return float(np.clip(pred, MIN_RATING, MAX_RATING))

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        u_inner = self._raw_to_inner_user.get(user_id)
        if u_inner is None:
            return [float(np.clip(self._global_mean, MIN_RATING, MAX_RATING))] * len(item_ids)

        item_arr = np.asarray(item_ids, dtype=np.int32)
        item_idx = np.array(
            [self._raw_to_inner_item.get(int(mid), -1) for mid in item_arr], dtype=np.int32
        )
        valid_mask = item_idx >= 0
        scores = np.full(len(item_ids), self._global_mean, dtype=np.float32)

        if valid_mask.any():
            valid_idx = item_idx[valid_mask]
            dots = self._qi[valid_idx] @ self._pu[u_inner]
            preds = self._global_mean + self._bu[u_inner] + self._bi[valid_idx] + dots
            np.clip(preds, MIN_RATING, MAX_RATING, out=preds)
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

        watched_inner = {
            idx
            for mid in watched_movie_ids
            for idx in [self._raw_to_inner_item.get(mid)]
            if idx is not None
        }

        scores = self._qi @ self._pu[u_inner]
        scores += self._global_mean + self._bu[u_inner] + self._bi

        for idx in watched_inner:
            scores[idx] = -np.inf

        n_items = self._qi.shape[0]
        top_k = min(top_n, n_items)
        if top_k <= 0:
            return []

        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            score = float(np.clip(scores[idx], MIN_RATING, MAX_RATING))
            results.append((int(self._inner_to_raw_item[idx]), score))
        return results

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10, valid_items: Optional[List[int]] = None
    ) -> List[int]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        if valid_items is not None:
            u_inner = self._raw_to_inner_user.get(user_id)
            if u_inner is None:
                fallback = [mid for mid in valid_items if mid not in watched_items]
                return fallback[:k]

            candidate_pairs = []
            unmapped_mids = []
            for mid in valid_items:
                if mid in watched_items:
                    continue
                idx = self._raw_to_inner_item.get(mid)
                if idx is not None:
                    candidate_pairs.append((mid, idx))
                else:
                    unmapped_mids.append(mid)

            if not candidate_pairs:
                return unmapped_mids[:k]

            mids_mapped = [p[0] for p in candidate_pairs]
            idxs_mapped = np.array([p[1] for p in candidate_pairs], dtype=np.int32)

            P_u = self._pu[u_inner]
            Q_subset = self._qi[idxs_mapped]
            bi_subset = self._bi[idxs_mapped]
            bu = self._bu[u_inner]

            scores = (Q_subset @ P_u) + self._global_mean + bu + bi_subset

            top_n = min(k, len(scores))
            top_sub_idx = np.argpartition(-scores, top_n - 1)[:top_n]
            top_sub_idx = top_sub_idx[np.argsort(-scores[top_sub_idx])]

            mapped_results = [mids_mapped[i] for i in top_sub_idx]
            if len(mapped_results) < k:
                mapped_results.extend(unmapped_mids[:k - len(mapped_results)])
            return mapped_results

        recs = self.recommend_for_user(user_id, list(watched_items), top_n=k)
        return [mid for mid, _ in recs]

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[Set[int]],
        k: int,
        valid_items: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        if not self._is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        n_users_batch = len(user_ids)
        if n_users_batch == 0 or k <= 0:
            return [[] for _ in range(n_users_batch)]

        if valid_items is not None:
            return [
                self.get_top_k_recommendations(uid, watched, k, valid)
                for uid, watched, valid in zip(user_ids, watched_items_list, valid_items)
            ]

        chunk_size = 256
        all_results: List[Optional[List[int]]] = [None] * n_users_batch
        Q = self._qi
        Bi = self._bi
        mu = self._global_mean
        n_items = Q.shape[0]

        for start in range(0, n_users_batch, chunk_size):
            end = min(start + chunk_size, n_users_batch)
            chunk_user_ids = user_ids[start:end]
            chunk_watched = watched_items_list[start:end]
            b = len(chunk_user_ids)

            u_inner_list = [self._raw_to_inner_user.get(uid) for uid in chunk_user_ids]

            P_batch = np.zeros((b, self.k_components), dtype=np.float32)
            bu_batch = np.zeros(b, dtype=np.float32)
            valid_mask = np.zeros(b, dtype=bool)

            for i, uid_inner in enumerate(u_inner_list):
                if uid_inner is not None:
                    P_batch[i] = self._pu[uid_inner]
                    bu_batch[i] = self._bu[uid_inner]
                    valid_mask[i] = True

            scores_matrix = np.full((b, n_items), mu, dtype=np.float32)
            if valid_mask.any():
                P_valid = np.ascontiguousarray(P_batch[valid_mask])
                sub = np.empty((int(valid_mask.sum()), n_items), dtype=np.float32)
                np.dot(P_valid, Q.T, out=sub)
                sub += mu
                np.add(sub, bu_batch[valid_mask, None], out=sub)
                np.add(sub, Bi, out=sub)
                np.clip(sub, MIN_RATING, MAX_RATING, out=sub)
                scores_matrix[valid_mask] = sub

            for i in range(b):
                if not valid_mask[i]:
                    watched = chunk_watched[i]
                    fallback: List[int] = []
                    for raw_id in self._popular_movies:
                        if raw_id not in watched:
                            fallback.append(raw_id)
                        if len(fallback) >= k:
                            break
                    all_results[start + i] = fallback
                    continue

                watched = chunk_watched[i]
                if watched:
                    watched_inner = [
                        self._raw_to_inner_item[mid]
                        for mid in watched
                        if mid in self._raw_to_inner_item
                    ]
                    if watched_inner:
                        scores_matrix[i, watched_inner] = -np.inf

                top_k = min(k, n_items)
                if top_k == 0:
                    all_results[start + i] = []
                    continue

                top_indices = np.argpartition(-scores_matrix[i], top_k - 1)[:top_k]
                top_indices = top_indices[np.argsort(-scores_matrix[i, top_indices])]
                all_results[start + i] = [int(self._inner_to_raw_item[idx]) for idx in top_indices]

        return all_results

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        if not self._is_fitted:
            return []

        inner_movie = self._raw_to_inner_item.get(movie_id)
        if inner_movie is None:
            return []

        liked_inner = [
            idx
            for mid in liked_items
            for idx in [self._raw_to_inner_item.get(mid)]
            if idx is not None
        ]
        if not liked_inner:
            return []

        Q_movie = self._qi[inner_movie]
        Q_liked = self._qi[liked_inner]

        movie_norm = np.linalg.norm(Q_movie) + 1e-8
        liked_norms = np.linalg.norm(Q_liked, axis=1) + 1e-8
        similarities = (Q_liked @ Q_movie) / (liked_norms * movie_norm)

        top_n = min(top_n_reasons, len(liked_inner))
        if top_n <= 0:
            return []

        top_indices = np.argpartition(-similarities, top_n - 1)[:top_n]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [
            {"movie_id": int(self._inner_to_raw_item[liked_inner[idx]]), "similarity": float(similarities[idx])}
            for idx in top_indices
        ]