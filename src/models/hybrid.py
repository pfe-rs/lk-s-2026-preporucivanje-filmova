import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
import numba as nb
from typing import Any, Dict, List, Optional
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger import LoggingConfig, StepLogger
from src.models.models import RecommenderProtocol

_F32_MIN = np.float32(0.5)
_F32_MAX = np.float32(5.0)

_CF_ATTRS = (
    "_pu", "_qi", "_bu", "_bi", "_global_mean",
    "_raw_to_inner_user", "_raw_to_inner_item",
)
_CB_ATTRS = (
    "user_profiles", "feature_matrix_norm", "soft_pop_arr",
    "movie_index", "user_index", "base_pop_norm_arr",
)

_E1F = np.empty(0, dtype=np.float32)
_E2F = np.empty((0, 0), dtype=np.float32)
_E1I = np.empty(0, dtype=np.int32)
_E1I64 = np.empty(0, dtype=np.int64)


def _cf_ready(m: Any) -> bool:
    return all(hasattr(m, a) for a in _CF_ATTRS)


def _cb_ready(m: Any) -> bool:
    return all(hasattr(m, a) for a in _CB_ATTRS)


def _build_lookup(keys, max_val: int) -> np.ndarray:
    if not keys:
        return np.array([-1], dtype=np.int32)
    arr = np.full(max_val + 1, -1, dtype=np.int32)
    arr[list(keys)] = np.arange(len(keys), dtype=np.int32)
    return arr


def _ensure_csr(mat: Any) -> sp.csr_matrix:
    if mat is None:
        return sp.csr_matrix((0, 0), dtype=np.float32)
    if sp.issparse(mat):
        return mat.tocsr().astype(np.float32)
    return sp.csr_matrix(mat, dtype=np.float32)


def _to_dense_f32(arr: Any) -> np.ndarray:
    if sp.issparse(arr):
        return arr.toarray().astype(np.float32)
    return np.asarray(arr, dtype=np.float32)


@nb.njit(nogil=True, fastmath=True)
def _compute_single_score(
    cf_inner, cb_inner,
    u_in_cf, pu, qi, bu, bi, cf_mu,
    u_in_cb, cb_u_data, cb_u_indices, cb_u_indptr,
    cb_i_data, cb_i_indices, cb_i_indptr,
    base_pop, soft_pop,
    alpha, beta
):
    score = 0.0
    if u_in_cf >= 0 and cf_inner >= 0:
        cf_val = cf_mu + bu[u_in_cf] + bi[cf_inner]
        for f in range(pu.shape[1]):
            cf_val += pu[u_in_cf, f] * qi[cf_inner, f]
        if cf_val < 0.5:
            cf_val = 0.5
        elif cf_val > 5.0:
            cf_val = 5.0
        score += alpha * cf_val

    if u_in_cb >= 0 and cb_inner >= 0:
        cb_val = base_pop[cb_inner]
        u_start = cb_u_indptr[u_in_cb]
        u_end = cb_u_indptr[u_in_cb + 1]
        i_start = cb_i_indptr[cb_inner]
        i_end = cb_i_indptr[cb_inner + 1]
        p1 = u_start
        p2 = i_start
        dot = 0.0
        while p1 < u_end and p2 < i_end:
            idx_u = cb_u_indices[p1]
            idx_i = cb_i_indices[p2]
            if idx_u == idx_i:
                dot += cb_u_data[p1] * cb_i_data[p2]
                p1 += 1
                p2 += 1
            elif idx_u < idx_i:
                p1 += 1
            else:
                p2 += 1
        cb_val += soft_pop[cb_inner] * dot
        score += beta * cb_val

    return score


@nb.njit(nogil=True, fastmath=True)
def _numba_get_top_k(
    movie_ids, cf_inner_indices, cb_inner_indices, watched_arr, k,
    u_in_cf, pu, qi, bu, bi, cf_mu,
    u_in_cb, cb_u_data, cb_u_indices, cb_u_indptr,
    cb_i_data, cb_i_indices, cb_i_indptr,
    base_pop, soft_pop, alpha, beta
):
    top_k_scores = np.full(k, -1e9, dtype=np.float32)
    top_k_ids = np.full(k, -1, dtype=np.int64)
    min_idx = 0
    min_score = -1e9

    n_movies = len(movie_ids)
    n_watched = len(watched_arr)

    for i in range(n_movies):
        mid = movie_ids[i]

        left = 0
        right = n_watched - 1
        is_watched = False
        while left <= right:
            mid_idx = (left + right) >> 1
            w_mid = watched_arr[mid_idx]
            if w_mid == mid:
                is_watched = True
                break
            elif w_mid < mid:
                left = mid_idx + 1
            else:
                right = mid_idx - 1

        if is_watched:
            continue

        cf_inner = cf_inner_indices[i]
        cb_inner = cb_inner_indices[i]

        score = _compute_single_score(
            cf_inner, cb_inner,
            u_in_cf, pu, qi, bu, bi, cf_mu,
            u_in_cb, cb_u_data, cb_u_indices, cb_u_indptr,
            cb_i_data, cb_i_indices, cb_i_indptr,
            base_pop, soft_pop, alpha, beta
        )

        if score > min_score:
            top_k_scores[min_idx] = score
            top_k_ids[min_idx] = mid
            new_min_score = top_k_scores[0]
            new_min_idx = 0
            for j in range(1, k):
                if top_k_scores[j] < new_min_score:
                    new_min_score = top_k_scores[j]
                    new_min_idx = j
            min_score = new_min_score
            min_idx = new_min_idx

    for i in range(k):
        for j in range(i + 1, k):
            if top_k_scores[i] < top_k_scores[j]:
                tmp_s = top_k_scores[i]
                top_k_scores[i] = top_k_scores[j]
                top_k_scores[j] = tmp_s
                tmp_id = top_k_ids[i]
                top_k_ids[i] = top_k_ids[j]
                top_k_ids[j] = tmp_id

    valid_cnt = 0
    for i in range(k):
        if top_k_ids[i] != -1:
            top_k_ids[valid_cnt] = top_k_ids[i]
            valid_cnt += 1

    return top_k_ids[:valid_cnt]


@nb.njit(nogil=True, fastmath=True)
def _numba_predict_scores(
    item_ids, cf_lookup, cf_max, cb_lookup, cb_max,
    u_in_cf, pu, qi, bu, bi, cf_mu,
    u_in_cb, cb_u_data, cb_u_indices, cb_u_indptr,
    cb_i_data, cb_i_indices, cb_i_indptr,
    base_pop, soft_pop, alpha, beta
):
    n = len(item_ids)
    res = np.empty(n, dtype=np.float32)
    for i in range(n):
        mid = item_ids[i]
        cf_inner = -1
        cb_inner = -1
        if 0 <= mid <= cf_max:
            cf_inner = cf_lookup[mid]
        if 0 <= mid <= cb_max:
            cb_inner = cb_lookup[mid]

        res[i] = _compute_single_score(
            cf_inner, cb_inner,
            u_in_cf, pu, qi, bu, bi, cf_mu,
            u_in_cb, cb_u_data, cb_u_indices, cb_u_indptr,
            cb_i_data, cb_i_indices, cb_i_indptr,
            base_pop, soft_pop, alpha, beta
        )
    return res


class HybridRecommender:
    def __init__(
        self,
        cf_model: RecommenderProtocol,
        cb_model: RecommenderProtocol,
        alpha: float = 0.5,
        logging_config: LoggingConfig = LoggingConfig(),
        use_gpu: bool = False,
        enable_cache: bool = False,
    ):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = np.float32(np.clip(alpha, 0.0, 1.0))
        self._beta = np.float32(1.0 - self.alpha)
        self.ratings_df = None
        self.movies_df = None
        self.config = logging_config
        self.step_logger = StepLogger(self.config)
        self.enable_cache = enable_cache
        self._cache = {} if enable_cache else None
        self._pool = ThreadPoolExecutor(max_workers=2)

        self._movie_ids = None
        self._cf_user_lookup = None
        self._cf_item_lookup = None
        self._cf_max_user_id = 0
        self._cf_max_item_id = 0
        self._cb_movie_lookup = None
        self._cb_user_lookup = None
        self._cb_max_movie_id = 0
        self._cb_max_user_id = 0
        self._cb_profiles_sparse = None
        self._cb_feats_sparse = None
        self._cb_base_pop_norm = None
        self._cb_soft_pop_norm = None

        self._numba_movie_ids = _E1I64
        self._numba_cf_inner = _E1I
        self._numba_cb_inner = _E1I
        self._nb_pu = _E2F
        self._nb_qi = _E2F
        self._nb_bu = _E1F
        self._nb_bi = _E1F
        self._nb_cf_mu = np.float32(0.0)
        self._nb_base_pop = _E1F
        self._nb_soft_pop = _E1F
        self._nb_cb_i_data = _E1F
        self._nb_cb_i_indices = _E1I
        self._nb_cb_i_indptr = _E1I
        self._nb_cb_u_data = _E1F
        self._nb_cb_u_indices = _E1I
        self._nb_cb_u_indptr = _E1I

    def _build_cf_lookups(self) -> None:
        if not _cf_ready(self.cf_model):
            return
        raw_users = self.cf_model._raw_to_inner_user
        raw_items = self.cf_model._raw_to_inner_item
        if raw_users and raw_items:
            self._cf_max_user_id = max(raw_users.keys())
            self._cf_max_item_id = max(raw_items.keys())
            self._cf_user_lookup = _build_lookup(raw_users, self._cf_max_user_id)
            self._cf_item_lookup = _build_lookup(raw_items, self._cf_max_item_id)

    def _build_cb_lookups(self) -> None:
        if not _cb_ready(self.cb_model):
            return
        cb = self.cb_model
        if cb.movie_index is not None and len(cb.movie_index):
            self._cb_max_movie_id = cb.movie_index.max()
            self._cb_movie_lookup = _build_lookup(
                dict(zip(cb.movie_index, range(len(cb.movie_index)))),
                self._cb_max_movie_id,
            )
        if cb.user_index is not None and len(cb.user_index):
            self._cb_max_user_id = cb.user_index.max()
            self._cb_user_lookup = _build_lookup(
                dict(zip(cb.user_index, range(len(cb.user_index)))),
                self._cb_max_user_id,
            )
        self._cb_profiles_sparse = _ensure_csr(cb.user_profiles)
        self._cb_feats_sparse = _ensure_csr(cb.feature_matrix_norm)
        base = _to_dense_f32(cb.base_pop_norm_arr)
        soft = _to_dense_f32(cb.soft_pop_arr)
        self._cb_base_pop_norm = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        self._cb_soft_pop_norm = np.nan_to_num(soft, nan=0.0, posinf=0.0, neginf=0.0)

    def _sync_movie_ids(self) -> None:
        if hasattr(self.cb_model, "movie_ids") and len(self.cb_model.movie_ids):
            self._movie_ids = np.asarray(self.cb_model.movie_ids, dtype=np.int64)
        elif hasattr(self.cb_model, "movie_id_to_idx"):
            self._movie_ids = np.fromiter(self.cb_model.movie_id_to_idx.keys(), dtype=np.int64)
        elif self.movies_df is not None:
            self._movie_ids = self.movies_df["movieId"].unique().astype(np.int64)
        else:
            self._movie_ids = None

    def _sync_numba_pointers(self) -> None:
        if self._movie_ids is None or len(self._movie_ids) == 0:
            return

        self._numba_movie_ids = self._movie_ids.astype(np.int64)
        n = len(self._movie_ids)

        self._numba_cf_inner = np.full(n, -1, dtype=np.int32)
        if _cf_ready(self.cf_model) and self._cf_item_lookup is not None:
            mask = (self._movie_ids >= 0) & (self._movie_ids <= self._cf_max_item_id)
            self._numba_cf_inner[mask] = self._cf_item_lookup[self._movie_ids[mask]]
            self._nb_pu = self.cf_model._pu.astype(np.float32)
            self._nb_qi = self.cf_model._qi.astype(np.float32)
            self._nb_bu = self.cf_model._bu.astype(np.float32)
            self._nb_bi = self.cf_model._bi.astype(np.float32)
            self._nb_cf_mu = np.float32(self.cf_model._global_mean)
        else:
            self._nb_pu = _E2F
            self._nb_qi = _E2F
            self._nb_bu = _E1F
            self._nb_bi = _E1F
            self._nb_cf_mu = np.float32(0.0)

        self._numba_cb_inner = np.full(n, -1, dtype=np.int32)
        if _cb_ready(self.cb_model) and self._cb_movie_lookup is not None:
            mask = (self._movie_ids >= 0) & (self._movie_ids <= self._cb_max_movie_id)
            self._numba_cb_inner[mask] = self._cb_movie_lookup[self._movie_ids[mask]]
            self._nb_base_pop = self._cb_base_pop_norm.astype(np.float32)
            self._nb_soft_pop = self._cb_soft_pop_norm.astype(np.float32)
            self._nb_cb_i_data = self._cb_feats_sparse.data.astype(np.float32)
            self._nb_cb_i_indices = self._cb_feats_sparse.indices.astype(np.int32)
            self._nb_cb_i_indptr = self._cb_feats_sparse.indptr.astype(np.int32)
        else:
            self._nb_base_pop = _E1F
            self._nb_soft_pop = _E1F
            self._nb_cb_i_data = _E1F
            self._nb_cb_i_indices = _E1I
            self._nb_cb_i_indptr = _E1I

        if self._cb_profiles_sparse is not None:
            self._nb_cb_u_data = self._cb_profiles_sparse.data.astype(np.float32)
            self._nb_cb_u_indices = self._cb_profiles_sparse.indices.astype(np.int32)
            self._nb_cb_u_indptr = self._cb_profiles_sparse.indptr.astype(np.int32)
        else:
            self._nb_cb_u_data = _E1F
            self._nb_cb_u_indices = _E1I
            self._nb_cb_u_indptr = _E1I

    def _blend(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        f_cf = self._pool.submit(self.cf_model.predict_scores, user_id, item_ids)
        f_cb = self._pool.submit(self.cb_model.predict_scores, user_id, item_ids)
        cf = np.asarray(f_cf.result(), dtype=np.float32)
        cb = np.asarray(f_cb.result(), dtype=np.float32)
        cf *= self.alpha
        cb *= self._beta
        cf += cb
        return cf

    def _filter_candidates(self, watched: set, valid_items: Optional[List[int]] = None) -> Optional[np.ndarray]:
        base_pool = self._movie_ids if valid_items is None else np.asarray(valid_items, dtype=np.int64)
        if base_pool is None:
            return None
        if not watched:
            return base_pool
        return base_pool[~np.isin(base_pool, np.fromiter(watched, dtype=np.int64, count=len(watched)))]

    @staticmethod
    def _select_topk(scores: np.ndarray, ids: np.ndarray, k: int) -> List[int]:
        n = len(scores)
        if n > k:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            idx = np.argsort(scores)[::-1]
        return ids[idx].tolist()

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "HybridRecommender":
        t0, c0 = time.perf_counter(), time.process_time()
        f_cf = self._pool.submit(self.cf_model.fit, ratings_df)
        f_cb = self._pool.submit(self.cb_model.fit, movies_df, ratings_df)
        f_cf.result()
        f_cb.result()
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self._sync_movie_ids()
        self._build_cf_lookups()
        self._build_cb_lookups()
        self._sync_numba_pointers()
        if self._cache is not None:
            self._cache.clear()
        self.step_logger.log_step(
            "Hybrid fit complete", t0, c0,
            {"movies_shape": str(movies_df.shape), "ratings_shape": str(ratings_df.shape)},
        )
        return self

    def fitted(
        self,
        cb_model: RecommenderProtocol,
        cf_model: RecommenderProtocol,
        movies_df: Optional[pd.DataFrame] = None,
        ratings_df: Optional[pd.DataFrame] = None,
    ) -> "HybridRecommender":
        self.cf_model = cf_model
        self.cb_model = cb_model
        if movies_df is not None:
            self.movies_df = movies_df
        if ratings_df is not None:
            self.ratings_df = ratings_df
        self._sync_movie_ids()
        self._build_cf_lookups()
        self._build_cb_lookups()
        self._sync_numba_pointers()
        if self._cache is not None:
            self._cache.clear()
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not item_ids:
            return []
        if self.enable_cache:
            key = (user_id, tuple(item_ids))
            if key in self._cache:
                return self._cache[key]

        if len(self._numba_movie_ids) == 0:
            out = self._blend(user_id, item_ids).tolist()
        else:
            u_in_cf = self._cf_user_lookup[user_id] if (self._cf_user_lookup is not None and 0 <= user_id <= self._cf_max_user_id) else -1
            u_in_cb = self._cb_user_lookup[user_id] if (self._cb_user_lookup is not None and 0 <= user_id <= self._cb_max_user_id) else -1
            arr = np.asarray(item_ids, dtype=np.int64)
            cfl = self._cf_item_lookup if self._cf_item_lookup is not None else _E1I
            cbl = self._cb_movie_lookup if self._cb_movie_lookup is not None else _E1I
            res = _numba_predict_scores(
                arr, cfl, self._cf_max_item_id, cbl, self._cb_max_movie_id,
                np.int32(u_in_cf), self._nb_pu, self._nb_qi, self._nb_bu, self._nb_bi, self._nb_cf_mu,
                np.int32(u_in_cb), self._nb_cb_u_data, self._nb_cb_u_indices, self._nb_cb_u_indptr,
                self._nb_cb_i_data, self._nb_cb_i_indices, self._nb_cb_i_indptr,
                self._nb_base_pop, self._nb_soft_pop, self.alpha, self._beta
            )
            out = res.tolist()

        if self.enable_cache:
            self._cache[key] = out
        return out

    def get_top_k_recommendations(
        self, user_id: int, watched_items: set, k: int = 10, valid_items: Optional[List[int]] = None
    ) -> List[int]:
        t0, c0 = time.perf_counter(), time.process_time()

        if valid_items is not None or len(self._numba_movie_ids) == 0:
            candidates = self._filter_candidates(watched_items, valid_items)
            if candidates is None or len(candidates) == 0:
                if valid_items is None and candidates is None:
                    return self.cf_model.get_top_k_recommendations(
                        user_id=user_id, watched_items=watched_items, k=k
                    )
                return []
            scores = self._blend(user_id, candidates.tolist())
            result = self._select_topk(scores, candidates, k)
            if self.config.log_per_prediction:
                self.step_logger.log_step("get_top_k_recommendations", t0, c0, {
                    "candidates": len(candidates), "user": user_id, "k": k
                })
            return result

        u_in_cf = self._cf_user_lookup[user_id] if (self._cf_user_lookup is not None and 0 <= user_id <= self._cf_max_user_id) else -1
        u_in_cb = self._cb_user_lookup[user_id] if (self._cb_user_lookup is not None and 0 <= user_id <= self._cb_max_user_id) else -1
        watched_arr = np.sort(np.fromiter(watched_items, dtype=np.int64))

        res = _numba_get_top_k(
            self._numba_movie_ids, self._numba_cf_inner, self._numba_cb_inner, watched_arr, k,
            np.int32(u_in_cf), self._nb_pu, self._nb_qi, self._nb_bu, self._nb_bi, self._nb_cf_mu,
            np.int32(u_in_cb), self._nb_cb_u_data, self._nb_cb_u_indices, self._nb_cb_u_indptr,
            self._nb_cb_i_data, self._nb_cb_i_indices, self._nb_cb_i_indptr,
            self._nb_base_pop, self._nb_soft_pop, self.alpha, self._beta
        )
        out = res.tolist()
        if self.config.log_per_prediction:
            self.step_logger.log_step("get_top_k_recommendations", t0, c0, {
                "candidates": len(self._numba_movie_ids) - len(watched_arr), "user": user_id, "k": k
            })
        return out

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int = 10,
        valid_items: Optional[List[List[int]]] = None,
    ) -> List[List[int]]:
        if not user_ids:
            return []

        if valid_items is not None:
            n_jobs = 1 if self.config.debug else -1
            return Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self.get_top_k_recommendations)(u, w, k, v)
                for u, w, v in zip(user_ids, watched_items_list, valid_items)
            )

        if len(self._numba_movie_ids) == 0:
            n_jobs = 1 if self.config.debug else -1
            return Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self.get_top_k_recommendations)(u, w, k)
                for u, w in zip(user_ids, watched_items_list)
            )

        u_cf_arr = np.array([
            self._cf_user_lookup[u] if (self._cf_user_lookup is not None and 0 <= u <= self._cf_max_user_id) else -1
            for u in user_ids
        ], dtype=np.int32)
        u_cb_arr = np.array([
            self._cb_user_lookup[u] if (self._cb_user_lookup is not None and 0 <= u <= self._cb_max_user_id) else -1
            for u in user_ids
        ], dtype=np.int32)

        def _worker(i):
            watched_arr = np.sort(np.fromiter(watched_items_list[i], dtype=np.int64))
            return _numba_get_top_k(
                self._numba_movie_ids, self._numba_cf_inner, self._numba_cb_inner, watched_arr, k,
                u_cf_arr[i], self._nb_pu, self._nb_qi, self._nb_bu, self._nb_bi, self._nb_cf_mu,
                u_cb_arr[i], self._nb_cb_u_data, self._nb_cb_u_indices, self._nb_cb_u_indptr,
                self._nb_cb_i_data, self._nb_cb_i_indices, self._nb_cb_i_indptr,
                self._nb_base_pop, self._nb_soft_pop, self.alpha, self._beta
            ).tolist()

        n_jobs = 1 if self.config.debug else -1
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_worker)(i) for i in range(len(user_ids))
        )

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        f_cb = self._pool.submit(self.cb_model.explain_recommendation, movie_id, liked_items, top_n_reasons)
        f_cf = self._pool.submit(self.cf_model.explain_recommendation, movie_id, liked_items, top_n_reasons)
        merged: Dict[int, List[float]] = {}
        for r in f_cb.result() + f_cf.result():
            merged.setdefault(r["movie_id"], []).append(r["similarity"])
        return sorted(
            [{"movie_id": mid, "similarity": float(np.mean(s))} for mid, s in merged.items()],
            key=lambda x: x["similarity"],
            reverse=True,
        )[:top_n_reasons]