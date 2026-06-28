import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger import LoggingConfig, StepLogger
from src.models.models import RecommenderProtocol

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

_F32_MIN = np.float32(0.5)
_F32_MAX = np.float32(5.0)
_FILL = np.float32(-1e9)

_CF_ATTRS = (
    "_pu", "_qi", "_bu", "_bi", "_global_mean",
    "_raw_to_inner_user", "_raw_to_inner_item",
)
_CB_ATTRS = (
    "user_profiles", "feature_matrix_norm", "soft_pop_arr",
    "movie_index", "user_index", "base_pop_norm_arr",
)


def _cf_ready(m: Any) -> bool:
    return all(hasattr(m, a) for a in _CF_ATTRS)


def _cb_ready(m: Any) -> bool:
    return all(hasattr(m, a) for a in _CB_ATTRS)


def _cf_score_matrix(cf: Any, user_ids: List[int], item_ids: np.ndarray) -> np.ndarray:
    nu, ni = len(user_ids), len(item_ids)

    u_in = np.fromiter((cf._raw_to_inner_user.get(int(x), -1) for x in user_ids), np.int32, nu)
    i_in = np.fromiter((cf._raw_to_inner_item.get(int(x), -1) for x in item_ids), np.int32, ni)

    vu = np.where(u_in >= 0)[0]
    vi = np.where(i_in >= 0)[0]
    vu_in, vi_in = u_in[vu], i_in[vi]

    out = np.full((nu, ni), float(cf._global_mean), np.float32)
    if not vu.size or not vi.size:
        return out

    use_gpu = HAS_GPU and getattr(cf, "use_gpu", False) and getattr(cf, "_pu_gpu", None) is not None

    if use_gpu:
        bu = cf._bu_gpu[vu_in]
        bi = cf._bi_gpu[vi_in]
        sub = (
            cf._global_mean
            + bu[:, None]
            + bi[None, :]
            + cf._pu_gpu[vu_in] @ cf._qi_gpu[vi_in].T
        )
        sub = cp.clip(sub, _F32_MIN, _F32_MAX).get().astype(np.float32)
        cp.get_default_memory_pool().free_all_blocks()
    else:
        bu, bi = cf._bu[vu_in], cf._bi[vi_in]
        sub = np.clip(
            cf._global_mean + bu[:, None] + bi[None, :] + cf._pu[vu_in] @ cf._qi[vi_in].T,
            _F32_MIN, _F32_MAX,
        ).astype(np.float32)

    out[np.ix_(vu, vi)] = sub
    return out


def _cb_score_matrix(cb: Any, user_ids: List[int], item_ids: np.ndarray) -> np.ndarray:
    nu, ni = len(user_ids), len(item_ids)

    ii = cb.movie_index.get_indexer(item_ids)
    vi = np.where(ii >= 0)[0]
    vi_in = ii[vi]

    out = np.zeros((nu, ni), np.float32)
    if not vi.size:
        return out

    ul = cb.user_index.get_indexer(np.asarray(user_ids, np.int64))
    gu = np.where(ul >= 0)[0]
    bu = np.where(ul < 0)[0]

    base = cb.base_pop_norm_arr[vi_in]
    if bu.size:
        out[np.ix_(bu, vi)] = base[None, :]

    if gu.size and cb.user_profiles is not None:
        profiles = cb.user_profiles[ul[gu]]
        feats = cb.feature_matrix_norm[vi_in]
        dots = (profiles @ feats.T).toarray().astype(np.float32)
        out[np.ix_(gu, vi)] = dots * cb.soft_pop_arr[vi_in][None, :]

    return out


def _apply_topk(
    scores: np.ndarray,
    item_ids: np.ndarray,
    watched_list: List[set],
    k: int,
) -> List[List[int]]:
    nu, ni = scores.shape
    k = min(k, ni)

    id_to_col = {int(x): c for c, x in enumerate(item_ids)}
    rows, cols = [], []
    for i, ws in enumerate(watched_list):
        for w in ws:
            c = id_to_col.get(int(w))
            if c is not None:
                rows.append(i)
                cols.append(c)
    if rows:
        scores[rows, cols] = _FILL

    fallback = np.arange(ni)
    top = np.argpartition(scores, -k, axis=1)[:, -k:] if ni > k else None
    threshold = _FILL * 0.5

    results = []
    for i in range(nu):
        idx = top[i] if top is not None else fallback
        order = idx[np.argsort(scores[i, idx])[::-1]]
        results.append(item_ids[order[scores[i, order] > threshold]].tolist())

    return results


class HybridRecommender:
    def __init__(
        self,
        cf_model: RecommenderProtocol,
        cb_model: RecommenderProtocol,
        alpha: float = 0.5,
        logging_config: LoggingConfig = LoggingConfig(),
        use_gpu: bool = True,
        enable_cache: bool = False,
    ):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = np.float32(alpha)
        self._beta = np.float32(1.0 - alpha)
        self.ratings_df = None
        self.movies_df = None
        self.config = logging_config
        self.step_logger = StepLogger(self.config)
        self.enable_cache = enable_cache
        self._movie_ids: Optional[np.ndarray] = None
        self._cache: Optional[Dict] = {} if enable_cache else None
        self._pool = ThreadPoolExecutor(max_workers=2)

    def _sync_movie_ids(self) -> None:
        if hasattr(self.cb_model, "movie_ids") and len(self.cb_model.movie_ids) > 0:
            self._movie_ids = np.asarray(self.cb_model.movie_ids, dtype=np.int64)
        elif hasattr(self.cb_model, "movie_id_to_idx"):
            self._movie_ids = np.fromiter(self.cb_model.movie_id_to_idx, dtype=np.int64)
        elif self.movies_df is not None:
            self._movie_ids = self.movies_df["movieId"].unique().astype(np.int64)
        else:
            self._movie_ids = None

    def _blend(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        f_cf = self._pool.submit(self.cf_model.predict_scores, user_id, item_ids)
        f_cb = self._pool.submit(self.cb_model.predict_scores, user_id, item_ids)
        cf = np.asarray(f_cf.result(), dtype=np.float32)
        cb = np.asarray(f_cb.result(), dtype=np.float32)
        cf *= self.alpha
        cb *= self._beta
        cf += cb
        return cf

    def _filter_candidates(self, watched: set) -> Optional[np.ndarray]:
        if self._movie_ids is None:
            return None
        if not watched:
            return self._movie_ids
        w = np.fromiter(watched, dtype=np.int64, count=len(watched))
        return self._movie_ids[~np.isin(self._movie_ids, w, assume_unique=True)]

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
        if self._cache is not None:
            self._cache.clear()
        self.step_logger.log_step("Hybrid fit complete", t0, c0, {
            "movies_shape": str(movies_df.shape),
            "ratings_shape": str(ratings_df.shape),
        })
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
        if self._cache is not None:
            self._cache.clear()
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.enable_cache:
            return self._blend(user_id, item_ids).tolist()
        key = (user_id, tuple(item_ids))
        if key not in self._cache:
            self._cache[key] = self._blend(user_id, item_ids).tolist()
        return self._cache[key]

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        t0, c0 = time.perf_counter(), time.process_time()
        candidates = self._filter_candidates(watched_items)
        if candidates is None:
            return self.cf_model.get_top_k_recommendations(user_id, watched_items, k)
        if len(candidates) == 0:
            return []
        scores = self._blend(user_id, candidates.tolist())
        result = self._select_topk(scores, candidates, k)
        if self.config.log_per_prediction:
            self.step_logger.log_step("get_top_k_recommendations", t0, c0, {
                "candidates": len(candidates),
                "user": user_id,
                "k": k,
            })
        return result

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int = 10,
    ) -> List[List[int]]:
        if not user_ids:
            return []

        all_ids = self._movie_ids
        if all_ids is None or not all_ids.size:
            return [
                self.get_top_k_recommendations(u, w, k)
                for u, w in zip(user_ids, watched_items_list)
            ]

        if _cf_ready(self.cf_model) and _cb_ready(self.cb_model):
            f_cf = self._pool.submit(_cf_score_matrix, self.cf_model, user_ids, all_ids)
            f_cb = self._pool.submit(_cb_score_matrix, self.cb_model, user_ids, all_ids)
            cf_mat = f_cf.result()
            cb_mat = f_cb.result()
            scores = self.alpha * cf_mat + self._beta * cb_mat
            del cf_mat, cb_mat
        else:
            iids = all_ids.tolist()
            scores = np.stack([
                np.asarray(self._blend(u, iids), np.float32) for u in user_ids
            ])

        return _apply_topk(scores, all_ids, watched_items_list, k)

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