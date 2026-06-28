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
    "_raw_to_inner_user", "_raw_to_inner_item", "_inner_to_raw_item",
    "_valid_item_mask", "_item_popularity", "_popular_movies",
)
_CB_ATTRS = (
    "user_profiles", "feature_matrix_norm", "soft_pop_arr",
    "movie_index", "user_index", "base_pop_norm_arr",
)


def _cf_ready(m: Any) -> bool:
    return all(hasattr(m, a) for a in _CF_ATTRS)


def _cb_ready(m: Any) -> bool:
    return all(hasattr(m, a) for a in _CB_ATTRS)


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


class CascadingHybridRecommender:
    def __init__(
        self,
        primary_model: RecommenderProtocol,
        secondary_model: RecommenderProtocol,
        primary_k: int = 50,
        logging_config: LoggingConfig = LoggingConfig(),
        use_gpu: bool = True,
        enable_cache: bool = False,
    ):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.primary_k = primary_k
        self.config = logging_config
        self.step_logger = StepLogger(self.config)
        self.enable_cache = enable_cache
        self._cache: Optional[Dict] = {} if enable_cache else None
        self._pool = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def _rerank(scores: np.ndarray, candidates: List[int], k: int) -> List[int]:
        n = len(scores)
        if n > k:
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
        else:
            idx = np.argsort(scores)[::-1]
        return [candidates[i] for i in idx]

    def fitted(
        self,
        primary_model: RecommenderProtocol,
        secondary_model: RecommenderProtocol,
    ) -> "CascadingHybridRecommender":
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        if self._cache is not None:
            self._cache.clear()
        return self

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame) -> "CascadingHybridRecommender":
        t0, c0 = time.perf_counter(), time.process_time()
        f_p = self._pool.submit(self.primary_model.fit, ratings_df)
        f_s = self._pool.submit(self.secondary_model.fit, movies_df, ratings_df)
        f_p.result()
        f_s.result()
        if self._cache is not None:
            self._cache.clear()
        self.step_logger.log_step("Cascading fit complete", t0, c0, {
            "movies_shape": str(movies_df.shape),
            "ratings_shape": str(ratings_df.shape),
        })
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.enable_cache:
            return self.secondary_model.predict_scores(user_id, item_ids)
        key = (user_id, tuple(item_ids))
        if key not in self._cache:
            self._cache[key] = self.secondary_model.predict_scores(user_id, item_ids)
        return self._cache[key]

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        t0, c0 = time.perf_counter(), time.process_time()
        broad = self.primary_model.get_top_k_recommendations(
            user_id=user_id, watched_items=watched_items, k=self.primary_k
        )
        if not broad:
            return []
        scores = np.asarray(self.secondary_model.predict_scores(user_id, broad), dtype=np.float32)
        result = self._rerank(scores, broad, k)
        if self.config.log_per_prediction:
            self.step_logger.log_step("get_top_k_recommendations", t0, c0, {
                "broad_candidates": len(broad),
                "final_candidates": len(result),
                "user": user_id,
                "k": k,
            })
        return result

    def _batch_primary_topk(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
    ) -> List[List[int]]:
        cf = self.primary_model
        n = len(user_ids)

        u_in = np.fromiter(
            (cf._raw_to_inner_user.get(int(u), -1) for u in user_ids), np.int32, n
        )
        results: List[Optional[List[int]]] = [None] * n

        for i in np.where(u_in == -1)[0]:
            ws = watched_items_list[i]
            results[i] = [m for m in cf._popular_movies if m not in ws][: self.primary_k]

        known = np.where(u_in >= 0)[0]
        if not known.size:
            return results

        ku = u_in[known]
        gpu = HAS_GPU and getattr(cf, "use_gpu", False) and getattr(cf, "_pu_gpu", None) is not None

        if gpu:
            bu = cf._bu_gpu[ku]
            raw = cp.clip(
                cf._global_mean + bu[:, None] + cf._bi_gpu[None, :] + cf._pu_gpu[ku] @ cf._qi_gpu.T,
                _F32_MIN,
                _F32_MAX,
            )
            ip_gpu = getattr(cf, "_item_popularity_gpu", None)
            if ip_gpu is None:
                ip_gpu = cp.asarray(cf._item_popularity, dtype=cp.float32)
            pen = (raw / ip_gpu[None, :] ** cf.alpha).get().astype(np.float32)
            cp.get_default_memory_pool().free_all_blocks()
        else:
            bu = cf._bu[ku]
            raw = np.clip(
                cf._global_mean + bu[:, None] + cf._bi[None, :] + cf._pu[ku] @ cf._qi.T,
                _F32_MIN,
                _F32_MAX,
            ).astype(np.float32)
            pen = raw / (cf._item_popularity[None, :] ** cf.alpha)

        n_items = pen.shape[1]
        valid_mat = np.tile(cf._valid_item_mask, (len(known), 1))

        for j, i in enumerate(known):
            wi = [cf._raw_to_inner_item[m] for m in watched_items_list[i] if m in cf._raw_to_inner_item]
            if wi:
                valid_mat[j, wi] = False

        pen[~valid_mat] = _FILL

        pk = self.primary_k
        tk = min(pk, n_items)
        top = np.argpartition(pen, -tk, axis=1)[:, -tk:] if n_items > tk else np.argsort(pen, axis=1)[:, ::-1]
        threshold = _FILL * 0.5

        for j, i in enumerate(known):
            idx = top[j]
            order = idx[np.argsort(pen[j, idx])[::-1]]
            valid = pen[j, order] > threshold
            results[i] = cf._inner_to_raw_item[order[valid][:pk]].tolist()

        return results

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int = 10,
    ) -> List[List[int]]:
        if not user_ids:
            return []

        fast_p = _cf_ready(self.primary_model)
        fast_s = _cb_ready(self.secondary_model)

        if fast_p:
            broad = self._batch_primary_topk(user_ids, watched_items_list)
        else:
            broad = [
                self.primary_model.get_top_k_recommendations(u, w, self.primary_k)
                for u, w in zip(user_ids, watched_items_list)
            ]

        out: List[List[int]] = [[] for _ in range(len(user_ids))]
        ne = [(i, user_ids[i], broad[i]) for i in range(len(user_ids)) if broad[i]]
        if not ne:
            return out

        ne_i, ne_u, ne_c = zip(*ne)

        if fast_s:
            unique = np.unique(np.concatenate([np.asarray(c, np.int64) for c in ne_c]))
            mat = _cb_score_matrix(self.secondary_model, list(ne_u), unique)
            col = {int(x): c for c, x in enumerate(unique)}

            for j, (orig, cands) in enumerate(zip(ne_i, ne_c)):
                ci = np.fromiter((col[int(x)] for x in cands), np.int32, len(cands))
                out[orig] = self._rerank(mat[j, ci], cands, k)
        else:
            for orig, uid, cands in zip(ne_i, ne_u, ne_c):
                s = np.asarray(self.secondary_model.predict_scores(uid, cands), np.float32)
                out[orig] = self._rerank(s, cands, k)

        return out

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        return self.secondary_model.explain_recommendation(movie_id, liked_items, top_n_reasons)