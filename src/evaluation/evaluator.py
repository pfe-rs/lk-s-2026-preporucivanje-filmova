import os
import time
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
import numba as nb


@nb.njit(parallel=True, cache=True, fastmath=True)
def _hits_parallel_bsearch(
    recs_idx: np.ndarray,
    rel_flat: np.ndarray,
    rel_offsets: np.ndarray,
    sentinel: np.int64,
) -> np.ndarray:
    n_users, max_k = recs_idx.shape
    hits = np.zeros((n_users, max_k), dtype=nb.bool_)
    for i in nb.prange(n_users):
        lo_off = rel_offsets[i]
        n_rel = rel_offsets[i + 1] - lo_off
        if n_rel == 0:
            continue
        for j in range(max_k):
            target = recs_idx[i, j]
            if target == sentinel:
                continue
            lo = np.int64(0)
            hi = np.int64(n_rel - 1)
            while lo <= hi:
                mid = (lo + hi) >> np.int64(1)
                v = rel_flat[lo_off + mid]
                if v == target:
                    hits[i, j] = True
                    break
                elif v < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
    return hits


class RecommendationEvaluator:
    def __init__(
        self,
        models: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        relevance_threshold: float = 4.0,
        user_sample_size: Optional[int] = None,
        random_state: int = 42,
        use_gpu: bool = True,
        item_universe: Optional[List[int]] = None,
        n_negative_samples: Optional[int] = None,
    ):
        self.models = models
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        self.n_negative_samples = n_negative_samples

        all_test_users = test_df["userId"].unique()
        if user_sample_size is not None and user_sample_size < len(all_test_users):
            np.random.seed(random_state)
            self.test_users = np.random.choice(
                all_test_users, size=user_sample_size, replace=False
            )
        else:
            self.test_users = all_test_users

        self.user_to_idx = {u: i for i, u in enumerate(self.test_users)}
        self.n_test_users = len(self.test_users)

        if item_universe is None:
            self.item_catalog = sorted(
                set(train_df["movieId"].unique()) | set(test_df["movieId"].unique())
            )
        else:
            self.item_catalog = sorted(set(item_universe))

        self.item_index = pd.Index(self.item_catalog)
        self.item_set = set(self.item_catalog)
        self.item_catalog_list = list(self.item_catalog)
        self.item_catalog_arr = np.asarray(self.item_catalog_list, dtype=np.int64)
        self.n_items = len(self.item_index)

        user_means = train_df.groupby("userId")["rating"].mean().to_dict()
        global_train_mean = (
            float(train_df["rating"].mean()) if len(train_df) else relevance_threshold
        )

        test_df_copy = test_df[["userId", "movieId", "rating"]].copy()
        test_df_copy["user_mean"] = (
            test_df_copy["userId"].map(user_means).fillna(global_train_mean)
        )

        relevance_mask = (
            (test_df_copy["rating"] >= relevance_threshold)
            & test_df_copy["userId"].isin(self.user_to_idx)
            & test_df_copy["movieId"].isin(self.item_set)
        )
        test_relevant = test_df_copy.loc[relevance_mask]

        self.rel_dict = (
            test_relevant.groupby("userId")["movieId"].apply(list).to_dict()
        )
        self.train_watched_items = (
            train_df.groupby("userId")["movieId"].apply(set).to_dict()
        )

        if len(test_relevant):
            user_pos_all = (
                test_relevant["userId"].map(self.user_to_idx).to_numpy(dtype=np.int64)
            )
            item_pos_all = self.item_index.get_indexer(
                test_relevant["movieId"].to_numpy()
            )
            order = np.argsort(user_pos_all, kind="stable")
            user_pos_sorted = user_pos_all[order]
            item_pos_sorted = item_pos_all[order]
        else:
            user_pos_sorted = np.array([], dtype=np.int64)
            item_pos_sorted = np.array([], dtype=np.int64)

        self.counts_arr = np.bincount(
            user_pos_sorted, minlength=self.n_test_users
        ).astype(np.float64)
        self.valid_user_mask = self.counts_arr > 0

        offsets = np.zeros(self.n_test_users + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(self.counts_arr).astype(np.int64)

        if len(item_pos_sorted) > 0:
            lex_order = np.lexsort((item_pos_sorted, user_pos_sorted))
            item_pos_lexsorted = item_pos_sorted[lex_order]
        else:
            item_pos_lexsorted = np.array([], dtype=np.int64)

        self._rel_flat = item_pos_lexsorted.astype(np.int64)
        self._rel_offsets = offsets.astype(np.int64)

        self.rel_indices_sorted = [
            self._rel_flat[offsets[i]:offsets[i + 1]]
            for i in range(self.n_test_users)
        ]

        self.user_negative_pools: Dict[int, list] = {}
        if self.n_negative_samples is not None:
            rng = np.random.default_rng(self.random_state)
            all_indices = np.arange(self.n_items, dtype=np.int64)
            id_to_idx_mapping = pd.Series(
                np.arange(self.n_items), index=self.item_index
            ).to_dict()
            for i, u in enumerate(self.test_users):
                watched = self.train_watched_items.get(u)
                if watched:
                    w_arr = np.fromiter(watched, dtype=np.int64, count=len(watched))
                    w_idx = np.unique(
                        [id_to_idx_mapping.get(mid, -1) for mid in w_arr]
                    ).astype(np.int64)
                    w_idx = w_idx[w_idx >= 0]
                else:
                    w_idx = np.empty(0, dtype=np.int64)

                rel_idx = self.rel_indices_sorted[i]
                exclude = np.union1d(w_idx, rel_idx)
                available_idx = np.setdiff1d(all_indices, exclude, assume_unique=True)

                n_avail = len(available_idx)
                n_samp = min(self.n_negative_samples, n_avail)
                chosen = rng.choice(n_avail, size=n_samp, replace=False)
                self.user_negative_pools[u] = self.item_catalog_arr[
                    available_idx[chosen]
                ].tolist()

        item_popularity = train_df["movieId"].value_counts()
        total_interactions = int(item_popularity.sum())
        default_novelty = (
            -np.log2(1.0 / total_interactions) if total_interactions > 0 else 0.0
        )

        if total_interactions > 0:
            pop_counts = item_popularity.reindex(
                self.item_catalog, fill_value=0
            ).to_numpy(dtype=np.float64)
            safe_counts = np.where(pop_counts > 0, pop_counts, 1.0)
            novelty_array = np.where(
                pop_counts > 0,
                -np.log2(safe_counts / total_interactions),
                default_novelty,
            )
        else:
            novelty_array = np.full(self.n_items, default_novelty, dtype=np.float64)

        self.novelty_array = np.append(novelty_array, default_novelty)
        self.results = None

        _warmup_recs = np.zeros((1, 1), dtype=np.int64)
        _warmup_flat = np.array([0], dtype=np.int64)
        _warmup_off = np.array([0, 0], dtype=np.int64)
        _hits_parallel_bsearch(_warmup_recs, _warmup_flat, _warmup_off, np.int64(0))

    def _get_valid_items_for_user(self, u: int) -> List[int]:
        if self.n_negative_samples is None:
            return self.item_catalog_list
        return self.rel_dict.get(u, []) + self.user_negative_pools.get(u, [])

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048,
    ) -> List[Dict[str, Any]]:
        logger.info("Evaluating '{}'", model_name)
        start_time = time.time()
        n_users = self.n_test_users

        valid_k_values = sorted(set(k for k in k_values if k <= max_recommendations))
        if len(valid_k_values) < len(k_values):
            dropped = sorted(set(k_values) - set(valid_k_values))
            logger.warning(
                "k values exceeding max_recommendations dropped for {}: {}",
                model_name,
                dropped,
            )
        if not valid_k_values:
            raise ValueError("No valid k values <= max_recommendations were provided.")

        max_k = min(max(valid_k_values), max_recommendations)

        recs_arr = np.full((n_users, max_recommendations), -1, dtype=np.int64)

        has_batch = hasattr(model, "get_top_k_recommendations_batch")
        sig = inspect.signature(
            model.get_top_k_recommendations_batch
            if has_batch
            else model.get_top_k_recommendations
        )
        supports_valid_items = "valid_items" in sig.parameters

        batch_failed = False

        if has_batch:
            for start in tqdm(
                range(0, n_users, batch_size), desc=f"{model_name}", leave=False
            ):
                end = min(start + batch_size, n_users)
                user_chunk = self.test_users[start:end]
                watched_list = [
                    self.train_watched_items.get(u, set()) for u in user_chunk
                ]
                try:
                    if supports_valid_items:
                        if self.n_negative_samples is not None:
                            valid_items_lists = [
                                self._get_valid_items_for_user(u) for u in user_chunk
                            ]
                        else:
                            valid_items_lists = [self.item_catalog_list for _ in user_chunk]
                        fetch_k = max(len(v) for v in valid_items_lists) if valid_items_lists else self.n_items
                    else:
                        valid_items_lists = None
                        fetch_k = self.n_items

                    kwargs: Dict[str, Any] = {
                        "user_ids": user_chunk,
                        "watched_items_list": watched_list,
                        "k": fetch_k,
                    }
                    if supports_valid_items and valid_items_lists is not None:
                        kwargs["valid_items"] = valid_items_lists

                    batch_recs = model.get_top_k_recommendations_batch(**kwargs)

                    item_set_local = self.item_set
                    for i, u in enumerate(user_chunk):
                        watched = self.train_watched_items.get(u, set())
                        current_recs = np.asarray(batch_recs[i]).tolist()
                        row = [
                            mid
                            for mid in current_recs
                            if mid in item_set_local and mid not in watched
                        ][:max_recommendations]
                        recs_arr[start + i, : len(row)] = row

                except Exception as e:
                    logger.warning(
                        "Batch error for {}: {}. Falling back to single mode.",
                        model_name,
                        e,
                    )
                    batch_failed = True
                    break

        if not has_batch or batch_failed:
            recs_arr = np.full((n_users, max_recommendations), -1, dtype=np.int64)
            item_set_local = self.item_set

            def fetch_for_user(u):
                try:
                    if supports_valid_items:
                        valid_items = self._get_valid_items_for_user(u)
                        fetch_k = len(valid_items) if valid_items else self.n_items
                    else:
                        valid_items = None
                        fetch_k = self.n_items

                    kwargs = {
                        "user_id": int(u),
                        "watched_items": self.train_watched_items.get(u, set()),
                        "k": fetch_k,
                    }
                    if supports_valid_items and valid_items is not None:
                        kwargs["valid_items"] = valid_items
                    recs = model.get_top_k_recommendations(**kwargs)
                    watched = self.train_watched_items.get(u, set())
                    return [
                        mid
                        for mid in recs
                        if mid in item_set_local and mid not in watched
                    ][:max_recommendations]
                except Exception:
                    return []

            with ThreadPoolExecutor(
                max_workers=min(32, os.cpu_count() or 4)
            ) as executor:
                results_list = list(
                    tqdm(
                        executor.map(fetch_for_user, self.test_users),
                        total=n_users,
                        desc=f"{model_name}",
                        leave=False,
                    )
                )
                for idx, row in enumerate(results_list):
                    recs_arr[idx, : len(row)] = row

        recs_idx = self.item_index.get_indexer(recs_arr.ravel()).reshape(recs_arr.shape)
        recs_idx_k = recs_idx[:, :max_k]
        recs_idx_safe = np.where(recs_idx_k < 0, self.n_items, recs_idx_k).astype(
            np.int64
        )

        hits = _hits_parallel_bsearch(
            recs_idx_safe,
            self._rel_flat,
            self._rel_offsets,
            np.int64(self.n_items),
        )

        novelty_vals = np.where(
            recs_idx_k != -1, self.novelty_array[recs_idx_safe], 0.0
        )

        discounts = 1.0 / np.log2(np.arange(2, max_k + 2, dtype=np.float64))
        cum_discounts = np.zeros(max_k + 1, dtype=np.float64)
        cum_discounts[1:] = np.cumsum(discounts)

        cumsum_hits = np.cumsum(hits, axis=1).astype(np.float64)
        positions = np.arange(1, max_k + 1, dtype=np.float64)
        ap_terms = (cumsum_hits / positions) * hits
        cumsum_ap = np.cumsum(ap_terms, axis=1)
        cumsum_dcg = np.cumsum(hits * discounts, axis=1)

        any_hit = hits.any(axis=1)
        first_hit_idx = np.where(any_hit, np.argmax(hits, axis=1), -1).astype(np.int32)

        min_counts_k = np.minimum(self.counts_arr[:, None], positions[None, :])
        ideal_dcg = np.where(
            min_counts_k > 0, cum_discounts[min_counts_k.astype(int)], 0.0
        )

        valid_mask = self.valid_user_mask
        n_valid_users = int(valid_mask.sum())

        results = []
        for k in valid_k_values:
            k_idx = min(k, max_k) - 1

            recs_k = recs_arr[:, :k]
            valid_recs_k = recs_k[recs_k != -1]
            coverage_k = (
                (len(np.unique(valid_recs_k)) / self.n_items) if self.n_items else 0.0
            )

            prec = cumsum_hits[valid_mask, k_idx] / k
            rec = np.divide(
                cumsum_hits[valid_mask, k_idx],
                self.counts_arr[valid_mask],
                out=np.zeros(n_valid_users),
                where=self.counts_arr[valid_mask] > 0,
            )
            dcg = cumsum_dcg[valid_mask, k_idx]
            ndcg = np.divide(
                dcg,
                ideal_dcg[valid_mask, k_idx],
                out=np.zeros_like(dcg),
                where=ideal_dcg[valid_mask, k_idx] > 0,
            )
            ap = np.divide(
                cumsum_ap[valid_mask, k_idx],
                np.minimum(self.counts_arr[valid_mask], k),
                out=np.zeros(n_valid_users),
                where=self.counts_arr[valid_mask] > 0,
            )
            mrr = np.where(
                (first_hit_idx[valid_mask] >= 0) & (first_hit_idx[valid_mask] < k),
                1.0 / (first_hit_idx[valid_mask] + 1),
                0.0,
            )
            nov = novelty_vals[valid_mask, :k].mean(axis=1)

            results.append(
                {
                    "model": model_name,
                    "k": k,
                    "precision": float(prec.mean()) if n_valid_users > 0 else 0.0,
                    "recall": float(rec.mean()) if n_valid_users > 0 else 0.0,
                    "ndcg": float(ndcg.mean()) if n_valid_users > 0 else 0.0,
                    "map": float(ap.mean()) if n_valid_users > 0 else 0.0,
                    "mrr": float(mrr.mean()) if n_valid_users > 0 else 0.0,
                    "novelty": float(nov.mean()) if n_valid_users > 0 else 0.0,
                    "coverage": coverage_k,
                    "n_users": int(
                        any_hit[valid_mask].sum()
                        if k >= max_k
                        else np.any(hits[valid_mask, :k], axis=1).sum()
                    ),
                }
            )

        logger.info("'{}' done in {:.1f}s", model_name, time.time() - start_time)
        return results

    def evaluate_all_models(
        self,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048,
    ) -> pd.DataFrame:
        all_results = []
        for model_name, model in self.models.items():
            try:
                all_results.extend(
                    self.evaluate_model(
                        model_name, model, k_values, max_recommendations, batch_size
                    )
                )
            except Exception as e:
                logger.error("Failed {}: {}", model_name, e)
                continue
        self.results = pd.DataFrame(all_results)
        return self.results

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("No results available. Run evaluate_all_models() first.")
        return self.results