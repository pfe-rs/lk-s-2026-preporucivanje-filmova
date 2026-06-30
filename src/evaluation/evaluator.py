import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import time
import os
import inspect
from loguru import logger
from tqdm import tqdm

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
        n_negative_samples: Optional[int] = None  # <-- Added for classic RecSys LOO benchmarks
    ):
        self.models = models
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        self.n_negative_samples = n_negative_samples

        all_test_users = test_df['userId'].unique()
        if user_sample_size is not None and user_sample_size < len(all_test_users):
            np.random.seed(random_state)
            self.test_users = np.random.choice(all_test_users, size=user_sample_size, replace=False)
        else:
            self.test_users = all_test_users

        self.user_to_idx = {u: i for i, u in enumerate(self.test_users)}
        self.n_test_users = len(self.test_users)

        if item_universe is None:
            self.item_catalog = sorted(set(train_df['movieId'].unique()) | set(test_df['movieId'].unique()))
        else:
            self.item_catalog = sorted(set(item_universe))
        self.item_index = pd.Index(self.item_catalog)
        self.item_set = set(self.item_catalog)
        self.item_catalog_list = list(self.item_catalog)
        self.n_items = len(self.item_index)

        user_means = train_df.groupby('userId')['rating'].mean().to_dict()
        global_train_mean = float(train_df['rating'].mean()) if len(train_df) else relevance_threshold

        test_df_copy = test_df.copy()
        test_df_copy['user_mean'] = test_df_copy['userId'].map(user_means).fillna(global_train_mean)

        test_relevant = test_df_copy[
            (test_df_copy['rating'] >= relevance_threshold) &
            (test_df_copy['rating'] >= test_df_copy['user_mean'])
        ]

        self.rel_dict = test_relevant.groupby('userId')['movieId'].apply(list).to_dict()
        self.train_watched_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()

        rel_indices = []
        for u in self.test_users:
            mids = self.rel_dict.get(u, [])
            valid_mids = [mid for mid in mids if mid in self.item_set]
            idxs = self.item_index.get_indexer(valid_mids).astype(np.int32)
            idxs.sort()
            rel_indices.append(idxs)

        self.rel_indices_sorted = rel_indices
        self.counts_arr = np.array([len(arr) for arr in rel_indices], dtype=np.float64)
        self.valid_user_mask = self.counts_arr > 0

        # Pre-build negative pools per user if sampling is enabled
        self.user_negative_pools = {}
        if self.n_negative_samples is not None:
            np.random.seed(self.random_state)
            for u in self.test_users:
                watched = self.train_watched_items.get(u, set())
                relevant_test = set(self.rel_dict.get(u, []))
                # Negatives cannot be watched in training, nor be part of the test truth
                forbidden = watched | relevant_test
                available_negatives = np.array(list(self.item_set - forbidden))
                
                if len(available_negatives) >= self.n_negative_samples:
                    sampled = np.random.choice(available_negatives, size=self.n_negative_samples, replace=False).tolist()
                else:
                    sampled = available_negatives.tolist()
                self.user_negative_pools[u] = sampled

        item_popularity = train_df['movieId'].value_counts().to_dict()
        total_interactions = sum(item_popularity.values())
        default_novelty = -np.log2(1.0 / total_interactions) if total_interactions > 0 else 0.0

        novelty_array = np.zeros(self.n_items, dtype=np.float64)
        for pos, mid in enumerate(self.item_catalog):
            cnt = item_popularity.get(mid, 0)
            if cnt > 0:
                novelty_array[pos] = -np.log2(cnt / total_interactions)
            else:
                novelty_array[pos] = default_novelty

        self.novelty_array = np.append(novelty_array, default_novelty)
        self.results = None

    def _get_valid_items_for_user(self, u: int) -> List[int]:
        """Helper to combine positive test targets and sampled negatives."""
        if self.n_negative_samples is None:
            return self.item_catalog_list
        return self.rel_dict.get(u, []) + self.user_negative_pools.get(u, [])

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048
    ) -> List[Dict[str, Any]]:
        logger.info("Evaluating '{}'", model_name)
        start_time = time.time()
        n_users = self.n_test_users

        valid_k_values = sorted(set(k for k in k_values if k <= max_recommendations))
        if len(valid_k_values) < len(k_values):
            dropped = sorted(set(k_values) - set(valid_k_values))
            logger.warning("k values exceeding max_recommendations dropped for {}: {}", model_name, dropped)
        if not valid_k_values:
            raise ValueError("No valid k values <= max_recommendations were provided.")

        max_k = min(max(valid_k_values), max_recommendations)
        over_fetch_k = min(max_recommendations * 4, self.n_items) if self.n_items else max_recommendations

        recs_arr = np.full((n_users, max_recommendations), -1, dtype=np.int64)

        has_batch = hasattr(model, 'get_top_k_recommendations_batch')
        if has_batch:
            sig = inspect.signature(model.get_top_k_recommendations_batch)
        else:
            sig = inspect.signature(model.get_top_k_recommendations)
        supports_valid_items = 'valid_items' in sig.parameters

        batch_failed = False

        if has_batch:
            for start in tqdm(range(0, n_users, batch_size), desc=f"{model_name}", leave=False):
                end = min(start + batch_size, n_users)
                user_chunk = self.test_users[start:end]
                chunk_indices = np.arange(start, end)
                watched_list = [self.train_watched_items.get(u, set()) for u in user_chunk]

                try:
                    kwargs = {
                        'user_ids': user_chunk,
                        'watched_items_list': watched_list,
                        'k': over_fetch_k
                    }
                    
                    if supports_valid_items:
                        if self.n_negative_samples is not None:
                            kwargs['valid_items'] = [self._get_valid_items_for_user(u) for u in user_chunk]
                        else:
                            kwargs['valid_items'] = self.item_catalog_list

                    batch_recs = model.get_top_k_recommendations_batch(**kwargs)

                    item_set_local = self.item_set
                    for i, idx in enumerate(chunk_indices):
                        u = user_chunk[i]
                        watched = self.train_watched_items.get(u, set())
                        current_recs = np.asarray(batch_recs[i]).tolist()

                        row = [mid for mid in current_recs if mid in item_set_local and mid not in watched][:max_recommendations]
                        recs_arr[idx, :len(row)] = row

                except Exception as e:
                    logger.warning("Batch error for {}: {}. Falling back to single mode.", model_name, e)
                    batch_failed = True
                    break

        if not has_batch or batch_failed:
            item_set_local = self.item_set

            def fetch_for_user(u):
                try:
                    kwargs = {
                        'user_id': int(u),
                        'watched_items': self.train_watched_items.get(u, set()),
                        'k': over_fetch_k
                    }
                    if supports_valid_items:
                        kwargs['valid_items'] = self._get_valid_items_for_user(u)

                    recs = model.get_top_k_recommendations(**kwargs)
                    watched = self.train_watched_items.get(u, set())

                    return [mid for mid in recs if mid in item_set_local and mid not in watched][:max_recommendations]
                except Exception:
                    return []

            with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
                results_list = list(tqdm(executor.map(fetch_for_user, self.test_users), total=n_users, desc=f"{model_name}", leave=False))
                for idx, row in enumerate(results_list):
                    recs_arr[idx, :len(row)] = row

        recs_idx = self.item_index.get_indexer(recs_arr.ravel()).reshape(recs_arr.shape)

        hits = np.zeros((n_users, max_k), dtype=bool)
        for i in range(n_users):
            rel_sorted = self.rel_indices_sorted[i]
            if len(rel_sorted) == 0:
                continue
            row_recs = recs_idx[i, :max_k]
            s = np.searchsorted(rel_sorted, row_recs)
            valid_mask = s < len(rel_sorted)
            hits[i, valid_mask] = rel_sorted[s[valid_mask]] == row_recs[valid_mask]

        novelty_vals = np.where(recs_idx[:, :max_k] != -1, self.novelty_array[recs_idx[:, :max_k]], 0.0)

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
        ideal_dcg = np.where(min_counts_k > 0, cum_discounts[min_counts_k.astype(int)], 0.0)

        valid_mask = self.valid_user_mask
        n_valid_users = int(valid_mask.sum())

        results = []
        for k in valid_k_values:
            k_idx = min(k, max_k) - 1

            recs_k = recs_arr[:, :k]
            valid_recs_k = recs_k[recs_k != -1]
            valid_recs_k = valid_recs_k[np.isin(valid_recs_k, self.item_catalog_list, assume_unique=False)]
            coverage_k = (len(np.unique(valid_recs_k)) / self.n_items) if self.n_items else 0.0

            prec = cumsum_hits[valid_mask, k_idx] / k
            rec = np.divide(cumsum_hits[valid_mask, k_idx], self.counts_arr[valid_mask], out=np.zeros(n_valid_users), where=self.counts_arr[valid_mask] > 0)
            dcg = cumsum_dcg[valid_mask, k_idx]
            ndcg = np.divide(dcg, ideal_dcg[valid_mask, k_idx], out=np.zeros_like(dcg), where=ideal_dcg[valid_mask, k_idx] > 0)
            ap = np.divide(cumsum_ap[valid_mask, k_idx], np.minimum(self.counts_arr[valid_mask], k), out=np.zeros(n_valid_users), where=self.counts_arr[valid_mask] > 0)
            mrr = np.where((first_hit_idx[valid_mask] >= 0) & (first_hit_idx[valid_mask] < k), 1.0 / (first_hit_idx[valid_mask] + 1), 0.0)
            nov = novelty_vals[valid_mask, :k].mean(axis=1)

            results.append({
                'model': model_name,
                'k': k,
                'precision': float(prec.mean()) if n_valid_users > 0 else 0.0,
                'recall': float(rec.mean()) if n_valid_users > 0 else 0.0,
                'ndcg': float(ndcg.mean()) if n_valid_users > 0 else 0.0,
                'map': float(ap.mean()) if n_valid_users > 0 else 0.0,
                'mrr': float(mrr.mean()) if n_valid_users > 0 else 0.0,
                'novelty': float(nov.mean()) if n_valid_users > 0 else 0.0,
                'coverage': coverage_k,
                'n_users': int(any_hit[valid_mask].sum() if k >= max_k else np.any(hits[valid_mask, :k], axis=1).sum())
            })

        logger.info("'{}' done in {:.1f}s", model_name, time.time() - start_time)
        return results
    def evaluate_all_models(
        self,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048
    ) -> pd.DataFrame:
        all_results = []
        for model_name, model in self.models.items():
            try:
                all_results.extend(self.evaluate_model(model_name, model, k_values, max_recommendations, batch_size))
            except Exception as e:
                logger.error("Failed {}: {}", model_name, e)
                continue
        self.results = pd.DataFrame(all_results)
        return self.results

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("No results available. Run evaluate_all_models() first.")
        return self.results