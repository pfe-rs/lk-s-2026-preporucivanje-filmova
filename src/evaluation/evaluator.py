import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Iterable
from tqdm import tqdm
from loguru import logger

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class RecommendationEvaluator:
    def __init__(
        self,
        models: Dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        relevance_threshold: float = 4.0,
        user_sample_size: Optional[int] = None,
        random_state: int = 42,
        use_gpu: bool = True
    ):
        self.models = models
        self.relevance_threshold = relevance_threshold
        self.random_state = random_state
        self.use_gpu = use_gpu and HAS_GPU

        all_test_users = test_df['userId'].unique()
        if user_sample_size is not None and user_sample_size < len(all_test_users):
            np.random.seed(random_state)
            self.test_users = np.random.choice(all_test_users, size=user_sample_size, replace=False)
        else:
            self.test_users = all_test_users

        self.user_to_idx = {u: i for i, u in enumerate(self.test_users)}
        self.n_test_users = len(self.test_users)

        test_filtered = test_df[test_df['rating'] >= relevance_threshold]
        rel_dict = test_filtered.groupby('userId')['movieId'].apply(list).to_dict()
        
        self.train_watched_items = train_df.groupby('userId')['movieId'].apply(set).to_dict()

        max_rel_len = max((len(items) for items in rel_dict.values()), default=0)
        self.padded_rel_items = np.full((self.n_test_users, max_rel_len), -1, dtype=np.int64)
        self.rel_counts = np.zeros(self.n_test_users, dtype=np.float32)

        for user_id, idx in self.user_to_idx.items():
            items = rel_dict.get(user_id, [])
            n_items = len(items)
            if n_items > 0:
                self.padded_rel_items[idx, :n_items] = items
                self.rel_counts[idx] = n_items

        self.all_items = set(test_df['movieId'].unique())
        self.item_popularity = train_df['movieId'].value_counts().to_dict()
        total_interactions = sum(self.item_popularity.values())

        max_train_id = train_df['movieId'].max() if not train_df.empty else 0
        max_test_id = test_df['movieId'].max() if not test_df.empty else 0
        max_movie_id = int(max(max_train_id, max_test_id))

        self._default_novelty = -np.log2(1.0 / total_interactions) if total_interactions > 0 else 0.0
        self.novelty_map = np.full(max_movie_id + 2, self._default_novelty, dtype=np.float32)
        
        for item, count in self.item_popularity.items():
            prob = count / total_interactions
            self.novelty_map[item] = -np.log2(prob) if prob > 0 else self._default_novelty
            
        self.novelty_map[-1] = 0.0

        if self.use_gpu:
            self.padded_rel_items = cp.asarray(self.padded_rel_items)
            self.rel_counts = cp.asarray(self.rel_counts)
            self.novelty_map = cp.asarray(self.novelty_map)

        self.results = None

    def _chunk_generator(self, iterable: Iterable, size: int):
        iterator = iter(iterable)
        for first in iterator:
            chunk = [first]
            for _ in range(size - 1):
                try:
                    chunk.append(next(iterator))
                except StopIteration:
                    break
            yield chunk

    def _compute_metrics_vectorized(
        self,
        recs: Any,
        u_idx: Any,
        k_values: List[int],
        max_k: int
    ) -> Dict[int, Dict[str, float]]:
        xp = cp if self.use_gpu else np
        
        rels = self.padded_rel_items[u_idx]
        counts = self.rel_counts[u_idx]

        hits_3d = recs[:, :, None] == rels[:, None, :]
        rel_mask = xp.any(hits_3d, axis=2)

        discounts = 1.0 / xp.log2(xp.arange(2, max_k + 2, dtype=xp.float32))
        idcg_cache = xp.concatenate([xp.zeros(1, dtype=xp.float32), xp.cumsum(discounts)])

        novelties = self.novelty_map[recs]

        results = {}
        for k in k_values:
            k_idx = min(k, max_k)
            mask_k = rel_mask[:, :k_idx]
            disc_k = discounts[:k_idx]

            hits = xp.sum(mask_k, axis=1, dtype=xp.float32)
            precision = hits / k_idx
            recall = xp.where(counts > 0, hits / counts, 0.0)

            dcg = xp.sum(mask_k * disc_k, axis=1)
            ideal_counts = xp.minimum(counts, k_idx).astype(xp.int32)
            ideal_dcg = idcg_cache[ideal_counts]
            ndcg = xp.where(ideal_dcg > 0, dcg / ideal_dcg, 0.0)

            cum_hits = xp.cumsum(mask_k, axis=1)
            positions = xp.arange(1, k_idx + 1, dtype=xp.float32)
            prec_at_i = cum_hits / positions
            ap = xp.sum(prec_at_i * mask_k, axis=1)
            map_score = xp.where(counts > 0, ap / xp.minimum(counts, k_idx), 0.0)

            has_rel = xp.any(mask_k, axis=1)
            first_rel = xp.argmax(mask_k, axis=1)
            mrr = xp.where(has_rel, 1.0 / (first_rel.astype(xp.float32) + 1.0), 0.0)

            novelty = xp.mean(novelties[:, :k_idx], axis=1)

            results[k] = {
                'precision': float(xp.mean(precision).item()),
                'recall': float(xp.mean(recall).item()),
                'ndcg': float(xp.mean(ndcg).item()),
                'map': float(xp.mean(map_score).item()),
                'mrr': float(xp.mean(mrr).item()),
                'novelty': float(xp.mean(novelty).item()),
                'n_users': int(xp.sum(hits > 0).item())
            }
            
        return results

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
        pad_val = -1
        k_max = min(max(k_values) if k_values else max_recommendations, max_recommendations)

        recs_arr = np.full((self.n_test_users, max_recommendations), pad_val, dtype=np.int64)
        
        has_batch_method = hasattr(model, 'get_top_k_recommendations_batch')

        for user_chunk in tqdm(self._chunk_generator(self.test_users, batch_size), 
                               total=(self.n_test_users + batch_size - 1) // batch_size, 
                               desc=f"{model_name}", leave=False):
            
            chunk_indices = [self.user_to_idx[u] for u in user_chunk]
            
            if has_batch_method:
                watched_list = [self.train_watched_items.get(u, set()) for u in user_chunk]
                try:
                    batch_recs = model.get_top_k_recommendations_batch(
                        user_ids=user_chunk,
                        watched_items_list=watched_list,
                        k=max_recommendations
                    )
                    for i, idx in enumerate(chunk_indices):
                        r = batch_recs[i][:max_recommendations]
                        recs_arr[idx, :len(r)] = r
                except Exception as e:
                    logger.warning("Batch error: {}", e)
            else:
                for u, idx in zip(user_chunk, chunk_indices):
                    watched = self.train_watched_items.get(u, set())
                    try:
                        r = model.get_top_k_recommendations(
                            user_id=int(u),
                            watched_items=watched,
                            k=max_recommendations
                        )
                        if r:
                            r = r[:max_recommendations]
                            recs_arr[idx, :len(r)] = r
                    except Exception as e:
                        logger.warning("Error for user {}: {}", u, e)

        unique_items = np.unique(recs_arr)
        unique_count = len(unique_items) - (1 if pad_val in unique_items else 0)
        coverage = unique_count / len(self.all_items) if self.all_items else 0.0

        xp = cp if self.use_gpu else np
        gpu_recs_arr = xp.asarray(recs_arr)
        gpu_u_idx = xp.arange(self.n_test_users)

        metrics_dict = self._compute_metrics_vectorized(
            gpu_recs_arr, gpu_u_idx, k_values, k_max
        )

        results = []
        for k in k_values:
            met = metrics_dict[k]
            results.append({
                'model': model_name,
                'k': k,
                'precision': met['precision'],
                'recall': met['recall'],
                'ndcg': met['ndcg'],
                'map': met['map'],
                'mrr': met['mrr'],
                'novelty': met['novelty'],
                'coverage': coverage,
                'n_users': met['n_users']
            })

        elapsed = time.time() - start_time
        logger.info("'{}' done in {:.1f}s", model_name, elapsed)
        
        return results

    def evaluate_all_models(
        self,
        k_values: List[int] = [5, 10, 20],
        max_recommendations: int = 20,
        batch_size: int = 2048
    ) -> pd.DataFrame:
        total_start = time.time()
        all_results = []
        
        for model_name, model in self.models.items():
            try:
                res = self.evaluate_model(model_name, model, k_values, max_recommendations, batch_size)
                all_results.extend(res)
            except Exception as e:
                logger.error("Failed {}: {}", model_name, e)
                continue
                
        self.results = pd.DataFrame(all_results)
        return self.results

    def get_results(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("No results available. Run evaluate_all_models() first.")
        return self.results