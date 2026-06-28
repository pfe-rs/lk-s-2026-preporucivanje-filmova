import pandas as pd
import numpy as np
from typing import List, Dict, Any


class PopularityBaseline:
    def __init__(self):
        self._sorted_items: np.ndarray = np.empty(0, dtype=np.int64)
        self._sorted_pops: np.ndarray = np.empty(0, dtype=np.float32)
        self._pop_lookup: Dict[int, float] = {}
        self.is_fitted = False

    def fit(self, ratings_df: pd.DataFrame) -> "PopularityBaseline":
        counts = ratings_df.groupby("movieId").size()
        order = np.argsort(-counts.values)
        self._sorted_items = counts.index.values[order].astype(np.int64)
        self._sorted_pops = counts.values[order].astype(np.float32)
        self._pop_lookup = {
            int(mid): float(pop)
            for mid, pop in zip(self._sorted_items, self._sorted_pops)
        }
        self.is_fitted = True
        return self

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return [self._pop_lookup.get(mid, 0.0) for mid in item_ids]

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = 10) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if not watched_items:
            return self._sorted_items[:k].tolist()
        result = []
        for mid in self._sorted_items:
            if int(mid) not in watched_items:
                result.append(int(mid))
                if len(result) == k:
                    break
        return result

    def get_top_k_recommendations_batch(
        self,
        user_ids: List[int],
        watched_items_list: List[set],
        k: int = 10,
    ) -> List[List[int]]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        results = []
        for watched in watched_items_list:
            if not watched:
                results.append(self._sorted_items[:k].tolist())
            else:
                w = np.fromiter(watched, np.int64, len(watched))
                results.append(self._sorted_items[~np.isin(self._sorted_items, w)][:k].tolist())
        return results

    def explain_recommendation(
        self, movie_id: int, liked_items: set, top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        return []