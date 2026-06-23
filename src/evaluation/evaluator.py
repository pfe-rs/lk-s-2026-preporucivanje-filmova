import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    map_at_k,
    mrr,
    catalog_coverage,
    intra_list_diversity,
    novelty,
)

logger = logging.getLogger(__name__)


def interaction_level_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits interactions per user to ensure all users are in both train and test sets.
    Crucial for evaluating Collaborative Filtering models.
    """
    np.random.seed(random_state)
    train_indices = []
    test_indices = []

    for user_id, group in df.groupby("userId"):
        n_interactions = len(group)
        if n_interactions < 2:
            train_indices.extend(group.index.tolist())
            continue

        n_test = max(1, int(n_interactions * test_size))
        shuffled_indices = group.sample(
            frac=1, random_state=random_state
        ).index.tolist()

        test_indices.extend(shuffled_indices[:n_test])
        train_indices.extend(shuffled_indices[n_test:])

    return df.loc[train_indices], df.loc[test_indices]


class ContentBasedFiltering:
    """Simple Content-Based Filtering using movie genres."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.movie_ids = None
        self.sim_matrix = None
        self.movie_id_to_idx = {}
        self.ratings_df = None

    def fit(self, df_ratings: pd.DataFrame) -> "ContentBasedFiltering":
        logger.info("Fitting Content-Based model...")
        self.ratings_df = df_ratings
        self.movie_ids = self.data_loader.movies_df["movieId"].values
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}

        genre_matrix = self.data_loader.get_genre_matrix()
        logger.info("Computing item-item similarity matrix...")
        self.sim_matrix = cosine_similarity(genre_matrix)
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if self.sim_matrix is None:
            raise ValueError("Model not fitted.")
        if movie_id not in self.movie_id_to_idx:
            return 3.0

        user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user_ratings.empty:
            return 3.0

        movie_idx = self.movie_id_to_idx[movie_id]
        weighted_sum, sim_sum = 0.0, 0.0

        for _, row in user_ratings.iterrows():
            rated_mid = row["movieId"]
            if rated_mid in self.movie_id_to_idx:
                sim = self.sim_matrix[movie_idx, self.movie_id_to_idx[rated_mid]]
                if sim > 0:
                    weighted_sum += sim * row["rating"]
                    sim_sum += sim

        return weighted_sum / sim_sum if sim_sum > 0 else user_ratings["rating"].mean()

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        if self.sim_matrix is None:
            raise ValueError("Model not fitted.")

        watched_indices = [
            self.movie_id_to_idx[mid]
            for mid in watched_movie_ids
            if mid in self.movie_id_to_idx
        ]

        if not watched_indices:
            pop = self.ratings_df.groupby("movieId").size().sort_values(ascending=False)
            return [(mid, float(score)) for mid, score in pop.head(top_n).items()]

        scores = np.mean(self.sim_matrix[watched_indices], axis=0)
        for idx in watched_indices:
            scores[idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(self.movie_ids[idx], float(scores[idx])) for idx in top_indices]


class HybridRecommender:
    """Combines CF and CBF by averaging their normalized scores."""

    def __init__(self, cf_model, cbf_model, alpha: float = 0.5):
        self.cf_model = cf_model
        self.cbf_model = cbf_model
        self.alpha = alpha  # Weight for CF (1-alpha for CBF)

    def predict_score(self, user_id: int, movie_id: int) -> float:
        return self.alpha * self.cf_model.predict_score(user_id, movie_id) + (
            1 - self.alpha
        ) * self.cbf_model.predict_score(user_id, movie_id)

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        cf_recs = self.cf_model.recommend_for_user(
            user_id, watched_movie_ids, top_n=top_n * 3
        )
        cbf_recs = self.cbf_model.recommend_for_user(
            user_id, watched_movie_ids, top_n=top_n * 3
        )

        cf_scores = {mid: score for mid, score in cf_recs}
        cbf_scores = {mid: score for mid, score in cbf_recs}
        all_movies = set(cf_scores.keys()).union(set(cbf_scores.keys()))

        def normalize(scores_dict):
            if not scores_dict:
                return {}
            vals = list(scores_dict.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 0.5 for k in scores_dict}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

        cf_norm, cbf_norm = normalize(cf_scores), normalize(cbf_scores)

        hybrid_scores = {}
        for mid in all_movies:
            hybrid_scores[mid] = self.alpha * cf_norm.get(mid, 0.0) + (
                1 - self.alpha
            ) * cbf_norm.get(mid, 0.0)

        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


class RecommendationEvaluator:
    def __init__(self, data_loader, k: int = 10):
        self.data_loader = data_loader
        self.k = k

        # Precompute for Beyond-Accuracy metrics
        self.item_popularity = (
            data_loader.ratings_df.groupby("movieId").size().to_dict()
        )
        self.total_items = len(data_loader.movies_df)

        self.item_features = {}
        genre_matrix = data_loader.get_genre_matrix()
        movie_ids = data_loader.movies_df["movieId"].values
        for i, mid in enumerate(movie_ids):
            self.item_features[mid] = genre_matrix[i]

    def evaluate_top_n(
        self,
        model,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        logger.info(f"Evaluating Top-N metrics for {model_name}...")

        test_user_items = test_df.groupby("userId")["movieId"].apply(list).to_dict()
        train_user_items = train_df.groupby("userId")["movieId"].apply(list).to_dict()

        metrics = defaultdict(list)
        all_recommended_items = []

        for user_id, relevant_items in test_user_items.items():
            watched_items = train_user_items.get(user_id, [])
            try:
                recs = model.recommend_for_user(user_id, watched_items, top_n=self.k)
                recommended_ids = [mid for mid, score in recs]
                all_recommended_items.append(recommended_ids)

                metrics["Precision"].append(
                    precision_at_k(recommended_ids, relevant_items, self.k)
                )
                metrics["Recall"].append(
                    recall_at_k(recommended_ids, relevant_items, self.k)
                )
                metrics["F1"].append(f1_at_k(recommended_ids, relevant_items, self.k))
                metrics["NDCG"].append(
                    ndcg_at_k(recommended_ids, relevant_items, self.k)
                )
                metrics["MAP"].append(map_at_k(recommended_ids, relevant_items, self.k))
                metrics["MRR"].append(mrr(recommended_ids, relevant_items))

                metrics["Diversity"].append(
                    intra_list_diversity(recommended_ids, self.item_features)
                )
                metrics["Novelty"].append(
                    novelty(recommended_ids, self.item_popularity)
                )

            except Exception as e:
                logger.warning(f"Failed to evaluate user {user_id}: {e}")
                continue

        results = {metric: np.mean(vals) for metric, vals in metrics.items() if vals}
        results["Catalog_Coverage"] = catalog_coverage(
            all_recommended_items, self.total_items
        )

        logger.info(f"Top-N Results for {model_name}: {results}")
        return results

    def evaluate_predictive(
        self, model, test_df: pd.DataFrame, model_name: str = "Model"
    ) -> Dict[str, float]:
        logger.info(f"Evaluating predictive metrics for {model_name}...")
        sq_errors, abs_errors = [], []

        eval_df = test_df.sample(min(5000, len(test_df)), random_state=42)

        for _, row in eval_df.iterrows():
            try:
                pred = model.predict_score(row["userId"], row["movieId"])
                sq_errors.append((row["rating"] - pred) ** 2)
                abs_errors.append(abs(row["rating"] - pred))
            except Exception:
                continue

        if not sq_errors:
            return {"RMSE": 0.0, "MAE": 0.0}

        results = {"RMSE": np.sqrt(np.mean(sq_errors)), "MAE": np.mean(abs_errors)}
        logger.info(f"Predictive Results for {model_name}: {results}")
        return results
