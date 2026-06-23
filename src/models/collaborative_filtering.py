import logging
import pandas as pd
from typing import List, Optional, Tuple
from surprise import SVD, Dataset, Reader

logger = logging.getLogger(__name__)

MIN_RATING = 1.0
MAX_RATING = 5.0
class CollaborativeFiltering:
    def __init__(self, k_components: int = 50, random_state: int = 42) -> None:
        self.n_components = k_components
        self.random_state = random_state

        self.svd_model: Optional[SVD] = None
        self.trainset = None

    def fit(self, df_ratings: pd.DataFrame) -> "CollaborativeFiltering":
        logger.info(
            f"Fitting Surprise SVD model with {self.n_components} components..."
        )

        reader = Reader(rating_scale=(MIN_RATING, MAX_RATING))

        data = Dataset.load_from_df(df_ratings[["userId", "movieId", "rating"]], reader)

        self.trainset = data.build_full_trainset()

        self.svd_model = SVD(
            n_factors=self.n_components, random_state=self.random_state
        )
        self.svd_model.fit(self.trainset)

        logger.info("SVD model successfully fitted.")
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float:
        if self.svd_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        prediction = self.svd_model.predict(user_id, movie_id)
        return float(prediction.est)

    def recommend_for_user(
        self, user_id: int, watched_movie_ids: List[int], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        if self.svd_model is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        all_movie_ids = self.trainset.all_items()
        all_raw_movie_ids = [self.trainset.to_raw_iid(m_id) for m_id in all_movie_ids]

        watched_set = set(watched_movie_ids)
        unwatched_movies = [
            m_id for m_id in all_raw_movie_ids if m_id not in watched_set
        ]

        predictions = [
            (movie_id, self.predict_score(user_id, movie_id))
            for movie_id in unwatched_movies
        ]

        predictions.sort(key=lambda x: x[1], reverse=True)

        return [(movie_id, score) for movie_id, score in predictions[:top_n]]
