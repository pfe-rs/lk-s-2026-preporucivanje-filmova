import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MOVIE_LENS_RAW_PATH = Path("data/raw/movielens/")


class MovieLensDataLoader:
    def __init__(self, data_path: str = "ml-latest-small"):
        self.data_path = Path(MOVIE_LENS_RAW_PATH / data_path)
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.links_df = None
        self.genre_matrix = None
        self.tag_matrix = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading MovieLens dataset...")

        try:
            self.movies_df = pd.read_csv(self.data_path / "movies.csv")
            self.ratings_df = pd.read_csv(self.data_path / "ratings.csv")
            self.tags_df = pd.read_csv(self.data_path / "tags.csv")
            self.links_df = pd.read_csv(self.data_path / "links.csv")

            logger.info(f"Loaded {len(self.movies_df)} movies")
            logger.info(f"Loaded {len(self.ratings_df)} ratings")
            logger.info(f"Loaded {len(self.tags_df)} tags")
            logger.info(f"Loaded {len(self.links_df)} links")

            return {
                "movies": self.movies_df,
                "ratings": self.ratings_df,
                "tags": self.tags_df,
                "links": self.links_df,
            }

        except FileNotFoundError as e:
            logger.error(f"Dataset files not found: {e}")
            raise

    def preprocess_movies(self) -> pd.DataFrame:
        if self.movies_df is None:
            raise ValueError("Movies data not loaded. Call load_data() first.")

        logger.info("Preprocessing movie data...")

        genre_dummies = self.movies_df["genres"].str.get_dummies("|")
        genre_dummies.columns = [f"genre_{col}" for col in genre_dummies.columns]

        self.genre_matrix = genre_dummies.values.astype(float)

        logger.info(f"Created genre matrix with shape {self.genre_matrix.shape}")
        return genre_dummies

    # TODO: CHANGE THIS SHIT
    def preprocess_tags(self) -> pd.DataFrame:
        """Preprocess tag data."""
        if self.tags_df is None:
            raise ValueError("Tags data not loaded. Call load_data() first.")

        logger.info("Preprocessing tag data...")

        tag_counts = self.tags_df["tag"].value_counts().reset_index()
        tag_counts.columns = ["tag", "count"]

        tag_matrix = self.tags_df.pivot_table(
            index="userId",
            columns="tag",
            values="timestamp",
            aggfunc="count",
            fill_value=0,
        )

        self.tag_matrix = tag_matrix
        logger.info(
            f"Created tag matrix with {tag_matrix.shape[1]} users and {tag_matrix.shape[0]} tags"
        )

        return tag_matrix

    def get_user_item_matrix(self) -> pd.DataFrame:
        """Create user-item rating matrix."""
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")

        logger.info("Creating user-item rating matrix...")

        user_item_matrix = self.ratings_df.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
            fill_value=0,
        )

        logger.info(f"Created user-item matrix: {user_item_matrix.shape}")
        return user_item_matrix

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")

        # Get unique users
        users = self.ratings_df["userId"].unique()

        # Split users into train and test
        np.random.seed(random_state)
        test_users = set(
            np.random.choice(users, size=int(len(users) * test_size), replace=False)
        )

        # Filter ratings for train and test
        train_ratings = self.ratings_df[~self.ratings_df["userId"].isin(test_users)]
        test_ratings = self.ratings_df[self.ratings_df["userId"].isin(test_users)]

        logger.info(
            f"Train ratings: {len(train_ratings)}, Test ratings: {len(test_ratings)}"
        )

        return train_ratings, test_ratings

    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        if self.movies_df is None:
            raise ValueError("Movies data not loaded. Call load_data() first.")

        movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
        if len(movie_info) == 0:
            return None

        return movie_info.iloc[0].to_dict()

    def get_user_rated_movies(self, user_id: int) -> List[int]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")

        return self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()

    def get_movie_genres(self, movie_id: int) -> List[str]:
        if self.movies_df is None:
            raise ValueError("Movies data not loaded. Call load_data() first.")

        movie_info = self.movies_df[self.movies_df["movieId"] == movie_id]
        if len(movie_info) == 0:
            return []

        genres = movie_info.iloc[0]["genres"].split("|")
        return genres
