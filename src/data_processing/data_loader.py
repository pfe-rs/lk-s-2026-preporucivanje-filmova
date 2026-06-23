import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
from letterboxdpy.movie import Movie
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.movie_data = []

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
<<<<<<< HEAD
   

    def fetch_single_movie(self, row):
        """Worker function to fetch data for one movie (runs in parallel)"""
        movie_id = row.movieId
        
        try:
            # Try TMDb first
            movie = Movie.from_tmdb(int(row.tmdbId))
            
            # Fallback to IMDb if TMDb returns None
            if movie is None and pd.notna(row.imdbId):
                movie = Movie.from_imdb(int(row.imdbId))
                
            if movie is None:
                return None

            # Extract features safely and optimize lookups using Walrus operator
            cast = movie.get_cast() or []
            fullcast = [slug for item in cast if (slug := item.get("slug"))][:5] # Auto-slices safely
            
            crew = movie.get_crew() or {}
            director_list = crew.get("director", [])
            director_l = director_list[0].get("slug") if director_list else None
            
            mainact = fullcast[0] if fullcast else None
        
            return {
                "title": movie.title,
                "year": movie.year,
                "crew": movie.crew,
                "cast": fullcast,
                "main_actor": mainact,
                "director": director_l,
                "rating": movie.rating,
                "runtime": movie.runtime,
            }

        except Exception as e:
            logger.error(f"Error fetching data for movie ID {movie_id}: {e}")
            return None

    def letterboxd_data(self):
        """Optimized main loop using concurrent threads"""
        rows = list(self.links_df.itertuples())
        
        # Adjust max_workers based on API rate limits (10-20 is usually a sweet spot)
        with ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to the thread pool
            futures = {executor.submit(self.fetch_single_movie, row): row for row in rows}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.movie_data.append(result)
    # def letterboxd_data(self):
    #     for  row in self.links_df.itertuples():
            
    #         movie_id = row.movieId
            
    #         letterboxd_id = int(row.tmdbId)
    #         #print(letterboxd_id)

    #         try:

    #             movie = Movie.from_tmdb(letterboxd_id)
    #             if movie is None:
    #                 letterboxd_id = int(row.imdbId)
    #                 movie = Movie.from_imdb(letterboxd_id)
    #                 if movie is None:
    #                     logger.warning(f"Letterboxd data not found for movie ID {movie_id}")
    #                     continue
    #             #print(movie.cast)
    #             cast = movie.get_cast()
                
    #             fullcast = [item["slug"] for item in cast if "slug" in item] 
                
    #             crew = movie.get_crew()
    #             director_l = crew["director"][0]["slug"] if "director" in crew and crew["director"] else None
    #             #print(director_l)
    #             #direc = json.loads(director_l)
    #             #print(direc["slug"])
                
    #             fullcast = fullcast[: 5] if len(fullcast) > 5 else fullcast
    #             mainact = fullcast[0] if fullcast else None
                
    #             self.movie_data.append({
    #                 "title": movie.title,
    #                 "year": movie.year,
    #                 "crew": movie.crew,
    #                 "cast": fullcast,
    #                 "main_actor": mainact,
    #                 "director": director_l,
    #                 "rating": movie.rating,
    #                 "runtime": movie.runtime,})
    #             #print(str(movie.year) + '\n' + str(movie.title) + '\n' + str(movie.crew) + '\n' + str(movie.cast) + '\n' + str(movie.rating) + '\n')
    #             # Process the movie data as needed
    #             #logger.info(f"Fetched Letterboxd data for movie ID {movie_id}: {movie.title}")
    #         except Exception as e:
    #             logger.error(f"Error fetching Letterboxd data for movie ID {movie_id}: {e}")
=======

    def get_genre_matrix(self) -> np.ndarray:
        if self.genre_matrix is None:
            self.preprocess_movies()
        return self.genre_matrix

    def get_movie_id_to_index(self) -> Dict[int, int]:
        if self.movies_df is None:
            self.load_data()
        return {mid: idx for idx, mid in enumerate(self.movies_df["movieId"])}
>>>>>>> 1b09a7b6da14bbaadfe3d3812ef0672ae9b1b08d
