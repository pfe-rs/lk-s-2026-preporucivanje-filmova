import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import logging
import asyncio
import aiohttp
import ast
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOVIE_LENS_RAW_PATH = Path("data/raw/movielens/")
TMDB_API_KEY = "99b5221ca8d50de2f0a8aaa057a2b86d"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

async def fetch_tmdb_movie_data(session: aiohttp.ClientSession, tmdb_id: int) -> Optional[Dict[str, Any]]:
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,keywords",
    }
    try:
        async with session.get(url, params=params, timeout=7) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 1))
                await asyncio.sleep(retry_after)
                return await fetch_tmdb_movie_data(session, tmdb_id)
            if response.status != 200:
                #logger.error(f"TMDB API error for tmdbId {tmdb_id}: HTTP Status {response.status}")
                return None
            return await response.json()
    except Exception as e:
        #logger.error(f"Network exception for tmdbId {tmdb_id}: {str(e)}")
        return None

async def process_single_row(session: aiohttp.ClientSession, row: Any, semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
    if pd.isna(row.tmdbId):
        return None
    async with semaphore:
        data = await fetch_tmdb_movie_data(session, int(row.tmdbId))
        if not data:
            return None
        try:
            credits_data = data.get('credits', {})
            cast_list = credits_data.get('cast', [])
            crew_list = credits_data.get('crew', [])
            vote_count = data.get('vote_count')
            fullcast = [c['name'] for c in cast_list[:5] if 'name' in c]
            main_actor = fullcast[0] if fullcast else None
            director_name = next((c['name'] for c in crew_list if c.get('job') == 'Director'), None)
            year = data.get('release_date', '').split('-')[0] if data.get('release_date') else None
            keywords_wrapper = data.get('keywords', {})
            keywords_list = keywords_wrapper.get('keywords', [])
            keywords = [kw.get('name') for kw in keywords_list if kw.get('name')]
            return {
                "movieId": int(row.movieId),
                "title": data.get('title', ''),
                "year": year,
                "cast": fullcast,
                "main_actor": main_actor,
                "director": director_name,
                "rating": data.get('vote_average', 0.0), 
                "runtime": data.get('runtime', 0),
                "keywords": keywords,
                "vote_count": vote_count
            }
        except Exception as e:
            logger.error(f"Error parsing JSON for movieId {getattr(row, 'movieId', '?')}: {e}")
            return None

class MovieLensDataLoader:
    def __init__(self, data_path: str = "ml-latest-small", cache_file: str = "data/processed/movie_metadata.csv"):
        self.data_path = Path(MOVIE_LENS_RAW_PATH / data_path)
        self.cache_path = Path(cache_file)
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
        genre_dummies = self.movies_df["genres"].str.get_dummies(sep='|')
        genre_dummies.columns = [f"genre_{col.lower()}" for col in genre_dummies.columns]
        self.genre_matrix = genre_dummies.values.astype(float)
        return genre_dummies
    
    def preprocess_tags(self) -> pd.DataFrame:
        if self.tags_df is None:
            raise ValueError("Tags data not loaded. Call load_data() first.")
        tag_matrix = self.tags_df.pivot_table(
            index="userId", columns="tag", values="timestamp", aggfunc="count", fill_value=0
        )
        self.tag_matrix = tag_matrix
        return tag_matrix

    def get_user_item_matrix(self) -> pd.DataFrame:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        return self.ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating", aggfunc="mean", fill_value=0
        )

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        users = self.ratings_df["userId"].unique()
        np.random.seed(random_state)
        test_users = set(np.random.choice(users, size=int(len(users) * test_size), replace=False))
        train_ratings = self.ratings_df[~self.ratings_df["userId"].isin(test_users)]
        test_ratings = self.ratings_df[self.ratings_df["userId"].isin(test_users)]
        return train_ratings, test_ratings

    async def letterboxd_data_async(self, max_concurrent_requests: int = 100):
        if self.links_df is None:
            raise ValueError("Links data not loaded. Call load_data() first.")

        cached_df = pd.DataFrame()
        existing_ids = set()

        if self.cache_path.exists():
            logger.info(f"Loading existing data from cache: {self.cache_path}")
            cached_df = pd.read_csv(self.cache_path)
            for col in ['cast', 'keywords']:
                if col in cached_df.columns:
                    cached_df[col] = cached_df[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
                    )
            if "movieId" in cached_df.columns:
                existing_ids = set(cached_df["movieId"].dropna().astype(int))

        missing_links = self.links_df[
            (~self.links_df["movieId"].astype(int).isin(existing_ids)) & 
            (self.links_df["tmdbId"].notna())
        ]

        if not missing_links.empty:
            total_to_fetch = len(missing_links)
            logger.info(f"Found {total_to_fetch} missing movies with valid TMDB IDs. Fetching...")
            rows = list(missing_links.itertuples())
            semaphore = asyncio.Semaphore(max_concurrent_requests)
            connector = aiohttp.TCPConnector(limit=max_concurrent_requests, ttl_dns_cache=300)
            
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = [process_single_row(session, row, semaphore) for row in rows]
                results = await tqdm_asyncio.gather(*tasks, desc="Downloading metadata")
                
            new_data = [r for r in results if r is not None]
            success_count = len(new_data)
            failed_count = total_to_fetch - success_count

            logger.info(f"DOWNLOAD SUMMARY -> Total Requested: {total_to_fetch} | Successfully Saved: {success_count} | Failed/Skipped: {failed_count}")

            if new_data:
                new_df = pd.DataFrame(new_data)
                cached_df = pd.concat([cached_df, new_df], ignore_index=True)
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cached_df.to_csv(self.cache_path, index=False)
                    logger.info(f"Cache successfully updated and saved to {self.cache_path}")
                except Exception as e:
                    logger.error(f"Error saving cache to disk: {e}")
            else:
                logger.error("Zero movies were successfully downloaded. Check the console logs above for specific TMDB API errors.")
        else:
            logger.info("All movies are already cached or have missing TMDB IDs. No network requests needed.")

        self.movie_data = cached_df.to_dict(orient='records')

    def get_genre_matrix(self) -> np.ndarray:
        if self.genre_matrix is None:
            self.preprocess_movies()
        return self.genre_matrix

    def get_movie_id_to_index(self) -> Dict[int, int]:
        if self.movies_df is None:
            self.load_data()
        return {mid: idx for idx, mid in enumerate(self.movies_df["movieId"])}