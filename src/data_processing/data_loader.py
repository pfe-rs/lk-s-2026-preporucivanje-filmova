import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import logging
import asyncio
import aiohttp
import orjson
from scipy.sparse import csr_matrix, coo_matrix
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOVIE_LENS_RAW_PATH = Path("data/raw/movielens/")
TMDB_API_KEY = "99b5221ca8d50de2f0a8aaa057a2b86d"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

class TMDBMetadataFetcher:
    def __init__(self, base_url: str = TMDB_BASE_URL, api_key: str = TMDB_API_KEY, max_concurrent: int = 100):
        self.base_url = base_url
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.connector = aiohttp.TCPConnector(limit=max_concurrent, ttl_dns_cache=300)

    async def fetch_movie_data(self, session: aiohttp.ClientSession, tmdb_id: int) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/movie/{tmdb_id}"
        params = {
            "api_key": self.api_key,
            "append_to_response": "credits,keywords",
        }
        for _ in range(3):
            try:
                async with session.get(url, params=params, timeout=7) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 1))
                        await asyncio.sleep(retry_after)
                        continue
                    if response.status != 200:
                        return None
                    return await response.json(loads=orjson.loads)
            except Exception:
                return None
        return None

    async def process_row(self, session: aiohttp.ClientSession, row: Any) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            data = await self.fetch_movie_data(session, int(row.tmdbId))
            if not data:
                return None
            try:
                credits_data = data.get("credits", {})
                cast_list = credits_data.get("cast", [])
                crew_list = credits_data.get("crew", [])
                
                fullcast = [c["name"] for c in cast_list[:5] if "name" in c]
                main_actor = fullcast[0] if fullcast else None
                director_name = next((c["name"] for c in crew_list if c.get("job") == "Director"), None)
                
                release_date = data.get("release_date")
                year = release_date.split("-")[0] if release_date else None
                
                keywords_list = data.get("keywords", {}).get("keywords", [])
                keywords = [kw.get("name") for kw in keywords_list if kw.get("name")]
                
                return {
                    "movieId": int(row.movieId),
                    "title": data.get("title", ""),
                    "year": year,
                    "cast": fullcast,
                    "main_actor": main_actor,
                    "director": director_name,
                    "rating": float(data.get("vote_average", 0.0)),
                    "runtime": int(data.get("runtime") or 0),
                    "keywords": keywords,
                    "vote_count": int(data.get("vote_count") or 0)
                }
            except Exception as e:
                logger.error(f"Error parsing JSON for movieId {getattr(row, 'movieId', '?')}: {e}")
                return None

    async def fetch_all(self, missing_links: pd.DataFrame, cache_path: Path, cached_df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        total_to_fetch = len(missing_links)
        logger.info(f"Found {total_to_fetch} missing movies with valid TMDB IDs. Fetching...")
        
        rows = list(missing_links.itertuples())
        
        async with aiohttp.ClientSession(connector=self.connector) as session:
            tasks = [self.process_row(session, row) for row in rows]
            
            new_data = []
            for i in range(0, total_to_fetch, batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await tqdm_asyncio.gather(*batch_tasks, desc=f"Downloading batch {i // batch_size + 1}")
                
                valid_results = [r for r in batch_results if r is not None]
                if valid_results:
                    new_data.extend(valid_results)
                    batch_df = pd.DataFrame(valid_results)
                    cached_df = pd.concat([cached_df, batch_df], ignore_index=True)
                    
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        cached_df.to_parquet(cache_path, index=False)
                    except Exception as e:
                        logger.error(f"Intermediate cache saving failed: {e}")

        success_count = len(new_data)
        failed_count = total_to_fetch - success_count
        logger.info(f"DOWNLOAD SUMMARY: Requested: {total_to_fetch} | Saved: {success_count} | Failed: {failed_count}")
        
        return cached_df


class MovieLensDataLoader:
    def __init__(self, data_path: str = "ml-latest-small", cache_file: str = "data/processed/movie_metadata.parquet"):
        self.data_path = Path(MOVIE_LENS_RAW_PATH / data_path)
        self.cache_path = Path(cache_file)
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.tags_df: Optional[pd.DataFrame] = None
        self.links_df: Optional[pd.DataFrame] = None
        self.genre_matrix: Optional[np.ndarray] = None
        self.movie_data: List[Dict[str, Any]] = []

    def load_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading MovieLens dataset...")
        try:
            dtypes_ratings = {"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64}
            dtypes_links = {"movieId": np.int32, "imdbId": np.int32, "tmdbId": pd.Int32Dtype()}
            dtypes_tags = {"userId": np.int32, "movieId": np.int32, "tag": "string", "timestamp": np.int64}

            self.movies_df = pd.read_csv(self.data_path / "movies.csv", engine="pyarrow")
            self.ratings_df = pd.read_csv(self.data_path / "ratings.csv", engine="pyarrow", dtype=dtypes_ratings)
            self.tags_df = pd.read_csv(self.data_path / "tags.csv", engine="pyarrow", dtype=dtypes_tags)
            self.links_df = pd.read_csv(self.data_path / "links.csv", engine="pyarrow", dtype=dtypes_links)
            
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
        genre_dummies = self.movies_df["genres"].str.get_dummies(sep="|").astype(np.float32)
        genre_dummies.columns = [f"genre_{col.lower()}" for col in genre_dummies.columns]
        self.genre_matrix = genre_dummies.values
        return genre_dummies
    
    def preprocess_tags(self) -> csr_matrix:
        if self.tags_df is None:
            raise ValueError("Tags data not loaded. Call load_data() first.")
            
        self.tags_df["tag"] = self.tags_df["tag"].astype("category")
        self.tags_df["userId"] = self.tags_df["userId"].astype("category")
        
        row = self.tags_df["userId"].cat.codes.values
        col = self.tags_df["tag"].cat.codes.values
        data = np.ones(len(self.tags_df), dtype=np.float32)
        
        return coo_matrix((data, (row, col))).tocsr()

    def get_user_item_matrix(self) -> Tuple[csr_matrix, List[int], List[int]]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
            
        user_c = pd.Categorical(self.ratings_df["userId"])
        movie_c = pd.Categorical(self.ratings_df["movieId"])
        
        rows = user_c.codes
        cols = movie_c.codes
        v = self.ratings_df["rating"].values
        
        matrix = coo_matrix((v, (rows, cols)), shape=(len(user_c.categories), len(movie_c.categories))).tocsr()
        return matrix, list(user_c.categories), list(movie_c.categories)

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        users = self.ratings_df["userId"].unique()
        np.random.seed(random_state)
        test_users = set(np.random.choice(users, size=int(len(users) * test_size), replace=False))
        mask = self.ratings_df["userId"].isin(test_users)
        return self.ratings_df[~mask], self.ratings_df[mask]

    async def letterboxd_data_async(self, max_concurrent_requests: int = 100):
        if self.links_df is None:
            raise ValueError("Links data not loaded. Call load_data() first.")

        cached_df = pd.DataFrame()
        existing_ids = set()

        if self.cache_path.exists():
            logger.info(f"Loading existing data from cache: {self.cache_path}")
            cached_df = pd.read_parquet(self.cache_path)
            if "movieId" in cached_df.columns:
                existing_ids = set(cached_df["movieId"].dropna().astype(np.int32))

        missing_links = self.links_df[
            (~self.links_df["movieId"].isin(existing_ids)) & 
            (self.links_df["tmdbId"].notna())
        ]

        if not missing_links.empty:
            fetcher = TMDBMetadataFetcher(max_concurrent=max_concurrent_requests)
            cached_df = await fetcher.fetch_all(missing_links, self.cache_path, cached_df)
        else:
            logger.info("All movies are already cached or have missing TMDB IDs. No network requests needed.")

        self.movie_data = cached_df.to_dict(orient="records")

    def get_genre_matrix(self) -> np.ndarray:
        if self.genre_matrix is None:
            self.preprocess_movies()
        return self.genre_matrix

    def get_movie_id_to_index(self) -> Dict[int, int]:
        if self.movies_df is None:
            self.load_data()
        return {int(mid): idx for idx, mid in enumerate(self.movies_df["movieId"])}