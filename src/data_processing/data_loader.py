import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import orjson
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.asyncio import tqdm_asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOVIE_LENS_RAW_PATH = Path("data/raw/movielens/")
TMDB_API_KEY = "99b5221ca8d50de2f0a8aaa057a2b86d"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=7)


class TMDBMetadataFetcher:
    __slots__ = ("base_url", "api_key", "max_concurrent", "_semaphore")

    def __init__(
        self,
        base_url: str = TMDB_BASE_URL,
        api_key: str = TMDB_API_KEY,
        max_concurrent: int = 100,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def fetch_movie_data(
        self, session: aiohttp.ClientSession, tmdb_id: int
    ) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/movie/{tmdb_id}"
        params = {"api_key": self.api_key, "append_to_response": "credits,keywords"}
        for _ in range(3):
            try:
                async with session.get(url, params=params, timeout=_REQUEST_TIMEOUT) as response:
                    if response.status == 429:
                        await asyncio.sleep(int(response.headers.get("Retry-After", 1)))
                        continue
                    if response.status != 200:
                        return None
                    return await response.json(loads=orjson.loads)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return None
        return None

    async def process_row(
        self, session: aiohttp.ClientSession, row: Any
    ) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            data = await self.fetch_movie_data(session, int(row.tmdbId))
            if not data:
                return None
            try:
                credits = data.get("credits", {})
                cast_list = credits.get("cast", [])[:5]
                crew_list = credits.get("crew", [])
                fullcast = [c["name"] for c in cast_list if "name" in c]
                release_date: Optional[str] = data.get("release_date")
                return {
                    "movieId": int(row.movieId),
                    "title": data.get("title", ""),
                    "year": release_date.split("-")[0] if release_date else None,
                    "cast": fullcast,
                    "main_actor": fullcast[0] if fullcast else None,
                    "director": next(
                        (c["name"] for c in crew_list if c.get("job") == "Director"), None
                    ),
                    "rating": float(data.get("vote_average", 0.0)),
                    "runtime": int(data.get("runtime") or 0),
                    "keywords": [
                        kw["name"]
                        for kw in data.get("keywords", {}).get("keywords", [])
                        if kw.get("name")
                    ],
                    "vote_count": int(data.get("vote_count") or 0),
                }
            except Exception as e:
                logger.error(f"Error parsing data for movieId {getattr(row, 'movieId', '?')}: {e}")
                return None

    async def fetch_all(
        self,
        missing_links: pd.DataFrame,
        cache_path: Path,
        cached_df: pd.DataFrame,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        total = len(missing_links)
        logger.info(f"Found {total} missing movies with valid TMDB IDs. Fetching...")

        rows = list(missing_links.itertuples(index=False))
        new_batches: List[pd.DataFrame] = []
        success_count = 0

        connector = aiohttp.TCPConnector(limit=self.max_concurrent, ttl_dns_cache=300)
        try:
            async with aiohttp.ClientSession(connector=connector, connector_owner=False) as session:
                for batch_start in range(0, total, batch_size):
                    batch_rows = rows[batch_start : batch_start + batch_size]
                    batch_num = batch_start // batch_size + 1

                    tasks = [self.process_row(session, row) for row in batch_rows]
                    results: List[Optional[Dict[str, Any]]] = await tqdm_asyncio.gather(
                        *tasks, desc=f"Downloading batch {batch_num}"
                    )
                    del tasks

                    valid = [r for r in results if r is not None]
                    del results

                    if not valid:
                        continue

                    batch_df = pd.DataFrame(valid)
                    del valid
                    new_batches.append(batch_df)
                    success_count += len(batch_df)

                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint = pd.concat([cached_df, *new_batches], ignore_index=True)
                        checkpoint.to_parquet(cache_path, index=False)
                        del checkpoint
                    except Exception as e:
                        logger.error(f"Intermediate cache saving failed: {e}")
        finally:
            await connector.close()

        logger.info(
            f"DOWNLOAD SUMMARY: Requested: {total} | Saved: {success_count} | Failed: {total - success_count}"
        )

        if new_batches:
            return pd.concat([cached_df, *new_batches], ignore_index=True)
        return cached_df


class MovieLensDataLoader:
    def __init__(
        self,
        data_path: str = "ml-latest-small",
        cache_file: str = "data/processed/movie_metadata.parquet",
    ) -> None:
        self.data_path = Path(MOVIE_LENS_RAW_PATH / data_path)
        self.cache_path = Path(cache_file)
        self.movies_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.tags_df: Optional[pd.DataFrame] = None
        self.links_df: Optional[pd.DataFrame] = None
        self.genre_matrix: Optional[np.ndarray] = None
        self.movie_data: List[Dict[str, Any]] = []
        self._movie_id_to_index: Optional[Dict[int, int]] = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("Loading MovieLens dataset...")
        try:
            self.movies_df = pd.read_csv(self.data_path / "movies.csv", engine="pyarrow")
            self.ratings_df = pd.read_csv(
                self.data_path / "ratings.csv",
                engine="pyarrow",
                dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32, "timestamp": np.int64},
            )
            self.tags_df = pd.read_csv(
                self.data_path / "tags.csv",
                engine="pyarrow",
                dtype={"userId": np.int32, "movieId": np.int32, "tag": "string", "timestamp": np.int64},
            )
            self.links_df = pd.read_csv(
                self.data_path / "links.csv",
                engine="pyarrow",
                dtype={"movieId": np.int32, "imdbId": np.int32, "tmdbId": pd.Int32Dtype()},
            )
            self._movie_id_to_index = None
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
        genre_dummies.columns = pd.Index([f"genre_{col.lower()}" for col in genre_dummies.columns])
        self.genre_matrix = genre_dummies.to_numpy()
        return genre_dummies

    def preprocess_tags(self) -> csr_matrix:
        if self.tags_df is None:
            raise ValueError("Tags data not loaded. Call load_data() first.")
        user_cat = pd.Categorical(self.tags_df["userId"])
        tag_cat = pd.Categorical(self.tags_df["tag"])
        data = np.ones(len(self.tags_df), dtype=np.float32)
        return coo_matrix(
            (data, (user_cat.codes, tag_cat.codes)),
            shape=(len(user_cat.categories), len(tag_cat.categories)),
        ).tocsr()

    def get_user_item_matrix(self) -> Tuple[csr_matrix, List[int], List[int]]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        user_cat = pd.Categorical(self.ratings_df["userId"])
        movie_cat = pd.Categorical(self.ratings_df["movieId"])
        matrix = coo_matrix(
            (self.ratings_df["rating"].to_numpy(), (user_cat.codes, movie_cat.codes)),
            shape=(len(user_cat.categories), len(movie_cat.categories)),
        ).tocsr()
        return matrix, list(user_cat.categories), list(movie_cat.categories)

    def get_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        rng = np.random.default_rng(random_state)
        users = self.ratings_df["userId"].unique()
        test_users = rng.choice(users, size=int(len(users) * test_size), replace=False)
        mask = self.ratings_df["userId"].isin(test_users)
        return self.ratings_df[~mask].copy(), self.ratings_df[mask].copy()

    async def letterboxd_data_async(self, max_concurrent_requests: int = 100) -> None:
        if self.links_df is None:
            raise ValueError("Links data not loaded. Call load_data() first.")

        cached_df = pd.DataFrame()
        existing_ids: set = set()

        if self.cache_path.exists():
            logger.info(f"Loading existing data from cache: {self.cache_path}")
            cached_df = pd.read_parquet(self.cache_path)
            if "movieId" in cached_df.columns:
                existing_ids = set(cached_df["movieId"].dropna().astype(np.int32))

        missing_links = self.links_df[
            (~self.links_df["movieId"].isin(existing_ids)) & self.links_df["tmdbId"].notna()
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
        if self._movie_id_to_index is not None:
            return self._movie_id_to_index
        if self.movies_df is None:
            self.movies_df = pd.read_csv(self.data_path / "movies.csv", engine="pyarrow")
        self._movie_id_to_index = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["movieId"])
        }
        return self._movie_id_to_index