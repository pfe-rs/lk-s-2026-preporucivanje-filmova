import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time
import psutil
import os
from loguru import logger
from tqdm import tqdm
from src.utils.logger import LoggingConfig, StepLogger

try:
    import cudf
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse import hstack as cp_hstack
    from cupyx.scipy.sparse import vstack as cp_vstack
    from cupyx.scipy.sparse import save_npz as cp_save_npz
    from cupyx.scipy.sparse import load_npz as cp_load_npz
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    from cuml.preprocessing import MinMaxScaler as cuMinMaxScaler
    from cuml.preprocessing import QuantileTransformer as cuQuantileTransformer
    from cuml.preprocessing import normalize as cu_normalize
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

from scipy.sparse import hstack, csr_matrix, vstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer as skTfidfVectorizer
from sklearn.preprocessing import MinMaxScaler as skMinMaxScaler
from sklearn.preprocessing import QuantileTransformer as skQuantileTransformer
from sklearn.preprocessing import normalize as sk_normalize
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ContentBasedConfig:
    main_actor_weight: float = 0.04
    director_weight: float = 0.30
    cast_weight: float = 0.10
    keywords_weight: float = 0.60
    genre_weight: float = 0.15
    numerical_weight: float = 0.30
    tfidf_sublinear_tf: bool = True
    tfidf_max_features: Optional[int] = None
    runtime_clip_percentiles: tuple = (0.01, 0.99)
    year_n_quantiles: int = 1000
    similarity_threshold: float = 0.1
    top_k_default: int = 10
    artifacts_dir: str = "data/processed/artifacts"
    fillna_strategy: str = "median"
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class ContentBasedRecommender:
    def __init__(self, config: Optional[ContentBasedConfig] = None):
        self.config = config or ContentBasedConfig()
        self.use_gpu = GPU_AVAILABLE

        self.tfidf_keywords = None
        self.tfidf_main_actor = None
        self.tfidf_director = None
        self.tfidf_cast = None
        self.scaler_runtime = None
        self.scaler_year = None
        self.scaler_main_actor_rating = None
        self.scaler_director_rating = None
        self.qt_year = None
        self.similarity_matrix = None
        self.feature_matrix = None
        self.feature_matrix_norm = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.all_movie_ids = set()
        self.user_profiles = {}
        self.movie_id_to_title = {}
        self.movie_popularity = {}
        self.movie_vote_counts = {}
        self.is_fitted = False
        self.step_logger = StepLogger(self.config.logging)

    @staticmethod
    def _clean_text_series_cpu(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.lower().str.replace(' ', '', regex=False)

    @staticmethod
    def _clean_text_series_gpu(series: cudf.Series) -> cudf.Series:
        return series.fillna("").astype(str).str.lower().str.replace(' ', '', regex=False)

    def _clean_text_series(self, series):
        if self.use_gpu:
            return self._clean_text_series_gpu(series)
        return self._clean_text_series_cpu(series)

    @staticmethod
    def _weight_cast_members_fast(cast_list: Any, max_weight: int = 3) -> str:
        if not isinstance(cast_list, (list, tuple, np.ndarray)):
            return str(cast_list).lower().replace(' ', '') if cast_list else ""
        weighted_cast = []
        for i, actor in enumerate(cast_list):
            weight = max(1, max_weight - i)
            actor_clean = str(actor).lower().replace(' ', '')
            if actor_clean:
                weighted_cast.extend([actor_clean] * weight)
        return ' '.join(weighted_cast)

    def _preprocess_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.use_gpu:
            return self._preprocess_numerical_features_gpu(df)
        return self._preprocess_numerical_features_cpu(df)

    def _preprocess_numerical_features_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        runtime_vals = df["runtime"].replace(0, np.nan)
        median_runtime = runtime_vals.median()
        df["runtime"] = runtime_vals.fillna(median_runtime)
        bounds = df["runtime"].quantile(list(self.config.runtime_clip_percentiles)).values
        df["runtime"] = df["runtime"].clip(bounds[0], bounds[1])
        self.scaler_runtime = skMinMaxScaler()
        df["runtime"] = self.scaler_runtime.fit_transform(df[["runtime"]])
        median_year = df["year"].astype(float).median()
        df["year"] = df["year"].fillna(median_year)
        self.qt_year = skQuantileTransformer(
            output_distribution="normal",
            n_quantiles=self.config.year_n_quantiles,
            random_state=42
        )
        year_transformed = self.qt_year.fit_transform(df[["year"]])
        self.scaler_year = skMinMaxScaler()
        df["year"] = self.scaler_year.fit_transform(year_transformed)
        median_votes = df["vote_count"].astype(float).median()
        df["vote_count"] = df['vote_count'].fillna(median_votes)
        qt = skQuantileTransformer(output_distribution='uniform')
        df["vote_count"] = qt.fit_transform(df[["vote_count"]].astype(float))
        return df

    def _preprocess_numerical_features_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        gdf = cudf.DataFrame.from_pandas(df)
        runtime_vals = gdf["runtime"].replace(0, np.nan)
        median_runtime = runtime_vals.median()
        gdf["runtime"] = runtime_vals.fillna(median_runtime)
        bounds = gdf["runtime"].quantile(list(self.config.runtime_clip_percentiles)).values
        gdf["runtime"] = gdf["runtime"].clip(bounds[0], bounds[1])
        self.scaler_runtime = cuMinMaxScaler()
        gdf["runtime"] = self.scaler_runtime.fit_transform(gdf[["runtime"]].astype(np.float32))
        median_year = gdf["year"].astype(float).median()
        gdf["year"] = gdf["year"].fillna(median_year)
        self.qt_year = cuQuantileTransformer(
            output_distribution="normal",
            n_quantiles=self.config.year_n_quantiles,
            random_state=42
        )
        year_transformed = self.qt_year.fit_transform(gdf[["year"]].astype(np.float32))
        self.scaler_year = cuMinMaxScaler()
        gdf["year"] = self.scaler_year.fit_transform(year_transformed)
        median_votes = gdf["vote_count"].astype(float).median()
        gdf["vote_count"] = gdf['vote_count'].fillna(median_votes)
        qt = cuQuantileTransformer(output_distribution='uniform')
        gdf["vote_count"] = qt.fit_transform(gdf[["vote_count"]].astype(np.float32))
        return gdf.to_pandas()

    def _preprocess_actor_director_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.use_gpu:
            return self._preprocess_actor_director_ratings_gpu(df)
        return self._preprocess_actor_director_ratings_cpu(df)

    def _preprocess_actor_director_ratings_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'rating' in df.columns:
            df['main_actor_rating'] = df.groupby('main_actor')['rating'].transform('mean')
            median_rating_actor = df["main_actor_rating"].replace(0, np.nan).median()
            df["main_actor_rating"] = df["main_actor_rating"].replace({0: median_rating_actor, np.nan: median_rating_actor})
            df["main_actor_rating"] = np.log1p(df["main_actor_rating"])
            self.scaler_main_actor_rating = skMinMaxScaler()
            df["main_actor_rating"] = self.scaler_main_actor_rating.fit_transform(df[["main_actor_rating"]])
        else:
            df['main_actor_rating'] = 0.0
        if 'rating' in df.columns:
            df['director_rating'] = df.groupby('director')['rating'].transform('mean')
            median_rating_director = df["director_rating"].replace(0, np.nan).median()
            df["director_rating"] = df["director_rating"].replace({0: median_rating_director, np.nan: median_rating_director})
            df["director_rating"] = np.log1p(df["director_rating"])
            self.scaler_director_rating = skMinMaxScaler()
            df["director_rating"] = self.scaler_director_rating.fit_transform(df[["director_rating"]])
        else:
            df['director_rating'] = 0.0
        return df

    def _preprocess_actor_director_ratings_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        gdf = cudf.DataFrame.from_pandas(df)
        if 'rating' in gdf.columns:
            gdf['main_actor_rating'] = gdf.groupby('main_actor')['rating'].transform('mean')
            median_rating_actor = gdf["main_actor_rating"].replace(0, np.nan).median()
            gdf["main_actor_rating"] = gdf["main_actor_rating"].fillna(median_rating_actor).replace(0, median_rating_actor)
            gdf["main_actor_rating"] = np.log1p(gdf["main_actor_rating"].astype(np.float32))
            self.scaler_main_actor_rating = cuMinMaxScaler()
            gdf["main_actor_rating"] = self.scaler_main_actor_rating.fit_transform(gdf[["main_actor_rating"]].astype(np.float32))
        else:
            gdf['main_actor_rating'] = 0.0
        if 'rating' in gdf.columns:
            gdf['director_rating'] = gdf.groupby('director')['rating'].transform('mean')
            median_rating_director = gdf["director_rating"].replace(0, np.nan).median()
            gdf["director_rating"] = gdf["director_rating"].fillna(median_rating_director).replace(0, median_rating_director)
            gdf["director_rating"] = np.log1p(gdf["director_rating"].astype(np.float32))
            self.scaler_director_rating = cuMinMaxScaler()
            gdf["director_rating"] = self.scaler_director_rating.fit_transform(gdf[["director_rating"]].astype(np.float32))
        else:
            gdf['director_rating'] = 0.0
        return gdf.to_pandas()

    def _fit_transform_tfidf(self, data, weight: float, use_gpu: bool = False):
        if use_gpu:
            vectorizer = cuTfidfVectorizer(
                sublinear_tf=self.config.tfidf_sublinear_tf,
                max_features=self.config.tfidf_max_features
            )
            if isinstance(data, cudf.Series):
                matrix = vectorizer.fit_transform(data.astype(str))
            else:
                matrix = vectorizer.fit_transform(cudf.Series(data).astype(str))
            return vectorizer, matrix * weight
        else:
            vectorizer = skTfidfVectorizer(
                sublinear_tf=self.config.tfidf_sublinear_tf,
                max_features=self.config.tfidf_max_features
            )
            matrix = vectorizer.fit_transform(data)
            return vectorizer, matrix * weight

    def _build_feature_matrix(self, df: pd.DataFrame) -> Any:
        if self.use_gpu:
            return self._build_feature_matrix_gpu(df)
        return self._build_feature_matrix_cpu(df)

    def _build_feature_matrix_cpu(self, df: pd.DataFrame) -> csr_matrix:
        main_actor_clean = self._clean_text_series_cpu(df["main_actor"])
        director_clean = self._clean_text_series_cpu(df["director"])
        cast_weighted = df["cast"].apply(self._weight_cast_members_fast)
        keywords_cleaned = df["keywords"].apply(
            lambda x: ' '.join([str(i).lower().replace(' ', '') for i in x if i])
            if isinstance(x, (list, tuple, np.ndarray)) else ""
        )
        with ThreadPoolExecutor(max_workers=4) as executor:
            f_actor = executor.submit(self._fit_transform_tfidf, main_actor_clean, self.config.main_actor_weight, False)
            f_director = executor.submit(self._fit_transform_tfidf, director_clean, self.config.director_weight, False)
            f_cast = executor.submit(self._fit_transform_tfidf, cast_weighted, self.config.cast_weight, False)
            f_keywords = executor.submit(self._fit_transform_tfidf, keywords_cleaned, self.config.keywords_weight, False)
            self.tfidf_main_actor, main_actor_tfidf = f_actor.result()
            self.tfidf_director, director_tfidf = f_director.result()
            self.tfidf_cast, cast_tfidf = f_cast.result()
            self.tfidf_keywords, keywords_tfidf = f_keywords.result()
        numerical_features = df[['runtime', 'year', 'main_actor_rating', 'director_rating', 'vote_count']].values
        numerical_matrix = csr_matrix(numerical_features) * self.config.numerical_weight
        genre_cols = [col for col in df.columns if col.startswith("genre_")]
        if genre_cols:
            genre_matrix = csr_matrix(df[genre_cols].values.astype(float)) * self.config.genre_weight
        else:
            genre_matrix = csr_matrix((df.shape[0], 0))
        combined = hstack([main_actor_tfidf, director_tfidf, cast_tfidf, keywords_tfidf, genre_matrix, numerical_matrix]).tocsr()
        return combined

    def _build_feature_matrix_gpu(self, df: pd.DataFrame) -> cp_csr_matrix:
        gdf = cudf.DataFrame.from_pandas(df)
        main_actor_clean = self._clean_text_series_gpu(gdf["main_actor"])
        director_clean = self._clean_text_series_gpu(gdf["director"])
        cast_weighted = gdf["cast"].apply(lambda x: self._weight_cast_members_fast(x))
        keywords_cleaned = gdf["keywords"].apply(
            lambda x: ' '.join([str(i).lower().replace(' ', '') for i in x if i]) if isinstance(x, (list, tuple, np.ndarray)) else ""
        )
        self.tfidf_main_actor, main_actor_tfidf = self._fit_transform_tfidf(main_actor_clean, self.config.main_actor_weight, True)
        self.tfidf_director, director_tfidf = self._fit_transform_tfidf(director_clean, self.config.director_weight, True)
        self.tfidf_cast, cast_tfidf = self._fit_transform_tfidf(cast_weighted, self.config.cast_weight, True)
        self.tfidf_keywords, keywords_tfidf = self._fit_transform_tfidf(keywords_cleaned, self.config.keywords_weight, True)
        num_cols = ['runtime', 'year', 'main_actor_rating', 'director_rating', 'vote_count']
        numerical = gdf[num_cols].astype(np.float32).values
        numerical_matrix = cp_csr_matrix(cp.asarray(numerical)) * self.config.numerical_weight
        genre_cols = [col for col in df.columns if col.startswith("genre_")]
        if genre_cols:
            genre_mat = cp_csr_matrix(cp.asarray(gdf[genre_cols].astype(np.float32).values)) * self.config.genre_weight
        else:
            genre_mat = cp_csr_matrix((gdf.shape[0], 0))
        combined = cp_hstack([main_actor_tfidf, director_tfidf, cast_tfidf, keywords_tfidf, genre_mat, numerical_matrix]).tocsr()
        return combined

    def _compute_similarity_matrix(self, features_norm, batch_size: int = 5000):
        if self.use_gpu:
            return self._compute_similarity_matrix_gpu(features_norm, batch_size)
        return self._compute_similarity_matrix_cpu(features_norm, batch_size)

    def _compute_similarity_matrix_cpu(self, features_norm: csr_matrix, batch_size: int = 5000) -> csr_matrix:
        n_samples = features_norm.shape[0]
        results = []
        total_batches = (n_samples + batch_size - 1) // batch_size
        iterator = range(0, n_samples, batch_size)
        if self.config.logging.show_progress_bars:
            iterator = tqdm(iterator, total=total_batches, desc="Similarity batches", unit="batch")
        for i in iterator:
            end = min(i + batch_size, n_samples)
            batch = features_norm[i:end]
            sim_batch = batch.dot(features_norm.T)
            sim_batch.data[sim_batch.data < self.config.similarity_threshold] = 0
            sim_batch.eliminate_zeros()
            results.append(sim_batch)
        return vstack(results)

    def _compute_similarity_matrix_gpu(self, features_norm: cp_csr_matrix, batch_size: int = 5000) -> cp_csr_matrix:
        n_samples = features_norm.shape[0]
        results = []
        total_batches = (n_samples + batch_size - 1) // batch_size
        iterator = range(0, n_samples, batch_size)
        if self.config.logging.show_progress_bars:
            iterator = tqdm(iterator, total=total_batches, desc="Similarity batches", unit="batch")
        for i in iterator:
            end = min(i + batch_size, n_samples)
            batch = features_norm[i:end]
            sim_batch = batch.dot(features_norm.T)
            sim_batch.data[sim_batch.data < self.config.similarity_threshold] = 0
            sim_batch.eliminate_zeros()
            results.append(sim_batch)
        return cp_vstack(results)

    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> 'ContentBasedRecommender':
        total_start = time.perf_counter()
        total_cpu = time.process_time()
        logger.info("Starting model fitting process")
        required_cols = ['movieId', 'main_actor', 'director', 'cast', 'runtime', 'year', 'keywords']
        missing = [col for col in required_cols if col not in movies_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.step_logger.log_step("Validation", total_start, total_cpu)
        df_processed = self._preprocess_numerical_features(movies_df.copy())
        df_processed = self._preprocess_actor_director_ratings(df_processed)
        self.step_logger.log_step("Preprocessing", total_start, total_cpu)
        features = self._build_feature_matrix(df_processed)
        self.feature_matrix = features
        if self.use_gpu:
            self.feature_matrix_norm = cu_normalize(features, norm='l2', axis=1)
        else:
            self.feature_matrix_norm = sk_normalize(features, norm='l2', axis=1)
        self.step_logger.log_step("Feature Matrix", total_start, total_cpu)
        self.similarity_matrix = self._compute_similarity_matrix(self.feature_matrix_norm)
        self.step_logger.log_step("Similarity Matrix", total_start, total_cpu)
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(df_processed['movieId'].values)}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        self.all_movie_ids = set(self.movie_id_to_idx.keys())
        self.step_logger.log_step("Index Mappings", total_start, total_cpu)
        if ratings_df is not None:
            self._build_user_profiles(ratings_df)
            if 'rating' in ratings_df.columns:
                self.movie_popularity = dict(ratings_df.groupby('movieId')['rating'].mean())
        else:
            if 'rating' in movies_df.columns:
                self.movie_popularity = dict(zip(movies_df['movieId'], movies_df['rating']))
        self.movie_vote_counts = dict(zip(df_processed['movieId'], df_processed['vote_count']))
        if 'title' in movies_df.columns:
            self.movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
        self.step_logger.log_step("User Profiles & Metadata", total_start, total_cpu)
        self.is_fitted = True
        if self.config.logging.log_data_shapes:
            fm = self.feature_matrix
            sm = self.similarity_matrix
            logger.info(f"Feature matrix: shape={fm.shape}, nnz={fm.nnz}")
            logger.info(f"Similarity matrix: shape={sm.shape}, nnz={sm.nnz}")
        total_wall = time.perf_counter() - total_start
        total_cpu_elapsed = time.process_time() - total_cpu
        mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
        logger.info(f"Fitting complete | wall={total_wall:.2f}s cpu={total_cpu_elapsed:.2f}s rss={mem_final:.1f}MB")
        return self

    def _build_user_profiles(self, ratings_df: pd.DataFrame):
        if self.use_gpu:
            self._build_user_profiles_gpu(ratings_df)
        else:
            self._build_user_profiles_cpu(ratings_df)

    def _build_user_profiles_cpu(self, ratings_df: pd.DataFrame):
        good = ratings_df[ratings_df['rating'] >= 4.0]
        valid = good[good['movieId'].isin(self.all_movie_ids)].copy()
        if valid.empty:
            self.user_profiles = {}
            return
        unique_users = valid['userId'].unique()
        user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        row_ind = valid['userId'].map(user_id_to_idx).values
        col_ind = valid['movieId'].map(self.movie_id_to_idx).values
        data = valid['rating'].values.astype(np.float32)
        user_item = csr_matrix((data, (row_ind, col_ind)), shape=(len(unique_users), len(self.movie_id_to_idx)))
        profiles_sparse = user_item.dot(self.feature_matrix)
        profiles_norm = sk_normalize(profiles_sparse, norm='l2', axis=1)
        self.user_profiles = {uid: profiles_norm[idx] for uid, idx in user_id_to_idx.items()}

    def _build_user_profiles_gpu(self, ratings_df: pd.DataFrame):
        good = ratings_df[ratings_df['rating'] >= 4.0]
        valid = good[good['movieId'].isin(self.all_movie_ids)].copy()
        if valid.empty:
            self.user_profiles = {}
            return
        unique_users = valid['userId'].unique()
        user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        row_ind = cp.asarray(valid['userId'].map(user_id_to_idx).values)
        col_ind = cp.asarray(valid['movieId'].map(self.movie_id_to_idx).values)
        data = cp.asarray(valid['rating'].values.astype(np.float32))
        user_item = cp_csr_matrix((data, (row_ind, col_ind)), shape=(len(unique_users), len(self.movie_id_to_idx)))
        fm = self.feature_matrix if isinstance(self.feature_matrix, cp_csr_matrix) else cp_csr_matrix(self.feature_matrix)
        profiles_sparse = user_item.dot(fm)
        profiles_norm = cu_normalize(profiles_sparse, norm='l2', axis=1)
        self.user_profiles = {uid: profiles_norm[idx] for uid, idx in user_id_to_idx.items()}

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if user_id not in self.user_profiles:
            return self._predict_popularity_scores(item_ids)
        profile = self.user_profiles[user_id]
        if self.use_gpu:
            return self._predict_scores_gpu(profile, item_ids)
        return self._predict_scores_cpu(profile, item_ids)

    def _predict_scores_cpu(self, profile, item_ids: List[int]) -> List[float]:
        valid = [(i, mid) for i, mid in enumerate(item_ids) if mid in self.movie_id_to_idx]
        scores = np.zeros(len(item_ids), dtype=np.float32)
        if valid:
            cand_indices = np.array([self.movie_id_to_idx[mid] for _, mid in valid])
            cand_vectors = self.feature_matrix_norm[cand_indices]
            dots = cand_vectors.dot(profile.T).toarray().flatten()
            for i, (orig_idx, mid) in enumerate(valid):
                pure = dots[i]
                pop = self.movie_vote_counts.get(mid, 0.0)
                soft_pop = 0.35 + (0.65 * pop)
                scores[orig_idx] = pure * soft_pop
        return scores.tolist()

    def _predict_scores_gpu(self, profile, item_ids: List[int]) -> List[float]:
        valid = [(i, mid) for i, mid in enumerate(item_ids) if mid in self.movie_id_to_idx]
        scores = np.zeros(len(item_ids), dtype=np.float32)
        if valid:
            cand_indices = cp.array([self.movie_id_to_idx[mid] for _, mid in valid])
            fm_norm = self.feature_matrix_norm if isinstance(self.feature_matrix_norm, cp_csr_matrix) else cp_csr_matrix(self.feature_matrix_norm)
            cand_vectors = fm_norm[cand_indices]
            if isinstance(profile, cp_csr_matrix):
                profile_vec = profile
            else:
                profile_vec = cp_csr_matrix(profile)
            dots = cand_vectors.dot(profile_vec.T).toarray().flatten()
            cpu_dots = cp.asnumpy(dots)
            for i, (orig_idx, mid) in enumerate(valid):
                pure = cpu_dots[i]
                pop = self.movie_vote_counts.get(mid, 0.0)
                soft_pop = 0.35 + (0.65 * pop)
                scores[orig_idx] = pure * soft_pop
        return scores.tolist()

    def _predict_popularity_scores(self, item_ids: List[int]) -> List[float]:
        scores = [self.movie_popularity.get(mid, 0.0) for mid in item_ids]
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s / max_score for s in scores]
        return scores

    def get_top_k_recommendations(self, user_id: int, watched_items: set, k: int = None) -> List[int]:
        if k is None:
            k = self.config.top_k_default
        candidate_ids = list(self.all_movie_ids - set(watched_items))
        if not candidate_ids:
            return []
        scores = self.predict_scores(user_id, candidate_ids)
        scored_items = list(zip(scores, candidate_ids))
        scored_items.sort(reverse=True, key=lambda x: x[0])
        return [mid for score, mid in scored_items[:k]]

    def get_top_k_with_titles(self, user_id: int, watched_items: set, k: int = None) -> List[Dict[str, Any]]:
        if k is None:
            k = self.config.top_k_default
        rec_ids = self.get_top_k_recommendations(user_id, watched_items, k)
        scores = self.predict_scores(user_id, rec_ids)
        results = []
        for mid, score in zip(rec_ids, scores):
            title = self.movie_id_to_title.get(mid, 'Unknown Title')
            results.append({'movieId': mid, 'title': title, 'score': round(float(score), 4)})
        return results

    def show_user_profile_and_recommendations(
        self, user_id: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
        k: int = 10, top_rated_count: int = 5, reasons_count: int = 3
    ) -> None:
        movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
        print("=" * 80)
        print(f"USER PROFILE: {user_id}")
        print("=" * 80)
        user_history = ratings_df[ratings_df['userId'] == user_id]
        if user_history.empty:
            print(f"User {user_id} not found in the database.")
            return
        print(f"\nGeneral Statistics:")
        print(f"   - Total ratings: {len(user_history)}")
        print(f"   - Average rating: {user_history['rating'].mean():.2f} / 5.0")
        print(f"   - Unique movies rated: {user_history['movieId'].nunique()}")
        high_rated = user_history[user_history['rating'] >= 4.0].sort_values('rating', ascending=False)
        if not high_rated.empty:
            limit = min(top_rated_count, len(high_rated))
            print(f"\nTOP {limit} FAVORITE MOVIES (Rating >= 4.0):")
            print("-" * 80)
            for i, (_, row) in enumerate(high_rated.head(limit).iterrows(), 1):
                movie_id = row['movieId']
                title = movie_titles.get(movie_id, f"Unknown (ID: {movie_id})")
                rating = row['rating']
                print(f"{i:2}. {title:<50} [Rating: {rating}/5.0]")
        else:
            print("\nNo highly rated movies (>= 4.0) found for this user.")
        print("\n" + "=" * 80)
        print(f"PERSONALIZED RECOMMENDATIONS FOR USER {user_id}")
        print("=" * 80)
        watched_items = set(user_history['movieId'].values)
        liked_items = set(user_history[user_history['rating'] >= 4.0]['movieId'].values)
        recommendations = self.get_top_k_with_titles(user_id, watched_items, k)
        if not recommendations:
            print("No recommendations could be generated.")
            return
        for i, rec in enumerate(recommendations, 1):
            rec_id = rec['movieId']
            rec_title = rec['title']
            rec_score = rec['score']
            print(f"\n{i:2}. {rec_title}")
            print(f"    Predicted Relevance Score: {rec_score:.4f}")
            reasons = self.explain_recommendation(rec_id, liked_items, top_n_reasons=reasons_count)
            if reasons:
                print(f"    Recommended because you liked:")
                for reason in reasons:
                    reason_id = reason['movie_id']
                    reason_title = movie_titles.get(reason_id, f"Unknown (ID: {reason_id})")
                    similarity = reason['similarity']
                    user_rating = user_history[user_history['movieId'] == reason_id]['rating'].values
                    rating_str = f" (You rated: {user_rating[0]}/5.0)" if len(user_rating) > 0 else ""
                    print(f"        - {reason_title}{rating_str} [Similarity: {similarity:.4f}]")
            else:
                print("    (No specific reasons found in your watch history)")
        print("\n" + "=" * 80)

    def save_artifacts(self, similarity_path=None, mapping_path=None, preprocessors_path=None):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving artifacts.")
        start = time.perf_counter()
        cpu_start = time.process_time()
        artifacts_dir = Path(self.config.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npz")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")

        if self.use_gpu:
            sm_cpu = self.similarity_matrix.get() if isinstance(self.similarity_matrix, cp_csr_matrix) else self.similarity_matrix
            save_npz(similarity_path, sm_cpu)
        else:
            save_npz(similarity_path, self.similarity_matrix)

        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id,
                'all_movie_ids': self.all_movie_ids
            }, f)

        fm_to_save = self.feature_matrix.get() if self.use_gpu and isinstance(self.feature_matrix, cp_csr_matrix) else self.feature_matrix
        preprocessors = {
            'tfidf_main_actor': self.tfidf_main_actor,
            'tfidf_director': self.tfidf_director,
            'tfidf_cast': self.tfidf_cast,
            'scaler_runtime': self.scaler_runtime,
            'scaler_year': self.scaler_year,
            'scaler_main_actor_rating': self.scaler_main_actor_rating,
            'scaler_director_rating': self.scaler_director_rating,
            'qt_year': self.qt_year,
            'config': self.config,
            'feature_matrix': fm_to_save,
            'user_profiles': self.user_profiles
        }
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)

        extra = {
            "similarity_file": str(Path(similarity_path).name),
            "mapping_file": str(Path(mapping_path).name),
            "preprocessors_file": str(Path(preprocessors_path).name)
        }
        self.step_logger.log_step("SaveArtifacts", start, cpu_start, extra)

    def load_artifacts(self, similarity_path=None, mapping_path=None, preprocessors_path=None):
        start = time.perf_counter()
        cpu_start = time.process_time()
        artifacts_dir = Path(self.config.artifacts_dir)
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npz")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")

        if similarity_path.endswith('.npy'):
            dense_matrix = np.load(similarity_path, allow_pickle=True)
            self.similarity_matrix = csr_matrix(dense_matrix)
        else:
            self.similarity_matrix = load_npz(similarity_path)

        with open(mapping_path, 'rb') as f:
            mappings = pickle.load(f)
            self.movie_id_to_idx = mappings['movie_id_to_idx']
            self.idx_to_movie_id = mappings['idx_to_movie_id']
            self.all_movie_ids = mappings['all_movie_ids']

        with open(preprocessors_path, 'rb') as f:
            preprocessors = pickle.load(f)
            self.tfidf_main_actor = preprocessors['tfidf_main_actor']
            self.tfidf_director = preprocessors['tfidf_director']
            self.tfidf_cast = preprocessors['tfidf_cast']
            self.scaler_runtime = preprocessors['scaler_runtime']
            self.scaler_year = preprocessors['scaler_year']
            self.scaler_main_actor_rating = preprocessors['scaler_main_actor_rating']
            self.scaler_director_rating = preprocessors['scaler_director_rating']
            self.qt_year = preprocessors['qt_year']
            self.config = preprocessors['config']
            self.feature_matrix = preprocessors.get('feature_matrix', None)
            self.user_profiles = preprocessors.get('user_profiles', {})

        if self.use_gpu:
            if isinstance(self.feature_matrix, np.ndarray) or isinstance(self.feature_matrix, csr_matrix):
                self.feature_matrix = cp_csr_matrix(self.feature_matrix)
            if isinstance(self.similarity_matrix, csr_matrix):
                self.similarity_matrix = cp_csr_matrix(self.similarity_matrix)
            if self.feature_matrix is not None:
                self.feature_matrix_norm = cu_normalize(self.feature_matrix, norm='l2', axis=1)
            self.user_profiles = {uid: cp_csr_matrix(prof) if isinstance(prof, csr_matrix) else prof
                                  for uid, prof in self.user_profiles.items()}
        else:
            if self.feature_matrix is not None:
                self.feature_matrix_norm = sk_normalize(self.feature_matrix, norm='l2', axis=1)

        self.is_fitted = True
        extra = {
            "similarity_file": str(Path(similarity_path).name),
            "mapping_file": str(Path(mapping_path).name),
            "preprocessors_file": str(Path(preprocessors_path).name)
        }
        self.step_logger.log_step("LoadArtifacts", start, cpu_start, extra)

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        if movie_id not in self.movie_id_to_idx:
            return []
        movie_idx = self.movie_id_to_idx[movie_id]
        if self.use_gpu:
            sm = self.similarity_matrix if isinstance(self.similarity_matrix, cp_csr_matrix) else cp_csr_matrix(self.similarity_matrix)
        else:
            sm = self.similarity_matrix
        row = sm[movie_idx]
        cols = row.indices
        if self.use_gpu:
            datas = cp.asnumpy(row.data)
            cols = cp.asnumpy(cols)
        else:
            datas = row.data
        sim_scores = list(zip(cols, datas))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in sim_scores:
            if idx == movie_idx:
                continue
            if len(results) >= top_n:
                break
            results.append({'movie_id': self.idx_to_movie_id[idx], 'similarity': float(score)})
        return results

    def explain_recommendation(self, movie_id: int, liked_items: set, top_n_reasons: int = 3) -> List[Dict[str, Any]]:
        if movie_id not in self.movie_id_to_idx:
            return []
        movie_idx = self.movie_id_to_idx[movie_id]
        liked_idxs = [self.movie_id_to_idx[lid] for lid in liked_items if lid in self.movie_id_to_idx]
        if not liked_idxs:
            return []
        if self.use_gpu:
            sm = self.similarity_matrix if isinstance(self.similarity_matrix, cp_csr_matrix) else cp_csr_matrix(self.similarity_matrix)
        else:
            sm = self.similarity_matrix
        row = sm[movie_idx]
        liked_idxs_arr = liked_idxs
        if self.use_gpu:
            sims = row[:, cp.array(liked_idxs_arr)].toarray()[0]
            sims = cp.asnumpy(sims)
        else:
            sims = row[:, liked_idxs_arr].toarray()[0]
        reasons = []
        for lid, sim in zip(liked_idxs, sims):
            if sim > 0:
                reasons.append({'movie_id': self.idx_to_movie_id[lid], 'similarity': float(sim)})
        reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return reasons[:top_n_reasons]