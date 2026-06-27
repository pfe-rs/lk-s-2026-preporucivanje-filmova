import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.sparse import hstack, csr_matrix, vstack, save_npz, load_npz
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import os
import time
import psutil

logger = logging.getLogger(__name__)

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

class ContentBasedRecommender:
    def __init__(self, config: Optional[ContentBasedConfig] = None):
        self.config = config or ContentBasedConfig()
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

    @staticmethod
    def _clean_text_series(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.lower().str.replace(' ', '', regex=False)

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
        df = df.copy()
        
        runtime_vals = df["runtime"].replace(0, np.nan)
        median_runtime = runtime_vals.median()
        df["runtime"] = runtime_vals.fillna(median_runtime)
        
        bounds = df["runtime"].quantile(list(self.config.runtime_clip_percentiles)).values
        df["runtime"] = df["runtime"].clip(bounds[0], bounds[1])
        
        self.scaler_runtime = MinMaxScaler()
        df["runtime"] = self.scaler_runtime.fit_transform(df[["runtime"]])
        
        median_year = df["year"].astype(float).median()
        df["year"] = df["year"].fillna(median_year)
        self.qt_year = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=self.config.year_n_quantiles,
            random_state=42
        )
        year_transformed = self.qt_year.fit_transform(df[["year"]])
        self.scaler_year = MinMaxScaler()
        df["year"] = self.scaler_year.fit_transform(year_transformed)
        
        median_votes = df["vote_count"].astype(float).median()
        df["vote_count"] = df['vote_count'].fillna(median_votes)
        qt = QuantileTransformer(output_distribution='uniform')
        df["vote_count"] = qt.fit_transform(df[["vote_count"]].astype(float))
        
        return df

    def _preprocess_actor_director_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if 'rating' in df.columns:
            df['main_actor_rating'] = df.groupby('main_actor')['rating'].transform('mean')
            median_rating_actor = df["main_actor_rating"].replace(0, np.nan).median()
            df["main_actor_rating"] = df["main_actor_rating"].replace({
                0: median_rating_actor,
                np.nan: median_rating_actor
            })
            df["main_actor_rating"] = np.log1p(df["main_actor_rating"])
            self.scaler_main_actor_rating = MinMaxScaler()
            df["main_actor_rating"] = self.scaler_main_actor_rating.fit_transform(
                df[["main_actor_rating"]]
            )
        else:
            df['main_actor_rating'] = 0.0
            
        if 'rating' in df.columns:
            df['director_rating'] = df.groupby('director')['rating'].transform('mean')
            median_rating_director = df["director_rating"].replace(0, np.nan).median()
            df["director_rating"] = df["director_rating"].replace({
                0: median_rating_director,
                np.nan: median_rating_director
            })
            df["director_rating"] = np.log1p(df["director_rating"])
            self.scaler_director_rating = MinMaxScaler()
            df["director_rating"] = self.scaler_director_rating.fit_transform(
                df[["director_rating"]]
            )
        else:
            df['director_rating'] = 0.0
            
        return df

    def _fit_transform_tfidf(self, data: pd.Series, weight: float) -> tuple:
        vectorizer = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features
        )
        matrix = vectorizer.fit_transform(data)
        return vectorizer, matrix * weight

    def _build_feature_matrix(self, df: pd.DataFrame) -> csr_matrix:
        main_actor_clean = self._clean_text_series(df["main_actor"])
        director_clean = self._clean_text_series(df["director"])
        cast_weighted = df["cast"].apply(self._weight_cast_members_fast)
        
        keywords_cleaned = df["keywords"].apply(
            lambda x: ' '.join([str(i).lower().replace(' ', '') for i in x if i]) 
            if isinstance(x, (list, tuple, np.ndarray)) else ""
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            f_actor = executor.submit(self._fit_transform_tfidf, main_actor_clean, self.config.main_actor_weight)
            f_director = executor.submit(self._fit_transform_tfidf, director_clean, self.config.director_weight)
            f_cast = executor.submit(self._fit_transform_tfidf, cast_weighted, self.config.cast_weight)
            f_keywords = executor.submit(self._fit_transform_tfidf, keywords_cleaned, self.config.keywords_weight)
            
            self.tfidf_main_actor, main_actor_tfidf = f_actor.result()
            self.tfidf_director, director_tfidf = f_director.result()
            self.tfidf_cast, cast_tfidf = f_cast.result()
            self.tfidf_keywords, keywords_tfidf = f_keywords.result()

        numerical_features = df[[
            'runtime', 
            'year', 
            'main_actor_rating',
            'director_rating',
            'vote_count'
        ]].values
        numerical_matrix = csr_matrix(numerical_features) * self.config.numerical_weight

        genre_cols = [col for col in df.columns if col.startswith("genre_")]
        if genre_cols:
            genre_matrix = csr_matrix(df[genre_cols].values.astype(float)) * self.config.genre_weight
        else:
            genre_matrix = csr_matrix((df.shape[0], 0))
        
        combined_features = hstack([
            main_actor_tfidf,
            director_tfidf,
            cast_tfidf,
            keywords_tfidf,
            genre_matrix,
            numerical_matrix
        ]).tocsr()
        
        return combined_features

    def _compute_similarity_matrix(self, features_norm: csr_matrix, batch_size: int = 5000) -> csr_matrix:
        n_samples = features_norm.shape[0]
        results = []
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch = features_norm[i:end]
            
            sim_batch = batch.dot(features_norm.T)
            
            sim_batch.data[sim_batch.data < self.config.similarity_threshold] = 0
            sim_batch.eliminate_zeros()
            results.append(sim_batch)
                
        return vstack(results)

    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> 'ContentBasedRecommender':
        """Fits the recommender model while logging execution time and memory usage."""
        
        # Helper function to track time and memory
        def _log_step(step_name: str, start_time: float) -> float:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            elapsed = time.time() - start_time
            logger.info(f"[{step_name}] Completed in {elapsed:.2f}s | Current Memory: {mem_mb:.2f} MB")
            return time.time() # Return new start time for the next step

        total_start = time.time()
        step_start = time.time()
        logger.info("Starting model fitting process...")

        # 1. Column Validation
        required_cols = ['movieId', 'main_actor', 'director', 'cast', 'runtime', 'year', 'keywords']
        missing_cols = [col for col in required_cols if col not in movies_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in movies_df: {missing_cols}")
        step_start = _log_step("Validation", step_start)
        
        # 2. Preprocessing
        logger.info("Preprocessing numerical and categorical features...")
        df_processed = self._preprocess_numerical_features(movies_df.copy())
        df_processed = self._preprocess_actor_director_ratings(df_processed)
        step_start = _log_step("Preprocessing", step_start)
        
        # 3. Building Feature Matrix
        logger.info("Building TF-IDF and combined feature matrices...")
        features_matrix = self._build_feature_matrix(df_processed)
        self.feature_matrix = features_matrix
        self.feature_matrix_norm = normalize(features_matrix, norm='l2', axis=1)
        step_start = _log_step("Feature Matrix Generation", step_start)
        
        # 4. Computing Similarity Matrix
        logger.info("Computing cosine similarity matrix in batches...")
        self.similarity_matrix = self._compute_similarity_matrix(self.feature_matrix_norm)
        step_start = _log_step("Similarity Matrix Computation", step_start)
        
        # 5. Building Index Mappings
        logger.info("Creating internal mapping dictionaries...")
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(df_processed['movieId'].values)
        }
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        self.all_movie_ids = set(self.movie_id_to_idx.keys())
        step_start = _log_step("Index Mappings", step_start)
        
        # 6. Building User Profiles & Popularity
        logger.info("Building user profiles and calculating movie popularity...")
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
        step_start = _log_step("User Profiles & Popularity", step_start)
        
        self.is_fitted = True
        
        # Final Summary
        total_elapsed = time.time() - total_start
        final_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        logger.info(f"=== Fitting Complete! ===")
        logger.info(f"Total Time: {total_elapsed:.2f}s | Final Memory Footprint: {final_mem:.2f} MB")
        
        return self

    def _build_user_profiles(self, ratings_df: pd.DataFrame):
        good_ratings_df = ratings_df[ratings_df['rating'] >= 4.0]
        valid_df = good_ratings_df[good_ratings_df['movieId'].isin(self.all_movie_ids)].copy()
        
        if valid_df.empty:
            self.user_profiles = {}
            return
            
        unique_users = valid_df['userId'].unique()
        user_id_to_idx = {uid: i for i, uid in enumerate(unique_users)}
        
        row_ind = valid_df['userId'].map(user_id_to_idx).values
        col_ind = valid_df['movieId'].map(self.movie_id_to_idx).values
        data = valid_df['rating'].values.astype(np.float32)
        
        user_item_matrix = csr_matrix(
            (data, (row_ind, col_ind)), 
            shape=(len(unique_users), len(self.movie_id_to_idx))
        )
        
        profiles_sparse = user_item_matrix.dot(self.feature_matrix)
        profiles_normalized = normalize(profiles_sparse, norm='l2', axis=1)
        
        self.user_profiles = {
            uid: profiles_normalized[idx] for uid, idx in user_id_to_idx.items()
        }

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        if user_id not in self.user_profiles:
            return self._predict_popularity_scores(item_ids)
        
        profile = self.user_profiles[user_id]
        
        valid_candidates = [
            (i, mid) for i, mid in enumerate(item_ids)
            if mid in self.movie_id_to_idx
        ]
        
        scores = np.zeros(len(item_ids), dtype=np.float32)
        
        if valid_candidates:
            cand_indices = np.array([self.movie_id_to_idx[mid] for _, mid in valid_candidates])
            cand_vectors_norm = self.feature_matrix_norm[cand_indices]
            
            dots = cand_vectors_norm.dot(profile.T).toarray().flatten()
            
            for i, (orig_idx, mid) in enumerate(valid_candidates):
                pure = dots[i]
                pop = self.movie_vote_counts.get(mid, 0.0)
                soft_popularity = 0.35 + (0.65 * pop)
                scores[orig_idx] = pure * soft_popularity
                
        return scores.tolist()

    def _predict_popularity_scores(self, item_ids: List[int]) -> List[float]:
        scores = []
        for mid in item_ids:
            score = self.movie_popularity.get(mid, 0.0)
            scores.append(score)
        
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
            results.append({
                'movieId': mid,
                'title': title,
                'score': round(float(score), 4)
            })
            
        return results

    def show_user_profile_and_recommendations(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        k: int = 10,
        top_rated_count: int = 5,
        reasons_count: int = 3
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

    def save_artifacts(
        self,
        similarity_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        preprocessors_path: Optional[str] = None
    ):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving artifacts.")
        
        artifacts_dir = Path(self.config.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npz")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")
        
        save_npz(similarity_path, self.similarity_matrix)
        
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'movie_id_to_idx': self.movie_id_to_idx,
                'idx_to_movie_id': self.idx_to_movie_id,
                'all_movie_ids': self.all_movie_ids
            }, f)
        
        with open(preprocessors_path, 'wb') as f:
            pickle.dump({
                'tfidf_main_actor': self.tfidf_main_actor,
                'tfidf_director': self.tfidf_director,
                'tfidf_cast': self.tfidf_cast,
                'scaler_runtime': self.scaler_runtime,
                'scaler_year': self.scaler_year,
                'scaler_main_actor_rating': self.scaler_main_actor_rating,
                'scaler_director_rating': self.scaler_director_rating,
                'qt_year': self.qt_year,
                'config': self.config,
                'feature_matrix': self.feature_matrix,
                'user_profiles': self.user_profiles
            }, f)

    def load_artifacts(
        self,
        similarity_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        preprocessors_path: Optional[str] = None
    ):
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
            if self.feature_matrix is not None:
                self.feature_matrix_norm = normalize(self.feature_matrix, norm='l2', axis=1)
            
            self.user_profiles = preprocessors.get('user_profiles', {})
        
        self.is_fitted = True

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        
        row = self.similarity_matrix[movie_idx]
        cols = row.indices
        datas = row.data
        
        sim_scores = list(zip(cols, datas))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sim_scores:
            if idx == movie_idx:
                continue
            if len(results) >= top_n:
                break
            results.append({
                'movie_id': self.idx_to_movie_id[idx],
                'similarity': float(score)
            })
        
        return results

    def explain_recommendation(
        self,
        movie_id: int,
        liked_items: set,
        top_n_reasons: int = 3
    ) -> List[Dict[str, Any]]:
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        liked_idxs = [self.movie_id_to_idx[lid] for lid in liked_items if lid in self.movie_id_to_idx]
        
        if not liked_idxs:
            return []
            
        row = self.similarity_matrix[movie_idx]
        sims = row[:, liked_idxs].toarray()[0]
        
        reasons = []
        for lid, sim in zip(liked_idxs, sims):
            if sim > 0:
                reasons.append({
                    'movie_id': self.idx_to_movie_id[lid],
                    'similarity': float(sim)
                })
                    
        reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return reasons[:top_n_reasons]