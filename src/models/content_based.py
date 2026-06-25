import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.all_movie_ids = set()
        
        self.user_profiles = {}
        self.movie_id_to_title = {} 
        self.movie_popularity = {}
        self.movie_vote_counts = {}
        
        self.is_fitted = False

    @staticmethod
    def _clean_text(x: Any) -> str:
        if isinstance(x, list):
            return [i.lower().replace(' ', '') for i in x if i]
        elif isinstance(x, str):
            return x.lower().replace(' ', '')
        return ""
    
    @staticmethod
    def _weight_cast_members(cast_list: list, max_weight: int = 3) -> str:
        if not isinstance(cast_list, list):
            return str(cast_list) if cast_list else ""
    
        weighted_cast = []
        for i, actor in enumerate(cast_list):
            weight = max(1, max_weight - i)
            weighted_cast.extend([actor] * weight)
        weighted_cast = ContentBasedRecommender._clean_text(weighted_cast)
        return ' '.join(weighted_cast)
    
    def _preprocess_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        median_runtime = df["runtime"].replace(0, np.nan).median()
        df["runtime"] = df["runtime"].replace({0: median_runtime, np.nan: median_runtime})
        lower_bound = df["runtime"].quantile(self.config.runtime_clip_percentiles[0])
        upper_bound = df["runtime"].quantile(self.config.runtime_clip_percentiles[1])
        df["runtime"] = df["runtime"].clip(lower_bound, upper_bound)
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
    
    def _build_feature_matrix(self, df: pd.DataFrame) -> csr_matrix:
        self.tfidf_main_actor = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features
        )
        main_actor_tfidf = self.tfidf_main_actor.fit_transform(
            df["main_actor"].fillna("").apply(self._clean_text)
        )
        
        self.tfidf_director = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features
        )
        director_tfidf = self.tfidf_director.fit_transform(
            df["director"].fillna("").apply(self._clean_text)
        )
        
        self.tfidf_cast = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features
        )
        cast_weighted = df["cast"].apply(
            lambda x: self._weight_cast_members(x) if isinstance(x, list) else str(x)
        )
        cast_tfidf = self.tfidf_cast.fit_transform(cast_weighted)

        self.tfidf_keywords = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features
        )
        keywords_cleaned = df["keywords"].apply(
            lambda x: ' '.join(self._clean_text(x)) if isinstance(x, list) else ""
        )
        keywords_tfidf = self.tfidf_keywords.fit_transform(keywords_cleaned)
        
        numerical_features = df[[
            'runtime', 
            'year', 
            'main_actor_rating',
            'director_rating',
            'vote_count'
        ]].values
        numerical_matrix = csr_matrix(numerical_features)

        genre_cols = [col for col in df.columns if col.startswith("genre_")]
        genre_matrix = csr_matrix(df[genre_cols].values.astype(float))
        
        main_actor_tfidf = main_actor_tfidf * self.config.main_actor_weight
        director_tfidf = director_tfidf * self.config.director_weight
        cast_tfidf = cast_tfidf * self.config.cast_weight
        keywords_tfidf = keywords_tfidf * self.config.keywords_weight
        genre_matrix = genre_matrix * self.config.genre_weight
        numerical_matrix = numerical_matrix * self.config.numerical_weight
        
        combined_features = hstack([
            main_actor_tfidf,
            director_tfidf,
            cast_tfidf,
            keywords_tfidf,
            genre_matrix,
            numerical_matrix
        ])
        
        return combined_features
    
    def _compute_similarity_matrix(self, features: csr_matrix) -> np.ndarray:
        similarity_matrix = cosine_similarity(features)
        similarity_matrix[similarity_matrix < self.config.similarity_threshold] = 0
        return similarity_matrix
    
    def fit(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None) -> 'ContentBasedRecommender':
        required_cols = ['movieId', 'main_actor', 'director', 'cast', 'runtime', 'year', 'keywords']
        missing_cols = [col for col in required_cols if col not in movies_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in movies_df: {missing_cols}")
        
        df_processed = self._preprocess_numerical_features(movies_df.copy())
        df_processed = self._preprocess_actor_director_ratings(df_processed)
        
        features_matrix = self._build_feature_matrix(df_processed)
        self.feature_matrix = features_matrix
        self.similarity_matrix = self._compute_similarity_matrix(features_matrix)
        
        self.movie_id_to_idx = {
            mid: idx for idx, mid in enumerate(df_processed['movieId'].values)
        }
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        self.all_movie_ids = set(self.movie_id_to_idx.keys())
        
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
        
        self.is_fitted = True
        return self

    def _build_user_profiles(self, ratings_df: pd.DataFrame):
        self.user_profiles = {}
        good_ratings_df = ratings_df[ratings_df['rating'] >= 4.0]
        valid_df = good_ratings_df[good_ratings_df['movieId'].isin(self.movie_id_to_idx)].copy()
        
        if valid_df.empty:
            return
            
        valid_df['movie_idx'] = valid_df['movieId'].map(self.movie_id_to_idx)
        
        for user_id, group in valid_df.groupby('userId'):
            idxs = group['movie_idx'].values
            weights = group['rating'].values
            
            user_features_sparse = self.feature_matrix[idxs]
            profile_sparse = csr_matrix(weights) * user_features_sparse
            profile = np.asarray(profile_sparse.todense()).reshape(-1)
            
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
                
            self.user_profiles[user_id] = profile

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
        
        scores = np.zeros(len(item_ids))
        
        if valid_candidates:
            cand_indices = np.array([
                self.movie_id_to_idx[mid] for _, mid in valid_candidates
            ])
            cand_vectors_sparse = self.feature_matrix[cand_indices]
            dots = cand_vectors_sparse.dot(profile)
            
            cand_norms = np.sqrt(np.asarray(cand_vectors_sparse.power(2).sum(axis=1))).reshape(-1)
            profile_norm = np.linalg.norm(profile)
            norms = cand_norms * profile_norm
            
            mask = norms > 0
            valid_scores = np.zeros_like(dots)
            valid_scores[mask] = dots[mask] / norms[mask]
            
            for i, (orig_idx, _) in enumerate(valid_candidates):
                pure = valid_scores[i] 
                pop = self.movie_vote_counts.get(_, 0.0)
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
                    
                    print(f"       - {reason_title}{rating_str} [Similarity: {similarity:.4f}]")
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
        
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npy")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")
        
        np.save(similarity_path, self.similarity_matrix)
        
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
                'config': self.config
            }, f)
    
    def load_artifacts(
        self,
        similarity_path: Optional[str] = None,
        mapping_path: Optional[str] = None,
        preprocessors_path: Optional[str] = None
    ):
        artifacts_dir = Path(self.config.artifacts_dir)
        
        similarity_path = similarity_path or str(artifacts_dir / "similarity_matrix.npy")
        mapping_path = mapping_path or str(artifacts_dir / "movie_id_to_idx.pkl")
        preprocessors_path = preprocessors_path or str(artifacts_dir / "preprocessors.pkl")
        
        self.similarity_matrix = np.load(similarity_path)
        
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
        
        self.is_fitted = True
    
    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        if movie_id not in self.movie_id_to_idx:
            return []
        
        movie_idx = self.movie_id_to_idx[movie_id]
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        similar_movies = sim_scores[1:top_n + 1]
        
        results = []
        for idx, score in similar_movies:
            results.append({
                'movie_id': self.idx_to_movie_id[idx],
                'similarity': score
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
        reasons = []
        
        for liked_id in liked_items:
            if liked_id in self.movie_id_to_idx:
                liked_idx = self.movie_id_to_idx[liked_id]
                sim = self.similarity_matrix[movie_idx, liked_idx]
                
                if sim > 0:
                    reasons.append({
                        'movie_id': liked_id,
                        'similarity': float(sim)
                    })
                    
        reasons.sort(key=lambda x: x['similarity'], reverse=True)
        return reasons[:top_n_reasons]