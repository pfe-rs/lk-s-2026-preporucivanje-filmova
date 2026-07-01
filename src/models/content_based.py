import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix, hstack, issparse, save_npz, load_npz
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
from loguru import logger
from tqdm.auto import tqdm

@dataclass
class ContentBasedConfig:
    main_actor_weight: float = 0.15
    director_weight: float = 0.20
    cast_weight: float = 0.15
    keywords_weight: float = 0.30
    genre_weight: float = 0.10
    numerical_weight: float = 0.10
    tfidf_sublinear_tf: bool = True
    tfidf_max_features: Optional[int] = 10000
    runtime_clip_percentiles: tuple = (0.01, 0.99)
    year_n_quantiles: int = 1000
    fillna_strategy: str = "median"
    similarity_threshold: float = 0.15
    top_k_per_item: int = 50
    top_k_default: int = 10
    pop_boost_weight: float = 0.05
    artifacts_dir: str = "data/processed/artifacts"
    show_progress_bars: bool = True


class ContentBasedRecommender:
    def __init__(self, config: Optional[ContentBasedConfig] = None):
        self.config = config or ContentBasedConfig()
        self.tfidf_main_actor = None
        self.tfidf_director = None
        self.tfidf_cast = None
        self.tfidf_keywords = None
        self.scaler_runtime = None
        self.scaler_year = None
        self.scaler_main_actor_rating = None
        self.scaler_director_rating = None
        self.qt_year = None
        self.similarity_matrix = None
        self.feature_matrix_norm = None
        self.movie_index = pd.Index([])
        self.movie_ids = np.array([], dtype=np.int64)
        self.user_index = pd.Index([])
        self.user_liked_indices = []
        self.popularity_array = np.array([], dtype=np.float32)
        self.movie_id_to_title = {}
        self.is_fitted = False

    @staticmethod
    def _build_cast_strings(cast_column):
        results = []
        for val in cast_column:
            if not isinstance(val, list) or not val:
                results.append("unknown")
                continue
            weighted = []
            for i, actor in enumerate(val):
                w = max(1, 3 - i)
                if w <= 0:
                    break
                actor_clean = str(actor).strip().lower()
                if actor_clean:
                    weighted.extend([actor_clean] * w)
            results.append(" ".join(weighted) if weighted else "unknown")
        return results

    @staticmethod
    def _build_keyword_strings(keyword_column):
        results = []
        for val in keyword_column:
            if not isinstance(val, list) or not val:
                results.append("unknown")
                continue
            tokens = [str(k).strip().lower() for k in val if len(str(k).strip()) >= 3]
            results.append(" ".join(tokens) if tokens else "unknown")
        return results

    @staticmethod
    def _clean_text_series(series: pd.Series) -> pd.Series:
        return (series.fillna("").astype(str).str.strip().str.lower()
                .replace(r'\s+', ' ', regex=True).replace("", "unknown"))

    def _preprocess_numerical_features(self, df: pd.DataFrame) -> None:
        def _fit_scaler(col, scaler_type, quant_transformer=None):
            data = pd.to_numeric(df[col], errors='coerce')
            valid = data.notna()
            if valid.sum() == 0:
                logger.warning(f"Column {col} has no valid values, filling with zeros")
                df[col] = 0.0
                if quant_transformer:
                    quant_transformer.fit(np.array([[0.0]]))
                scaler = scaler_type()
                scaler.fit(np.array([[0.0]]))
                return scaler
            if col == 'runtime':
                low = data.quantile(self.config.runtime_clip_percentiles[0])
                high = data.quantile(self.config.runtime_clip_percentiles[1])
                data = data.clip(low, high)
                logger.debug(f"Clipping {col} between {low} and {high}")
            fill_val = data.median() if self.config.fillna_strategy == "median" else 0.0
            if pd.isna(fill_val):
                fill_val = 0.0
            data = data.fillna(fill_val)
            if quant_transformer:
                data = quant_transformer.fit_transform(data.values.reshape(-1, 1))
            else:
                data = data.values.reshape(-1, 1)
            scaler = scaler_type()
            scaled = scaler.fit_transform(data).astype(np.float32).ravel()
            df[col] = scaled
            return scaler

        logger.info("Preprocessing numerical features")
        self.scaler_runtime = _fit_scaler('runtime', MinMaxScaler)
        self.qt_year = QuantileTransformer(n_quantiles=self.config.year_n_quantiles,
                                           output_distribution='uniform', random_state=42)
        self.scaler_year = _fit_scaler('year', MinMaxScaler, self.qt_year)

    def _preprocess_actor_director_ratings(self, df: pd.DataFrame) -> None:
        if 'rating' not in df.columns:
            logger.warning("No rating column found, setting actor/director ratings to zero")
            df['main_actor_avg_rating'] = 0.0
            df['director_avg_rating'] = 0.0
            self.scaler_main_actor_rating = MinMaxScaler()
            self.scaler_director_rating = MinMaxScaler()
            self.scaler_main_actor_rating.fit(np.array([[0.0]]))
            self.scaler_director_rating.fit(np.array([[0.0]]))
            return
            
        logger.info("Computing actor and director average ratings")
        main_actor_clean = self._clean_text_series(df['main_actor'])
        director_clean = self._clean_text_series(df['director'])
        
        actor_rating = df.groupby(main_actor_clean)['rating'].mean().fillna(0)
        actor_rating = np.log1p(actor_rating.where(actor_rating > 0, 0))
        
        director_rating = df.groupby(director_clean)['rating'].mean().fillna(0)
        director_rating = np.log1p(director_rating.where(director_rating > 0, 0))
        
        df['main_actor_avg_rating'] = main_actor_clean.map(actor_rating).fillna(0).values
        df['director_avg_rating'] = director_clean.map(director_rating).fillna(0).values
        
        self.scaler_main_actor_rating = MinMaxScaler()
        self.scaler_director_rating = MinMaxScaler()
        
        df['main_actor_avg_rating'] = self.scaler_main_actor_rating.fit_transform(
            df[['main_actor_avg_rating']]).astype(np.float32).ravel()
        df['director_avg_rating'] = self.scaler_director_rating.fit_transform(
            df[['director_avg_rating']]).astype(np.float32).ravel()
            
        logger.info(f"Actor rating range: {df['main_actor_avg_rating'].min():.4f} - {df['main_actor_avg_rating'].max():.4f}")

    def _fit_transform_tfidf(self, data: pd.Series, weight: float) -> Tuple[Any, csr_matrix]:
        vectorizer = TfidfVectorizer(
            sublinear_tf=self.config.tfidf_sublinear_tf,
            max_features=self.config.tfidf_max_features,
            dtype=np.float32
        )
        tfidf_matrix = vectorizer.fit_transform(data)
        if weight != 1.0:
            tfidf_matrix = tfidf_matrix * weight
        return vectorizer, tfidf_matrix

    def _build_feature_matrix(self, df: pd.DataFrame) -> csr_matrix:
        logger.info("Building feature matrix")
        texts = {
            'main_actor': df['main_actor_text'],
            'director': df['director_text'],
            'cast': df['cast_weighted'],
            'keywords': df['keywords_flat']
        }
        weights = {
            'main_actor': self.config.main_actor_weight,
            'director': self.config.director_weight,
            'cast': self.config.cast_weight,
            'keywords': self.config.keywords_weight
        }
        
        results = []
        for key in ['main_actor', 'director', 'cast', 'keywords']:
            results.append(self._fit_transform_tfidf(texts[key], weights[key]))
            
        self.tfidf_main_actor, main_actor_mat = results[0]
        self.tfidf_director, director_mat = results[1]
        self.tfidf_cast, cast_mat = results[2]
        self.tfidf_keywords, keywords_mat = results[3]
        
        logger.debug(f"TF-IDF shapes: main_actor={main_actor_mat.shape}, director={director_mat.shape}, "
                     f"cast={cast_mat.shape}, keywords={keywords_mat.shape}")

        genre_cols = [c for c in df.columns if c.startswith('genre_')]
        if genre_cols:
            logger.info(f"Using {len(genre_cols)} genre columns")
            genre_mat = csr_matrix(df[genre_cols].astype(np.float32).values)
            if self.config.genre_weight != 1.0:
                genre_mat = genre_mat * self.config.genre_weight
        else:
            genre_mat = csr_matrix((df.shape[0], 0), dtype=np.float32)

        numerical_features = ['runtime', 'year', 'main_actor_avg_rating', 'director_avg_rating']
        for feat in numerical_features:
            if feat not in df.columns:
                df[feat] = 0.0
        num_mat = csr_matrix(df[numerical_features].astype(np.float32).values)
        if self.config.numerical_weight != 1.0:
            num_mat = num_mat * self.config.numerical_weight

        feature_matrix = hstack(
            [main_actor_mat, director_mat, cast_mat, keywords_mat, genre_mat, num_mat],
            format='csr', dtype=np.float32
        )
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        return feature_matrix

    def _compute_similarity_matrix(self, features_norm: csr_matrix) -> csr_matrix:
        n_items = features_norm.shape[0]
        if n_items == 0:
            logger.warning("No items to compute similarity matrix")
            return csr_matrix((0, 0), dtype=np.float32)
        
        k = min(self.config.top_k_per_item, n_items - 1)
        if k <= 0:
            return csr_matrix((n_items, n_items), dtype=np.float32)

        logger.info(f"Computing top-{k} similarity matrix for {n_items} items")
        features_norm_T = features_norm.T.tocsr()
        
        batch_size = 1000
        threshold = self.config.similarity_threshold
        
        all_data = []
        all_indices = []
        indptr = [0]
        
        total_blocks = (n_items + batch_size - 1) // batch_size
        logger.info(f"Processing {n_items} items in {total_blocks} blocks of size {batch_size}")
        
        start_time_total = time.time()
        
        for block_idx, start in enumerate(range(0, n_items, batch_size), 1):
            block_start_time = time.time()
            end = min(start + batch_size, n_items)
            current_batch_size = end - start
            
            S_sparse = features_norm[start:end].dot(features_norm_T)
            S_dense = S_sparse.toarray().astype(np.float32)
            
            row_indices = np.arange(current_batch_size)
            col_indices = np.arange(start, end)
            S_dense[row_indices, col_indices] = -1.0
            
            S_dense[S_dense < threshold] = -1.0
            
            if k > 0:
                top_k_indices = np.argpartition(S_dense, -k, axis=1)[:, -k:]
                top_k_values = np.take_along_axis(S_dense, top_k_indices, axis=1)
                
                valid_mask = top_k_values > 0.0
                
                for i in range(current_batch_size):
                    valid_vals = top_k_values[i][valid_mask[i]]
                    if valid_vals.size > 0:
                        valid_idxs = top_k_indices[i][valid_mask[i]]
                        sort_order = np.argsort(-valid_vals)
                        all_data.append(valid_vals[sort_order])
                        all_indices.append(valid_idxs[sort_order])
                        indptr.append(indptr[-1] + valid_vals.size)
                    else:
                        indptr.append(indptr[-1])
            else:
                indptr.extend([indptr[-1]] * current_batch_size)
            
            block_elapsed = time.time() - block_start_time
            total_elapsed = time.time() - start_time_total
            eta = (total_elapsed / block_idx) * (total_blocks - block_idx)
                
        total_elapsed = time.time() - start_time_total
        logger.info(f"Similarity computation completed in {total_elapsed:.2f}s")
                
        if all_data:
            data_concat = np.concatenate(all_data)
            indices_concat = np.concatenate(all_indices)
            indptr = np.array(indptr, dtype=np.int64)
        else:
            data_concat = np.array([], dtype=np.float32)
            indices_concat = np.array([], dtype=np.int32)
            indptr = np.zeros(n_items + 1, dtype=np.int64)

        sim_csr = csr_matrix((data_concat, indices_concat, indptr),
                             shape=(n_items, n_items), dtype=np.float32)
        sim_csr.eliminate_zeros()
        logger.info(f"Similarity matrix nnz: {sim_csr.nnz}")
        return sim_csr

    def _build_popularity_arrays(self, ratings_df: Optional[pd.DataFrame] = None):
        n_movies = len(self.movie_index)
        if ratings_df is not None and 'movieId' in ratings_df.columns:
            logger.info("Building popularity array from ratings")
            counts = ratings_df.groupby('movieId').size()
            self.popularity_array = np.zeros(n_movies, dtype=np.float32)
            aligned_counts = counts.reindex(self.movie_ids, fill_value=0)
            log_counts = np.log1p(aligned_counts.values.astype(np.float64))
            max_val = log_counts.max()
            if max_val > 0:
                self.popularity_array = (log_counts / max_val).astype(np.float32)
            logger.info(f"Popularity scores: min={self.popularity_array.min():.4f}, max={self.popularity_array.max():.4f}")
        else:
            logger.info("No ratings DataFrame provided, popularity set to zero")
            self.popularity_array = np.zeros(n_movies, dtype=np.float32)

    def _build_user_profiles(self, ratings_df: pd.DataFrame):
        if ratings_df is None or ratings_df.empty:
            logger.info("No ratings data for user profiles")
            self.user_index = pd.Index([])
            self.user_liked_indices = []
            return
        logger.info("Building user profiles from ratings")
        liked = ratings_df[ratings_df['rating'] >= 4.0]
        if liked.empty:
            self.user_index = pd.Index([])
            self.user_liked_indices = []
            return
        movie_id_to_idx = pd.Series(np.arange(len(self.movie_ids)), index=self.movie_ids)
        liked['idx'] = liked['movieId'].map(movie_id_to_idx)
        liked = liked.dropna(subset=['idx'])
        liked['idx'] = liked['idx'].astype(np.int32)
        grouped = liked.groupby('userId')['idx'].apply(list)
        self.user_index = pd.Index(grouped.index, name='userId')
        self.user_liked_indices = [np.array(lst, dtype=np.int32) for lst in grouped]
        logger.info(f"Created profiles for {len(self.user_index)} users")

    def _get_user_embedding(self, liked_indices: np.ndarray) -> np.ndarray:
        if len(liked_indices) == 0:
            return np.zeros(self.feature_matrix_norm.shape[1], dtype=np.float32)
        rows = self.feature_matrix_norm[liked_indices]
        if issparse(rows):
            return rows.mean(axis=0).A1.astype(np.float32)
        return rows.mean(axis=0).astype(np.float32)

    def _score_items_for_user(self, liked_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        if len(item_indices) == 0:
            return np.zeros(0, dtype=np.float32)
        user_emb = self._get_user_embedding(liked_indices)
        selected_features = self.feature_matrix_norm[item_indices]
        if issparse(selected_features):
            scores = selected_features.dot(user_emb)
            if issparse(scores):
                scores = scores.toarray().ravel()
            else:
                scores = np.asarray(scores).ravel()
        else:
            scores = np.dot(selected_features, user_emb)
        scores = scores.astype(np.float32)
        pop = self.popularity_array[item_indices]
        return (1.0 - self.config.pop_boost_weight) * scores + self.config.pop_boost_weight * pop

    def _final_scores_for_candidates(self, liked_indices: np.ndarray,
                                     candidate_idx: np.ndarray) -> np.ndarray:
        return self._score_items_for_user(liked_indices, candidate_idx)

    def predict_scores(self, user_id: int, item_ids: List[int]) -> List[float]:
        if not self.is_fitted:
            return [0.0] * len(item_ids)
        item_indexer = self.movie_index.get_indexer(item_ids)
        valid_mask = item_indexer >= 0
        liked = (self.user_liked_indices[self.user_index.get_loc(user_id)]
                 if user_id in self.user_index else np.array([], dtype=np.int32))
        candidates = np.array(item_indexer[valid_mask], dtype=np.int32)
        full_scores = np.zeros(len(item_ids), dtype=np.float32)
        if len(candidates) > 0:
            full_scores[valid_mask] = self._score_items_for_user(liked, candidates)
        return full_scores.tolist()

    def get_top_k_recommendations(self, user_id: int, watched_items: Set[int],
                                  k: int = None,
                                  valid_items: Optional[List[int]] = None) -> List[int]:
        if not self.is_fitted:
            logger.warning("Model not fitted, cannot recommend")
            return []
        k = k or self.config.top_k_default
        if k <= 0:
            return []
        if valid_items is not None:
            candidate_mask = np.isin(self.movie_ids, list(valid_items))
        else:
            candidate_mask = np.ones(len(self.movie_ids), dtype=bool)
        if watched_items:
            candidate_mask &= ~np.isin(self.movie_ids, list(watched_items))
        candidate_idx = np.where(candidate_mask)[0]
        if len(candidate_idx) == 0:
            return []
        liked = (self.user_liked_indices[self.user_index.get_loc(user_id)]
                 if user_id in self.user_index else np.array([], dtype=np.int32))
        scores = self._final_scores_for_candidates(liked, candidate_idx)
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        top_ind = np.argpartition(scores, -k)[-k:]
        top_ind = top_ind[np.argsort(-scores[top_ind])]
        top_idx = candidate_idx[top_ind]
        return self.movie_ids[top_idx].tolist()

    def get_top_k_recommendations_batch(self, user_ids: List[int],
                                        watched_items_list: List[Set[int]],
                                        k: int,
                                        valid_items: Optional[List[int]] = None) -> List[List[int]]:
        if not self.is_fitted:
            return [[] for _ in user_ids]
        n_items = len(self.movie_ids)
        if valid_items is not None:
            valid_idx = self.movie_index.get_indexer(valid_items)
            valid_idx = valid_idx[valid_idx >= 0]
            if len(valid_idx) == 0:
                return [[] for _ in user_ids]
            candidate_mask = np.zeros(n_items, dtype=bool)
            candidate_mask[valid_idx] = True
        else:
            candidate_mask = np.ones(n_items, dtype=bool)

        logger.info(f"Batch recommending for {len(user_ids)} users")
        results = []
        for uid, watched in tqdm(zip(user_ids, watched_items_list),
                                 total=len(user_ids),
                                 disable=not self.config.show_progress_bars,
                                 desc="Batch recommendations"):
            liked = (self.user_liked_indices[self.user_index.get_loc(uid)]
                     if uid in self.user_index else np.array([], dtype=np.int32))
            watched_idx = np.array([], dtype=np.int32)
            if watched:
                idx_arr = self.movie_index.get_indexer(list(watched))
                watched_idx = idx_arr[idx_arr >= 0]
            cand_mask = candidate_mask.copy()
            if len(watched_idx) > 0:
                cand_mask[watched_idx] = False
            cand_idx = np.where(cand_mask)[0]
            if len(cand_idx) == 0:
                results.append([])
                continue
            scores = self._final_scores_for_candidates(liked, cand_idx)
            k_eff = min(k, len(scores))
            if k_eff == 0:
                results.append([])
                continue
            top = np.argpartition(scores, -k_eff)[-k_eff:]
            top = top[np.argsort(-scores[top])]
            results.append(self.movie_ids[cand_idx[top]].tolist())
        logger.info("Batch recommendation complete")
        return results

    def show_user_profile_and_recommendations(self, user_id, ratings_df, movies_df,
                                              k=10, top_rated_count=5, reasons_count=3):
        if not self.is_fitted:
            print("Model not fitted.")
            return
        print(f"\n=== User {user_id} Profile & Recommendations ===\n")
        if user_id in self.user_index:
            liked_idx = self.user_liked_indices[self.user_index.get_loc(user_id)]
            liked_ids = self.movie_ids[liked_idx]
            print(f"Liked movies ({len(liked_ids)}):")
            for mid in liked_ids[:top_rated_count]:
                title = self.movie_id_to_title.get(mid, str(mid))
                print(f"  {mid} - {title}")
        else:
            liked_ids = set()
            print("No liked movies found.")
        recs = self.get_top_k_with_titles(user_id, set(), k)
        print(f"\nTop {k} recommendations:")
        for rec in recs:
            print(f"  {rec['movieId']} - {rec['title']} (score: {rec['score']:.4f})")
            if reasons_count > 0 and liked_ids:
                explain = self.explain_recommendation(rec['movieId'], set(liked_ids), reasons_count)
                for exp in explain:
                    print(f"    because you liked {exp['movie_id']} ({exp['similarity']:.3f})")

    def get_top_k_with_titles(self, user_id, watched_items, k=10) -> List[Dict[str, Any]]:
        rec_ids = self.get_top_k_recommendations(user_id, watched_items, k=k)
        if not rec_ids:
            return []
        rec_indices = self.movie_index.get_indexer(rec_ids)
        liked = (self.user_liked_indices[self.user_index.get_loc(user_id)]
                 if user_id in self.user_index else np.array([], dtype=np.int32))
        scores = self._score_items_for_user(liked, rec_indices)
        result = []
        for mid, score in zip(rec_ids, scores):
            title = self.movie_id_to_title.get(mid, str(mid))
            result.append({'movieId': int(mid), 'title': title, 'score': float(score)})
        return result

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        if not self.is_fitted or movie_id not in self.movie_index:
            return []
        idx = self.movie_index.get_loc(movie_id)
        if self.similarity_matrix.shape[0] <= idx:
            return []
        row = self.similarity_matrix.getrow(idx)
        if row.nnz == 0:
            return []
        similarities = row.toarray().ravel()
        top_indices = np.argsort(-similarities)
        top_indices = top_indices[top_indices != idx]
        results = []
        for i in top_indices[:top_n]:
            if similarities[i] > 0:
                results.append({'movie_id': int(self.movie_ids[i]), 'similarity': float(similarities[i])})
        return results

    def explain_recommendation(self, movie_id, liked_items: Set[int],
                               top_n_reasons=3) -> List[Dict[str, Any]]:
        if not self.is_fitted or movie_id not in self.movie_index:
            return []
        target_idx = self.movie_index.get_loc(movie_id)
        if self.similarity_matrix.shape[0] <= target_idx:
            return []
        row = self.similarity_matrix.getrow(target_idx).toarray().ravel()
        liked_indices = self.movie_index.get_indexer(list(liked_items))
        liked_indices = liked_indices[liked_indices >= 0]
        if len(liked_indices) == 0:
            return []
        sims = row[liked_indices]
        top = np.argsort(-sims)[:top_n_reasons]
        results = []
        for i in top:
            if sims[i] > 0:
                results.append({
                    'movie_id': int(self.movie_ids[liked_indices[i]]),
                    'similarity': float(sims[i])
                })
        return results

    def save_artifacts(self, similarity_path=None, mapping_path=None, preprocessors_path=None):
        if not self.is_fitted:
            logger.warning("Model not fitted, no artifacts saved.")
            return
        base = Path(self.config.artifacts_dir)
        base.mkdir(parents=True, exist_ok=True)
        sim_path = similarity_path or str(base / "similarity.npz")
        map_path = mapping_path or str(base / "mapping.pkl")
        prep_path = preprocessors_path or str(base / "preprocessors.pkl")
        save_npz(sim_path, self.similarity_matrix)
        mapping = {
            'movie_index': self.movie_index,
            'movie_ids': self.movie_ids,
            'user_index': self.user_index,
            'movie_id_to_title': self.movie_id_to_title
        }
        with open(map_path, 'wb') as f:
            pickle.dump(mapping, f)
        preps = {
            'config': self.config,
            'tfidf_main_actor': self.tfidf_main_actor,
            'tfidf_director': self.tfidf_director,
            'tfidf_cast': self.tfidf_cast,
            'tfidf_keywords': self.tfidf_keywords,
            'scaler_runtime': self.scaler_runtime,
            'scaler_year': self.scaler_year,
            'scaler_main_actor_rating': self.scaler_main_actor_rating,
            'scaler_director_rating': self.scaler_director_rating,
            'qt_year': self.qt_year,
            'user_liked_indices': self.user_liked_indices,
            'popularity_array': self.popularity_array
        }
        with open(prep_path, 'wb') as f:
            pickle.dump(preps, f)
        logger.info(f"Artifacts saved to {base}")

    def load_artifacts(self, similarity_path=None, mapping_path=None, preprocessors_path=None):
        base = Path(self.config.artifacts_dir)
        sim_path = similarity_path or str(base / "similarity.npz")
        map_path = mapping_path or str(base / "mapping.pkl")
        prep_path = preprocessors_path or str(base / "preprocessors.pkl")
        logger.info(f"Loading artifacts from {base}")
        self.similarity_matrix = load_npz(sim_path)
        with open(map_path, 'rb') as f:
            mapping = pickle.load(f)
        self.movie_index = mapping['movie_index']
        self.movie_ids = mapping['movie_ids']
        self.user_index = mapping['user_index']
        self.movie_id_to_title = mapping.get('movie_id_to_title', {})
        with open(prep_path, 'rb') as f:
            preps = pickle.load(f)
        self.config = preps['config']
        self.tfidf_main_actor = preps['tfidf_main_actor']
        self.tfidf_director = preps['tfidf_director']
        self.tfidf_cast = preps['tfidf_cast']
        self.tfidf_keywords = preps['tfidf_keywords']
        self.scaler_runtime = preps['scaler_runtime']
        self.scaler_year = preps['scaler_year']
        self.scaler_main_actor_rating = preps['scaler_main_actor_rating']
        self.scaler_director_rating = preps['scaler_director_rating']
        self.qt_year = preps['qt_year']
        self.user_liked_indices = preps['user_liked_indices']
        self.popularity_array = preps['popularity_array']
        self.is_fitted = True
        logger.info("Artifacts loaded successfully.")

    def fit(self, movies_df: pd.DataFrame,
            ratings_df: Optional[pd.DataFrame] = None) -> 'ContentBasedRecommender':
        t_start = time.time()
        logger.info("Starting model fitting")
        required_cols = ['main_actor', 'director', 'cast', 'keywords']
        for col in required_cols:
            if col not in movies_df.columns:
                logger.warning(f"Missing column {col}, filling with empty.")
                movies_df[col] = ""
        df = movies_df.copy()
        logger.info("Cleaning text columns")
        df['main_actor_text'] = self._clean_text_series(df['main_actor'])
        df['director_text'] = self._clean_text_series(df['director'])

        logger.info("Building cast and keyword strings")
        if self.config.show_progress_bars:
            logger.info("Processing cast entries")
            df['cast_weighted'] = self._build_cast_strings(tqdm(df['cast'], desc="Building cast strings", disable=not self.config.show_progress_bars))
            df['keywords_flat'] = self._build_keyword_strings(tqdm(df['keywords'], desc="Building keyword strings", disable=not self.config.show_progress_bars))
        else:
            df['cast_weighted'] = self._build_cast_strings(df['cast'].tolist())
            df['keywords_flat'] = self._build_keyword_strings(df['keywords'].tolist())

        logger.info("Preprocessing numerical and rating features")
        self._preprocess_numerical_features(df)
        self._preprocess_actor_director_ratings(df)

        logger.info("Building normalized feature matrix")
        feat_mat = self._build_feature_matrix(df)
        self.feature_matrix_norm = normalize(feat_mat, norm='l2', copy=False)
        self.movie_index = pd.Index(df['movieId'], name='movieId')
        self.movie_ids = df['movieId'].values.astype(np.int64)
        self.movie_id_to_title = dict(zip(df['movieId'], df.get('title', df['movieId'])))

        self.similarity_matrix = self._compute_similarity_matrix(self.feature_matrix_norm)
        self._build_popularity_arrays(ratings_df)
        self._build_user_profiles(ratings_df)

        self.is_fitted = True
        elapsed = time.time() - t_start
        logger.info(f"Fitting complete in {elapsed:.2f}s. "
                    f"Movies: {len(self.movie_ids)}, users: {len(self.user_index)}, "
                    f"similarity nnz: {self.similarity_matrix.nnz}")
        return self