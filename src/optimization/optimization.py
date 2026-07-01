#!/usr/bin/env python3
"""
Hyperparameter optimisation for all recommendation models.
Uses Optuna to maximise NDCG@10 on a held‑out validation set,
then reports the best configurations.
"""
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

# ------------------------------------------------------------
# Ensure the project root is on sys.path so that src imports work
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_loader import MovieLensDataLoader
from src.data_processing.splitters import DataSplitter
from src.models.content_based import ContentBasedRecommender, ContentBasedConfig
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.hybrid import HybridRecommender
from src.models.cascading_hybrid import CascadingHybridRecommender
from src.models.popular_baseline_model import PopularityBaseline
from src.evaluation.evaluator import RecommendationEvaluator

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET = "ml-latest-small"          # or "ml-latest"
CACHE_METADATA = "data/processed/movie_metadata.parquet"
STUDY_DB = "sqlite:///optuna_studies.db"
N_TRIALS = 100                       # per model
VAL_SIZE = 0.2                       # fraction of train users used for validation
RANDOM_SEED = 42
METRIC = "ndcg"                      # metric to optimise
K_OPT = 10                           # k value used for optimisation
NEGATIVE_SAMPLES = 99
TOP_K_EVAL = 20                      # max_recommendations for evaluation

# ----------------------------------------------------------------------
# Data loading and splitting (cache to avoid repeated I/O)
# ----------------------------------------------------------------------
async def load_movielens() -> Tuple[pd.DataFrame, pd.DataFrame]:
    loader = MovieLensDataLoader(DATASET, cache_file=CACHE_METADATA)
    data_dict = loader.load_data()
    await loader.letterboxd_data_async(max_concurrent_requests=50)

    movies_df = pd.DataFrame(loader.movie_data)
    genre_features = loader.preprocess_movies()
    movies_df = pd.concat([movies_df, genre_features], axis=1)
    movies_df = movies_df.dropna().reset_index(drop=True)

    ratings_df = data_dict["ratings"]
    return movies_df, ratings_df

def create_train_val_splits(ratings_df: pd.DataFrame, val_frac: float, seed: int):
    splitter = DataSplitter(ratings_df)
    # full train / test (temporal) split
    splits = splitter.leave_one_out()
    train_full = splits["train"]
    test_full = splits["test"]

    # Further split train_full into train and validation by users
    rng = np.random.default_rng(seed)
    all_users = train_full["userId"].unique()
    n_val_users = max(1, int(len(all_users) * val_frac))
    val_users = rng.choice(all_users, size=n_val_users, replace=False)
    val_mask = train_full["userId"].isin(val_users)
    train_sub = train_full[~val_mask].copy()
    val_sub = train_full[val_mask].copy()
    return train_sub, val_sub, test_full

# ----------------------------------------------------------------------
# Objective functions for Optuna
# ----------------------------------------------------------------------
def evaluate_model_on_val(
    model: Any,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    item_universe: List[int],
    k_values: List[int] = [K_OPT],
    max_rec: int = TOP_K_EVAL,
) -> float:
    """Return the primary metric (NDCG@K_OPT) on validation set."""
    evaluator = RecommendationEvaluator(
        models={model_name: model},
        train_df=train_df,
        test_df=val_df,
        relevance_threshold=4.0,
        user_sample_size=None,
        random_state=RANDOM_SEED,
        item_universe=item_universe,
        n_negative_samples=NEGATIVE_SAMPLES,
    )
    res_df = evaluator.evaluate_all_models(k_values=k_values, max_recommendations=max_rec)
    if res_df.empty:
        return 0.0
    # Extract the metric for the given model and k
    row = res_df[(res_df["model"] == model_name) & (res_df["k"] == K_OPT)]
    if row.empty:
        return 0.0
    return row.iloc[0][METRIC]

# ----- Collaborative Filtering -----
def objective_cf(trial: optuna.Trial, train_df, val_df, item_universe) -> float:
    params = {
        "k_components": trial.suggest_int("k_components", 20, 200, step=10),
        "reg_all": trial.suggest_float("reg_all", 1e-3, 0.5, log=True),
        "lr_all": trial.suggest_float("lr_all", 1e-4, 0.1, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 5, 30),
        "alpha": trial.suggest_float("alpha", 0.1, 0.9),
        "min_ratings": trial.suggest_int("min_ratings", 1, 10),
    }
    model = CollaborativeFiltering(random_state=RANDOM_SEED, **params)
    model.fit(train_df)
    metric = evaluate_model_on_val(model, "CF", train_df, val_df, item_universe)
    return metric

# ----- Content-Based -----
def objective_cb(trial: optuna.Trial, train_df, val_df, movies_df, item_universe) -> float:
    # Weights (they will be normalised inside ContentBasedConfig sum to 1)
    kw = trial.suggest_float("keywords_weight", 0.01, 0.6)
    actor = trial.suggest_float("main_actor_weight", 0.01, 0.6)
    director = trial.suggest_float("director_weight", 0.01, 0.6)
    cast = trial.suggest_float("cast_weight", 0.01, 0.6)
    genre = trial.suggest_float("genre_weight", 0.01, 0.6)
    num = trial.suggest_float("numerical_weight", 0.01, 0.4)
    # Normalise to sum 1 later inside config (we pass them raw)
    cfg = ContentBasedConfig(
        main_actor_weight=actor,
        director_weight=director,
        cast_weight=cast,
        keywords_weight=kw,
        genre_weight=genre,
        numerical_weight=num,
        tfidf_sublinear_tf=trial.suggest_categorical("tfidf_sublinear_tf", [True, False]),
        tfidf_max_features=trial.suggest_int("tfidf_max_features", 1000, 15000, step=1000),
        similarity_threshold=trial.suggest_float("similarity_threshold", 0.02, 0.25),
        top_k_per_item=trial.suggest_int("top_k_per_item", 20, 80, step=5),
        pop_boost_weight=trial.suggest_float("pop_boost_weight", 0.0, 0.3),
        show_progress_bars=False,
    )
    model = ContentBasedRecommender(config=cfg)
    model.fit(movies_df=movies_df, ratings_df=train_df)
    metric = evaluate_model_on_val(model, "CB", train_df, val_df, item_universe)
    return metric

# ----- Hybrid (CF + CB blended) -----
def objective_hybrid(trial: optuna.Trial, train_df, val_df, movies_df, item_universe) -> float:
    # Optimise CF and CB sub‑parameters as well (nested)
    cf_params = {
        "k_components": trial.suggest_int("cf_k_components", 20, 200, step=10),
        "reg_all": trial.suggest_float("cf_reg_all", 1e-3, 0.5, log=True),
        "n_epochs": trial.suggest_int("cf_n_epochs", 5, 20),
        "alpha": trial.suggest_float("cf_alpha_pop", 0.1, 0.9),
        "min_ratings": trial.suggest_int("cf_min_ratings", 1, 10),
    }
    cb_weights = {
        "keywords_weight": trial.suggest_float("cb_keywords_weight", 0.01, 0.6),
        "main_actor_weight": trial.suggest_float("cb_main_actor_weight", 0.01, 0.6),
        "director_weight": trial.suggest_float("cb_director_weight", 0.01, 0.6),
        "cast_weight": trial.suggest_float("cb_cast_weight", 0.01, 0.6),
        "genre_weight": trial.suggest_float("cb_genre_weight", 0.01, 0.6),
        "numerical_weight": trial.suggest_float("cb_numerical_weight", 0.01, 0.4),
    }
    cb_cfg = ContentBasedConfig(
        **cb_weights,
        tfidf_sublinear_tf=trial.suggest_categorical("cb_tfidf_sublinear_tf", [True, False]),
        tfidf_max_features=trial.suggest_int("cb_tfidf_max_features", 1000, 15000, step=1000),
        similarity_threshold=trial.suggest_float("cb_similarity_threshold", 0.02, 0.25),
        top_k_per_item=trial.suggest_int("cb_top_k_per_item", 20, 80, step=5),
        pop_boost_weight=trial.suggest_float("cb_pop_boost_weight", 0.0, 0.3),
        show_progress_bars=False,
    )
    alpha = trial.suggest_float("hybrid_alpha", 0.2, 0.9)
    
    cf = CollaborativeFiltering(random_state=RANDOM_SEED, **cf_params)
    cb = ContentBasedRecommender(config=cb_cfg)
    cf.fit(train_df)
    cb.fit(movies_df=movies_df, ratings_df=train_df)
    
    hybrid = HybridRecommender(
        cf_model=cf,
        cb_model=cb,
        alpha=alpha,
    )
    hybrid.fitted(cf_model=cf, cb_model=cb, movies_df=movies_df, ratings_df=train_df)
    metric = evaluate_model_on_val(hybrid, "Hybrid", train_df, val_df, item_universe)
    return metric

# ----- Cascading Hybrid -----
def objective_cascade(trial: optuna.Trial, train_df, val_df, movies_df, item_universe) -> float:
    # CF and CB params (simplified to avoid too many dimensions)
    cf_params = {
        "k_components": trial.suggest_int("cf_k", 20, 200, step=10),
        "reg_all": trial.suggest_float("cf_reg", 1e-3, 0.5, log=True),
        "n_epochs": trial.suggest_int("cf_epochs", 5, 20),
        "alpha": trial.suggest_float("cf_alpha", 0.1, 0.9),
        "min_ratings": trial.suggest_int("cf_minr", 1, 10),
    }
    cb_cfg = ContentBasedConfig(
        keywords_weight=trial.suggest_float("cb_kw", 0.01, 0.6),
        main_actor_weight=trial.suggest_float("cb_actor", 0.01, 0.6),
        director_weight=trial.suggest_float("cb_dir", 0.01, 0.6),
        cast_weight=trial.suggest_float("cb_cast", 0.01, 0.6),
        genre_weight=trial.suggest_float("cb_genre", 0.01, 0.6),
        numerical_weight=trial.suggest_float("cb_num", 0.01, 0.4),
        tfidf_sublinear_tf=trial.suggest_categorical("cb_tfidf_sub", [True, False]),
        tfidf_max_features=trial.suggest_int("cb_tfidf_maxf", 1000, 15000, step=1000),
        similarity_threshold=trial.suggest_float("cb_sim_thr", 0.02, 0.25),
        top_k_per_item=trial.suggest_int("cb_topk", 20, 80, step=5),
        pop_boost_weight=trial.suggest_float("cb_pop", 0.0, 0.3),
        show_progress_bars=False,
    )
    primary_k = trial.suggest_int("primary_k", 20, 120, step=10)

    cf = CollaborativeFiltering(random_state=RANDOM_SEED, **cf_params)
    cb = ContentBasedRecommender(config=cb_cfg)
    cf.fit(train_df)
    cb.fit(movies_df=movies_df, ratings_df=train_df)

    cascade = CascadingHybridRecommender(
        primary_model=cf,
        secondary_model=cb,
        primary_k=primary_k,
    )
    cascade.fitted(primary_model=cf, secondary_model=cb)
    metric = evaluate_model_on_val(cascade, "Cascade", train_df, val_df, item_universe)
    return metric

# ----------------------------------------------------------------------
# Study runner
# ----------------------------------------------------------------------
def run_optimization(
    objective,
    study_name: str,
    n_trials: int,
    direction: str = "maximize",
    storage: str = STUDY_DB,
):
    """Create or continue an Optuna study."""
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def main():
    print("=== Loading data ===")
    movies_df, ratings_df = await load_movielens()
    print(f"Movies: {movies_df.shape}, Ratings: {ratings_df.shape}")

    # Create train/val/test splits
    train_df, val_df, test_df = create_train_val_splits(ratings_df, VAL_SIZE, RANDOM_SEED)
    item_universe = sorted(set(movies_df["movieId"].unique()))

    # Pre-train popularity baseline (not tuned)
    pop_model = PopularityBaseline()
    pop_model.fit(train_df)

    studies = {}

    # 1. Content-Based
    print("\n=== Tuning Content-Based ===")
    cb_obj = lambda trial: objective_cb(trial, train_df, val_df, movies_df, item_universe)
    study_cb = run_optimization(cb_obj, "ContentBased", N_TRIALS)
    studies["CB"] = study_cb
    print(f"Best CB trial: {study_cb.best_trial.params}, value: {study_cb.best_value}")

    # 2. Collaborative Filtering
    print("\n=== Tuning Collaborative Filtering ===")
    cf_obj = lambda trial: objective_cf(trial, train_df, val_df, item_universe)
    study_cf = run_optimization(cf_obj, "CollaborativeFiltering", N_TRIALS)
    studies["CF"] = study_cf
    print(f"Best CF trial: {study_cf.best_trial.params}, value: {study_cf.best_value}")

    # 3. Hybrid
    print("\n=== Tuning Hybrid ===")
    hyb_obj = lambda trial: objective_hybrid(trial, train_df, val_df, movies_df, item_universe)
    study_hyb = run_optimization(hyb_obj, "Hybrid", N_TRIALS)
    studies["Hybrid"] = study_hyb
    print(f"Best Hybrid trial: {study_hyb.best_trial.params}, value: {study_hyb.best_value}")

    # 4. Cascading Hybrid
    print("\n=== Tuning Cascading Hybrid ===")
    cas_obj = lambda trial: objective_cascade(trial, train_df, val_df, movies_df, item_universe)
    study_cas = run_optimization(cas_obj, "CascadeHybrid", N_TRIALS)
    studies["Cascade"] = study_cas
    print(f"Best Cascade trial: {study_cas.best_trial.params}, value: {study_cas.best_value}")

    # Save best parameters to a JSON file
    best_params = {}
    for name, study in studies.items():
        best_params[name] = {
            "params": study.best_trial.params,
            "value": study.best_value,
        }
    output_path = PROJECT_ROOT / "best_hyperparameters.json"
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest parameters saved to {output_path}")

asyncio.run(main())