# Content‑Based Recommendation System Documentation

---

## 1. Overview
This document describes the **content‑based recommendation pipeline** used in the *film‑recommendations* project. It covers the data sources, core interfaces, type definitions, configuration, and notebook walkthrough. No implementation code is included—only the public API, data schemas, and usage examples.

---

## 2. Data Description

### 2.1 `data/processed/movie_metadata.csv`
| Column | Type | Description |
|--------|------|-------------|
| `title` | `str` | Movie title |
| `year` | `int` | Release year |
| `cast` | `list[str]` | List of main cast members (string representation of a Python list) |
| `main_actor` | `str` | Primary actor/actress |
| `director` | `str` | Director name |
| `rating` | `float` | Average user rating (e.g., from MovieLens) |
| `runtime` | `float` | Runtime in minutes |
| `genre_<GenreName>` | `int (0/1)` | Binary indicator for each genre (one‑hot) |

*Sample row*:
```csv
Toy Story,1995,"['Tom Hanks', 'Tim Allen', 'Don Rickles', 'Jim Varney', 'Wallace Shawn']",Tom Hanks,John Lasseter,7.978,81.0,0,0,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0
```

### 2.2 `data/processed/processed_movie_data.csv`
Contains **engineered movie feature vectors** (e.g., TF‑IDF vectors for cast, director, plot, etc.). The exact columns are not listed here but are stored as a pandas DataFrame and later converted to a NumPy array.

### 2.3 Similarity Matrices (`data/processed/*.npy`)
Pre‑computed similarity matrices stored as NumPy arrays:

| File | Shape | Meaning |
|------|-------|---------|
| `similarity_matrix.npy` | `(N, N)` | Overall item‑item similarity |
| `tfidf_cast_matrix.npy` | `(N, N)` | Cast‑based similarity |
| `tfidf_director_matrix.npy` | `(N, N)` | Director‑based similarity |
| `tfidf_main_actor_matrix.npy` | `(N, N)` | Main‑actor‑based similarity |

`N` = number of movies in the dataset.

### 2.4 User Profiles (`data/processed/user_profiles.csv` – inferred)
Each user is represented by a **profile vector** (typically a weighted sum of the item vectors they have interacted with). The profile is stored as a NumPy array or a dictionary mapping feature names to scores.

---

## 3. Core Interfaces

### 3.1 Data Processing Layer (`src/data_processing/*`)

| Class / Function | Module | Signature (public) | Description |
|------------------|--------|--------------------|-------------|
| `DataLoader` | `data_loader.py` | `load_metadata(csv_path: str) -> pd.DataFrame` | Loads `movie_metadata.csv` into a DataFrame. |
|  |  | `load_features(csv_path: str) -> pd.DataFrame` | Loads processed feature vectors. |
|  |  | `load_similarity(matrix_path: str) -> np.ndarray` | Loads a pre‑computed similarity matrix. |
| `SimilarityCalculator` (abstract) | — | `compute_similarity(features: np.ndarray) -> np.ndarray` | Calculates pair‑wise similarity (e.g., cosine) from raw features. |
| `FeatureExtractor` (abstract) | — | `extract_features(df: pd.DataFrame) -> np.ndarray` | Transforms raw metadata into a numeric feature vector. |

### 3.2 Model Layer (`src/models/*`)

| Class | Module | Public Methods | Signature |
|-------|--------|----------------|-----------|
| `ContentBasedModel` | `content_based.py` | `__init__(self, similarity_matrix: np.ndarray, item_ids: List[int])` | Constructor stores the similarity matrix and mapping from index → `item_id`. |
|  |  | `fit(self, user_profiles: Dict[int, np.ndarray]) -> None` | Learns any internal state (e.g., user‑item relevance) from user profiles. |
|  |  | `recommend(self, user_id: int, top_k: int = 10) -> List[int]` | Returns a list of `item_id`s ranked by predicted relevance for the given `user_id`. |
|  |  | `predict_rating(self, user_id: int, movie_id: int) -> float` | Predicts an expected rating for a specific `(user_id, movie_id)` pair. |
| `Evaluator` (abstract base) | `evaluation/metrics.py` | `evaluate(self, recommendations: List[int], ground_truth: Set[int]) -> Dict[str, float]` | Computes metrics such as precision@k, recall@k, NDCG, etc. |

### 3.3 Configuration (`src/config.py`)

| Variable | Type | Default | Meaning |
|----------|------|---------|---------|
| `DATA_PATH` | `Path` | `Path("data/processed")` | Base directory for all processed data files. |
| `SIMILARITY_MATRIX_PATH` | `Path` | `Path("data/processed/similarity_matrix.npy")` | Path to the main similarity matrix. |
| `FEATURE_VECTORS_PATH` | `Path` | `Path("data/processed/processed_movie_data.csv")` | Path to the feature CSV. |
| `TOP_K` | `int` | `10` | Number of recommendations to return. |
| `SIMILARITY_THRESHOLD` | `float` | `0.5` | Minimum similarity value to consider a match. |
| `RANDOM_STATE` | `int` | `42` | Seed for reproducibility. |

---

## 4. Type Definitions

```python
MovieId = int                                 # Unique identifier for a movie
UserId = int                                  # Unique identifier for a user
ItemId = int                                  # Alias for MovieId (used in internal mappings)
SimilarityMatrix = np.ndarray                 # Shape = (num_items, num_items)
FeatureVector = np.ndarray                    # Shape = (num_features,) for a single item
UserProfile = Dict[int, FeatureVector]        # Mapping from user_id → user feature vector
RecommendationList = List[ItemId]             # List of recommended item ids
```

---

## 5. Notebook Walkthrough

| Notebook | Purpose | Key Cells (high‑level) |
|----------|---------|------------------------|
| `01_eda_movielens.ipynb` | Exploratory data analysis of the raw MovieLens data. | Load CSV, basic statistics, genre distribution plots. |
| `02_collaboration_filtering.ipynb` | Baseline collaborative‑filtering model (Surprise SVD). | Data preparation, model training, evaluation. |
| `03_content_based_filtering.ipynb` | Content‑based pipeline setup (feature extraction, similarity). | Load metadata, compute TF‑IDF vectors, build similarity matrices. |
| `04_metrics_and_evaluation.ipynb` | Metric computation and error analysis. | Load ground‑truth splits, compute precision/recall/NDCG, visualise results. |
| `05_content_based.ipynb` | Final model training, recommendation generation, and export. | Fit `ContentBasedModel`, generate top‑k recommendations, save outputs. |

Each notebook follows a **cell‑by‑cell** progression: data ingestion → preprocessing → model training → evaluation → result visualization.

---

## 6. Example Usage (Signatures Only)

```python
# ------------------------------------------------------------------
# 1. Load raw data
# ------------------------------------------------------------------
metadata_df = DataLoader.load_metadata("data/processed/movie_metadata.csv")
features_df = DataLoader.load_features("data/processed/processed_movie_data.csv")
sim_matrix = SimilarityCalculator.compute_similarity(features_df.values)

# ------------------------------------------------------------------
# 2. Initialise the content‑based model
# ------------------------------------------------------------------
model = ContentBasedModel(
    similarity_matrix=sim_matrix,
    item_ids=metadata_df["movieId"].tolist()
)

# ------------------------------------------------------------------
# 3. Prepare user profiles (example dict)
# ------------------------------------------------------------------
user_profiles = {
    user_id: np.random.rand(len(metadata_df))  # placeholder vectors
}
model.fit(user_profiles)

# ------------------------------------------------------------------
# 4. Generate top‑k recommendations for a user
# ------------------------------------------------------------------
user_id = 123
recommendations: RecommendationList = model.recommend(user_id, top_k=10)
print(f"Top 10 recommendations for user {user_id}: {recommendations}")

# ------------------------------------------------------------------
# 5. Predict a rating for a specific (user, movie) pair
# ------------------------------------------------------------------
pred_rating = model.predict_rating(user_id, movie_id=42)
print(f"Predicted rating: {pred_rating:.3f}")
```

*All signatures above reflect the public API; implementation details are omitted.*

---

## 7. Configuration Details

```python
# src/config.py
from pathlib import Path

DATA_PATH = Path("data/processed")
SIMILARITY_MATRIX_PATH = DATA_PATH / "similarity_matrix.npy"
FEATURE_VECTORS_PATH = DATA_PATH / "processed_movie_data.csv"
TOP_K = 10
SIMILARITY_THRESHOLD = 0.5
RANDOM_STATE = 42
```

These constants are imported by the data‑loading and model modules to ensure consistent paths and hyper‑parameters.

---

## 8. Implementation Notes (High‑Level)

* **Similarity Computation** – Uses cosine similarity on TF‑IDF vectors for cast, director, and main‑actor fields; the final similarity matrix is a weighted combination of these sub‑matrices.
* **Recommendation Generation** – For a given user profile, scores each item by the dot product between the profile vector and the item’s similarity vector; top‑k items are returned.
* **Evaluation** – Ground‑truth interaction matrices are used to compute standard recommendation metrics (precision, recall, NDCG). Visualisations are generated in `04_metrics_and_evaluation.ipynb`.

---

*End of Documentation*