import numpy as np
from typing import List, Dict


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return len(set(rec_k) & set(relevant)) / k


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)


def f1_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def dcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain"""
    dcg = dcg_at_k(recommended, relevant, k)
    idcg = dcg_at_k(relevant, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def map_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    rec_k = recommended[:k]
    hits = 0
    sum_precs = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precs += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precs / min(len(relevant), k)


def mrr(recommended: List[int], relevant: List[int]) -> float:
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0

def catalog_coverage(all_recommended: List[List[int]], total_items: int) -> float:
    unique_recommended = set()
    for rec_list in all_recommended:
        unique_recommended.update(rec_list)
    if total_items == 0:
        return 0.0
    return len(unique_recommended) / total_items


def intra_list_diversity(
    recommended: List[int], item_features: Dict[int, np.ndarray]
) -> float:
    if len(recommended) < 2:
        return 0.0
    similarities = []
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            feat_i = item_features.get(recommended[i])
            feat_j = item_features.get(recommended[j])
            if feat_i is not None and feat_j is not None:
                dot = np.dot(feat_i, feat_j)
                norm = np.linalg.norm(feat_i) * np.linalg.norm(feat_j)
                sim = dot / norm if norm > 0 else 0.0
                similarities.append(sim)
    if not similarities:
        return 0.0
    return 1.0 - np.mean(similarities)


def novelty(recommended: List[int], item_popularity: Dict[int, int]) -> float:
    total_interactions = sum(item_popularity.values())
    if total_interactions == 0:
        return 0.0

    novelties = []
    for item in recommended:
        pop = item_popularity.get(item, 1)
        prob = pop / total_interactions
        novelties.append(-np.log2(prob))
    return np.mean(novelties) if novelties else 0.0
