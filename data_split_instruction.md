# The Ultimate Guide to Evaluating Recommender Systems on MovieLens

This guide provides a comprehensive, step-by-step instruction on how to properly define relevance and split data specifically for the **MovieLens dataset** (ML-100K, ML-1M, ML-20M, ML-25M). 

Since MovieLens contains **explicit feedback** (ratings) and **timestamps**, this guide strictly focuses on explicit evaluation protocols.

---

## Table of Contents
1. [Defining Relevance in MovieLens](#1-defining-relevance-in-movielens)
2. [Data Splitting Strategies](#2-data-splitting-strategies)
3. [Handling the "Empty Relevant" Edge Case](#3-handling-the-empty-relevant-edge-case)
4. [The Evaluation Protocol (Negative Sampling)](#4-the-evaluation-protocol-negative-sampling)
5. [Complete Pandas Implementation](#5-complete-pandas-implementation)

---

## 1. Defining Relevance in MovieLens

In MovieLens, users rate movies on a scale (1 to 5, or 0.5 to 5.0). Because this is **explicit feedback**, relevance is defined by setting a rating threshold. 

*Crucial Rule:* You must decide what to do with the "neutral" zone (usually 3.0 or 3.5). Including neutral ratings as "relevant" introduces massive noise into your evaluation.

### Strategy A: The Academic Standard (Recommended)
**Definition:** `Relevant = rating >= 4.0` | `Discard = rating == 3.0` | `Negative = rating <= 2.0`
- **How it works:** You completely drop 3.0 ratings from your dataset. Only 4s and 5s are treated as positive items the model should recommend.
- **Pros:** This is the exact protocol used in 95% of top-tier RecSys papers (e.g., LightGCN, BERT4Rec, SASRec). It makes your results directly comparable to State-of-the-Art.
- **Cons:** Discards a significant portion of the data (all 3.0 ratings).

### Strategy B: The Strict Threshold
**Definition:** `Relevant = rating == 5.0` | `Negative = rating < 5.0`
- **How it works:** Only absolute masterpieces (5 stars) are considered relevant.
- **Pros:** Extremely high confidence that the user loved the movie.
- **Cons:** The dataset becomes incredibly sparse. The model will struggle to find patterns because the "positive" signal is too rare.

### Strategy C: The Lenient Threshold
**Definition:** `Relevant = rating >= 3.5` | `Negative = rating <= 2.5`
- **How it works:** You treat "okay" movies (3.5) as relevant.
- **Pros:** Maximizes the amount of training data.
- **Cons:** Dilutes the quality of recommendations. The model will be rewarded for recommending mediocre movies. **Not recommended for rigorous evaluation.**

---

## 2. Data Splitting Strategies

⚠️ **THE GOLDEN RULE OF RECSYS:** **NEVER use a random split (`train_test_split`)**. 
MovieLens has a `timestamp` column. Random splitting leaks future interactions into the training set, resulting in artificially inflated metrics that will fail in the real world. You **must** use chronological splits.

### Strategy 1: Leave-One-Out (LOO)
**The Academic Benchmark Standard.**
- **How it works:** For every user, sort their interactions by `timestamp`. 
  - The **last** interaction goes to the **Test** set.
  - The **second to last** goes to the **Validation** set.
  - All remaining interactions go to the **Train** set.
- **When to use:** When you want to evaluate the model's ability to predict the *very next* movie a user will love. This is the standard for papers evaluating ranking metrics (NDCG@10, Recall@10).
- **Characteristics:** Every user has exactly 1 item in the test set.

### Strategy 2: Global Temporal Split (80/20)
**The Industry Standard.**
- **How it works:** Sort the *entire dataset* by `timestamp`. Find the timestamp at the 80th percentile. 
  - All interactions before this timestamp go to **Train**.
  - All interactions after go to **Test**.
- **When to use:** When you want to simulate a real-world production environment where the model is trained on historical data and deployed to predict future behavior for the whole platform.
- **Characteristics:** Users will have varying numbers of items in the test set (some might have 0, some might have 10).

### Strategy 3: Chronological K-Fold (Per User)
**The Robustness Check.**
- **How it works:** For each user, sort their history by time and divide it into $K$ chronological chunks (e.g., 5 folds). 
  - Fold 1: Train on chunk 1, test on chunk 2.
  - Fold 2: Train on chunks 1-2, test on chunk 3.
  - ...and so on.
- **When to use:** When you want to ensure your model's performance is stable over time and not just lucky on a specific month's data. You average the metrics across all folds.

---

## 3. Handling the "Empty Relevant" Edge Case

If you use a **Global Temporal Split**, some users might have rated movies in the past (Train), but didn't rate *any* movies $\ge 4.0$ in the future (Test). Their `relevant` list is empty.

**How to handle it:**
1. **Filter them out (Standard for LOO/Temporal):** Before calculating metrics, remove users from the test set who have `len(relevant) == 0`. You cannot calculate Recall or NDCG if the ground truth is empty.
2. **Zero-out (Strict Business Evaluation):** Keep them in the test set, assign their metrics a score of `0.0`, and include them in the final average. This penalizes the model for failing to engage a portion of the user base.

*Recommendation for MovieLens:* Filter them out. Report your metrics as "Evaluated on users with at least one positive interaction in the test set."

---

## 4. The Evaluation Protocol (Negative Sampling)

When calculating ranking metrics (NDCG, Recall) on MovieLens, you cannot ask the model to rank all 20,000 movies in the catalog for every user—it is too slow.

**The Standard Protocol (1:99 or 1:100 Negative Sampling):**
1. Take the **1 positive item** from the user's test set.
2. Randomly sample **99 negative items** that the user has *never interacted with* in the training set.
3. Ask the model to predict scores for these **100 items**.
4. Sort the 100 items by predicted score.
5. Calculate Recall@10 and NDCG@10 based on whether the 1 positive item appears in the top 10.

---