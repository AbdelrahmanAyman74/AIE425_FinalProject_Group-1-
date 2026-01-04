"""
AIE425 Final Project GROUP 1 
Abdelrahman Ayman Samy Mohamed, 222100930
Yassmin Mohamed Mahmoud Metwally, 222101910
Shahd Mamdouh Ali Hassan, 222102250
Seif Amr Abdelhafez abdo , 222102312

============================================================================
Complete Recommendation System Pipeline
Tasks: Data Loading → Content-Based → Item-CF → SVD-CF → Hybrid → Evaluation
============================================================================
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack, save_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import SVD, Dataset, Reader
import os

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RECOMMENDATION SYSTEM PIPELINE - Starting")
print("="*80)

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1/7] Loading data...")

interactions = pd.read_csv('../data/user_interactions.csv')
courses = pd.read_csv('../data/courses_processed.csv')

# Validate course_index
interactions = interactions[interactions['course_index'].between(0, len(courses)-1)].copy()

print(f"  ✓ Interactions: {interactions.shape[0]:,} ratings")
print(f"  ✓ Courses: {len(courses):,}")
print(f"  ✓ Users: {interactions['user_id'].nunique():,}")

# Train/test split
train_df, test_df = train_test_split(interactions, test_size=0.2, random_state=42)
print(f"  ✓ Train: {len(train_df):,} | Test: {len(test_df):,}")

# ============================================================================
# STEP 2: Build Content-Based Features
# ============================================================================
print("\n[2/7] Building content-based features...")

TITLE_COL = 'course_title' if 'course_title' in courses.columns else courses.columns[0]

# Build combined features from available columns
available_cols = [TITLE_COL]
for col in ['category', 'subcategory', 'level', 'platform', 'description']:
    if col in courses.columns:
        available_cols.append(col)

print(f"  Using columns: {available_cols}")

courses['combined_features'] = courses[available_cols].fillna('').astype(str).agg(' '.join, axis=1)

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
item_feature_matrix = vectorizer.fit_transform(courses['combined_features'])

print(f"  ✓ Item feature matrix: {item_feature_matrix.shape}")
save_npz('../results/item_feature_matrix.npz', item_feature_matrix)

# ============================================================================
# STEP 3: Content-Based Recommender
# ============================================================================
print("\n[3/7] Training content-based recommender...")

train_users = train_df['user_id'].unique()
user_id_to_idx = {u: i for i, u in enumerate(train_users)}

# Build user profiles
profiles = []
for u in train_users:
    u_rows = train_df[train_df['user_id'] == u]
    idx = u_rows['course_index'].values.astype(int)
    r = u_rows['rating'].values.astype(float)
    
    feats = item_feature_matrix[idx]
    weighted_sum = (feats.T.multiply(r)).T.sum(axis=0)
    profile = weighted_sum / (r.sum() + 1e-10)
    profiles.append(csr_matrix(profile))

user_profile_matrix = vstack(profiles, format='csr')
CB_scores = cosine_similarity(user_profile_matrix, item_feature_matrix)

print(f"  ✓ CB scores: {CB_scores.shape}")

# ============================================================================
# STEP 4: Item-Based Collaborative Filtering
# ============================================================================
print("\n[4/7] Training item-based collaborative filtering...")

n_users = len(train_users)
n_items = len(courses)

train_df2 = train_df.copy()
train_df2['user_u'] = train_df2['user_id'].map(user_id_to_idx)

R_train = csr_matrix(
    (train_df2['rating'].values,
     (train_df2['user_u'].values, train_df2['course_index'].values)),
    shape=(n_users, n_items)
)

item_user_matrix = R_train.T
item_item_sim = cosine_similarity(item_user_matrix, dense_output=False)
item_item_sim.setdiag(0)

print(f"  ✓ Item-item similarity: {item_item_sim.shape}, nnz={item_item_sim.nnz:,}")

def predict_rating_itemcf(user_id, target_item_idx, k_neighbors=30, min_neighbors=3):
    if user_id not in user_id_to_idx:
        return 3.0
    u = user_id_to_idx[user_id]
    user_row = R_train[u]
    rated_items = user_row.indices
    rated_ratings = user_row.data
    
    if rated_items.size == 0:
        return 3.0
    
    user_mean = float(np.mean(rated_ratings))
    sims = item_item_sim[target_item_idx, rated_items].toarray().ravel()
    
    nonzero_mask = sims != 0
    if nonzero_mask.sum() < min_neighbors:
        return float(np.clip(user_mean, 1, 5))
    
    sims = sims[nonzero_mask]
    rated_ratings_filtered = rated_ratings[nonzero_mask]
    
    if len(sims) > k_neighbors:
        top_k_idx = np.argsort(np.abs(sims))[::-1][:k_neighbors]
        sims = sims[top_k_idx]
        rated_ratings_filtered = rated_ratings_filtered[top_k_idx]
    
    denom = np.sum(np.abs(sims)) + 1e-10
    pred = np.sum(sims * rated_ratings_filtered) / denom
    return float(np.clip(pred, 1, 5))

# ============================================================================
# STEP 5: SVD-Based Collaborative Filtering (Surprise)
# ============================================================================
print("\n[5/7] Training SVD collaborative filtering...")

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[['user_id', 'course_index', 'rating']], reader)
trainset = data.build_full_trainset()

svd_algo = SVD(n_factors=20, random_state=42)
svd_algo.fit(trainset)

print(f"  ✓ SVD trained with {svd_algo.n_factors} factors")

# ============================================================================
# STEP 6: Hybrid Recommender (Weighted CB + SVD)
# ============================================================================
print("\n[6/7] Building hybrid recommender...")

def minmax_2d(M):
    mn = M.min()
    mx = M.max()
    return np.zeros_like(M) if (mx - mn) < 1e-12 else (M - mn) / (mx - mn)

CB_norm = minmax_2d(CB_scores)

# Get SVD predictions for train users
def get_svd_scores(user_id):
    scores = []
    for item_idx in range(n_items):
        pred = svd_algo.predict(user_id, item_idx)
        scores.append(pred.est)
    return np.array(scores)

BEST_ALPHA = 0.7  # From task 9.1 validation

def recommend_hybrid(user_id, n=10):
    if user_id not in user_id_to_idx:
        # Cold start: use content-based only
        return recommend_content_based(user_id, n)
    
    u = user_id_to_idx[user_id]
    cb_scores = CB_norm[u]
    svd_scores = get_svd_scores(user_id)
    svd_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min() + 1e-10)
    
    hybrid_scores = BEST_ALPHA * cb_scores + (1 - BEST_ALPHA) * svd_norm
    
    # Exclude rated
    rated = set(train_df.loc[train_df['user_id'] == user_id, 'course_index'].values)
    hybrid_scores = hybrid_scores.copy()
    if rated:
        hybrid_scores[list(rated)] = -1
    
    top_idx = np.argsort(hybrid_scores)[::-1][:n]
    return [(int(i), float(hybrid_scores[i])) for i in top_idx]

def recommend_content_based(user_id, n=10):
    if user_id not in user_id_to_idx:
        return []
    u = user_id_to_idx[user_id]
    scores = CB_norm[u].copy()
    rated = set(train_df.loc[train_df['user_id'] == user_id, 'course_index'].values)
    if rated:
        scores[list(rated)] = -1
    top_idx = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in top_idx]

print(f"  ✓ Hybrid ready (alpha={BEST_ALPHA})")

# ============================================================================
# STEP 7: Evaluation & Baselines
# ============================================================================
print("\n[7/7] Evaluating models...")

# Baselines
item_pop = train_df.groupby('course_index').size().sort_values(ascending=False)
mostpop_list = item_pop.index.to_numpy()

def recommend_random(exclude=set(), k=10, seed=42):
    rng = np.random.default_rng(seed)
    pool = np.array([i for i in range(n_items) if i not in exclude], dtype=int)
    if len(pool) <= k:
        return pool.tolist()
    return rng.choice(pool, size=k, replace=False).tolist()

def recommend_mostpop(exclude=set(), k=10):
    recs = []
    for i in mostpop_list:
        if int(i) not in exclude:
            recs.append(int(i))
        if len(recs) == k:
            break
    return recs

# Metrics
def precision_at_k(recs, relevant, k=10):
    if k == 0:
        return 0.0
    recs_k = recs[:k]
    hits = sum(1 for x in recs_k if x in relevant)
    return hits / k

def recall_at_k(recs, relevant, k=10):
    if len(relevant) == 0:
        return np.nan
    recs_k = recs[:k]
    hits = sum(1 for x in recs_k if x in relevant)
    return hits / len(relevant)

def ndcg_at_k(recs, relevant, k=10):
    recs_k = recs[:k]
    gains = np.array([1.0 if x in relevant else 0.0 for x in recs_k])
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_gains = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal_gains * discounts))
    return 0.0 if idcg == 0 else dcg / idcg

# Evaluate on test users
K = 10
REL_TH = 4.0
test_users = test_df['user_id'].unique()
test_users_in_train = [u for u in test_users if u in user_id_to_idx]

print(f"  Evaluating on {len(test_users_in_train)} test users...")

rows = []
eval_count = 0
for u in test_users_in_train:
    u_test = test_df[test_df['user_id'] == u]
    relevant = set(u_test.loc[u_test['rating'] >= REL_TH, 'course_index'].astype(int).tolist())
    if len(relevant) == 0:
        continue
    
    exclude = set(train_df.loc[train_df['user_id'] == u, 'course_index'].astype(int).tolist())
    
    rec_random = recommend_random(exclude=exclude, k=K)
    rec_pop = recommend_mostpop(exclude=exclude, k=K)
    rec_cb = [x[0] for x in recommend_content_based(u, n=K)]
    rec_hybrid = [x[0] for x in recommend_hybrid(u, n=K)]
    
    for name, recs in [("Random", rec_random), ("MostPop", rec_pop), 
                       ("ContentBased", rec_cb), ("Hybrid", rec_hybrid)]:
        rows.append({
            "user_id": u,
            "model": name,
            f"P@{K}": precision_at_k(recs, relevant, k=K),
            f"R@{K}": recall_at_k(recs, relevant, k=K),
            f"NDCG@{K}": ndcg_at_k(recs, relevant, k=K),
        })
    
    eval_count += 1
    if eval_count >= 500:  # Limit for speed
        break

results = pd.DataFrame(rows)
summary = results.groupby("model")[[f"P@{K}", f"R@{K}", f"NDCG@{K}"]].mean().reset_index()
summary = summary.sort_values(by=f"NDCG@{K}", ascending=False)
summary[[f"P@{K}", f"R@{K}", f"NDCG@{K}"]] = summary[[f"P@{K}", f"R@{K}", f"NDCG@{K}"]].round(4)

print("\n" + "="*80)
print("EVALUATION RESULTS (Task 11.2)")
print("="*80)
print(summary.to_string(index=False))
print(f"\nEvaluated on {results['user_id'].nunique()} users")

# Save results
summary.to_csv('../results/baseline_comparison.csv', index=False)
print("✓ Results saved to ../results/baseline_comparison.csv")

# ============================================================================
# Demo Recommendations
# ============================================================================
print("\n" + "="*80)
print("DEMO: Top-10 Hybrid Recommendations")
print("="*80)

demo_user = test_users_in_train[0]
top10 = recommend_hybrid(demo_user, n=10)

print(f"\nUser: {demo_user}")
print("Rank | Course Index | Score   | Title")
print("-" * 80)
for rank, (idx, score) in enumerate(top10, 1):
    title = courses.iloc[idx][TITLE_COL][:50]
    print(f"{rank:4d} | {idx:12d} | {score:.4f} | {title}")

print("\n" + "="*80)
print("PIPELINE COMPLETE ✓")
print("="*80)
