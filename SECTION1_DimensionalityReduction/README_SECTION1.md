# AIE 425 Final Project – Intelligent Recommender Systems  
## Section 1: Dimensionality Reduction Methods for Recommendation

**Course**: AIE 425 Intelligent Recommender Systems  
**Institution**: Galala University  
**Academic Year**: 2025–2026  
**Group**: 1  

## Group Members
- **Abdelrahman Ayman Samy Mohamed** — 222100930  
- **Yassmin Mohamed Mahmoud Metwally** — 222101910  
- **Shahd Mamdouh Ali Hassan** — 222102250  
- **Seif Amr Abdelhafez Abdo** — 222102312  

---

## Overview
This section investigates **dimensionality reduction techniques** for collaborative filtering–based recommendation systems using the **MovieLens dataset**. Three methods are implemented and compared:

1. PCA with mean-filling  
2. PCA with MLE-based item covariance  
3. SVD-based matrix factorization  

The goal is to analyze how global and local latent structures affect rating prediction quality.

---

## Project Structure

SECTION1_DimensionalityReduction/
│
├── data/
│ ├── ratings.csv
│ ├── ratings_subset.csv
│ ├── Target_Users_U1_U2_U3.csv
│ └── Target_Items_I1_I2.csv
│
├── results/
│ ├── tables/
│ └── plots/
│
├── code/
│ ├── pca_mean_filling.ipynb
│ ├── pca_mle.ipynb
│ ├── svd_analysis.ipynb
│
│ 
└── README_SECTION1.md


---

## Dataset

- **Source**: MovieLens ratings dataset
- **Full dataset**: ~138,000 users × 27,000 movies (original MovieLens)
- **Project subset**: 11,000 users × 600 items with 3.1M ratings
- **Target entities**: 
  - Users: U1=8405, U2=118205, U3=88604
  - Items: I1=1562, I2=2701
- **Rating scale**: 1.0 to 5.0 stars

---

## Implemented Methods

### Part 1: PCA with Mean-Filling
Applies PCA on a mean-filled user–item matrix to capture **global latent structure**.  
Evaluated using MAE and RMSE across different latent dimensions.

### Part 2: PCA with MLE Covariance
Computes **item–item covariance** using pairwise-complete observations and predicts ratings via weighted peer aggregation.  
Focuses on **local similarity under sparsity**.

### Part 3: SVD Matrix Factorization
Directly factorizes the sparse rating matrix into user and item latent factors.  
Avoids imputation bias and is suitable for large sparse datasets.

---

## Key Findings
- PCA mean-filling provides a strong global baseline but introduces imputation bias  
- MLE covariance captures local similarity more effectively under sparsity  
- SVD offers the most robust formulation for scalable recommender systems  
- Top-5 neighbors capture most similarity signal; larger k adds marginal benefit  

---

## Execution Order
1. `pca_mean_filling.ipynb`  
2. `pca_mle.ipynb`  
3. `svd_analysis.ipynb`  

Each notebook is self-contained and produces outputs in `results/tables/` or `results/plots/`.



