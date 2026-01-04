# Section 2 – Domain-Specific Course Recommender System

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

This section implements a **hybrid course recommender system** that integrates **content-based filtering** and **collaborative filtering** to generate personalized course recommendations. The system is designed to address **rating sparsity** and **cold-start users**, and its performance is evaluated using standard **Top-K ranking metrics**.

---

## Folder Structure

```
SECTION2_DomainRecommender/
│
├── data/                         # Input datasets
│
├── code/                         # Implementation notebooks and script
│   ├── data_preprocessing.ipynb  # Data loading, cleaning, and splitting
│   ├── content_based.ipynb       # TF-IDF feature extraction & CB model
│   ├── collaborative.ipynb       # Item-based CF and SVD (Surprise)
│   ├── hybrid.ipynb              # Hybrid recommendation logic
│   └── main.py                   # End-to-end execution pipeline
│
├── results/                      # Generated outputs
│
└── README_SECTION2.md            # This file
```

> All modules except `main.py` are implemented as **Jupyter notebooks (`.ipynb`)** for clarity and experimentation. The final pipeline is executed through `main.py`.

---

## Methodology Summary

### Content-Based Filtering

* Course metadata is transformed using **TF-IDF** (unigrams + bigrams).
* User profiles are computed as **rating-weighted averages** of course vectors.
* Recommendations are generated using **cosine similarity**.

### Collaborative Filtering

* **Item-based CF** with cosine similarity and Top-K neighbors.
* **SVD matrix factorization** using the Surprise library with bias terms.

### Hybrid Model

* Normalized CB and CF scores are combined using a **weighted hybrid**:
  [ Score(u,i) = 0.7 \times CB(u,i) + 0.3 \times CF(u,i) ]
* Emphasizes content-based signals to improve cold-start performance.

---

## Evaluation

Models compared:

* Random
* Most Popular
* Content-Based
* Hybrid (CB + SVD)

**Metrics (K = 10):**

* Precision@10
* Recall@10
* NDCG@10

Relevance threshold: **rating ≥ 4.0**.
Results are saved to `results/baseline_comparison.csv`.

---

## Execution

From the `code/` directory:

```bash
py -3.11 main.py
```


**Institution**: Galala University
**Academic Year**: 2025–2026
