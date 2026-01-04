# AIE 425 Final Project – Intelligent Recommender Systems

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

## Project Overview

This project implements and evaluates recommender system techniques as part of the AIE 425 course. It is organized into **two main sections**:

* **Section 1 – Dimensionality Reduction & Analysis**: Exploratory analysis and dimensionality reduction using PCA and SVD.
* **Section 2 – Domain-Specific Course Recommender**: A hybrid recommender system combining content-based and collaborative filtering for course recommendation.

The project follows standard academic methodologies and evaluates performance using established metrics.

---

## Repository Structure

```
AIE425_FinalProject_Group1/
│
├── README.md                     # Root documentation (this file)
├── requirements.txt              # Python dependencies
├── Final_Report.docx             # Final written report
│
├── SECTION1_DimensionalityReduction/
│   ├── README_SECTION1.md
│   ├── code/
│   │   ├── pca_mean_filling.ipynb
│   │   ├── pca_mle.ipynb
│   │   ├── svd_analysis.ipynb
│   │   └── utils.ipynb
│   ├── data/
│   │   ├── ratings.csv
│   │   └── ratings_subset.csv
│   └── results/
│       ├── plots/
│       └── tables/
│
└── SECTION2_DomainRecommender/
    ├── README_SECTION2.md
    ├── code/
    │   ├── data_preprocessing.ipynb
    │   ├── content_based.ipynb
    │   ├── collaborative.ipynb
    │   ├── hybrid.ipynb
    │   └── main.py
    ├── data/
    │   ├── coursera_data.csv
    │   ├── udemy_courses.csv
    │   ├── courses_processed.csv
    │   └── user_interactions.csv
    └── results/
```

---

## Section 1 – Dimensionality Reduction

**Objective:** Analyze rating data and apply dimensionality reduction techniques.

Methods used:

* Principal Component Analysis (PCA) with mean filling
* PCA with Maximum Likelihood Estimation (MLE)
* Singular Value Decomposition (SVD)

All experiments are implemented as Jupyter notebooks. Outputs include plots and summary tables saved in the `results/` directory.

Refer to `SECTION1_DimensionalityReduction/README_SECTION1.md` for details.

---

## Section 2 – Domain Recommender System

**Objective:** Build a personalized course recommender system.

Implemented models:

* Content-Based Filtering (TF-IDF + cosine similarity)
* Item-Based Collaborative Filtering
* SVD-based Collaborative Filtering
* Weighted Hybrid Recommender

Evaluation metrics:

* Precision@10
* Recall@10
* NDCG@10

All development notebooks are provided for transparency, while `main.py` executes the full end-to-end pipeline.

Refer to `SECTION2_DomainRecommender/README_SECTION2.md` for full methodology and results.

---

## Installation & Requirements

* **Python**: 3.11 (recommended)
* **Key Libraries**:

  * numpy
  * pandas
  * scipy
  * scikit-learn
  * scikit-surprise
  * jupyter

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Section 1 (Analysis)

```bash
cd SECTION1_DimensionalityReduction
jupyter notebook
```

Run notebooks in logical order as listed in the `code/` directory.

### Section 2 (Recommender System)

```bash
cd SECTION2_DomainRecommender/code
py -3.11 main.py
```

---

## Academic Notes

* The project adheres to standard recommender system theory taught in AIE 425.
* All code and analysis represent original academic work.
* External methods are cited and implemented following course guidelines.

---

## Contributors

Group 1 – AIE 425

---

## License

Academic Use Only.
Developed for educational purposes as part of AIE 425 coursework.

---

**Last Updated**: January 2026
