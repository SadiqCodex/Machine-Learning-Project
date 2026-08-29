# Machine Learning Project

**A structured, hands-on Machine Learning portfolio — from data preprocessing and classical algorithms to ensemble methods, imbalance handling, and unsupervised learning.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Machine Learning](https://img.shields.io/badge/Focus-Machine%20Learning-success)](https://github.com/SadiqCodex/Machine-Learning-Project)
[![Repo](https://img.shields.io/badge/GitHub-Machine--Learning--Project-181717?logo=github)](https://github.com/SadiqCodex/Machine-Learning-Project)

---

## Overview

This repository is a practical Machine Learning learning journey implemented primarily in **Jupyter notebooks**. It covers end-to-end workflows: loading CSV datasets, exploratory analysis, feature encoding and scaling, train/test splitting, model training, evaluation with classification and regression metrics, decision-boundary visualization, and comparison of multiple algorithms.

The notebooks progress from fundamentals (linear/logistic regression, preprocessing) to intermediate and advanced topics (polynomial features, decision trees, KNN, Naive Bayes, SVM, bias–variance trade-off, imbalanced data resampling, ensemble learning, clustering, and association rule mining).

> **Note:** Content is based on the notebooks, datasets, and model artifacts present in this repository. Filenames retain their original spellings as committed in the project.

---

## Learning Objectives

By working through this repository, you will be able to:

- Prepare tabular data with encoding, imputation, and feature scaling
- Build and evaluate **regression** and **classification** models with scikit-learn
- Apply **polynomial feature expansion** for non-linear decision boundaries and curves
- Interpret **confusion matrices**, precision, recall, and F1-score
- Compare distance-based (**KNN**), probabilistic (**Naive Bayes**), tree-based, and margin-based (**SVM**) learners
- Explore **bias–variance** effects via polynomial degree on logistic models
- Handle **class imbalance** using undersampling and oversampling (`imblearn`)
- Combine models with **voting**, **bagging**, **random forests**, **AdaBoost**, and **gradient boosting**
- Apply **unsupervised learning**: K-Means, hierarchical clustering, Apriori, and FP-Growth
- Use **GridSearchCV** for hyperparameter tuning and feature selection utilities

---

## Machine Learning Topics Covered

| Area | Topics confirmed in notebooks |
|------|-------------------------------|
| **Supervised — Regression** | Linear regression, multiple linear regression, polynomial regression, KNN regressor, decision tree regressor, SVR, Ridge & Lasso regularization |
| **Supervised — Classification** | Logistic regression (binary & multiclass), polynomial logistic classification, decision trees, KNN, Naive Bayes variants, SVC |
| **Ensemble Learning** | Voting (hard/soft-style setups), bagging, random forest, AdaBoost, gradient boosting |
| **Unsupervised Learning** | K-Means (`k-means++`), hierarchical / agglomerative clustering, association rules (Apriori, FP-Growth) |
| **Data Preparation** | Label / one-hot / ordinal encoding, simple imputation, StandardScaler, MinMaxScaler |
| **Evaluation & Diagnostics** | Train/test split, accuracy, precision, recall, F1, MAE / MSE / RMSE / R², confusion matrix, silhouette score, cross-validation & GridSearchCV |
| **Core ML Concepts** | Bias–variance trade-off, imbalanced datasets, OvR vs multinomial multiclass logistic regression, feature selection |

---

## Algorithms & Techniques

### Regression
- `LinearRegression`, `PolynomialFeatures` + linear models
- `KNeighborsRegressor`, `DecisionTreeRegressor`, `SVR`
- Regularization: `Ridge`, `Lasso` (with `StandardScaler`)

### Classification
- `LogisticRegression` (including `multi_class="multinomial"`)
- `DecisionTreeClassifier` (Gini / entropy, depth control)
- `KNeighborsClassifier`
- `GaussianNB`, `MultinomialNB`, `BernoulliNB`
- `SVC` (e.g. linear kernel)

### Ensembles
- `VotingClassifier` / `VotingRegressor`
- `BaggingClassifier` / `BaggingRegressor`
- `RandomForestClassifier` / `RandomForestRegressor`
- `AdaBoostClassifier`, `GradientBoostingRegressor`

### Clustering & Pattern Mining
- `KMeans`, `AgglomerativeClustering`, SciPy `linkage`
- `TransactionEncoder` + `apriori` / `fpgrowth` (mlxtend)

### Resampling & Search
- `RandomUnderSampler`, `RandomOverSampler` (imbalanced-learn)
- `GridSearchCV`, `cross_val_score`
- `SequentialFeatureSelector` (mlxtend) for backward feature selection

### Visualization & Metrics
- Matplotlib, Seaborn, `mlxtend.plotting` decision regions
- `confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
- `mean_absolute_error`, `mean_squared_error`, `r2_score`, `silhouette_score`

---

## Project / Notebook Overview

| Notebook | Purpose |
|----------|---------|
| `practice1.ipynb` | Data cleaning & encoding on loan data (`SimpleImputer`, `LabelEncoder`, `OneHotEncoder`, `OrdinalEncoder`) |
| `practice2.ipynb` | Exploratory analysis and scaling (`StandardScaler`, `MinMaxScaler`) on loan features |
| `practice3.ipynb` | Backward sequential feature selection with logistic regression on diabetes data |
| `practice4.ipynb` | Train/test splitting workflow on house price data |
| `dataset_preprocessing.ipynb` | Scaling demos, linear/logistic baselines, KNN & decision tree classifiers, MAE/MSE/RMSE/R² walkthrough |
| `package_predictor.ipynb` | Simple linear regression: CGPA → package |
| `multiple_linear.reg.ipynb` | Multiple linear regression: age & experience → salary |
| `polynomial_salary_model.ipynb` | Polynomial regression on level–salary data |
| `regression_predictor.ipynb` | House-price regression with Linear / Lasso / Ridge after scaling |
| `k_nearest-regrassor.ipynb` | K-Nearest Neighbors regression on salary data |
| `multiple_decision_tree.ipynb` | Decision tree regression on salary features |
| `job_predictino.ipynb` | Logistic regression for placement prediction (CGPA + score → placed) |
| `subscription_prediction.ipynb` | Scaled logistic regression for purchase / subscription prediction |
| `polynomial_classifier.ipynb` | Polynomial features + logistic regression for non-linear classification |
| `polynomial_classifier_test.ipynb` | Baseline logistic regression comparison on the same logistic dataset |
| `confusion_matrix.ipynb` | Classification metrics & confusion matrix on placement data |
| `multiple_classification.ipynb` | Multiclass logistic regression on iris (OvR vs multinomial) |
| `decision_tree.ipynb` | Decision tree classification (depth / criterion) on subscription data |
| `k_nearest-classificatoin.ipynb` | KNN classification with neighbor search / scoring |
| `bayes_theroam_dataset.ipynb` | Gaussian, Multinomial, and Bernoulli Naive Bayes on placement data |
| `Support Vector Machines(SVM) - Classification.ipynb` | SVC/SVR demos, decision tree regressors, and `GridSearchCV` tuning |
| `Bias–Variance.ipynb` | Polynomial logistic models at degrees 1 / 3 / 10 to illustrate complexity vs test accuracy |
| `imbalance_dataset.ipynb` | Logistic regression on imbalanced subscription-style purchase data |
| `imblearn.ipynb` | Random undersampling & oversampling before logistic classification |
| `esemble_learning.ipynb` | Voting, bagging, random forests, AdaBoost, gradient boosting comparisons |
| `unsupervised_k_mean_cluster.ipynb` | K-Means, hierarchical clustering, Apriori & FP-Growth on iris / grocery data |

---

## Repository Structure

```text
Machine-Learning-Project/
├── datasets/                          # CSV datasets used across notebooks
│   ├── cgpa_package.csv
│   ├── cgpa_score_placement.csv
│   ├── subscription_dataset.csv
│   ├── salary_dataset.csv
│   ├── salary_new_dataset.csv
│   ├── level_salary.csv
│   ├── logistic_dataset.csv
│   ├── iris_*.csv
│   ├── loan_data.csv
│   ├── house_price.csv
│   ├── grocery.csv
│   ├── diabetes.csv
│   ├── student.csv / student_data.csv
│   ├── regularization_house_dataset.csv
│   └── ...                            # additional CSVs for future / related work
├── models/                            # Saved model artifacts
│   ├── quote_lstm_model.h5
│   ├── tokenizer.pkl
│   └── max_len.pkl
├── *.ipynb                            # Topic-focused Jupyter notebooks
└── README.md
```

Notebooks typically load data with relative paths such as `../datasets/<file>.csv`. Run them from an environment where that path resolves correctly (project root or adjust paths as needed).

---

## Datasets & Models

### Datasets used in notebooks

| Dataset | Typical use in this repo |
|---------|--------------------------|
| `cgpa_package.csv` | Linear regression & ensemble regression demos (CGPA → package) |
| `cgpa_score_placement.csv` | Placement classification (CGPA, score → placed) |
| `subscription_dataset.csv` | Purchase classification; bias–variance, trees, KNN, imbalance labs |
| `salary_dataset.csv` / `salary_new_dataset.csv` | Multiple linear / tree / KNN regression |
| `level_salary.csv` | Polynomial salary modeling; SVM-related experiments |
| `logistic_dataset.csv` | Polynomial vs linear logistic classification |
| `iris_multiclass.csv` | Multiclass logistic regression |
| `iris_data.csv` / `iris_data_species.csv` | Clustering experiments |
| `loan_data.csv` | Preprocessing, encoding, scaling practice |
| `house_price.csv` | Train/test split practice |
| `regularization_house_dataset.csv` | Linear / Ridge / Lasso house-price modeling |
| `diabetes.csv` | Feature selection with logistic regression |
| `grocery.csv` | Market-basket / association rules |
| `student.csv` / `student_data.csv` / `citizen_data.csv` | Preprocessing & small-scale modeling demos |

### Additional artifacts

- Extra CSVs under `datasets/` (e.g. `qoute_dataset.csv`, `bank_customer_dl_dataset.csv`, `dl_cgpa_score_placement.csv`) are present for related or future experiments.
- `models/` contains saved artifacts (`quote_lstm_model.h5`, `tokenizer.pkl`, `max_len.pkl`) associated with quote-related modeling work.

---

## Tools & Technologies

| Category | Libraries / tools |
|----------|-------------------|
| Language | Python 3.x |
| Environment | Jupyter Notebook |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, mlxtend plotting |
| ML core | scikit-learn |
| Imbalance | imbalanced-learn (`imblearn`) |
| Extra ML utilities | mlxtend (plotting, frequent patterns, feature selection) |
| Scientific | SciPy (hierarchical clustering / linkage) |

---

## Installation / Environment Setup

```bash
# Clone the repository
git clone https://github.com/SadiqCodex/Machine-Learning-Project.git
cd Machine-Learning-Project

# (Recommended) create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# Install core dependencies
pip install notebook pandas numpy matplotlib seaborn scikit-learn imbalanced-learn mlxtend scipy
```

Optional (if you work with the saved Keras/H5 artifacts under `models/`):

```bash
pip install tensorflow
```

---

## How to Run the Notebooks

```bash
# From the project root
jupyter notebook
```

Or with JupyterLab:

```bash
jupyter lab
```

Then open any `.ipynb` file and run cells top-to-bottom (**Run All**).

**Tips**

1. Confirm dataset paths (`../datasets/...`) match your working directory.
2. Prefer running topic notebooks in the order suggested in [Workflow / Learning Pipeline](#workflow--learning-pipeline).
3. Some notebooks include interactive plots or decision-region visualizations — ensure a GUI/inline backend (`%matplotlib inline` where used).

---

## Workflow / Learning Pipeline

A practical path through the repository:

```text
1. Data foundations
   practice1 → practice2 → practice4 → dataset_preprocessing

2. Regression fundamentals
   package_predictor → multiple_linear.reg → polynomial_salary_model
   → regression_predictor → k_nearest-regrassor → multiple_decision_tree

3. Classification fundamentals
   job_predictino → subscription_prediction → confusion_matrix
   → polynomial_classifier → multiple_classification

4. Classical algorithms
   decision_tree → k_nearest-classificatoin → bayes_theroam_dataset
   → Support Vector Machines(SVM) - Classification

5. Model behavior & data challenges
   Bias–Variance → imbalance_dataset → imblearn

6. Ensembles & unsupervised learning
   esemble_learning → unsupervised_k_mean_cluster

7. Feature engineering extras
   practice3 (sequential feature selection)
```

---

## Key Concepts Demonstrated

- **Supervised learning pipeline** — features / labels → split → fit → predict → score
- **Feature scaling** — `StandardScaler` and `MinMaxScaler` before distance- or regularized models
- **Non-linear modeling** — `PolynomialFeatures` for regression curves and classification boundaries
- **Tree depth & criterion** — controlling decision tree complexity (`max_depth`, `entropy` / Gini)
- **Neighbor sensitivity** — varying `n_neighbors` for KNN classification
- **Probabilistic classifiers** — Gaussian / Multinomial / Bernoulli Naive Bayes
- **Margin-based learning** — linear SVC / SVR
- **Bias–variance intuition** — rising polynomial degree can reduce test accuracy on held-out data
- **Class imbalance** — resampling minority/majority classes before logistic training
- **Ensemble aggregation** — combining diverse base learners via voting, bagging, and boosting
- **Unsupervised structure discovery** — cluster count selection cues (e.g. silhouette) and frequent itemsets
- **Hyperparameter search** — `GridSearchCV` on decision tree regressors
- **Evaluation literacy** — confusion matrix and regression error metrics beyond raw accuracy

---

## Results / Observations

The following observations are taken from **executed notebook outputs** in this repository (values may vary if you re-run with different random splits or library versions):

| Experiment | Observed result (from notebook output) |
|------------|----------------------------------------|
| Bias–variance (`Bias–Variance.ipynb`) | Polynomial logistic regression test accuracy: **degree 1 → 82.50%**, **degree 3 → 77.50%**, **degree 10 → 72.50%** (160 train / 40 test on subscription data) |
| Ensemble voting (`esemble_learning.ipynb`) | Example voting classifier training score reported as **Voting Accuracy: 97.625**; AdaBoost test **Accuracy: 0.872** |
| Ensemble regressors | Example scores printed for LR / SVR / DT / voting regressor on package-style data (e.g. voting ~**89.3** train / ~**88.1** related score prints in notebook) |
| Preprocessing metrics demo (`dataset_preprocessing.ipynb`) | Sample classification metrics: **Accuracy ≈ 0.714**, **Precision/Recall/F1 = 0.75**; regression demos include MAE/MSE/RMSE and an R² example of **0.0002** on a weak fit case |
| SVM / tree search (`Support Vector Machines(SVM) - Classification.ipynb`) | Grid/random search best scores on the order of **~99.9** (as printed for tuned decision tree regressor setups) |

These figures illustrate experimental outcomes in the learning notebooks; they are **not** claimed as production benchmarks.

---

## Future Improvements

- Add a pinned `requirements.txt` / `environment.yml` for fully reproducible installs
- Normalize dataset paths and notebook naming for consistency
- Consolidate repeated EDA/training boilerplate into shared utility modules
- Add markdown explanations and learning notes inside code-heavy notebooks
- Expand evaluation with learning curves, ROC-AUC where applicable, and clearer train vs test reporting
- Document or add notebooks that load the saved artifacts under `models/`
- Add unit-style smoke tests for data loading and minimal model fits
- Introduce a simple project license file

---

## Contributing

Contributions that improve clarity, reproducibility, or coverage are welcome.

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/your-improvement`)  
3. Commit clear, focused changes  
4. Open a pull request describing what you changed and why  

Please keep notebooks runnable, avoid committing large unrelated binaries when possible, and prefer documenting observed metrics rather than unverified claims.

---

## Author

**Sadik Mohammad** ([SadiqCodex](https://github.com/SadiqCodex))

- Repository: [Machine-Learning-Project](https://github.com/SadiqCodex/Machine-Learning-Project)
- Focus: practical Machine Learning implementations with Python & Jupyter

---

## Closing

This repository is a **hands-on Machine Learning portfolio**: structured notebooks, real CSV workflows, and progressive coverage from preprocessing through supervised, ensemble, and unsupervised techniques. Clone it, run the notebooks in order, experiment with hyperparameters, and treat each notebook as a building block toward stronger applied ML intuition.

If this project helps your learning journey, consider starring the repository and exploring related experiments on [GitHub @SadiqCodex](https://github.com/SadiqCodex).
