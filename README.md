# ML Assignment 2 — Wine Quality Classification

**Student ID:** 2025AA05369  
**Course:** M.Tech (AIML/DSE) — Machine Learning  
**Institution:** BITS Pilani (WILP)

---

## a. Problem Statement

The dataset contains physicochemical properties of red and white wines. Based on these input features, we predict the **quality tier** of the wine — whether it is of Low, Mid, or High quality. Given a set of chemical measurements for a wine sample, the goal is to classify it into one of three quality tiers using multiple ML classification models and evaluate each model using standard performance metrics.

---

## b. Dataset Description

**Source:** Wine Quality Dataset — UCI Machine Learning Repository / Kaggle  
**Total Samples:** 6,497 (Red: 1,599 + White: 4,898)  
**Input Features:** 11  
**Target Column:** `quality_tier` (1 = Low, 2 = Mid, 3 = High)

| Feature | Description |
|---------|-------------|
| `fixed acidity` | Fixed acidity of the wine |
| `volatile acidity` | Volatile acidity — higher values give a vinegar taste |
| `citric acid` | Citric acid content — adds freshness |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt content in the wine |
| `free sulfur dioxide` | Free SO₂ — prevents microbial growth |
| `total sulfur dioxide` | Total SO₂ (free + bound forms) |
| `density` | Density of the wine |
| `pH` | Acidity or basicity level |
| `sulphates` | Sulphate content — acts as a preservative |
| `alcohol` | Alcohol percentage by volume |

**Class Distribution (Test Set — 1,100 samples):**

| Quality Tier | Score Range | Test Samples | % |
|-------------|-------------|-------------|---|
| 1 — Low | 3–4 | 43 | 3.9% |
| 2 — Mid | 5–6 | 831 | 75.5% |
| 3 — High | 7–9 | 226 | 20.5% |

> ⚠️ Dataset has significant class imbalance — the Mid class dominates at ~75%. Models were trained with `class_weight='balanced'` and `sample_weight` for XGBoost to handle this.

---

## c. Models Used

### Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.2582 | 0.5176 | 0.6239 | 0.2582 | 0.2791 | 0.0257 |
| Decision Tree | 0.3191 | 0.5441 | 0.6359 | 0.3191 | 0.3599 | 0.0439 |
| KNN | 0.7218 | 0.5291 | 0.6092 | 0.7218 | 0.6508 | 0.0001 |
| Naive Bayes | 0.7555 | 0.5534 | 0.5707 | 0.7555 | 0.6502 | 0.0000 |
| Random Forest (Ensemble) | 0.7555 | 0.5133 | 0.5707 | 0.7555 | 0.6502 | 0.0000 |
| XGBoost (Ensemble) | 0.6845 | 0.5101 | 0.6325 | 0.6845 | 0.6540 | 0.0431 |

---

### Model Observations

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| **Logistic Regression** | With `class_weight='balanced'`, the model now attempts to predict all three classes but overall accuracy drops to 25.8%. Precision is reasonable at 0.62 but recall is very low at 0.26, indicating the model still struggles with the highly imbalanced wine quality boundaries. The linear decision boundary is insufficient for this non-linear dataset. |
| **Decision Tree** | With `class_weight='balanced'`, accuracy is 31.9% with improved MCC (0.044). The model now makes more balanced predictions across all three tiers, but the overall accuracy appears lower because it no longer defaults to Mid for everything. Precision of 0.64 shows it is more careful when predicting each class. |
| **KNN** | With `weights='distance'`, achieves 72.2% accuracy and best F1 Score (0.6508) among all models. Distance weighting helps closer neighbours contribute more, making it the most practically useful model in terms of balanced performance across classes. |
| **Naive Bayes** | Unchanged from baseline — GaussianNB does not support class_weight. Accuracy of 75.5% but MCC of 0.0 confirms it still only predicts the Mid class. The feature independence assumption does not hold for correlated wine chemical properties. |
| **Random Forest** | Despite `class_weight='balanced'` with 100 trees, MCC remains 0.0 suggesting it still collapses to predicting Mid. The ensemble is dominated by the majority class signal even with balanced weights. Would benefit from SMOTE oversampling. |
| **XGBoost** | With balanced `sample_weight`, achieves 68.5% accuracy and best MCC (0.043) among ensemble models. The F1 Score of 0.654 shows it makes the most balanced predictions overall. Gradient boosting with sample weighting handles class imbalance better than tree-based ensembles. |

> **Key Insight:** Applying `class_weight='balanced'` redistributes the learning focus across all three quality tiers. KNN with distance weighting emerges as the best overall performer (F1: 0.6508), while XGBoost shows the strongest balance between precision and recall across classes. Naive Bayes and Random Forest remain unaffected due to dataset complexity.

---

## Project Structure

```
MLAssignment2/
├── app.py
├── requirements.txt
├── README.md
├── wine_quality.csv
├── wine_quality - Test.csv
├── 2025AA05369_MLAssignment2.ipynb
├── Screenshot_BITSLabScreenshot.png
└── model/
    ├── LogisticRegression_model.pkl
    ├── DecisionTreeClassifier_model.pkl
    ├── KNeighborsClassifier_model.pkl
    ├── GaussianNB_model.pkl
    ├── RandomForestClassifier_model.pkl
    └── XGBClassifier_model.pkl
```

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit App Features

- 📥 Download sample test CSV (`wine_quality - Test.csv`)
- 📤 Upload your own test CSV with `quality_tier` column
- 🤖 Model selection dropdown (6 classifiers)
- 📊 Display of all 6 evaluation metrics (Accuracy, ROC AUC, Precision, Recall, F1, MCC)
- 🔍 Confusion Matrix heatmap
- 📄 Classification Report table
- 📈 Prediction distribution bar chart
