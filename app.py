import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

# -------------------------------------------------
# High Contrast Professional UI
# -------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f8fafc;
}

/* Main Title */
h1 {
    text-align: center;
    color: #7c2d12;
    font-weight: 700;
}

/* Section Headers */
h2, h3 {
    color: #1e293b;
    border-left: 5px solid #7c2d12;
    padding-left: 10px;
}

/* Buttons */
.stButton>button {
    background-color: #7c2d12;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.6em 1.2em;
    border: none;
}

.stButton>button:hover {
    background-color: #991b1b;
    color: white;
}

/* Download Button */
.stDownloadButton>button {
    background-color: #0f766e;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.6em 1.2em;
    border: none;
}

.stDownloadButton>button:hover {
    background-color: #115e59;
}

/* File uploader */
.stFileUploader {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: white;
    color: black;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: white;
    border-radius: 8px;
}

/* Classification Report Table */
.report-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
    background-color: white;
    color: black;
}

.report-table th {
    background-color: #7c2d12;
    color: white;
    padding: 10px;
    text-align: center;
}

.report-table td {
    padding: 8px;
    text-align: center;
    border-bottom: 1px solid #e5e7eb;
}

.report-table tr:hover {
    background-color: #fef3c7;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Title & Description
# -------------------------------------------------
st.title("🍷 Wine Quality Prediction System")
st.markdown("Upload test data and evaluate trained ML models on Wine Quality classification.")

# -------------------------------------------------
# Load trained models
# -------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression":  pickle.load(open("model/LogisticRegression_model.pkl", "rb")),
        "Decision Tree":        pickle.load(open("model/DecisionTreeClassifier_model.pkl", "rb")),
        "Random Forest":        pickle.load(open("model/RandomForestClassifier_model.pkl", "rb")),
        "Naive Bayes":          pickle.load(open("model/GaussianNB_model.pkl", "rb")),
        "K-Nearest Neighbors":  pickle.load(open("model/KNeighborsClassifier_model.pkl", "rb")),
        "XGBoost":              pickle.load(open("model/XGBClassifier_model.pkl", "rb")),
    }

try:
    models = load_models()
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")
    st.stop()

# -------------------------------------------------
# Download Sample Test CSV
# -------------------------------------------------
st.subheader("📥 Sample Test Data (CSV)")

try:
    with open("wine_quality - Test.csv", "rb") as f:
        st.download_button(
            label="⬇️ Download Sample Test CSV",
            data=f,
            file_name="wine_quality - Test.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning("Place `wine_quality - Test.csv` in project folder.")

# -------------------------------------------------
# Upload Dataset
# -------------------------------------------------
st.subheader("📤 Upload Your Test Data")

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV) — must contain `quality_tier` column",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())
    st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

    if "quality_tier" not in data.columns:
        st.error("CSV must contain `quality_tier` column.")
        st.stop()

    X_test = data.drop("quality_tier", axis=1)
    y_test = data["quality_tier"]

    # Model selection
    st.subheader("🤖 Select Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))
    model = models[model_name]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    if model_name == "XGBoost":
        y_pred = y_pred + 1

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob, multi_class="ovr")
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")
    mcc  = matthews_corrcoef(y_test, y_pred)

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy",  f"{acc:.3f}")
    col2.metric("ROC AUC",   f"{auc:.3f}")
    col3.metric("Precision", f"{prec:.3f}")
    col4.metric("Recall",    f"{rec:.3f}")
    col5.metric("F1 Score",  f"{f1:.3f}")
    col6.metric("MCC",       f"{mcc:.3f}")

    # -------------------------------------------------
    # Smaller Confusion Matrix
    # -------------------------------------------------
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=["Low", "Mid", "High"],
        yticklabels=["Low", "Mid", "High"],
        annot_kws={"size": 10},
        cbar=False,
        ax=ax
    )

    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(model_name, fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # Classification Report
    st.subheader("📄 Classification Report")

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Mid", "High"],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose().round(3)

    st.markdown(
        report_df.to_html(classes="report-table", border=0),
        unsafe_allow_html=True
    )

    # Prediction Distribution
    st.subheader("📈 Prediction Distribution")

    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    pred_labels = {1: "Low", 2: "Mid", 3: "High"}
    pred_counts.index = [pred_labels.get(i, i) for i in pred_counts.index]

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    pred_counts.plot(kind="bar", ax=ax2, color=["#b91c1c", "#f59e0b", "#16a34a"])
    ax2.set_title("Predicted Class Distribution")
    ax2.set_xlabel("Quality Tier")
    ax2.set_ylabel("Count")

    st.pyplot(fig2)
