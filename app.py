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
# Page Config  (must be FIRST Streamlit call)
# -------------------------------------------------
st.set_page_config(page_title="Wine Quality Classifier", layout="wide")

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
        "Logistic Regression":  pickle.load(open("model/LogisticRegression_model.pkl",     "rb")),
        "Decision Tree":        pickle.load(open("model/DecisionTreeClassifier_model.pkl",      "rb")),
        "Random Forest":        pickle.load(open("model/RandomForestClassifier_model.pkl",      "rb")),
        "Naive Bayes":          pickle.load(open("model/GaussianNB_model.pkl",     "rb")),
        "K-Nearest Neighbors":  pickle.load(open("model/KNeighborsClassifier_model.pkl",     "rb")),
        "XGBoost":              pickle.load(open("model/XGBClassifier_model.pkl",     "rb")),
    }

try:
    models = load_models()
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")
    st.info("Make sure all .pkl files are inside a `model/` folder in the same directory as app.py")
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
    st.warning("Sample CSV file not found. Place `wine_quality - Test.csv` in the project folder.")

# -------------------------------------------------
# Upload Dataset (CSV)
# -------------------------------------------------
st.subheader("📤 Upload Your Test Data")

uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV) — must contain `quality_tier` column  |  use `wine_quality - Test.csv`",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Preview
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(data.head())
    st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

    # Validate target column
    if "quality_tier" not in data.columns:
        st.error("Uploaded file must contain a `quality_tier` column as the target label (1=Low, 2=Mid, 3=High).")
        st.stop()

    # Split features and target
    X_test = data.drop("quality_tier", axis=1)
    y_test = data["quality_tier"]

    # -------------------------------------------------
    # Model Selection
    # -------------------------------------------------
    st.subheader("🤖 Select Model")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))
    model = models[model_name]

    # -------------------------------------------------
    # Predict
    # -------------------------------------------------
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Fix label shift for XGBoost (predicts 0,1,2 → need 1,2,3)
    if model_name == "XGBoost":
        y_pred = y_pred + 1

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
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
    # Confusion Matrix
    # -------------------------------------------------
    st.subheader("🔍 Confusion Matrix")

    plt.close("all")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=["Low", "Mid", "High"],
        yticklabels=["Low", "Mid", "High"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11)
    ax.tick_params(axis='both', labelsize=9)

    st.pyplot(fig)

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------
    st.subheader("📄 Classification Report")

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Mid", "High"],
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose().round(3)

    st.markdown("""
    <style>
    .report-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }
    .report-table th {
        background-color: #7f1d1d;
        color: white;
        padding: 10px;
        text-align: center;
    }
    .report-table td {
        padding: 8px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    .report-table tr:hover {
        background-color: #fef2f2;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        report_df.to_html(classes="report-table", border=0),
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # Class Distribution of Predictions
    # -------------------------------------------------
    st.subheader("📈 Prediction Distribution")

    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    pred_labels = {1: "Low", 2: "Mid", 3: "High"}
    pred_counts.index = [pred_labels.get(i, i) for i in pred_counts.index]

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    pred_counts.plot(kind="bar", ax=ax2,
                     color=["#ef4444", "#f59e0b", "#22c55e"],
                     edgecolor="none")
    ax2.set_title("Predicted Class Distribution", fontsize=11)
    ax2.set_xlabel("Quality Tier")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=0)
    plt.tight_layout()

    st.pyplot(fig2)
