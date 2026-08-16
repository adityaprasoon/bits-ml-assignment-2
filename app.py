from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def get_positive_label(y):
    labels = pd.Series(y).unique()
    return 0 if 0 in labels else 1


def get_model_classes(model):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].classes_
    return model.classes_


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    classes = get_model_classes(model)
    positive_label = get_positive_label(y)

    if positive_label not in classes:
        positive_label = int(classes[0])

    positive_index = list(classes).index(positive_label)
    if hasattr(model, "predict_proba"):
        prob_values = model.predict_proba(X)[:, positive_index]
        auc_score = roc_auc_score((y == positive_label).astype(int), prob_values)
    else:
        auc_score = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc_score,
        "Precision": precision_score(y, y_pred, pos_label=positive_label),
        "Recall": recall_score(y, y_pred, pos_label=positive_label),
        "F1": f1_score(y, y_pred, pos_label=positive_label),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


@st.cache_data
def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    default_path = Path(__file__).resolve().parent / "test_data.csv"
    if default_path.exists():
        return pd.read_csv(default_path)

    return None


@st.cache_resource
def load_model(model_name):
    model_path = Path(__file__).resolve().parent / "model" / MODEL_FILES[model_name]
    return joblib.load(model_path)


def build_comparison_table(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    rows = []

    for model_name in MODEL_FILES:
        model = load_model(model_name)
        metrics, _ = evaluate_model(model, X, y)
        rows.append({"ML Model Name": model_name, **metrics})

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by=["F1", "AUC", "Accuracy"], ascending=False)
    return comparison_df.reset_index(drop=True)


def main():
    st.set_page_config(page_title="Breast Cancer Classifier Dashboard", layout="wide")
    st.title("Breast Cancer Classification Dashboard")
    st.caption("This app evaluates the trained models on the provided test dataset.")

    uploaded_file = st.sidebar.file_uploader("Upload test CSV (optional)", type=["csv"])
    dataset = load_dataset(uploaded_file)

    if dataset is None:
        st.error("No dataset found. Please upload a CSV file containing a 'target' column or keep the default test_data.csv in the project folder.")
        st.stop()

    if "target" not in dataset.columns:
        st.error("The uploaded CSV must contain a 'target' column.")
        st.stop()

    feature_columns = [col for col in dataset.columns if col != "target"]
    X = dataset[feature_columns]
    y = dataset["target"]

    st.sidebar.write(f"Loaded rows: {len(dataset)}")
    st.sidebar.write(f"Feature columns: {len(feature_columns)}")
    st.sidebar.write(f"Target distribution: {y.value_counts().to_dict()}")

    selected_model = st.selectbox("Choose a model", list(MODEL_FILES.keys()))

    model = load_model(selected_model)
    metrics, y_pred = evaluate_model(model, X, y)

    st.subheader(f"Selected model: {selected_model}")
    metric_cols = st.columns(6)
    metric_values = [
        ("Accuracy", metrics["Accuracy"]),
        ("AUC", metrics["AUC"]),
        ("Precision", metrics["Precision"]),
        ("Recall", metrics["Recall"]),
        ("F1", metrics["F1"]),
        ("MCC", metrics["MCC"]),
    ]

    for col, (label, value) in zip(metric_cols, metric_values):
        col.metric(label, f"{value:.4f}")

    positive_label = get_positive_label(y)
    labels = [0, 1]
    cm = confusion_matrix(y, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Confusion Matrix - {selected_model}")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = pd.DataFrame(
        {
            "Class": ["Malignant", "Benign"],
            "Precision": [
                precision_score(y, y_pred, pos_label=positive_label, zero_division=0),
                precision_score(y, y_pred, pos_label=1 - positive_label, zero_division=0),
            ],
            "Recall": [
                recall_score(y, y_pred, pos_label=positive_label, zero_division=0),
                recall_score(y, y_pred, pos_label=1 - positive_label, zero_division=0),
            ],
            "F1": [
                f1_score(y, y_pred, pos_label=positive_label, zero_division=0),
                f1_score(y, y_pred, pos_label=1 - positive_label, zero_division=0),
            ],
        }
    )
    st.dataframe(report, use_container_width=True)

    st.subheader("Model Comparison")
    comparison_df = build_comparison_table(dataset)
    st.dataframe(comparison_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "MCC": "{:.4f}",
    }), use_container_width=True)


if __name__ == "__main__":
    main()
