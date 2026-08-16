# Breast Cancer Classification using Multiple ML Models

## a. Problem statement
This assignment requires the implementation of multiple classification models on a public dataset, evaluation using standard ML metrics, and deployment of a Streamlit web application for interactive demonstration. The goal is to compare different models on the same dataset and identify the best-performing model for breast cancer classification.

The project uses the Breast Cancer Wisconsin Diagnostic dataset, which is a binary classification task for distinguishing between malignant and benign tumors. The trained models are evaluated on a held-out test set, and the results are displayed in a small Streamlit dashboard.

## b. Dataset description
The project uses the scikit-learn `load_breast_cancer` dataset.

- Dataset name: Breast Cancer Wisconsin Diagnostic
- Source: scikit-learn
- Problem type: Binary classification
- Number of samples: 569
- Number of features: 30 numeric feature columns
- Target classes:
  - 0 = malignant
  - 1 = benign
- Data quality: no missing values detected
- Class distribution: 62.7% benign, 37.3% malignant (moderate imbalance, acceptable for baseline evaluation with stratified train/test split)

This dataset is well suited for classification benchmarking because it is clean, relatively compact, and widely used in medical ML applications.

## c. GitHub Repository Link
GitHub Repository Link: https://github.com/adityaprasoon/bits-ml-assignment-2

The repository contains the complete project source code, including:
- `app.py`
- `requirements.txt`
- `test_data.csv`
- `model/` folder with serialized trained models
- `README.md`

## d. Models used
The project implements the following five classification models as required:
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor (KNN) Classifier
- Gaussian Naive Bayes
- Random Forest Classifier

## Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Random Forest | 0.9561 | 0.9931 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |
| KNN | 0.9561 | 0.9788 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |
| Gaussian Naive Bayes | 0.9298 | 0.9868 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| Decision Tree | 0.9123 | 0.9157 | 0.8478 | 0.9286 | 0.8864 | 0.8174 |

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall model on this dataset, with highest accuracy, AUC, F1, and MCC. It provides the strongest balance of sensitivity and specificity for the malignant classification task. |
| Decision Tree | Performs reasonably well but has lower predictive power compared to the top models. It is more prone to local decision boundaries and weaker generalization. |
| KNN | Strong performance, especially with scaled features. It is competitive but remains slightly below Logistic Regression in overall validation metrics. |
| Naive Bayes | Performs well with a good AUC, but accuracy and precision are lower than Logistic Regression and Random Forest. |
| Random Forest | Excellent ensemble performance with strong AUC and competitive accuracy; it is close to Logistic Regression, but the latter is still the leading model on this dataset. |

Overall Winner for this dataset: Logistic Regression

## Live Streamlit App Link
Live Streamlit App Link: <replace-with-your-streamlit-cloud-app-url>

The Streamlit app includes:
- dataset upload option (CSV)
- model selection dropdown
- evaluation metrics display
- confusion matrix
- classification report
- model comparison table

## Local execution
Clone the repository to your local machine, then run the following commands from the root of the cloned project folder:

```bash
git clone <this-github-repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
