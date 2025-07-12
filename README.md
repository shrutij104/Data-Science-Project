# Data-Science-Project
# 📊 Customer Churn Prediction – Sony Research Assignment

This project was completed as part of a take-home assignment used in the recruitment process for Data Science roles at Sony Research. The objective is to analyze a telecom company's customer data to predict churn and provide actionable business insights.

---

## 📁 Dataset

The dataset used is: `Data_Science_Challenge.csv`, which contains telecom customer data with several numerical and categorical features, including a churn indicator.

---

## 🧠 Objectives

1. Perform **exploratory data analysis** and extract insights.
2. **Preprocess** the data and engineer features.
3. **Split** the dataset into training and testing sets with appropriate reasoning.
4. **Build and evaluate** various machine learning models for churn prediction.
5. Choose the best-performing model based on relevant **evaluation metrics**.
6. Discuss potential **deployment issues**.

---

## 📌 Steps Covered in the Notebook

### 1. 📋 Data Inspection
- Shape, data types, null values, duplicates
- Summary statistics
- Class imbalance
- Outlier detection using IQR

### 2. 🧹 Data Preprocessing
- Convert column types
- Drop irrelevant features
- Impute or remove missing values

### 3. 📊 Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis
- Distribution plots (histograms, bar plots)
- Correlation matrix for numeric features
- Association matrix for categorical features
- Feature impact on churn

### 4. 🏗 Feature Engineering
- Handling non-normal distributions
- New feature creation (e.g., binary indicators)
- Categorical encoding:
  - Direct mapping
  - Frequency encoding
  - Target encoding
- Scaling with `RobustScaler` and `StandardScaler`

### 5. 🧪 Train-Test Split
- Stratified splitting to preserve class balance

### 6. 📉 Model Building
Implemented multiple classifiers:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Naive Bayes
- Support Vector Machines
- K-Nearest Neighbors

### 7. ⚙️ Model Evaluation
Used metrics:
- Accuracy
- Precision, Recall, F1-Score
- ROC AUC Score
- Confusion Matrix

### 8. 🔍 Hyperparameter Tuning
- GridSearchCV
- RandomizedSearchCV

### 9. 🏁 Final Model and Pipeline
- Best model saved using `joblib`
- Pipeline constructed using `sklearn.pipeline.Pipeline`
- Visualization using SHAP for interpretability

---

## ✅ Final Model Performance (LightGBM)

- **Accuracy**: **87.25%**
- **Precision**: *Available in notebook*
- **Recall**: *Available in notebook*
- **F1-Score**: *Available in notebook*
- **ROC AUC**: *Available in notebook*

---

## 🧪 Evaluation Metrics

Models were evaluated on:
- ROC AUC
- F1-Score
- Confusion Matrix Analysis
- Cross-Validation Performance

---

## 🚧 Deployment Considerations

- **Data Drift**: Regular monitoring for changes in customer behavior.
- **Imbalanced Classes**: Used ADASYN for oversampling; needs evaluation post-deployment.
- **Model Interpretability**: SHAP values included for transparency.
- **Pipeline Integration**: Final model saved with full preprocessing and ready for production.

---

## 🛠 Technologies & Libraries

- Python (Pandas, NumPy, Seaborn, Matplotlib)
- Scikit-Learn
- XGBoost, LightGBM
- Imbalanced-learn (ADASYN)
- SHAP for explainability
- Joblib for model saving

---

## 📂 File Structure


