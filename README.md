# CreditWise Loan Approval System

## Project Overview

The **CreditWise Loan Approval System** is a Machine Learning classification project designed to automate the initial assessment of loan applications.

The problem statement describes a financial institution, that currently relies on manual verification of applicant income, employment details, credit history, and other information. This process can be time-consuming, inconsistent, and susceptible to human error. The proposed system uses historical applicant data to predict whether a loan should be **Approved (`1`)** or **Rejected (`0`)** before final human verification.

The implementation in this project performs data preprocessing, Exploratory Data Analysis (EDA), categorical encoding, feature scaling, correlation analysis, feature engineering, and comparison of multiple classification algorithms.

> **Important:** The model is intended as a decision-support/initial screening system. The original problem statement specifies prediction before final human verification; it should not be treated as a fully autonomous lending decision.

---

## Problem Statement

financial institution receives loan applications from customers across urban and rural regions of India. The existing manual verification process requires loan officers to evaluate financial, employment, credit, and personal information.

The financial institution faces two key challenges:

1. Good customers may be rejected, resulting in lost business.
2. High-risk customers may be approved, resulting in potential financial losses.

The objective is therefore to develop an intelligent Machine Learning system that can analyze historical loan application records and predict whether an application should be **Approved or Rejected**.

The dataset contains applicant, financial, employment, loan, and credit-related attributes, with `Loan_Approved` as the target variable.

---

## Project Objectives

The main objectives of this project are:

- Analyze historical loan application data.
- Identify and handle missing values.
- Perform Exploratory Data Analysis (EDA).
- Analyze numerical and categorical variables.
- Identify relationships between applicant attributes and loan approval.
- Convert categorical variables into machine-readable numerical features.
- Scale numerical features before model training.
- Train and compare multiple classification algorithms.
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Perform additional feature engineering using important numerical features.
- Develop a baseline loan approval prediction pipeline that can support human verification.

---

## Dataset Description

The dataset used by the notebook is:

```text
loan_approval_data.csv
```

The notebook contains **1,000 rows and 20 columns**. The initial inspection shows that **950 records contain values in the dataset columns**, with missing values present across the data and subsequently handled using imputation.

### Target Variable

```text
Loan_Approved
```

Target interpretation:

| Value | Meaning |
|------:|---------|
| `1` | Loan Approved |
| `0` | Loan Rejected |

---

## Column Dictionary

| Column Name | Description |
|---|---|
| `Applicant_ID` | Unique applicant identifier |
| `Applicant_Income` | Monthly income of the applicant |
| `Coapplicant_Income` | Monthly income of the co-applicant |
| `Employment_Status` | Applicant employment status such as Salaried, Self-employed, or Unemployed |
| `Age` | Applicant age |
| `Marital_Status` | Applicant marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit bureau score |
| `Existing_Loans` | Number of existing/running loans |
| `DTI_Ratio` | Debt-to-Income ratio |
| `Savings` | Applicant savings balance |
| `Collateral_Value` | Value of collateral provided |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan duration in months |
| `Loan_Purpose` | Purpose of the loan, such as Home, Education, Personal, or Business |
| `Property_Area` | Property location category: Urban, Semiurban, or Rural |
| `Education_Level` | Applicant education level |
| `Gender` | Applicant gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target variable: `1 = Approved`, `0 = Rejected` |

The problem statement also defines the major financial and applicant attributes used by the system, including income, credit score, existing loans, DTI ratio, savings, collateral, loan amount, loan term, employment, education, and property area. 

---

## Technologies and Libraries

The project is implemented in Python using:

- **Python**
- **Pandas** – data loading and manipulation
- **NumPy** – numerical operations
- **Matplotlib** – visualization
- **Seaborn** – statistical visualization
- **Scikit-learn** – preprocessing, model training, and evaluation
- **Jupyter Notebook** – development environment

### Scikit-learn Components Used

- `train_test_split`
- `SimpleImputer`
- `LabelEncoder`
- `OneHotEncoder`
- `StandardScaler`
- `LogisticRegression`
- `KNeighborsClassifier`
- `GaussianNB`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- Classification metrics and confusion matrix

---

# Methodology and Steps Taken

## 1. Data Loading and Initial Inspection

The dataset is loaded using Pandas:

```python
df = pd.read_csv("loan_approval_data.csv")
```

The notebook performs:

- `head()`
- `info()`
- `isnull().sum()`
- `describe()`

The initial dataset contains:

- **1,000 rows**
- **20 columns**
- Numerical and categorical features
- Missing values

The notebook identifies numerical and categorical columns using Pandas data types.

---

## 2. Handling Missing Values

Missing values are handled using `SimpleImputer`.

### Numerical Features

Numerical columns are imputed using the **mean**:

```python
num_imp = SimpleImputer(strategy="mean")
```

### Categorical Features

Categorical columns are imputed using the **most frequent value**:

```python
cat_imp = SimpleImputer(strategy="most_frequent")
```

This allows the subsequent EDA and model-training stages to operate on a complete dataset.

---

## 3. Exploratory Data Analysis (EDA)

EDA is performed to understand the distribution of the target and individual applicant characteristics.

### Target Distribution

A pie chart is used to examine the distribution of:

```text
Loan_Approved
```

This helps determine how balanced the approval and rejection classes are.

### Categorical Analysis

The notebook analyzes:

- Gender
- Education Level
- Employment Status
- Marital Status
- Loan Purpose
- Property Area
- Employer Category

Bar plots are used to visualize the frequency of each category.

### Numerical Analysis

Histograms are used to inspect:

- Applicant Income
- Coapplicant Income
- Credit Score

Box plots are used to analyze:

- Applicant Income
- Credit Score
- DTI Ratio
- Savings

The notebook also compares numerical variables against `Loan_Approved` to understand their relationship with the target.

---

## 4. Feature Removal

`Applicant_ID` is removed before model training because it is an identifier rather than a meaningful predictive feature.

```python
df = df.drop("Applicant_ID", axis=1)
```

---

## 5. Categorical Encoding

Two encoding approaches are used.

### Label Encoding

`LabelEncoder` is used for:

- `Education_Level`
- `Loan_Approved`

The target is converted into numerical form for classification.

### One-Hot Encoding

The following categorical features are one-hot encoded:

```text
Employment_Status
Marital_Status
Loan_Purpose
Property_Area
Gender
Employer_Category
```

The implementation uses:

```python
OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)
```

`drop="first"` reduces redundant dummy variables, while `handle_unknown="ignore"` allows unseen categories to be handled during transformation.

---

## 6. Correlation Analysis

After encoding the categorical variables, a correlation matrix is calculated for numerical features.

A correlation heatmap is generated using Seaborn.

The notebook also sorts correlations against the target:

```python
num_cols.corr()["Loan_Approved"].sort_values(ascending=False)
```

In the implemented dataset, `Credit_Score` shows a positive correlation with loan approval, while `DTI_Ratio` and `Loan_Amount` show negative correlations with the target.

Correlation is used here as an exploratory tool and should not be interpreted as proof of causation.

---

## 7. Train-Test Split

The target is separated from the feature matrix:

```python
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]
```

The dataset is split into:

- **80% training data**
- **20% testing data**

with:

```python
random_state=42
```

The test set contains **200 records**.

---

## 8. Feature Scaling

Numerical model inputs are standardized using:

```python
StandardScaler()
```

The notebook creates:

```text
X_train_scaled
X_test_scaled
```

Scaling is especially relevant for models such as Logistic Regression and KNN because these models can be affected by differences in feature magnitude.

---

# Machine Learning Models

The project compares several supervised classification algorithms.

## 1. Logistic Regression

Logistic Regression is used as a linear classification baseline.

## 2. K-Nearest Neighbors (KNN)

KNN is implemented with:

```python
KNeighborsClassifier(n_neighbors=3)
```

## 3. Gaussian Naive Bayes

Gaussian Naive Bayes is used to provide a probabilistic classification approach.

## 4. Random Forest

Random Forest is used as a tree-based ensemble classifier.

## 5. Gradient Boosting

Gradient Boosting is used as another ensemble-based classification approach.

---

# Model Evaluation

The models are evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

The first model comparison in the notebook produced the following results:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.885 | 0.852 | 0.754 | 0.800 |
| KNN | 0.740 | 0.574 | 0.574 | 0.574 |
| Naive Bayes | 0.845 | 0.778 | 0.689 | 0.730 |
| Random Forest | 0.880 | 0.814 | 0.787 | 0.800 |
| Gradient Boosting | **0.890** | 0.820 | **0.820** | **0.820** |

These values are from the model-evaluation outputs saved in the supplied notebook.

### Gradient Boosting Confusion Matrix

The first Gradient Boosting evaluation produced:

```text
[[128, 11],
 [ 11, 50]]
```

Using the notebook's class encoding, this corresponds to:

- True Negatives: 128
- False Positives: 11
- False Negatives: 11
- True Positives: 50

---

# Feature Engineering

After the initial model comparison, additional features are created from important numerical variables.

### Squared Features

The notebook creates:

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

The original:

```text
DTI_Ratio
Credit_Score
```

features are then removed from the feature matrix for this second experiment.

The dataset is subsequently split and scaled again before retraining the models.

---

## Feature Engineering Results

After feature engineering, the notebook reports:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.870 | 0.830 | 0.721 | 0.772 |
| KNN | 0.735 | 0.571 | 0.525 | 0.547 |
| Naive Bayes | 0.860 | 0.811 | 0.705 | 0.754 |
| Gradient Boosting | **0.890** | 0.820 | 0.820 | 0.820 |

The notebook's Gradient Boosting result remains:

```text
Accuracy  : 0.890
Precision : 0.820
Recall    : 0.820
F1-score  : 0.820
```

---

# Key Findings

Based on the implemented analysis:

### 1. Credit Score

`Credit_Score` has a positive correlation with the loan approval target in the correlation analysis. The notebook reports a correlation of approximately:

```text
0.451
```

with `Loan_Approved`.

### 2. DTI Ratio

`DTI_Ratio` has a relatively strong negative correlation with loan approval in the implemented dataset:

```text
-0.445
```

### 3. Applicant Income

`Applicant_Income` has a positive but smaller correlation with loan approval:

```text
0.120
```

### 4. Loan Amount

`Loan_Amount` has a negative correlation with the target:

```text
-0.126
```

### 5. Model Comparison

The first model comparison shows that Gradient Boosting achieved:

- Accuracy: **89.0%**
- Precision: **82.0%**
- Recall: **82.0%**
- F1-score: **82.0%**

Logistic Regression also performed strongly, achieving **88.5% accuracy** in the first experiment.

---

# Project Workflow

```text
                    ┌─────────────────────┐
                    │  Loan Application   │
                    │       Dataset       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Loading &      │
                    │ Initial Inspection  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Missing Value       │
                    │ Imputation          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │ Analysis (EDA)      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Remove Applicant_ID │
                    │ & Encode Categories │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Correlation &      │
                    │ Feature Analysis    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train-Test Split    │
                    │       80 / 20       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Scaling     │
                    │  StandardScaler     │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │        Model Comparison         │
              │                                 │
              │ Logistic Regression             │
              │ KNN                             │
              │ Naive Bayes                     │
              │ Random Forest                   │
              │ Gradient Boosting               │
              └────────────────┬────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ Accuracy / P / R /  │
                    │ F1 / Confusion Mat. │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ DTI² / Credit²      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Experimental  │
                    │ Model Comparison    │
                    └─────────────────────┘
```

---

# Project Structure

A recommended repository structure for this project is:

```text
CreditWise-Loan-Approval-System/
│
├── loan_approval_data.csv
├── loan_aproval_system.ipynb
├── README.md
│
└── assets/
    └── plots/
```

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd CreditWise-Loan-Approval-System
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
loan_aproval_system.ipynb
```

## 5. Run the Notebook

Make sure:

```text
loan_approval_data.csv
```

is available in the same working directory as the notebook, then execute the notebook cells from top to bottom.

---

# Model Performance Summary

The first experiment provides the following comparison:

```text
Gradient Boosting
Accuracy  : 89.0%
Precision : 82.0%
Recall    : 82.0%
F1-score  : 82.0%

Logistic Regression
Accuracy  : 88.5%
Precision : 85.2%
Recall    : 75.4%
F1-score  : 80.0%

Random Forest
Accuracy  : 88.0%
Precision : 81.4%
Recall    : 78.7%
F1-score  : 80.0%

Naive Bayes
Accuracy  : 84.5%
Precision : 77.8%
Recall    : 68.9%
F1-score  : 73.0%

KNN
Accuracy  : 74.0%
Precision : 57.4%
Recall    : 57.4%
F1-score  : 57.4%
```

These are **experimental notebook results**, not a production performance guarantee.

---

# Limitations and Future Improvements

The current notebook is a baseline Machine Learning implementation. Several improvements can be considered for a production-grade loan approval system:

1. **Cross-validation**
   - Use K-Fold or Stratified K-Fold cross-validation instead of relying on a single train-test split.

2. **Hyperparameter tuning**
   - Optimize models using `GridSearchCV` or `RandomizedSearchCV`.

3. **Class imbalance analysis**
   - Evaluate class distribution and consider techniques such as class weights or resampling where appropriate.

4. **Better feature engineering**
   - Explore financially meaningful ratios and interactions such as income-to-loan amount and total household income.

5. **Model interpretability**
   - Use feature importance and explainability techniques to understand model predictions.

6. **Fairness evaluation**
   - Since the problem statement explicitly emphasizes unbiased decisions, evaluate model behavior across relevant applicant groups before real-world deployment.

7. **Pipeline implementation**
   - Combine preprocessing, feature transformation, and modeling into a reproducible Scikit-learn pipeline.

8. **Production validation**
   - Validate the model on temporally or operationally representative data before deployment.

9. **Human verification**
   - Keep human review in the decision workflow, consistent with the original problem statement.

---

# Technical Note

The supplied notebook currently performs:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

For a production ML pipeline, the test set should normally be transformed using the scaler fitted on the training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents information from the test-set feature distribution from being used during preprocessing.

---

# Conclusion

The **CreditWise Loan Approval System** demonstrates an end-to-end Machine Learning workflow for loan approval prediction.

The project covers:

- Data loading
- Missing-value handling
- Exploratory Data Analysis
- Categorical encoding
- Feature scaling
- Correlation analysis
- Train-test splitting
- Multiple classification algorithms
- Model evaluation
- Feature engineering
- Model comparison

In the first experiment, **Gradient Boosting** produced an accuracy of **89.0%**, with precision, recall, and F1-score of approximately **82.0%** each. Logistic Regression also achieved **88.5% accuracy**.

The system is designed as an initial predictive layer that can help analyze applications before **final human verification**, matching the intended workflow described in the project problem statement.
