# Credit Card Fraud Detection Using Machine Learning

## Project Overview
Credit Card Fraud Detection is a machine learning classification project that identifies potentially fraudulent credit card transactions. The project uses transaction data and machine learning algorithms to distinguish between legitimate and fraudulent transactions.

The implementation was carried out using **Google Colab** and Python.

## Objectives
- Understand and analyze credit card transaction data.
- Perform data cleaning and preprocessing.
- Explore transaction patterns using data visualization.
- Handle the highly imbalanced target classes.
- Train and compare multiple machine learning models.
- Evaluate models using suitable classification metrics.

## Dataset
**Source:** [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- Total records: 284,807
- Total columns: 31
- Target column: `Class`
- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The dataset contains anonymized numerical features (`V1` to `V28`), transaction time, and transaction amount.

## Technologies Used
- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow
1. Dataset collection and loading
2. Data understanding and exploration
3. Data cleaning and preprocessing
4. Exploratory Data Analysis (EDA)
5. Statistical analysis
6. Train, validation, and test split
7. Machine learning model development
8. Model comparison and hyperparameter tuning
9. Model evaluation

## Machine Learning Models
- Dummy Classifier (Baseline)
- Logistic Regression
- Decision Tree
- Random Forest

## Evaluation Metrics
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand transaction patterns and identify important characteristics of fraudulent and legitimate transactions.

### Visualizations

- **Class Distribution:** Analyzes the number of legitimate and fraudulent transactions.
- **Transaction Amount Distribution:** Uses histograms to understand transaction amount patterns.
- **Transaction Time Distribution:** Examines transaction activity over time.
- **Amount by Transaction Class:** Uses box plots to compare legitimate and fraudulent transaction amounts.
- **Correlation Heatmap:** Identifies relationships between numerical features and the target variable.

### Statistical Analysis

The Mann-Whitney U test is used to determine whether there is a statistically significant difference between legitimate and fraudulent transaction amounts.

These analyses help understand the highly imbalanced dataset and support the development of effective fraud detection models.

## Validation Results
The following results were obtained during baseline model comparison on the validation dataset at a classification threshold of 0.5.

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.0508 | 0.8873 | 0.0962 | 0.9717 |
| Decision Tree |
