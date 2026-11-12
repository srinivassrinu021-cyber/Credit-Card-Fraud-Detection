
## Additional Analysis Modules

This project includes additional Python modules to improve data analysis, visualization, and model evaluation.

### 1. Fraud Analysis (`fraud_analysis.py`)

- Displays dataset information and transaction statistics.
- Identifies missing values.
- Calculates the percentage of fraudulent transactions.
- Analyzes transaction amount statistics.
- Generates fraud distribution charts.

### 2. Model Comparison (`model_comparison.py`)

Compares different machine learning classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-Score

The comparison results are exported into a CSV file.

### 3. Fraud Detection Visualization (`fraud_detection_visualization.py`)

Generates visual insights from transaction data:

- Fraud vs Legitimate Transaction Distribution
- Transaction Amount Comparison
- Fraud Percentage Analysis
- Transaction Amount Distribution
- Feature Correlation Analysis

### Running the Additional Modules

Run the scripts individually:

```bash
python fraud_analysis.py

python model_comparison.py

python fraud_detection_visualization.py
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Note:** Make sure the dataset file `creditcard.csv` is available in the project directory before running these scripts.
