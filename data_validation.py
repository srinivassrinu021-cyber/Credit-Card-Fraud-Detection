
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("creditcard.csv")

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Time and Amount columns
preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Time", "Amount"])
    ],
    remainder="passthrough"
)

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Get fraud probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Test different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Threshold": threshold,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON")
print(results_df.to_string(index=False))

# Find best threshold based on F1 score
best_result = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\nBest Threshold:", best_result["Threshold"])
print("Best F1 Score:", best_result["F1 Score"])

import pandas as pd
import sys

def validate_dataset(file_path):

    print("=" * 50)
    print("CREDIT CARD DATASET VALIDATION")
    print("=" * 50)

    try:
        df = pd.read_csv(file_path)

        # 1. Dataset Information
        print("\n1. Dataset Information")
        print("Total Rows:", df.shape[0])
        print("Total Columns:", df.shape[1])

        # 2. Missing Values
        print("\n2. Missing Values")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.any() else "No missing values found")

        # 3. Duplicate Records
        print("\n3. Duplicate Records")
        duplicates = df.duplicated().sum()
        print("Duplicate Rows:", duplicates)

        # 4. Required Columns
        print("\n4. Required Column Validation")

        required_columns = ["Amount", "Class"]

        for column in required_columns:
            if column in df.columns:
                print(column, "- Available")
            else:
                print(column, "- Missing")

        if not all(col in df.columns for col in required_columns):
            print("\nValidation stopped: Required columns are missing.")
            return

        # 5. Transaction Distribution
        print("\n5. Transaction Distribution")

        class_counts = df["Class"].value_counts()
        print(class_counts)

        # 6. Fraud Percentage
        fraud_percentage = (df["Class"] == 1).mean() * 100

        print("\n6. Fraud Percentage")
        print(f"{fraud_percentage:.4f}%")

        # 7. Amount Statistics
        print("\n7. Transaction Amount Statistics")
        print(df["Amount"].describe())

        # 8. Target Validation
        print("\n8. Target Class Validation")
        print("Unique Classes:", df["Class"].unique())

        # 9. Final Report
        report = [
            "CREDIT CARD DATASET VALIDATION REPORT",
            "=" * 40,
            f"Total Rows: {len(df)}",
            f"Total Columns: {len(df.columns)}",
            f"Missing Values: {df.isnull().sum().sum()}",
            f"Duplicate Rows: {duplicates}",
            f"Fraud Percentage: {fraud_percentage:.4f}%",
            f"Fraud Transactions: {(df['Class'] == 1).sum()}",
            f"Legitimate Transactions: {(df['Class'] == 0).sum()}"
        ]

        with open("validation_report.txt", "w") as file:
            file.write("\n".join(report))

        print("\nValidation Report Saved Successfully!")

    except FileNotFoundError:
        print("Error: Dataset file not found.")

    except Exception as error:
        print("Error:", error)


if __name__ == "__main__":

    dataset_path = (
        sys.argv[1] if len(sys.argv) > 1
        else "creditcard.csv"
    )

    validate_dataset(dataset_path)
