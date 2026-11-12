
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("creditcard.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Transactions:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fraud and Legitimate Transaction Count
fraud_count = df["Class"].value_counts()

print("\nTransaction Distribution:")
print(fraud_count)

print("\nFraud Percentage:",
      round((df["Class"].mean() * 100), 2), "%")

# Transaction Amount Statistics
print("\nTransaction Amount Statistics:")
print(df["Amount"].describe())

# Visualize Transaction Distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("transaction_distribution.png")
plt.show()

# Visualize Transaction Amount
plt.figure(figsize=(8, 5))
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.savefig("transaction_amount.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Credit Card Transaction Correlation")
plt.savefig("correlation_heatmap.png")
plt.show()

print("\nFraud Analysis Completed Successfully!")
