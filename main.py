import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load dataset
df = pd.read_csv("data/tickets.csv")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# Remove missing rows
df = df.dropna()

# Target distribution
category_counts = df["Topic_group"].value_counts()

print("\nCategory Counts:")
print(category_counts)

# Save category counts
category_counts.to_csv("outputs/category_distribution.csv")

# Plot category distribution
plt.figure(figsize=(10, 6))
category_counts.plot(kind="bar")
plt.title("Support Ticket Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/category_distribution.png")
plt.close()

# Features and Target
X = df["Document"]
y = df["Topic_group"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# NLP Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Classification Report
report = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report)

# Save report
with open("outputs/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("\nOutputs saved successfully inside outputs folder.")