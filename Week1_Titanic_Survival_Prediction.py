## WEEK 1 TASK 1


import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

df = pd.read_csv("train.csv")

print(df.head())

print(df.shape)

print("Dataset Shape:", df.shape)

print(df.columns)

df.info()

# Check missing values
print(df.isnull().sum())


# Fill missing values in Age with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing values in Embarked with the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop the Cabin column
df.drop("Cabin", axis=1, inplace=True)


# Check missing values
print(df.isnull().sum())

print("\n")

# Check dataset information
df.info()


# ===============================
# 1) EDA
# ===============================
plt.figure()
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Passengers")
plt.tight_layout()
plt.show()


# Viz 2: Survival rate by Sex
plt.figure()
sns.barplot(data=df, x="Sex", y="Survived", estimator=np.mean, errorbar=None)
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# Viz 3: Survival rate by Passenger Class
plt.figure()
sns.barplot(data=df, x="Pclass", y="Survived", estimator=np.mean, errorbar=None)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, element="step")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.show()


# -------------------------------------------------
# 2) Feature Engineering (at least 2 new features)
# -------------------------------------------------

df_fe = df.copy()
# Feature 1: FamilySize
df_fe["FamilySize"] = df_fe["SibSp"] + df_fe["Parch"] + 1

# Feature 2: IsAlone
df_fe["IsAlone"] = (df_fe["FamilySize"] == 1).astype(int)

# Feature 3: Title
df_fe["Title"] = df_fe["Name"].str.extract(
    r",\s*([^\.]+)\.", expand=False
).str.strip()

# Standardize rare titles
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"
}
df_fe["Title"] = df_fe["Title"].replace(title_map)

rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
               "Rev", "Sir", "Jonkheer", "Dona"]
df_fe["Title"] = df_fe["Title"].replace(rare_titles, "Rare")

# If Cabin still exists, create HasCabin and optionally drop Cabin
if "Cabin" in df_fe.columns:
    df_fe["HasCabin"] = df_fe["Cabin"].notna().astype(int)

print(df_fe[["Name", "FamilySize", "IsAlone", "Title"]].head())

# 3) Select Features and Target
# -------------------------------------------------
target = "Survived"

drop_cols = [target]
for c in ["PassengerId", "Name", "Ticket", "Cabin"]:
    if c in df_fe.columns:
        drop_cols.append(c)

X = df_fe.drop(columns=drop_cols)
y = df_fe[target]


# Identify column types
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# One-Hot Encode categorical columns
df_encoded = pd.get_dummies(
    df_fe,
    columns=["Sex", "Embarked", "Title"],
    drop_first=True
)

print(df_encoded.head())


df_encoded = df_encoded.drop(["PassengerId", "Name", "Ticket"], axis=1, errors="ignore")



from sklearn.model_selection import train_test_split

# Features (Input)
X = df_encoded.drop("Survived", axis=1)

# Target (Output)
y = df_encoded["Survived"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)

# ==================================================
# Step 8: Train Logistic Regression Model
# ==================================================

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import accuracy metric
from sklearn.metrics import accuracy_score

# Create the model
lr_model = LogisticRegression(max_iter=1000)

# Train the model using training data
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred_lr = lr_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_lr)
# Print accuracy
print("Logistic Regression Accuracy:", accuracy)

# ==================================================
# Step 9: Train Random Forest Model
# ==================================================

# Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# Import accuracy metric
from sklearn.metrics import accuracy_score

# Create the model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

# Print accuracy
print("Random Forest Accuracy:", rf_accuracy)

# ==================================================
# Step 10: Model Evaluation
# ==================================================

from sklearn.metrics import classification_report

# Logistic Regression Report
print("=" * 50)
print("Logistic Regression Report")
print("=" * 50)

print(classification_report(y_test, y_pred_lr))

# Random Forest Report
print("=" * 50)
print("Random Forest Report")
print("=" * 50)

print(classification_report(y_test, y_pred_rf))


# ==================================================
# Step 11: Confusion Matrix
# ==================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create confusion matrix for Random Forest
cm = confusion_matrix(y_test, y_pred_rf)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")
plt.show()

# ==================================================
# Step 12: Model Comparison Figure
# ==================================================

import matplotlib.pyplot as plt
import pandas as pd

# Create comparison DataFrame
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy, rf_accuracy]
})

# Plot
plt.figure(figsize=(6,4))
plt.bar(comparison["Model"], comparison["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show accuracy values on bars
for i, v in enumerate(comparison["Accuracy"]):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)

plt.show()

