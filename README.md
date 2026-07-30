# Titanic Survival Prediction

## Live Pakistan Machine Learning Internship

**Track:** Machine Learning

**Week:** 1 of 4

**Project:** Titanic Survival Prediction

**Author:** Muhammad Aoun

**Language:** Python

**Libraries Used:**
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Project Overview

This project was completed as part of the Live Pakistan Machine Learning Internship Week 1 assignment.

The objective of this project is to build a supervised machine learning classification model that predicts whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, fare, and family details.

The project follows a complete machine learning workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, model evaluation, and performance comparison.

---

# Problem Statement

The Titanic disaster is one of the most well-known historical events. Using historical passenger records, this project aims to predict passenger survival based on available information.

The developed model can help understand which factors had the greatest influence on survival and demonstrates the end-to-end workflow of a supervised machine learning classification project.

---

# Dataset Information

Dataset Name:
Titanic - Machine Learning from Disaster

Source:
Kaggle

Dataset Size:
- Rows: 891
- Columns: 12

Target Variable:
- Survived
    - 0 = Did Not Survive
    - 1 = Survived

Main Features:
- PassengerId
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

---

# Project Workflow

## Step 1 – Import Libraries

Imported all required Python libraries for:

- Data manipulation
- Data visualization
- Machine Learning
- Model evaluation

Libraries include:

- pandas
- numpy
- matplotlib
- seaborn
- sklearn

---

## Step 2 – Load Dataset

The Titanic dataset was loaded using Pandas.

```python
pd.read_csv("train.csv")
```

---

## Step 3 – Data Exploration

Performed initial exploration using:

- df.head()
- df.shape
- df.columns
- df.info()
- df.isnull().sum()

This helped identify:

- Dataset size
- Data types
- Missing values
- Feature names

---

## Step 4 – Data Cleaning

Missing values were handled as follows:

### Age

Missing values were filled using the median because Age is a numerical feature and the median is less affected by outliers.

### Embarked

Missing values were filled using the most frequent value (mode).

### Cabin

The Cabin column contained more than 75% missing values.

Since most records were empty, this feature was removed from the dataset.

---

# Exploratory Data Analysis (EDA)

Several visualizations were created to understand the dataset.

### Visualization 1

Survival Count

Purpose:

Shows the number of passengers who survived and did not survive.

---

### Visualization 2

Survival Rate by Gender

Purpose:

Shows how survival varied between male and female passengers.

Observation:

Female passengers had a significantly higher survival rate.

---

### Visualization 3

Survival Rate by Passenger Class

Purpose:

Shows survival percentages across passenger classes.

Observation:

Passengers in First Class had the highest survival rate.

---

### Visualization 4

Age Distribution by Survival

Purpose:

Shows the age distribution of survivors and non-survivors.

Observation:

Children generally had better survival chances than older passengers.

---

# Feature Engineering

Three new features were created.

## FamilySize

Formula:

FamilySize = SibSp + Parch + 1

Purpose:

Represents the total family members traveling together.

---

## IsAlone

Formula:

If FamilySize == 1

Then

IsAlone = 1

Otherwise

IsAlone = 0

Purpose:

Indicates whether the passenger traveled alone.

---

## Title

Passenger titles were extracted from the Name column.

Examples:

- Mr
- Mrs
- Miss
- Master

Rare titles were grouped into a single category called "Rare".

Purpose:

Titles often indicate gender, age, and social status, making them useful for prediction.

---

# Data Encoding

Machine learning algorithms require numerical inputs.

Categorical variables were converted using One-Hot Encoding.

Encoded columns included:

- Sex
- Embarked
- Title

---

# Train-Test Split

The dataset was divided into:

Training Data:
80%

Testing Data:
20%

Random State:
42

---

# Machine Learning Models

Two classification models were trained.

## Logistic Regression

A baseline linear classification algorithm.

Accuracy:

81.01%

---

## Random Forest

An ensemble learning algorithm consisting of multiple decision trees.

Accuracy:

83.80%

---

# Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Logistic Regression

Accuracy:
81.01%

Performed well but was slightly weaker than Random Forest.

---

## Random Forest

Accuracy:
83.80%

Achieved the highest performance.

Also obtained better:

- Precision
- Recall
- F1-Score

Therefore it was selected as the final model.

---

# Confusion Matrix

A confusion matrix was generated for the Random Forest model.

Results:

- True Negatives: 90
- False Positives: 15
- False Negatives: 14
- True Positives: 60

This shows that the model correctly classified most passengers.

---

# Model Comparison

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 81.01% |
| Random Forest | 83.80% |

Best Model:

Random Forest

---

# Results

The Random Forest classifier produced the best performance among the trained models.

Performance Summary:

- Accuracy: 83.80%
- Better Precision
- Better Recall
- Better F1-Score

---

# Project Structure

```
Week1_Titanic_Survival_Prediction/

│── train.csv
│── Week1_Titanic_Survival_Prediction.py
│── README.md
│── Screenshots/
│   │── Survival_Count.png
│   │── Survival_by_Sex.png
│   │── Survival_by_Class.png
│   │── Confusion_Matrix.png
│   │── Model_Comparison.png
```

---

# Requirements

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# How to Run

Run the Python script:

```bash
python Week1_Titanic_Survival_Prediction.py
```

The program will:

- Load the dataset
- Clean the data
- Perform EDA
- Create new features
- Train machine learning models
- Evaluate performance
- Display charts
- Compare model performance

---

# Learning Outcomes

Through this project, the following machine learning concepts were practiced:

- Data Cleaning
- Missing Value Handling
- Exploratory Data Analysis (EDA)
- Feature Engineering
- One-Hot Encoding
- Train-Test Split
- Logistic Regression
- Random Forest Classification
- Model Evaluation
- Confusion Matrix Interpretation
- Model Comparison

---

# Conclusion

This project successfully developed a supervised machine learning model to predict Titanic passenger survival. After cleaning the dataset and engineering meaningful features, two classification models were trained and evaluated. Random Forest achieved the highest accuracy of **83.80%**, outperforming Logistic Regression across all evaluation metrics. The project demonstrates a complete end-to-end machine learning workflow, from data preprocessing and visualization to model evaluation and interpretation, making it a solid foundation for future classification projects.

---

# Acknowledgements

- Live Pakistan Machine Learning Internship Program
- Kaggle Titanic Dataset
- Scikit-learn Documentation
- Pandas Documentation
- Matplotlib & Seaborn Documentation