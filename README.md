# Predictive Maintenance Using Contextual Data Fusion

### Infotact Solutions — Data Science & Machine Learning Internship

A machine learning project focused on predicting potential equipment failures using operational and sensor-related data from the **AI4I 2020 Predictive Maintenance Dataset**. The project combines contextual data fusion, feature engineering, class imbalance handling, LightGBM classification, and model robustness evaluation.

---

## 📌 Project Overview

Predictive maintenance uses machine learning to identify potential equipment failures before they occur. This helps industries reduce unexpected downtime, improve equipment reliability, and support proactive maintenance decisions.

This project explores the development of a predictive maintenance classification model using operational and sensor data. Contextual data fusion and feature engineering are applied to enhance the dataset, followed by model development and evaluation.

## 🎯 Project Objectives

- Predict potential machine failures using sensor and operational data.
- Perform data cleaning and preprocessing.
- Apply contextual data fusion and feature engineering.
- Handle class imbalance using SMOTE.
- Develop a LightGBM classification model.
- Evaluate model performance using multiple metrics.
- Perform stratified cross-validation.
- Conduct noise sensitivity analysis.
- Evaluate decision thresholds to improve classification performance.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, LightGBM |
| Class Imbalance Handling | SMOTE |
| Data Visualization | Matplotlib, Seaborn |
| Development Environment | Jupyter Notebook |
| Version Control | Git, GitHub |

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains machine operational information and sensor measurements used for predicting equipment failures.

### Features explored

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Failure Indicators

The dataset was enhanced through contextual data fusion and feature engineering.

## 🔄 Project Workflow

1. Dataset Collection and Preparation
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Contextual Data Fusion
5. Feature Engineering
6. Train-Test Split
7. Class Imbalance Handling using SMOTE
8. LightGBM Model Development
9. Model Evaluation
10. Stratified Cross-Validation
11. Noise Sensitivity Analysis
12. Threshold Tuning
13. Documentation and Reporting

## 📅 Weekly Progress

### Week 1 — Repository Setup & Data Preparation

**Objective:** Establish the project environment and prepare the initial dataset.

#### Completed Tasks

- Created the GitHub repository.
- Set up the project structure.
- Selected the AI4I dataset.
- Performed initial data preprocessing.
- Conducted exploratory data analysis.
- Prepared project planning and documentation.

### Week 2 — Contextual Data Fusion & Feature Engineering

**Objective:** Enhance the dataset using contextual features.

#### Completed Tasks

- Prepared the enhanced dataset.
- Added contextual features:
  - Ambient Temperature
  - Load Density
- Engineered additional features:
  - Temperature Difference
  - Power Index
- Generated `enhanced_dataset.csv`.
- Completed enhanced dataset analysis.
- Prepared feature importance documentation.

### Week 3 — Model Development

**Objective:** Develop and evaluate a predictive maintenance model using contextual data fusion and machine learning.

#### Completed Tasks

- Prepared the final training dataset.
- Performed Train-Test Split.
- Applied SMOTE to handle class imbalance.
- Trained a LightGBM classification model.
- Evaluated the model using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Generated a classification report.
- Created a confusion matrix.
- Analyzed feature importance.
- Performed 5-Fold Stratified Cross-Validation.

#### Results

- Successfully trained the LightGBM model on the enhanced dataset.
- Balanced the training data using SMOTE.
- Evaluated model performance using multiple metrics.
- Completed cross-validation for model reliability.
- Generated visualizations for the confusion matrix and feature importance.

**Status:** Week 3 modeling tasks completed successfully.

### Week 4 — Model Robustness Evaluation

**Objective:** Evaluate model robustness under noisy sensor conditions and identify a suitable classification threshold.

#### Completed Tasks

- Noise Sensitivity Analysis
- Threshold Tuning
- Precision-Recall Curve Analysis
- Model Performance Comparison
- Final Documentation

#### Noise Sensitivity Analysis

Gaussian noise was added to the test dataset to evaluate model robustness.

The analysis showed that model performance decreased when noisy sensor data was introduced, highlighting the importance of clean and reliable sensor measurements for predictive maintenance.

#### Threshold Tuning

Three decision thresholds were evaluated.

| Threshold | Precision | Recall | F1-Score | Accuracy |
|---|---:|---:|---:|---:|
| 0.3 | 0.62 | 0.97 | 0.75 | 0.98 |
| 0.5 | 0.77 | 0.97 | 0.86 | 0.99 |
| 0.7 | 0.87 | 0.97 | 0.92 | 0.99 |

**Observation:** Threshold 0.7 achieved the best overall performance among the reported threshold evaluations, with a precision of 0.87, recall of 0.97, F1-score of 0.92, and accuracy of 0.99.

*Note: These are the reported threshold evaluation results. They should not be interpreted as independently verified production performance.*

## 🔍 Key Findings

- LightGBM was used to develop the predictive maintenance classification model.
- SMOTE was applied to address class imbalance.
- Contextual data fusion supported the feature engineering process.
- 5-Fold Stratified Cross-Validation was performed to assess model reliability.
- Noise sensitivity analysis highlighted the impact of noisy sensor data.
- Threshold tuning helped identify a decision threshold with improved reported F1-score.

## 👥 Team Members

- **Durga Anil** — Team Lead
- Binit Binu
- Krutika Thakur
- Priyadharshini

## 🤝 Team Contributions

### Durga Anil — Team Lead

- Repository setup and project management
- Contextual data fusion
- Feature engineering
- LightGBM model development
- SMOTE implementation
- Cross-validation
- Model evaluation
- Noise sensitivity analysis
- Threshold tuning
- Final documentation

### Binit Binu

- Dataset validation
- Data quality analysis
- Class imbalance analysis
- Data preprocessing

### Krutika Thakur

- Visualizations
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Feature importance charts

### Priyadharshini

- Literature review
- Documentation
- Feature comparison study
- Modeling workflow documentation

## 🚀 Future Improvements

- Integrate real-time sensor data.
- Develop a Streamlit-based predictive maintenance dashboard.
- Improve robustness using advanced noise-handling techniques.
- Explore deep learning-based predictive maintenance models.
- Deploy the model on a cloud platform.

## 📌 Project Status

**Completed:** This project was completed as part of the Data Science & Machine Learning Internship at Infotact Solutions.

The project covered data preparation, contextual data fusion, feature engineering, model development, evaluation, robustness testing, and documentation.

## 💼 Internship Experience

**Data Science & Machine Learning Intern — Infotact Solutions**

This project provided practical experience in machine learning workflows, contextual data fusion, feature engineering, classification, model evaluation, and team collaboration.

---

⭐ Thank you for visiting this repository!

