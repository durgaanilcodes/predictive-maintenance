# Predictive Maintenance Using Contextual Data Fusion

### Infotact Solutions — Data Science & Machine Learning Internship

## 📌 Project Overview

This project focuses on developing a machine learning-based predictive maintenance system using the AI4I 2020 Predictive Maintenance Dataset.

The system combines operational and sensor-related data to identify potential equipment failures. Contextual data fusion and machine learning techniques are used to improve failure prediction and support proactive maintenance decisions.

## 🎯 Project Objectives

- Predict potential machine failures using sensor and operational data.
- Perform contextual data fusion and feature engineering.
- Handle class imbalance using SMOTE.
- Develop a LightGBM classification model.
- Evaluate model performance using multiple metrics.
- Analyze model robustness under noisy conditions.
- Perform threshold tuning to improve prediction performance.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- SMOTE (Imbalanced-learn)
- Matplotlib
- Seaborn
- Jupyter Notebook
- Git & GitHub

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

The dataset contains machine operational information and sensor measurements used for predicting equipment failures.

Features explored include:
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure indicators

## 🔍 Project Workflow

1. Dataset collection and preparation
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Contextual Data Fusion
5. Feature Engineering
6. Train-Test Split
7. Class Imbalance Handling using SMOTE
8. LightGBM Model Development
9. Model Evaluation
10. 5-Fold Stratified Cross-Validation
11. Noise Sensitivity Analysis
12. Threshold Tuning
13. Documentation and Reporting

## 📅 Weekly Progress

### Week 1 — Repository Setup & Data Preparation

- Created GitHub repository.
- Set up project structure.
- Selected the AI4I dataset.
- Performed initial data preprocessing.
- Conducted exploratory data analysis.
- Prepared project planning and documentation.

### Week 2 — Contextual Data Fusion & Feature Engineering

- Prepared the enhanced dataset.
- Added contextual features:
  - Ambient temperature
  - Load density
- Engineered additional features:
  - Temperature difference
  - Power index
- Generated `enhanced_dataset.csv`.
- Completed enhanced data analysis.
- Prepared feature importance documentation.

### Week 3 — Model Development

**Objective:** Develop and evaluate a predictive maintenance model using contextual data fusion and machine learning.

Completed tasks:
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

### Week 4 — Model Robustness Evaluation

Completed tasks:
- Noise Sensitivity Analysis
- Threshold Tuning
- Precision-Recall Curve Analysis
- Model Performance Comparison
- Final Documentation

#### Noise Sensitivity Analysis

Gaussian noise was added to the test dataset to evaluate model robustness.

The analysis showed that model performance decreased significantly when noisy sensor data was introduced, highlighting the importance of reliable sensor data for predictive maintenance.

#### Threshold Tuning

Three decision thresholds were evaluated.

| Threshold | Precision | Recall | F1-Score | Accuracy |
|---|---:|---:|---:|---:|
| 0.3 | 0.62 | 0.97 | 0.75 | 0.98 |
| 0.5 | 0.77 | 0.97 | 0.86 | 0.99 |
| 0.7 | 0.87 | 0.97 | 0.92 | 0.99 |

**Observation:** Threshold 0.7 achieved the best overall performance in the reported threshold evaluation, with precision of 0.87, recall of 0.97, F1-score of 0.92, and accuracy of 0.99.

## 📈 Key Findings

- LightGBM was used to develop the predictive maintenance classification model.
- SMOTE was applied to address class imbalance.
- Contextual data fusion supported the feature engineering process.
- 5-Fold Stratified Cross-Validation was performed to assess model reliability.
- Noise sensitivity analysis highlighted the effect of noisy sensor data.
- Threshold tuning helped identify a decision threshold with improved reported F1-score.

## 👩‍💻 Team Members

- Durga Anil — Team Lead
- Binit Binu
- Krutika Thakur
- Priyadharshini

## 👩‍💻 Team Contributions

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

**Completed:** Four-week internship project covering data preparation, contextual data fusion, model development, evaluation, robustness testing, and documentation.

## 📄 Internship

This project was completed as part of the Data Science & Machine Learning Internship at Infotact Solutions.

