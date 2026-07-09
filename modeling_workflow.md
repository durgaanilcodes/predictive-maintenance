# Modeling Workflow

## Step 1: Data Preparation

The enhanced dataset is loaded and checked for missing values and data quality issues.

## Step 2: Feature Selection

Internal Features:
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Contextual Features:
- Ambient_Temperature
- Load_Density
- Temp_Difference
- Power_Index

## Step 3: Handling Class Imbalance

SMOTE is applied to balance the minority failure class.

## Step 4: Cross Validation

5-Fold Stratified Cross Validation is used to ensure reliable model evaluation.

## Step 5: Model Training

LightGBM is used to train the predictive maintenance model.

## Step 6: Model Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Step 7: Future Ablation Study

Model performance will be compared:
1. Internal features only.
2. Internal + contextual features.
