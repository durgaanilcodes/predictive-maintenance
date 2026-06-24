Data Quality Report

Project
Predictive Maintenance Dataset Analysis

Week 3 Activities

Activity 1: Validate Final Dataset Quality Before Modeling

Validation checks performed:

- Checked dataset shape
- Verified column names
- Verified data types
- Checked missing values
- Checked duplicate rows

Results:

- Total Rows: 10000
- Total Columns: 14
- Missing Values: 0
- Duplicate Rows: 0

Dataset quality is suitable for machine learning modeling.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Activity 2: Verify Feature Distributions and Class Imbalance

Features analyzed:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

Observations:

- Air temperature follows a near normal distribution.
- Process temperature follows a near normal distribution.
- Rotational speed shows moderate variation.
- Torque values are approximately normally distributed.
- Tool wear values are spread across the dataset.

Class Imbalance Analysis:

Machine Failure Distribution:

- No Failure (0): 9661
- Failure (1): 339

Observation:

The dataset is imbalanced because failure cases are much fewer than non-failure cases.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Activity 3: Preprocessing Improvements Review

Preprocessing checks performed:

- Missing value verification
- Duplicate row verification
- Feature range analysis
- Categorical feature inspection
- Class imbalance review

Recommendations:

1. Apply StandardScaler to numerical features.
2. Encode the Type column before model training.
3. Remove identifier columns such as UDI and Product ID.
4. Consider imbalance handling techniques during model training.
5. Verify engineered features before model development.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

Conclusion

The dataset is clean and ready for predictive maintenance modeling.

Key findings:

- No missing values
- No duplicate rows
- Feature distributions are acceptable
- Dataset is class imbalanced
- Feature scaling is recommended
