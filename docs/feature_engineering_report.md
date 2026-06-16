Feature Engineering Report

Project

AI4I 2020 Predictive Maintenance Dataset

---

Task 1: Perform Feature Engineering on the Dataset

The following engineered features were created:

1. temp_difference

Process temperature minus Air temperature.

python
df["temp_difference"] =
df["Process temperature [K]"] -
df["Air temperature [K]"]

2. power_feature

Torque multiplied by Rotational Speed.

python
df["power_feature"] =
df["Torque [Nm]"] *
df["Rotational speed [rpm]"]

3. wear_speed_ratio

Tool Wear divided by Rotational Speed.

python
df["wear_speed_ratio"] =
df["Tool wear [min]"] /
df["Rotational speed [rpm]"]

4. wear_torque

Tool Wear multiplied by Torque.

python
df["wear_torque"] =
df["Tool wear [min]"] *
df["Torque [Nm]"]

____________________________________________________________________________________________________________________________________________________________________

Task 2: Create Additional Statistical and Derived Features

The following statistical features were created using Z-score normalization.

1. torque_zscore

python
df["torque_zscore"]

2. speed_zscore

python
df["speed_zscore"]

3. temp_zscore

python
df["temp_zscore"]

These features help identify values that deviate from normal operating conditions.

____________________________________________________________________________________________________________________________________________________________________

Task 3: Improve Preprocessing Pipeline

The preprocessing pipeline was improved through the following steps:

Label Encoding

Converted the categorical Type column into numerical values.

python
LabelEncoder()

Feature Selection

Removed:

- UDI
- Product ID

These columns do not contribute to machine failure prediction.

Standardization

Applied StandardScaler to numerical features:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

cal features
- Encoded categorical data
- Standardized numerical data

The dataset is prepared for machine learning model training and predictive maintenance analysis.
