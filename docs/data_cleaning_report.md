Data Cleaning Report

Dataset
AI4I 2020 Predictive Maintenance Dataset

Dataset Size
- Rows: 10000
- Columns: 14

Missing Values Check

Command Used:

python:
df.isnull().sum()

df.duplicated().sum()

df.describe()

df['Machine failure'].value_counts()
