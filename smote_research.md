# SMOTE Research

## What is SMOTE?

SMOTE (Synthetic Minority Over-sampling Technique) is a method used to handle imbalanced datasets by generating synthetic samples for the minority class.

## Why is SMOTE needed?

In predictive maintenance datasets, machine failures occur very rarely. Most records belong to the normal class, creating a class imbalance problem.

## Advantages of SMOTE

- Balances the dataset
- Improves Recall and F1 Score
- Reduces bias toward the majority class
- Helps machine learning models detect failures better

## Application in this Project

SMOTE will be applied to the training dataset before building the LightGBM model to improve failure prediction performance.

## Expected Outcome

The use of SMOTE is expected to improve:
- Recall
- F1 Score
- Failure detection capability
