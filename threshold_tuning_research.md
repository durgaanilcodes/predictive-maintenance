# Threshold Tuning Research

## What is Threshold Tuning?

Threshold tuning is the process of selecting the best probability threshold for classifying machine failures.

By default, machine learning models classify predictions using a threshold of 0.5. However, changing this threshold can improve Precision or Recall depending on project requirements.

## Why Threshold Tuning?

- Improve prediction performance
- Reduce false positives
- Reduce false negatives
- Find the best balance between Precision and Recall

## Example

Threshold = 0.5 → Default prediction

Threshold = 0.7 → Higher Precision

Threshold = 0.3 → Higher Recall

## In Our Project

Threshold tuning will be used after the LightGBM model predicts failure probabilities to determine the optimal decision threshold.
