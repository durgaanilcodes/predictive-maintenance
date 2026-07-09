# Week 4 - Model Robustness Evaluation

## Completed Tasks

- Noise Sensitivity Analysis
- Threshold Tuning
- Precision-Recall Curve
- Model Performance Comparison
- Documentation

## Noise Sensitivity Analysis

Gaussian noise was added to the test dataset to evaluate model robustness.

Results showed that the model performance decreased significantly on noisy data, demonstrating the importance of clean sensor data for predictive maintenance.

## Threshold Tuning

Three thresholds were evaluated.

| Threshold | Precision | Recall | F1 Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| 0.3 | 0.62 | 0.97 | 0.75 | 0.98 |
| 0.5 | 0.77 | 0.97 | 0.86 | 0.99 |
| 0.7 | 0.87 | 0.97 | 0.92 | 0.99 |

Threshold 0.7 achieved the best overall performance.

## Conclusion

The LightGBM model performs well under normal conditions and achieves excellent predictive performance. Noise sensitivity analysis highlighted the impact of noisy sensor data, while threshold tuning identified the optimal decision threshold for reliable failure prediction.
