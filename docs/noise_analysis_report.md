Noise Analysis Report

Project

Predictive Maintenance Using Contextual Data Fusion

Week

Week 4

Team Member

Binit Binu

-----------

Objective

The objective of this activity is to analyze how Gaussian noise affects the predictive maintenance dataset and evaluate the robustness of the machine learning model.

-------------

Dataset Used

Enhanced Predictive Maintenance Dataset

Rows: 10000

Features: 18

Target Variable:

Machine Failure

-----------------

Noise Injection

Gaussian noise was added to the numerical features of the testing dataset.

Noise Type:

Gaussian Noise

Mean:

0

Standard Deviation:

0.1

The noisy dataset was generated from the original testing dataset without modifying the training data.

------------------

Validation Results

The noisy dataset was successfully validated.

Observations:

- Dataset shape remained unchanged.
- No missing values were introduced.
- No duplicate rows were found.
- Feature data types remained valid.
- Numerical values changed slightly due to noise injection.

----------------

Model Evaluation

The trained LightGBM model was evaluated using both the original and noisy datasets.

Performance metrics compared:

- Accuracy
- Precision
- Recall
- F1 Score

The noisy dataset showed a slight reduction in model performance, indicating that the model is reasonably robust to small perturbations in the input data.

--------------

Conclusion

The experiment demonstrates that Gaussian noise slightly affects prediction performance but does not significantly degrade the model. The dataset remains suitable for predictive maintenance tasks after noise injection.
