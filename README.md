# Loan Default Prediction Using Machine Learning

End-to-end machine learning pipeline for predicting loan defaults with business-impact optimisation.

# Results

- F1-score improved from 0.72 → 0.84 through feature engineering and model benchmarking
- Business-impact threshold optimisation — classification cutoff selected to maximise net financial outcome, not default 0.5
- Benchmarked Logistic Regression, Random Forest, and Gradient Boosting using 5-fold cross-validation

# Dataset

- 58,000+ borrower records with demographic, loan, and credit history features
- 9 engineered features including age groups, income bands, debt burden categories, and interaction terms

# Tech Stack

Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Jupyter Notebook

# Key Features

- Full Scikit-learn Pipeline with ColumnTransformer for automated preprocessing
- One-hot encoding and standard scaling for clean, reproducible data flow
- ROC curves, confusion matrices, and feature importance visualisations
