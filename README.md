# Loan Default Prediction Using Machine Learning

Notebook project for classifying loan default risk with scikit-learn.

## Validation results

Results use a 20% stratified validation set with 11,729 rows.

| Model | F1 | AUC |
|---|---:|---:|
| Logistic Regression | 0.654 | 0.917 |
| Random Forest | 0.795 | 0.932 |
| Gradient Boosting | 0.787 | 0.940 |
| Tuned Random Forest | 0.795 | 0.935 |

The tuned Random Forest has a reported 5-fold cross-validation F1 of 0.794. GridSearchCV used 3-fold cross-validation for parameter tuning.

A cost-based business-impact function selected a probability threshold of 0.15, with recall 0.86 and precision 0.48. The dollar figure in the notebook is a simulation based on stated assumptions: a rejected defaulter saves the loan amount, while a good loan earns 10%. It is not a real saving.

## Run

The notebook expects `train.csv` and `test.csv` in the repository root. These files are not included. Supply those inputs before running the notebook.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook loan_default_prediction.ipynb
```

Run the notebook cells in order.

## Topics

credit-risk, loan-default, machine-learning, scikit-learn, jupyter-notebook
