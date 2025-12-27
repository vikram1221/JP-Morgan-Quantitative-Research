# Credit Risk Analysis & Rating Framework

##Overview

This project implements a credit risk pipeline that:
1. estimates Probability of Default (PD) and Expected Loss, and
2. maps FICO scores to credit ratings using multiple quantization methods.
   
The focus is on transparent, interpretable credit modeling, not black-box ML.

## Data
- Task 3 and 4_Loan_Data.csv
    - Borrower financials, FICO scores, and default outcomes

Components:

1. Probability of Default & Expected Loss
- Logistic regression (statsmodels)
- Outputs borrower-level PD and Expected Loss

2. Credit Rating Maps
FICO scores are discretized into ratings using three methods:
- Quantile (equal-frequency)
- MSE-optimized
- Log-likelihood (bucket-level PD)

Lower rating = better credit quality.
