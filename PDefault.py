import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load loan data
loan_df = pd.read_csv("Task 3 and 4_Loan_Data.csv")
print(loan_df.head())

# Prepare features and target
y = loan_df["default"]

X = loan_df.drop(columns=["default", "customer_id"])
X = X.fillna(X.mean())

# Add intercept (required for statsmodels)
X = sm.add_constant(X)

# Fit logistic regression
logit_model = sm.Logit(y, X)
pd_model = logit_model.fit(disp=False)

print(pd_model.summary())

# Predict probability of default
loan_df["PD"] = pd_model.predict(X)
loan_df[["PD", "default"]].head()


# Expected loss formula (Given)
## Expected Loss = PD * (1-0.10) * Exposure

RECOVERY_RATE = 0.10

def expected_loss(pd, exposure, recovery_rate=RECOVERY_RATE):
    return pd * (1 - recovery_rate) * exposure

def price_loan_expected_loss(borrower_features):
    """
    borrower_features: pandas Series or dict
    Returns expected loss
    """

    if isinstance(borrower_features, dict):
        borrower_features = pd.DataFrame([borrower_features])
    else:
        borrower_features = borrower_features.to_frame().T

    # Add constant
    borrower_features = sm.add_constant(borrower_features, has_constant="add")

    # FORCE column alignment with training data
    borrower_features = borrower_features[pd_model.model.exog_names]

    # Predict Probability of Default
    pd_hat = pd_model.predict(borrower_features)[0]

    # Exposure (adjust name if needed)
    exposure = borrower_features["loan_amt_outstanding"].iloc[0]

    return pd_hat, expected_loss(pd_hat, exposure)

if __name__ == "__main__":

    sample_borrower = loan_df.iloc[0].drop("default")

    pd_hat, el = price_loan_expected_loss(sample_borrower)

    print(f"Probability of Default: {pd_hat:.3f}")
    print(f"Expected Loss: {el:.2f}")

    # Export to csv

    loan_df["Expected_loss"] = (
        loan_df["PD"] * (1 - RECOVERY_RATE) * loan_df["loan_amt_outstanding"]
    )
    loan_df.to_csv("loan_pd_expected_loss.csv", index=False)

    print("Exported loan PD and Expected Loss to loan_pd_expected_loss.csv")