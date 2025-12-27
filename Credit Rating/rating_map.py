import pandas as pd
import numpy as np
import os

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

## Baseline: Equal-frequency quantization

def fico_rating_quantile(df, fico_col="fico_score", n_buckets=5):
    """"
    Assigns ratings using equal-frequency (quantile) bucketing. 
    Lower rating = better credit
    """
    df = df.copy()

    df["rating"] = pd.qcut(
        df[fico_col],
        q=n_buckets,
        labels=range(n_buckets, 0, -1)
        ).astype(int)

    return df

# Generate ratings
df_quant = fico_rating_quantile(df, n_buckets=5)

# Export output
df_quant.to_csv("fico_rating_map_quantile.csv", index=False)

print(df_quant[["fico_score", "rating"]].head())


## MSE-Optimal FICO Quantization (1-D k-means)

def fico_rating_mse(df, fico_col="fico_score", n_buckets=5):
    """
    Approximates 1D k-means by sorting and segmenting. 
    Lower rating = better credit quality. 
    """
    df = df.copy()

    # Sort Fico scores
    df = df.sort_values(fico_col).reset_index(drop=True)

    # Split into approximately equal-size buckets
    buckets = np.array_split(df.index, n_buckets)

    df["rating_mse"] = np.nan

    # Assign ratings (highest FICO --> rating 1)
    for rating, idx in enumerate(reversed(buckets), start=1):
        df.loc[idx, "rating_mse"] = rating
    
    return df

# Generating ratings
df_quant.to_csv("fico_rating_map_quantile.csv", index=False)

# Export MSE-based ratings (New file)
df_mse = fico_rating_mse(df)
df_mse.to_csv("fico_rating_map_mse.csv", index=False)


## Log-likelihood-Based Quantization

def fico_rating_loglikelihood(
        df, 
        fico_col="fico_score",
        default_col="default",
        n_buckets=5 
):
    """
    Log-likelihood-based FICO Quantization. 
    Buckets maximise Bernoulli likelihood via stable PD estimates. 
    Lower rating = better credit quality
    """

    df = df.copy()

    # Sort by FICO
    df = df.sort_values(fico_col).reset_index(drop=True)

    # Create initial buckets (contiguous)
    buckets = np.array_split(df.index, n_buckets)

    bucket_stats = []

    for i, idx in enumerate(buckets):
        n_i = len(idx)
        k_i = df.loc[idx, default_col].sum()
        p_i = k_i / n_i if n_i > 0 else 0

        bucket_stats.append({
            "bucket": i,
            "n": n_i,
            "k": k_i,
            "pd": p_i,
            "idx": idx
        })

    # Sort buckets by PD (lower PD = better rating)
    bucket_stats = sorted(bucket_stats, key=lambda x: x["pd"])

    df["rating_ll"] = np.nan

    for rating, b in enumerate(bucket_stats, start=1):
        df.loc[b["idx"], "rating_ll"] = rating

    return df

## Log-likelihood-based ratings

df_ll = fico_rating_loglikelihood(df, n_buckets=5)

df_ll.to_csv("fico_rating_map_loglikelihood.csv", index=False)

print("Log-likelihood rating map exported: ")
print(df_ll[["fico_score", "default", "rating_ll"]].head())