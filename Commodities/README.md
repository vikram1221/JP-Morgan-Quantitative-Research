# Natural Gas Seasonal Regression & Storage Pricing


## Overview
This project models natural gas prices using a seasonal regression and applies the model to value a storage contract under realistic operational constraints.
The focus is on interpretability, not black-box forecasting.

## Data
- Nat_Gas.csv
- Dates: monthly dates
- Prices: natural gas spot prices

## Method
Prices are modeled as a linear time trend + monthly seasonality using OLS.

The fitted model is used to:
  - interpolate historical prices,
  - forecast future monthly prices.

Estimated prices are then used to value a storage contract with:
  - injection/withdrawal limits,
  - capacity constraints,
  - daily storage costs.

## Key Components
- Seasonal regression with month dummies
- Price interpolation & extrapolation
- Storage contract valuation logic  

