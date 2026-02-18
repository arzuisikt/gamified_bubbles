#!/usr/bin/env python3
"""
05_regressions.py (small-sample safe)

- Merges session-level volatility and feedback
- Runs minimal OLS (no robust covariance)
- Prints coefficients
- Saves regression summary to results/regression_results.txt
"""

import os
import pandas as pd
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

vol_path = os.path.join(RESULTS_DIR, "volatility_by_session.csv")
fb_path = os.path.join(RESULTS_DIR, "feedback_by_session.csv")

if not os.path.exists(vol_path):
    raise FileNotFoundError("Missing volatility_by_session.csv")
if not os.path.exists(fb_path):
    raise FileNotFoundError("Missing feedback_by_session.csv")

vol = pd.read_csv(vol_path)
fb = pd.read_csv(fb_path)

# Merge datasets
df = vol.merge(fb, on="trading_session_uuid", how="inner")

print("\nMerged session-level dataset:")
print(df)

# Save merged dataset
merged_out = os.path.join(RESULTS_DIR, "session_level_dataset.csv")
df.to_csv(merged_out, index=False)
print("\nSaved merged dataset to:", merged_out)

if len(df) < 3:
    print("\n⚠️ WARNING: Very small sample size. Results are illustrative only.\n")

# -------------------------
# Regression: Volatility ~ Feedback
# -------------------------

y = df["realized_vol_logret"]
X = sm.add_constant(df[["feedback_rate"]])

model = sm.OLS(y, X).fit()

print("\nRegression coefficients:")
print(model.params)

print("\nR-squared:", model.rsquared)

# Save full regression summary to file
reg_out = os.path.join(RESULTS_DIR, "regression_results.txt")
with open(reg_out, "w") as f:
    f.write(str(model.summary()))

print("\nSaved regression summary to:", reg_out)
