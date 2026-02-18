#!/usr/bin/env python3
"""
03_compute_volatility.py
- Extracts the price series from transactions_clean.csv
- Computes session-level metrics:
   - simple returns
   - rolling volatility (e.g., a 10-trade window)
   - realized volatility (standard deviation of returns)
- The paper’s period-based surge/crash definition requires period information;
  here we provide a proxy instead.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(rolling_window_trades: int = 10, filter_self_trades: bool = False) -> None:
    # --- CSV input now
    tx_path = os.path.join(RESULTS_DIR, "transactions_clean.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Run 01_clean_data.py first. Missing: {tx_path}")

    tx = pd.read_csv(tx_path)

    # --- Required columns
    required = {
        "trading_session_uuid",
        "price",
        "timestamp_dt",
        "bid_trader_uuid",
        "ask_trader_uuid",
    }
    missing = required - set(tx.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Parse timestamp_dt (CSV stores it as string)
    tx["timestamp_dt"] = pd.to_datetime(tx["timestamp_dt"], errors="coerce", utc=True)

    # --- Optional self-trade filter
    if filter_self_trades:
        tx = tx[tx["bid_trader_uuid"] != tx["ask_trader_uuid"]].copy()

    # --- Clean numeric price
    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")

    tx = tx.dropna(subset=["price", "timestamp_dt"]).copy()
    tx = tx.sort_values(["trading_session_uuid", "timestamp_dt"]).reset_index(drop=True)

    # --- Compute returns within each session
    tx["log_price"] = np.log(tx["price"])

    tx["log_ret"] = tx.groupby("trading_session_uuid")["log_price"].diff()
    tx["ret"] = tx.groupby("trading_session_uuid")["price"].pct_change()

    # Rolling volatility of log returns
    tx["rolling_vol_logret"] = (
        tx.groupby("trading_session_uuid")["log_ret"]
        .rolling(rolling_window_trades)
        .std()
        .reset_index(level=0, drop=True)
    )

    # --- Session-level volatility summary
    summary = (
        tx.groupby("trading_session_uuid", as_index=False)
        .agg(
            n_trades=("price", "size"),
            price_first=("price", "first"),
            price_last=("price", "last"),
            realized_vol_logret=("log_ret", "std"),
            realized_vol_ret=("ret", "std"),
        )
    )

    # --- Save outputs
    tx_out = os.path.join(RESULTS_DIR, "transactions_with_volatility.csv")
    summary_out = os.path.join(RESULTS_DIR, "volatility_by_session.csv")

    tx.to_csv(tx_out, index=False)
    summary.to_csv(summary_out, index=False)

    print("✅ Wrote:")
    print(" -", tx_out)
    print(" -", summary_out)

    print("\nVolatility summary (head):")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main(rolling_window_trades=10, filter_self_trades=False)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
