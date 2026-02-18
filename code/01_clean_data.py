#!/usr/bin/env python3
"""
01_clean_data.py
- Reads orders.csv and transactions.csv
- Parses timestamps
- Standardizes column types
- Reports duplicates and obvious inconsistencies
- Writes cleaned CSV outputs to the results/ directory
"""

from __future__ import annotations

import os
import sys
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def ensure_dirs() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def parse_iso_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def main() -> None:
    ensure_dirs()

    orders_path = os.path.join(DATA_DIR, "orders.csv")
    tx_path = os.path.join(DATA_DIR, "transactions.csv")

    orders = safe_read_csv(orders_path)
    tx = safe_read_csv(tx_path)

    # --- Standard column name normalization
    orders.columns = [c.strip() for c in orders.columns]
    tx.columns = [c.strip() for c in tx.columns]

    # --- Parse timestamps
    if "timestamp" in orders.columns:
        orders["timestamp_dt"] = parse_iso_datetime(orders["timestamp"])

    if "timestamp" in tx.columns:
        tx["timestamp_dt"] = parse_iso_datetime(tx["timestamp"])

    # created_ts as numeric
    if "created_ts" in orders.columns:
        orders["created_ts"] = pd.to_numeric(orders["created_ts"], errors="coerce")

    if "created_ts" in tx.columns:
        tx["created_ts"] = pd.to_numeric(tx["created_ts"], errors="coerce")

    # --- Type conversions
    for col in ["amount", "price", "order_type"]:
        if col in orders.columns:
            orders[col] = pd.to_numeric(orders[col], errors="coerce")

    for col in ["quantity", "price"]:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce")

    # --- Basic QA checks (console only)
    print("\nCleaning report summary:")

    if "timestamp_dt" in orders.columns:
        print("orders_missing_timestamp_dt:", orders["timestamp_dt"].isna().sum())

    if "timestamp_dt" in tx.columns:
        print("tx_missing_timestamp_dt:", tx["timestamp_dt"].isna().sum())

    if "order_id" in orders.columns and "trading_session_uuid" in orders.columns:
        dup_orders = orders.duplicated(
            subset=["trading_session_uuid", "order_id", "status"]
        ).sum()
        print("  orders_duplicates_session_orderid:", dup_orders)

    if "transaction_id" in tx.columns:
        dup_tx = tx.duplicated(subset=["transaction_id"]).sum()
        print("  tx_duplicates_transaction_id:", dup_tx)

    if "bid_trader_uuid" in tx.columns and "ask_trader_uuid" in tx.columns:
        self_trade_count = (tx["bid_trader_uuid"] == tx["ask_trader_uuid"]).sum()
        tx_count = len(tx)
        self_trade_rate = self_trade_count / tx_count if tx_count else None
        print("  self_trade_count:", self_trade_count)
        print("  tx_count:", tx_count)
        print("  self_trade_rate:", self_trade_rate)

    # --- Save cleaned versions as CSV
    orders_out = os.path.join(RESULTS_DIR, "orders_clean.csv")
    tx_out = os.path.join(RESULTS_DIR, "transactions_clean.csv")

    orders.to_csv(orders_out, index=False)
    tx.to_csv(tx_out, index=False)

    print("\n✅ Wrote:")
    print(f" - {orders_out}")
    print(f" - {tx_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
