#!/usr/bin/env python3
"""
02_reconstruct_inventory.py
- Reconstructs trader-level inventory (net position) from transactions_clean.csv
- For each transaction:
    bid_trader (buyer) +quantity
    ask_trader (seller) -quantity
- Optionally filters out self-trades (default: does not filter)
"""

from __future__ import annotations

import os
import sys
import pandas as pd


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(filter_self_trades: bool = False) -> None:
    tx_path = os.path.join(RESULTS_DIR, "transactions_clean.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Run 01_clean_data.py first. Missing: {tx_path}")

    tx = pd.read_csv(tx_path)

    needed = {"trading_session_uuid", "bid_trader_uuid", "ask_trader_uuid", "quantity"}
    missing = needed - set(tx.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if filter_self_trades:
        tx = tx[tx["bid_trader_uuid"] != tx["ask_trader_uuid"]].copy()

    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx = tx.dropna(subset=["quantity"]).copy()

    # Inventory changes
    bids = tx[["trading_session_uuid", "bid_trader_uuid", "quantity"]].copy()
    bids.columns = ["trading_session_uuid", "trader_uuid", "inv_change"]
    bids["inv_change"] = bids["inv_change"].astype(float)

    asks = tx[["trading_session_uuid", "ask_trader_uuid", "quantity"]].copy()
    asks.columns = ["trading_session_uuid", "trader_uuid", "inv_change"]
    asks["inv_change"] = -asks["inv_change"].astype(float)

    inv = pd.concat([bids, asks], ignore_index=True)

    inv_summary = (
        inv.groupby(["trading_session_uuid", "trader_uuid"], as_index=False)["inv_change"]
        .sum()
        .rename(columns={"inv_change": "net_inventory"})
    )

    out_path = os.path.join(RESULTS_DIR, "inventory_by_session_trader.csv")
    inv_summary.to_csv(out_path, index=False)

    print("✅ Wrote:", out_path)
    print(inv_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main(filter_self_trades=False)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
