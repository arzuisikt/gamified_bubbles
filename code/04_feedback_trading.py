#!/usr/bin/env python3
"""
04_feedback_trading.py
- Constructs a proxy measure of feedback trading
- The trader-type classification in the paper (Haruvy & Noussair) requires a period structure.
  Here, we instead build a transaction-time proxy.

Definition (proxy):
- A trade is classified as feedback if a trader makes a net BUY after a price increase.
- A trade is classified as feedback if a trader makes a net SELL after a price decrease.

Notes:
- Each transaction row contains both a buyer and a seller.
- For the buyer, sign = +1; for the seller, sign = -1.
- "Price change" is defined as the trade-to-trade price change within a session.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(filter_self_trades: bool = False) -> None:
    tx_path = os.path.join(RESULTS_DIR, "transactions_clean.csv")
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Run 01_clean_data.py first. Missing: {tx_path}")

    tx = pd.read_csv(tx_path)

    required = {
        "trading_session_uuid",
        "transaction_id",
        "price",
        "quantity",
        "timestamp_dt",
        "bid_trader_uuid",
        "ask_trader_uuid",
    }
    missing = required - set(tx.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamp_dt from CSV string to datetime
    tx["timestamp_dt"] = pd.to_datetime(tx["timestamp_dt"], errors="coerce", utc=True)

    if filter_self_trades:
        tx = tx[tx["bid_trader_uuid"] != tx["ask_trader_uuid"]].copy()

    tx["price"] = pd.to_numeric(tx["price"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")

    tx = tx.dropna(subset=["price", "quantity", "timestamp_dt"]).copy()
    tx = tx.sort_values(["trading_session_uuid", "timestamp_dt"]).reset_index(drop=True)

    # Price change per session
    tx["price_change"] = tx.groupby("trading_session_uuid")["price"].diff()

    # Build trader-level trade events (two rows per transaction: buyer and seller)
    buyer = tx[
        ["trading_session_uuid", "transaction_id", "timestamp_dt", "price", "price_change", "bid_trader_uuid", "quantity"]
    ].copy()
    buyer = buyer.rename(columns={"bid_trader_uuid": "trader_uuid"})
    buyer["signed_qty"] = +buyer["quantity"].astype(float)

    seller = tx[
        ["trading_session_uuid", "transaction_id", "timestamp_dt", "price", "price_change", "ask_trader_uuid", "quantity"]
    ].copy()
    seller = seller.rename(columns={"ask_trader_uuid": "trader_uuid"})
    seller["signed_qty"] = -seller["quantity"].astype(float)

    events = pd.concat([buyer, seller], ignore_index=True)

    # Remove first trade per session where price_change is NaN
    events = events.dropna(subset=["price_change"]).copy()

    # Feedback indicator:
    # if price_change > 0 and signed_qty > 0 => buy after uptick
    # if price_change < 0 and signed_qty < 0 => sell after downtick
    events["feedback_trade"] = np.where(
        ((events["price_change"] > 0) & (events["signed_qty"] > 0)) |
        ((events["price_change"] < 0) & (events["signed_qty"] < 0)),
        1, 0
    )

    # Summaries
    trader_summary = (
        events.groupby(["trading_session_uuid", "trader_uuid"], as_index=False)
        .agg(
            n_trades=("feedback_trade", "size"),
            feedback_rate=("feedback_trade", "mean"),
        )
    )

    session_summary = (
        events.groupby("trading_session_uuid", as_index=False)
        .agg(
            n_events=("feedback_trade", "size"),
            feedback_rate=("feedback_trade", "mean"),
        )
    )

    out_events = os.path.join(RESULTS_DIR, "feedback_events.csv")
    out_trader = os.path.join(RESULTS_DIR, "feedback_by_session_trader.csv")
    out_session = os.path.join(RESULTS_DIR, "feedback_by_session.csv")

    events.to_csv(out_events, index=False)
    trader_summary.to_csv(out_trader, index=False)
    session_summary.to_csv(out_session, index=False)

    print("✅ Wrote:")
    print(" -", out_events)
    print(" -", out_trader)
    print(" -", out_session)
    print("\nSession feedback summary (head):")
    print(session_summary.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main(filter_self_trades=False)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
