import pandas as pd
import numpy as np

# folder structure
raw_dir = "../raw_data"
out_dir = "../processed_data"

date = "2026-03-12"  # date of record for database
sessions = ["0xs4vw9m"]

# load raw data and retain relevant sessions
# -------------------
intro = pd.read_csv(f"{raw_dir}/intro_{date}.csv")
intro = intro[intro["session.code"].isin(sessions)]

post_exp = pd.read_csv(f"{raw_dir}/post_exp_{date}.csv")
post_exp = post_exp[post_exp["session.code"].isin(sessions)]

app = pd.read_csv(f"{raw_dir}/trader_bridge_app_{date}.csv")
app = app[app["session.code"].isin(sessions)]
app = app[app["participant._current_page_name"] == "FinalForProlific"]

mbo = pd.read_csv(f"{raw_dir}/trader_bridge_app_custom_export_mbo_{date}.csv")

# ------------------------------------------------------------
# Build trader-market-day panel w/ demographics and treatment
# ------------------------------------------------------------


def rename_columns(app):
    app = app.rename(
        columns={
            "session.code": "session_code",
            "participant.code": "participant_code",
            "participant.payoff": "payoff",
            "player.trader_uuid": "trader_uuid",
            "player.assigned_initial_cash": "initial_cash",
            "player.forecast_price_next_day": "forecast",
            "player.forecast_confidence_next_day": "forecast_confidence",
            "subsession.round_number": "trading_day",
            "group.noise_trader_present": "algo_present",
            "group.market_design": "gamified",
            "group.group_composition": "hybrid",
            "group.trading_session_uuid": "market_uuid",
        }
    )
    cols_app = [
        "session_code",
        "participant_code",
        "market_uuid",
        "trader_uuid",
        "gamified",
        "hybrid",
        "algo_present",
        "trading_day",
        "payoff",
        "initial_cash",
        "forecast",
        "forecast_confidence",
    ]

    # select relevant columns
    trader_day = app[cols_app]
    return trader_day


trader_day = rename_columns(app)

post_exp = post_exp.merge(
    intro[["participant.code", "player.self_assesment"]], on="participant.code"
)
post_exp["gender_female"] = np.where(post_exp["player.gender"] == "Female", 1, 0)
post_exp = post_exp.rename(
    columns={
        "participant.code": "participant_code",
        "player.payoff_for_trade": "trade_payoff",
        "player.gender": "gender",
        "player.age": "age",
        "player.course_financial": "finance_course",
        "player.trading_experience": "trading_experience",
        "player.self_assesment": "self_assessment",
    }
)
post_exp["fin_quiz_score"] = (
    post_exp["player.num_correct_answers"] / post_exp["player.num_quiz_questions"]
)
post_exp["high_education"] = np.where(
    post_exp["player.education"].isin(
        [
            "MBA",
            "PhD",
            "master",
            "undergraduate: 1st year",
            "undergraduate: 2nd year",
            "undergraduate: 3rd year",
            "undergraduate: 4th year",
        ]
    ),
    1,
    0,
)
post_exp["overconfidence"] = (
    post_exp["self_assessment"] / 10 - post_exp["fin_quiz_score"]
)
cols_post = [
    "participant_code",
    "trade_payoff",
    "fin_quiz_score",
    "self_assessment",
    "overconfidence",
    "gender_female",
    "age",
    "finance_course",
    "trading_experience",
    "high_education",
]
post2 = post_exp[cols_post]

trader_day = trader_day.merge(post2, on="participant_code")
trader_day["gamified"] = np.where(trader_day["gamified"] == 1, 1, 0)
trader_day["hybrid"] = np.where(trader_day["hybrid"] == "human_only", 0, 1)
trader_day["repetition"] = np.where(trader_day["trading_day"] <= 15, 1, 2)
trader_day["trading_day"] = np.where(
    trader_day["trading_day"] > 15,
    trader_day["trading_day"] - 15,
    trader_day["trading_day"],
)

################
# 2. Build trading panels
################
trades = mbo[mbo["record_kind"] == "trade"]
trades["fundamental_value"] = 8 * (16 - trades["trading_day"])
trades = trades.rename(
    columns={
        "bid_trader_uuid": "buyer_uuid",
        "ask_trader_uuid": "seller_uuid",
        "trading_session_uuid": "market_uuid",
        "market_number": "repetition",
    }
)
trades["event_ts"] = pd.to_datetime(trades["event_ts"])
trades["diff_time"] = (
    trades.groupby("market_uuid")["event_ts"]
    .diff()
    .apply(lambda x: x.seconds)
    .shift(-1)
)
cols_trade = [
    "market_uuid",
    "repetition",
    "trading_day",
    "event_ts",
    "diff_time",
    "buyer_uuid",
    "seller_uuid",
    "aggressor_side",
    "price",
    "fundamental_value",
]
trades = trades[cols_trade].reset_index(drop=True)
trades["mispricing"] = trades["price"] - trades["fundamental_value"]
trades["abs_mispricing"] = np.abs(trades["mispricing"])

## Market-period panel
mp = (
    trades.groupby(["market_uuid", "repetition", "trading_day"])
    .agg(
        n_trades=("price", "count"),
        avg_mispricing=("mispricing", "mean"),
        avg_abs_mispricing=("abs_mispricing", "mean"),
        closing_price=("price", "last"),
        opening_price=("price", "first"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        fundamental_value=("fundamental_value", "first"),
    )
    .reset_index()
)
mp["abs_mispricing_ratio"] = mp["avg_abs_mispricing"] / mp["fundamental_value"]
mp = mp.merge(
    trader_day[["market_uuid", "repetition", "trading_day"]].drop_duplicates(),
    how="outer",
)
mp["n_trades"] = mp["n_trades"].fillna(0)
mp["closing_price"] = mp.groupby("market_uuid")["closing_price"].ffill()
mp["fundamental_value"] = 8 * (16 - mp["trading_day"])
mp["return"] = mp.groupby("market_uuid")["closing_price"].pct_change()
