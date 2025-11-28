import time
import os
import argparse
import json
from pathlib import Path
import pandas as pd
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import (
    get_sel_df,
    get_all_markets,
    get_all_results,
    get_markets,
    add_volatility_to_df,
)
from gspread_dataframe import set_with_dataframe
import traceback

# Auto-manage configuration (defaults can be overridden via env)
AUTO_MANAGE_SELECTED = os.getenv("AUTO_MANAGE_SELECTED", "1") == "1"
AUTO_MAX_MARKETS = int(os.getenv("AUTO_MAX_MARKETS", "10"))
AUTO_DEFAULT_TRADE_SIZE = float(os.getenv("AUTO_DEFAULT_TRADE_SIZE", "10"))
AUTO_DEFAULT_MAX_SIZE = float(os.getenv("AUTO_DEFAULT_MAX_SIZE", "50"))
AUTO_PARAM_TYPE = os.getenv("AUTO_PARAM_TYPE", "mid")
AUTO_STALE_HOURS = float(
    os.getenv("AUTO_STALE_HOURS", "12")
)  # Increased from 6 to reduce premature removal
AUTO_MIN_GM_REWARD = float(os.getenv("AUTO_MIN_GM_REWARD", "0.5"))  # Minimum reward threshold
AUTO_ACTIVITY_FILE = Path(os.getenv("AUTO_ACTIVITY_FILE", "data/market_activity.json"))

# Scoring weights for candidate ranking (tuned for better reward/risk balance)
AUTO_VOL_WEIGHT = float(os.getenv("AUTO_VOL_WEIGHT", "0.3"))  # Reduced penalty for volatility
AUTO_SPREAD_WEIGHT = float(os.getenv("AUTO_SPREAD_WEIGHT", "0.5"))  # Reduced spread penalty
AUTO_REWARD_WEIGHT = float(os.getenv("AUTO_REWARD_WEIGHT", "2.0"))  # Bonus weight for high rewards
AUTO_LIQUIDITY_WEIGHT = float(os.getenv("AUTO_LIQUIDITY_WEIGHT", "0.5"))  # Bonus for good liquidity
AUTO_PRICE_PROXIMITY_WEIGHT = float(
    os.getenv("AUTO_PRICE_PROXIMITY_WEIGHT", "1.0")
)  # Bonus for prices near edges

# Allow min_size up to this multiple of trade_size when auto-selecting
AUTO_MIN_SIZE_MULT = float(os.getenv("AUTO_MIN_SIZE_MULT", "2.0"))
# Always rebuild Selected Markets instead of preserving existing rows
AUTO_RESET_SELECTED = os.getenv("AUTO_RESET_SELECTED", "0") == "1"

# Maximum short-term volatility (3-hour) allowed for selection
AUTO_MAX_3HR_VOLATILITY = float(os.getenv("AUTO_MAX_3HR_VOLATILITY", "50"))
# Prefer markets with moderate spreads (not too tight, not too wide)
AUTO_IDEAL_SPREAD_MIN = float(os.getenv("AUTO_IDEAL_SPREAD_MIN", "0.01"))
AUTO_IDEAL_SPREAD_MAX = float(os.getenv("AUTO_IDEAL_SPREAD_MAX", "0.05"))

# Liquidity requirements - filter out illiquid markets
# Minimum bid price (must have real bids, not just 0.01)
AUTO_MIN_BID = float(os.getenv("AUTO_MIN_BID", "0.05"))
# Maximum ask price (must have real asks, not just 0.99)
AUTO_MAX_ASK = float(os.getenv("AUTO_MAX_ASK", "0.95"))
# Maximum allowed spread (too wide = illiquid)
AUTO_MAX_SPREAD = float(os.getenv("AUTO_MAX_SPREAD", "0.15"))

# Initialize global variables
spreadsheet = get_spreadsheet()
client = get_clob_client()

wk_all = spreadsheet.worksheet("All Markets")
wk_vol = spreadsheet.worksheet("Volatility Markets")

sel_df = get_sel_df(spreadsheet, "Selected Markets")


def update_sheet(data, worksheet):
    all_values = worksheet.get_all_values()
    existing_num_rows = len(all_values)
    existing_num_cols = len(all_values[0]) if all_values else 0

    num_rows, num_cols = data.shape
    max_rows = max(num_rows, existing_num_rows)
    max_cols = max(num_cols, existing_num_cols)

    # Create a DataFrame with the maximum size and fill it with empty strings
    padded_data = pd.DataFrame("", index=range(max_rows), columns=range(max_cols))

    # Update the padded DataFrame with the original data and its columns
    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [""] * (max_cols - num_cols)

    # Update the sheet with the padded DataFrame, including column headers
    set_with_dataframe(
        worksheet, padded_data, include_index=False, include_column_header=True, resize=True
    )


def sort_df(df):
    # Calculate the mean and standard deviation for each column
    mean_gm = df["gm_reward_per_100"].mean()
    std_gm = df["gm_reward_per_100"].std()

    mean_volatility = df["volatility_sum"].mean()
    std_volatility = df["volatility_sum"].std()

    # Standardize the columns
    df["std_gm_reward_per_100"] = (df["gm_reward_per_100"] - mean_gm) / std_gm
    df["std_volatility_sum"] = (df["volatility_sum"] - mean_volatility) / std_volatility

    # Define a custom scoring function for best_bid and best_ask
    def proximity_score(value):
        if 0.1 <= value <= 0.25:
            return (0.25 - value) / 0.15
        elif 0.75 <= value <= 0.9:
            return (value - 0.75) / 0.15
        else:
            return 0

    df["bid_score"] = df["best_bid"].apply(proximity_score)
    df["ask_score"] = df["best_ask"].apply(proximity_score)

    # Create a composite score (higher is better for rewards, lower is better for volatility, with proximity scores)
    df["composite_score"] = (
        df["std_gm_reward_per_100"] - df["std_volatility_sum"] + df["bid_score"] + df["ask_score"]
    )

    # Sort by the composite score in descending order
    sorted_df = df.sort_values(by="composite_score", ascending=False)

    # Drop the intermediate columns used for calculation
    sorted_df = sorted_df.drop(
        columns=[
            "std_gm_reward_per_100",
            "std_volatility_sum",
            "bid_score",
            "ask_score",
            "composite_score",
        ]
    )

    return sorted_df


def _load_activity():
    try:
        if AUTO_ACTIVITY_FILE.exists():
            return json.loads(AUTO_ACTIVITY_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_activity(activity):
    try:
        AUTO_ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        AUTO_ACTIVITY_FILE.write_text(json.dumps(activity))
    except Exception as ex:
        print(f"Warning: could not persist activity file: {ex}")


def _update_activity_map(activity_map, df):
    now = time.time()
    for _, row in df.iterrows():
        cid = str(row.get("condition_id", ""))
        if not cid:
            continue
        # Mark activity when there is any bid/ask present
        try:
            if float(row.get("best_bid", 0)) > 0 or float(row.get("best_ask", 0)) > 0:
                activity_map[cid] = now
        except Exception:
            pass
        # Initialize unseen markets to now to give them a grace period
        activity_map.setdefault(cid, now)
    return activity_map


def _compute_price_proximity_score(best_bid, best_ask):
    """
    Calculate a score based on how favorable the price is for market making.

    Prices near the edges (0.1-0.25 or 0.75-0.9) are more profitable for market makers
    because there's less probability of losing on the position.
    Prices near 0.5 are riskier as outcomes are uncertain.
    """
    if best_bid is None or best_ask is None:
        return 0

    try:
        best_bid = float(best_bid)
        best_ask = float(best_ask)
    except (TypeError, ValueError):
        return 0

    mid_price = (best_bid + best_ask) / 2

    # Score based on proximity to favorable price ranges
    # Best: prices near 0.1-0.25 or 0.75-0.9
    # Worst: prices near 0.5

    if 0.1 <= mid_price <= 0.25:
        # Near low end - good for buying YES cheap
        return (0.25 - mid_price) / 0.15  # 0 to 1
    elif 0.75 <= mid_price <= 0.9:
        # Near high end - good for selling YES expensive
        return (mid_price - 0.75) / 0.15  # 0 to 1
    elif 0.25 < mid_price < 0.4:
        # Moderate low - still reasonable
        return 0.3
    elif 0.6 < mid_price < 0.75:
        # Moderate high - still reasonable
        return 0.3
    else:
        # Near 0.5 - highest uncertainty
        return 0


def _compute_liquidity_score(spread, best_bid, best_ask):
    """
    Score based on market liquidity indicators.

    Moderate spreads are actually good - too tight means no profit margin,
    too wide means harder to trade.
    """
    try:
        spread = float(spread) if spread else 0
        best_bid = float(best_bid) if best_bid else 0
        best_ask = float(best_ask) if best_ask else 0
    except (TypeError, ValueError):
        return 0

    # No liquidity
    if best_bid <= 0 or best_ask <= 0:
        return 0

    # Ideal spread range (1-5 cents) - room for profit but not too illiquid
    if AUTO_IDEAL_SPREAD_MIN <= spread <= AUTO_IDEAL_SPREAD_MAX:
        return 1.0
    elif spread < AUTO_IDEAL_SPREAD_MIN:
        # Very tight spread - low profit margin
        return 0.3
    elif spread <= AUTO_IDEAL_SPREAD_MAX * 2:
        # Moderate spread - still tradeable
        return 0.6
    else:
        # Wide spread - illiquid, harder to trade
        return 0.2


def _compute_volatility_score(row):
    """
    Score based on volatility profile.

    We want LOW short-term volatility (less immediate risk) but
    some moderate activity (not completely dead markets).
    """
    try:
        vol_3h = float(row.get("3_hour", 0) or 0)
        vol_24h = float(row.get("24_hour", 0) or 0)
        vol_sum = float(row.get("volatility_sum", 0) or 0)
    except (TypeError, ValueError):
        return 0

    # Penalize high short-term volatility heavily
    if vol_3h > AUTO_MAX_3HR_VOLATILITY:
        return -1.0  # Strong penalty

    # Low volatility is good - less risk of adverse movement
    if vol_sum < 5:
        return 1.0
    elif vol_sum < 15:
        return 0.7
    elif vol_sum < 30:
        return 0.4
    elif vol_sum < 50:
        return 0.1
    else:
        return -0.3  # Penalty for very high volatility


def _compute_order_book_depth_score(row):
    """
    Score based on order book depth indicators.
    
    Deeper order books mean:
    - More reliable order execution
    - Easier to enter/exit positions
    - Less slippage on larger orders
    - More active market = more trade opportunities
    """
    try:
        best_bid = float(row.get("best_bid", 0) or 0)
        best_ask = float(row.get("best_ask", 1) or 1)
        spread = float(row.get("spread", 1) or 1)
        daily_rate = float(row.get("rewards_daily_rate", 0) or 0)
    except (TypeError, ValueError):
        return 0
    
    # Higher daily rate indicates more volume/activity
    # Scale: typically ranges from 0 to 5000+ for active markets
    if daily_rate > 1000:
        volume_score = 1.0
    elif daily_rate > 500:
        volume_score = 0.8
    elif daily_rate > 200:
        volume_score = 0.6
    elif daily_rate > 100:
        volume_score = 0.4
    elif daily_rate > 50:
        volume_score = 0.2
    else:
        volume_score = 0.1  # Low volume markets - fewer trades possible
    
    # Tight spreads with good volume = excellent for market making
    if spread < 0.02 and daily_rate > 200:
        # Very active, tight market
        return volume_score * 1.5
    elif spread < 0.05 and daily_rate > 100:
        # Active market with reasonable spread
        return volume_score * 1.2
    else:
        return volume_score


def _compute_reward_efficiency_score(row):
    """
    Score based on reward relative to volatility (risk-adjusted return).
    """
    try:
        reward = float(row.get("gm_reward_per_100", 0) or 0)
        vol_sum = float(row.get("volatility_sum", 1) or 1)  # Avoid division by zero
    except (TypeError, ValueError):
        return 0

    if vol_sum <= 0:
        vol_sum = 1

    # Reward per unit of volatility
    efficiency = reward / vol_sum

    # Normalize to a reasonable scale
    if efficiency >= 1.0:
        return 1.0
    elif efficiency >= 0.5:
        return 0.8
    elif efficiency >= 0.2:
        return 0.5
    elif efficiency >= 0.1:
        return 0.3
    else:
        return 0.1


def compute_market_score(row, hyperparams=None):
    """
    Comprehensive market scoring function that balances multiple factors:
    - Raw reward potential
    - Risk (volatility)
    - Liquidity (spread)
    - Price proximity to favorable ranges
    - Reward efficiency (reward per unit risk)
    - Order book depth / market activity

    Returns a composite score where higher is better.
    """
    try:
        reward = float(row.get("gm_reward_per_100", 0) or 0)
        vol_sum = float(row.get("volatility_sum", 0) or 0)
        spread = float(row.get("spread", 0) or 0)
        best_bid = float(row.get("best_bid", 0) or 0)
        best_ask = float(row.get("best_ask", 0) or 0)
    except (TypeError, ValueError):
        return -999  # Invalid data

    # Skip markets with no liquidity
    if best_bid <= 0 or best_ask <= 0:
        return -999

    # Component scores
    price_score = _compute_price_proximity_score(best_bid, best_ask)
    liquidity_score = _compute_liquidity_score(spread, best_bid, best_ask)
    volatility_score = _compute_volatility_score(row)
    efficiency_score = _compute_reward_efficiency_score(row)
    depth_score = _compute_order_book_depth_score(row)  # NEW: order book depth

    # Weighted composite score
    # Primary factor: reward (normalized to ~0-5 range by dividing by typical max)
    reward_normalized = min(reward / 5.0, 2.0)  # Cap at 2.0 to prevent outlier domination

    composite = (
        AUTO_REWARD_WEIGHT * reward_normalized
        + AUTO_PRICE_PROXIMITY_WEIGHT * price_score
        + AUTO_LIQUIDITY_WEIGHT * liquidity_score
        + AUTO_LIQUIDITY_WEIGHT * depth_score  # NEW: factor in order book depth
        + volatility_score  # Already includes penalty
        - AUTO_VOL_WEIGHT * (vol_sum / 50.0)  # Normalized volatility penalty
        - AUTO_SPREAD_WEIGHT * (spread / 0.1)  # Normalized spread penalty
    )

    return composite


def _cancel_orders_for_market(client, condition_id, token1=None, token2=None):
    """
    Cancel all open orders for a market before removing it from Selected Markets.
    
    Tries multiple approaches:
    1. Cancel by market ID (condition_id)
    2. Cancel by individual token IDs (token1, token2) if provided
    """
    if client is None:
        return
    
    cancelled = False
    
    # Try cancelling by market ID first
    try:
        client.cancel_all_market(str(condition_id))
        print(f"Cancelled all orders for market {condition_id}")
        cancelled = True
    except Exception as ex:
        print(f"Warning: cancel_all_market failed for {condition_id}: {ex}")
    
    # Also try cancelling by individual tokens if provided
    if token1:
        try:
            client.cancel_all_asset(str(token1))
            print(f"  Cancelled orders for token1: {token1[:20]}...")
            cancelled = True
        except Exception as ex:
            pass  # Silent - may have no orders for this token
    
    if token2:
        try:
            client.cancel_all_asset(str(token2))
            print(f"  Cancelled orders for token2: {token2[:20]}...")
            cancelled = True
        except Exception as ex:
            pass  # Silent - may have no orders for this token
    
    if not cancelled:
        print(f"Warning: could not cancel any orders for market {condition_id}")


def _build_row_from_candidate(row, columns):
    # Defaults for custom columns
    defaults = {
        "param_type": AUTO_PARAM_TYPE,
        "trade_size": AUTO_DEFAULT_TRADE_SIZE,
        "max_size": AUTO_DEFAULT_MAX_SIZE,
        "min_size": row.get("min_size", AUTO_DEFAULT_TRADE_SIZE),
        "multiplier": "",
        "neg_risk": row.get("neg_risk", ""),
    }

    base = {
        "question": row.get("question", ""),
        "answer1": row.get("answer1", ""),
        "answer2": row.get("answer2", ""),
        "market_slug": row.get("market_slug", ""),
        "condition_id": row.get("condition_id", ""),
        "token1": row.get("token1", ""),
        "token2": row.get("token2", ""),
    }
    base.update(defaults)

    # Fill remaining columns to keep sheet shape intact
    for col in columns:
        if col not in base:
            base[col] = row.get(col, "")
    return base


def load_hyperparameters(spreadsheet):
    """
    Load hyperparameters (type -> param -> value) from the Google Sheet.
    """
    hyper = {}
    try:
        wk = spreadsheet.worksheet("Hyperparameters")
        records = wk.get_all_records()
    except Exception as ex:
        print(f"Warning: could not fetch Hyperparameters sheet: {ex}")
        return hyper

    current_type = None
    for record in records:
        type_val = record.get("type")
        if type_val and str(type_val).strip() and str(type_val).lower() != "nan":
            current_type = str(type_val).strip()
        if not current_type:
            continue
        param = record.get("param")
        value = record.get("value")
        if param is None:
            continue
        try:
            if isinstance(value, str) and value.replace(".", "", 1).replace("-", "", 1).isdigit():
                value = float(value)
            elif isinstance(value, (int, float)):
                value = float(value)
        except Exception:
            pass
        hyper.setdefault(current_type, {})[param] = value
    return hyper


def auto_manage_selected_markets(new_df, worksheet, client, hyperparams=None):
    """
    Keep Selected Markets capped, add top candidates, remove stale ones (after cancelling orders).

    Uses improved scoring algorithm that considers:
    - Reward potential (gm_reward_per_100)
    - Risk (volatility profile)
    - Liquidity (spread, bid/ask presence)
    - Price proximity to favorable ranges
    - Reward efficiency (reward per unit of volatility)
    """
    if hyperparams is None:
        hyperparams = {}
    try:
        current_sel = pd.DataFrame(worksheet.get_all_records())
    except Exception as ex:
        print(f"Warning: could not read Selected Markets: {ex}")
        return

    # Determine columns to preserve
    sel_columns = (
        list(current_sel.columns)
        if len(current_sel.columns) > 0
        else [
            "question",
            "answer1",
            "answer2",
            "market_slug",
            "condition_id",
            "token1",
            "token2",
            "neg_risk",
            "param_type",
            "trade_size",
            "max_size",
            "min_size",
            "multiplier",
        ]
    )
    if current_sel.empty:
        current_sel = pd.DataFrame(columns=sel_columns)

    if AUTO_RESET_SELECTED:
        print("AUTO_RESET_SELECTED enabled; clearing Selected Markets and rebuilding.")
        if client is not None and "condition_id" in current_sel.columns:
            for _, row in current_sel.iterrows():
                cid = str(row.get("condition_id", ""))
                t1 = str(row.get("token1", "")) if "token1" in row else None
                t2 = str(row.get("token2", "")) if "token2" in row else None
                _cancel_orders_for_market(client, cid, t1, t2)
        current_sel = pd.DataFrame(columns=sel_columns)

    # Update activity map with the latest book data
    activity_map = _load_activity()
    activity_map = _update_activity_map(activity_map, new_df)
    _save_activity(activity_map)

    now = time.time()

    if "condition_id" not in current_sel.columns:
        print("Selected Markets sheet missing condition_id; skipping auto-manage.")
        return

    # Merge stats onto selected to evaluate removal
    stats_cols = [
        "condition_id",
        "gm_reward_per_100",
        "best_bid",
        "best_ask",
        "spread",
        "volatility_sum",
        "3_hour",
        "24_hour",
    ]
    available_cols = [c for c in stats_cols if c in new_df.columns]
    merged = current_sel.merge(
        new_df[available_cols], on="condition_id", how="left", suffixes=("", "_stats")
    )

    # Determine markets where we're currently active (positions or orders)
    active_markets = set()
    if client is not None:
        try:
            positions = client.get_all_positions()
            if "condition_id" in positions.columns:
                active_markets.update(positions["condition_id"].astype(str))
        except Exception as ex:
            print(f"Warning: could not fetch positions for active check: {ex}")
        try:
            open_orders = client.get_all_orders()
            if "market" in open_orders.columns:
                active_markets.update(open_orders["market"].astype(str))
            elif "condition_id" in open_orders.columns:
                active_markets.update(open_orders["condition_id"].astype(str))
        except Exception as ex:
            print(f"Warning: could not fetch orders for active check: {ex}")

    stale_ids = []
    removal_reasons = {}

    for _, row in merged.iterrows():
        cid = str(row.get("condition_id", ""))
        if not cid:
            continue
        if cid in active_markets:
            continue  # Don't remove markets with active trades

        last_seen = float(activity_map.get(cid, now))
        time_stale = (now - last_seen) > (AUTO_STALE_HOURS * 3600)

        reward = row.get("gm_reward_per_100", 0)
        try:
            reward_val = float(reward) if reward == reward else 0  # handles NaN
        except Exception:
            reward_val = 0

        # Check for 3-hour volatility spike
        try:
            vol_3h = float(row.get("3_hour", 0) or 0)
        except (TypeError, ValueError):
            vol_3h = 0

        # Check for no/low liquidity
        best_bid = float(row.get("best_bid", 0) or 0)
        best_ask = float(row.get("best_ask", 1) or 1)
        spread = float(row.get("spread", 1) or 1)

        no_liquidity = (best_bid == 0) and (best_ask == 0)
        low_liquidity = (
            (best_bid < AUTO_MIN_BID) or (best_ask > AUTO_MAX_ASK) or (spread > AUTO_MAX_SPREAD)
        )

        reward_low = reward_val < AUTO_MIN_GM_REWARD
        high_short_term_vol = vol_3h > AUTO_MAX_3HR_VOLATILITY

        if time_stale:
            stale_ids.append(cid)
            removal_reasons[cid] = "stale (no activity)"
        elif no_liquidity:
            stale_ids.append(cid)
            removal_reasons[cid] = "no liquidity"
        elif reward_low:
            stale_ids.append(cid)
            removal_reasons[cid] = f"low reward ({reward_val:.2f} < {AUTO_MIN_GM_REWARD})"
        elif high_short_term_vol:
            stale_ids.append(cid)
            removal_reasons[cid] = (
                f"high 3-hour volatility ({vol_3h:.1f} > {AUTO_MAX_3HR_VOLATILITY})"
            )
        elif low_liquidity:
            stale_ids.append(cid)
            removal_reasons[cid] = (
                f"low liquidity (bid={best_bid:.2f}, ask={best_ask:.2f}, spread={spread:.2f})"
            )

    if stale_ids:
        print(f"Auto-manage: removing {len(stale_ids)} market(s):")
        for cid in stale_ids:
            print(f"  - {cid}: {removal_reasons.get(cid, 'unknown reason')}")
        if client is not None:
            # Cancel orders for each removed market using both market ID and token IDs
            for cid in stale_ids:
                row = current_sel[current_sel["condition_id"].astype(str) == cid]
                t1 = str(row["token1"].iloc[0]) if len(row) > 0 and "token1" in row.columns else None
                t2 = str(row["token2"].iloc[0]) if len(row) > 0 and "token2" in row.columns else None
                _cancel_orders_for_market(client, cid, t1, t2)
        # Remove stale rows
        current_sel = current_sel[~current_sel["condition_id"].astype(str).isin(stale_ids)]

    # Enforce max markets by trimming lowest-scoring if still above cap
    current_sel = current_sel.reset_index(drop=True)
    if len(current_sel) > AUTO_MAX_MARKETS:
        # Re-score existing markets and trim worst performers
        trimmed = current_sel.merge(new_df[available_cols], on="condition_id", how="left")
        trimmed["_score"] = trimmed.apply(lambda r: compute_market_score(r, hyperparams), axis=1)
        trimmed = trimmed.sort_values("_score", ascending=False)
        keep = trimmed.head(AUTO_MAX_MARKETS)["condition_id"].astype(str)
        drop_ids = set(trimmed["condition_id"].astype(str)) - set(keep)
        if drop_ids:
            print(
                f"Auto-manage: trimming {len(drop_ids)} market(s) to enforce cap (keeping top {AUTO_MAX_MARKETS})"
            )
            if client is not None:
                # Cancel orders for each trimmed market using both market ID and token IDs
                for cid in drop_ids:
                    row = current_sel[current_sel["condition_id"].astype(str) == cid]
                    t1 = str(row["token1"].iloc[0]) if len(row) > 0 and "token1" in row.columns else None
                    t2 = str(row["token2"].iloc[0]) if len(row) > 0 and "token2" in row.columns else None
                    _cancel_orders_for_market(client, cid, t1, t2)
            current_sel = current_sel[current_sel["condition_id"].astype(str).isin(keep)]

    # Compute how many slots are open
    slots = max(0, AUTO_MAX_MARKETS - len(current_sel))

    # Build candidate pool not already selected
    selected_ids = set(current_sel["condition_id"].astype(str))
    candidates = new_df[~new_df["condition_id"].astype(str).isin(selected_ids)].copy()

    # Filter candidates by basic requirements
    candidates = candidates[
        (candidates["gm_reward_per_100"] >= AUTO_MIN_GM_REWARD)
        & (candidates["best_bid"] > 0)
        & (candidates["best_ask"] > 0)
    ]

    # Filter out illiquid markets with stricter liquidity requirements
    # Must have real bid (not just 0.01) and real ask (not just 0.99)
    candidates = candidates[
        (candidates["best_bid"] >= AUTO_MIN_BID) & (candidates["best_ask"] <= AUTO_MAX_ASK)
    ]

    # Filter out markets with spreads too wide (illiquid)
    if "spread" in candidates.columns:
        spread_numeric = pd.to_numeric(candidates["spread"], errors="coerce")
        candidates = candidates[spread_numeric <= AUTO_MAX_SPREAD]

    # Log how many candidates remain after liquidity filter
    print(
        f"Auto-manage: {len(candidates)} candidates after liquidity filter (bid>={AUTO_MIN_BID}, ask<={AUTO_MAX_ASK}, spread<={AUTO_MAX_SPREAD})"
    )

    # Skip markets where the minimum size is far above our trade size
    if "min_size" in candidates.columns:
        min_size_numeric = pd.to_numeric(candidates["min_size"], errors="coerce")
        candidates = candidates[min_size_numeric <= AUTO_DEFAULT_TRADE_SIZE * AUTO_MIN_SIZE_MULT]

    # Filter on 3-hour volatility threshold
    if "3_hour" in candidates.columns:
        vol_numeric = pd.to_numeric(candidates["3_hour"], errors="coerce")
        candidates = candidates[vol_numeric <= AUTO_MAX_3HR_VOLATILITY]

    # Also check hyperparameters volatility threshold if available
    param_cfg = hyperparams.get(AUTO_PARAM_TYPE, {}) if hyperparams else {}
    vol_threshold = param_cfg.get("volatility_threshold")
    try:
        vol_threshold = float(vol_threshold)
    except (TypeError, ValueError):
        vol_threshold = None
    if vol_threshold is not None and "3_hour" in candidates.columns:
        vol_numeric = pd.to_numeric(candidates["3_hour"], errors="coerce")
        candidates = candidates[vol_numeric <= vol_threshold]

    # Score all candidates using the comprehensive scoring function
    candidates["_score"] = candidates.apply(lambda r: compute_market_score(r, hyperparams), axis=1)

    # Filter out invalid scores
    candidates = candidates[candidates["_score"] > -100]

    # Sort by score (best first)
    candidates = candidates.sort_values("_score", ascending=False)

    additions = []
    if slots > 0 and len(candidates) > 0:
        print(
            f"Auto-manage: evaluating top {min(slots * 2, len(candidates))} candidates for {slots} slot(s):"
        )
        for _, row in candidates.head(slots).iterrows():
            score = row["_score"]
            reward = row.get("gm_reward_per_100", 0)
            vol_sum = row.get("volatility_sum", 0)
            spread = row.get("spread", 0)
            question = row.get("question", "")[:50]
            print(
                f"  + Adding: score={score:.2f}, reward={reward:.2f}, vol={vol_sum:.1f}, spread={spread:.3f}"
            )
            print(f"    Question: {question}...")
            additions.append(_build_row_from_candidate(row, sel_columns))

    if additions:
        print(f"Auto-manage: adding {len(additions)} market(s) to Selected")
        additions_df = pd.DataFrame(additions)
        # Ensure all columns exist
        for col in sel_columns:
            if col not in additions_df.columns:
                additions_df[col] = ""
        additions_df = additions_df[sel_columns]
        current_sel = pd.concat([current_sel, additions_df], ignore_index=True)

    # De-dup just in case
    current_sel = current_sel.drop_duplicates(subset=["condition_id"])

    # Keep ID columns as strings to avoid scientific notation
    id_cols = ["condition_id", "token1", "token2"]
    for col in id_cols:
        if col in current_sel.columns:
            current_sel[col] = current_sel[col].astype(str)

    # Normalize sizing columns using defaults and ensure we can place orders
    min_map = (
        pd.to_numeric(new_df.set_index("condition_id")["min_size"], errors="coerce")
        if "condition_id" in new_df.columns
        else {}
    )
    min_series = (
        current_sel["condition_id"].map(min_map) if "condition_id" in current_sel.columns else None
    )
    if min_series is None:
        min_series = pd.Series(AUTO_DEFAULT_TRADE_SIZE, index=current_sel.index)
    min_series = min_series.fillna(AUTO_DEFAULT_TRADE_SIZE)
    current_sel["min_size"] = min_series

    trade_series = min_series.where(min_series > AUTO_DEFAULT_TRADE_SIZE, AUTO_DEFAULT_TRADE_SIZE)
    current_sel["trade_size"] = trade_series

    max_series = trade_series.where(trade_series > AUTO_DEFAULT_MAX_SIZE, AUTO_DEFAULT_MAX_SIZE)
    current_sel["max_size"] = max_series

    # Write back
    update_sheet(current_sel, worksheet)
    print(f"Auto-manage: Selected Markets now {len(current_sel)} rows (cap {AUTO_MAX_MARKETS})")


def fetch_and_process_data():
    global spreadsheet, client, wk_all, wk_vol, sel_df

    spreadsheet = get_spreadsheet()
    client = get_clob_client()

    wk_all = spreadsheet.worksheet("All Markets")
    wk_vol = spreadsheet.worksheet("Volatility Markets")
    wk_selected = spreadsheet.worksheet("Selected Markets")
    wk_full = spreadsheet.worksheet("Full Markets")

    sel_df = get_sel_df(spreadsheet, "Selected Markets")
    hyperparams = load_hyperparameters(spreadsheet)

    all_df = get_all_markets(client)
    print("Got all Markets")
    all_results = get_all_results(all_df, client)
    print("Got all Results")
    m_data, all_markets = get_markets(all_results, sel_df, maker_reward=0.75)
    print("Got all orderbook")

    print(f'{pd.to_datetime("now")}: Fetched all markets data of length {len(all_markets)}.')
    new_df = add_volatility_to_df(all_markets)
    new_df["volatility_sum"] = new_df["24_hour"] + new_df["7_day"] + new_df["14_day"]

    new_df = new_df.sort_values("volatility_sum", ascending=True)
    new_df["volatilty/reward"] = (
        (new_df["gm_reward_per_100"] / new_df["volatility_sum"]).round(2)
    ).astype(str)

    # Add comprehensive market score for visibility
    new_df["market_score"] = new_df.apply(
        lambda r: round(compute_market_score(r, hyperparams), 2), axis=1
    )

    new_df = new_df[
        [
            "question",
            "answer1",
            "answer2",
            "spread",
            "rewards_daily_rate",
            "gm_reward_per_100",
            "sm_reward_per_100",
            "bid_reward_per_100",
            "ask_reward_per_100",
            "volatility_sum",
            "volatilty/reward",
            "market_score",
            "min_size",
            "1_hour",
            "3_hour",
            "6_hour",
            "12_hour",
            "24_hour",
            "7_day",
            "30_day",
            "best_bid",
            "best_ask",
            "volatility_price",
            "max_spread",
            "tick_size",
            "neg_risk",
            "market_slug",
            "token1",
            "token2",
            "condition_id",
        ]
    ]

    volatility_df = new_df.copy()
    volatility_df = volatility_df[new_df["volatility_sum"] < 20]
    # Sort by market_score (best opportunities first)
    volatility_df = volatility_df.sort_values("market_score", ascending=False)

    # Sort main df by market_score for better visibility of top opportunities
    new_df = new_df.sort_values("market_score", ascending=False)

    print(f'{pd.to_datetime("now")}: Fetched select market of length {len(new_df)}.')

    # Ensure ID columns remain strings (avoid scientific notation in Sheets)
    id_cols = ["token1", "token2", "condition_id"]
    for col in id_cols:
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(str)
        if col in m_data.columns:
            m_data[col] = m_data[col].astype(str)

    if len(new_df) > 50:
        update_sheet(new_df, wk_all)
        update_sheet(volatility_df, wk_vol)
        update_sheet(m_data, wk_full)
        if AUTO_MANAGE_SELECTED:
            auto_manage_selected_markets(new_df, wk_selected, client, hyperparams)
    else:
        print(f'{pd.to_datetime("now")}: Not updating sheet because of length {len(new_df)}.')


def main():
    parser = argparse.ArgumentParser(description="Fetch and update market data in a loop")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=int(os.getenv("UPDATE_INTERVAL_MINUTES", "60")),
        help="Minutes to wait between runs (default: 60 or UPDATE_INTERVAL_MINUTES env var)",
    )
    args = parser.parse_args()

    interval_minutes = max(1, args.interval_minutes)
    print(f"Running update loop every {interval_minutes} minute(s)")

    while True:
        try:
            fetch_and_process_data()
        except Exception as e:
            traceback.print_exc()
            print(str(e))
        try:
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("Received KeyboardInterrupt, exiting loop")
            break


if __name__ == "__main__":
    main()
