import time
import os
import argparse
import json
from pathlib import Path
import pandas as pd
from data_updater.trading_utils import get_clob_client
from data_updater.google_utils import get_spreadsheet
from data_updater.find_markets import get_sel_df, get_all_markets, get_all_results, get_markets, add_volatility_to_df
from gspread_dataframe import set_with_dataframe
import traceback

# Auto-manage configuration (defaults can be overridden via env)
AUTO_MANAGE_SELECTED = os.getenv("AUTO_MANAGE_SELECTED", "1") == "1"
AUTO_MAX_MARKETS = int(os.getenv("AUTO_MAX_MARKETS", "10"))
AUTO_DEFAULT_TRADE_SIZE = float(os.getenv("AUTO_DEFAULT_TRADE_SIZE", "10"))
AUTO_DEFAULT_MAX_SIZE = float(os.getenv("AUTO_DEFAULT_MAX_SIZE", "50"))
AUTO_PARAM_TYPE = os.getenv("AUTO_PARAM_TYPE", "mid")
AUTO_STALE_HOURS = float(os.getenv("AUTO_STALE_HOURS", "6"))
AUTO_MIN_GM_REWARD = float(os.getenv("AUTO_MIN_GM_REWARD", "0"))
AUTO_ACTIVITY_FILE = Path(os.getenv("AUTO_ACTIVITY_FILE", "data/market_activity.json"))
# Scoring weights for candidate ranking
AUTO_VOL_WEIGHT = float(os.getenv("AUTO_VOL_WEIGHT", "0.5"))  # penalty per unit of volatility_sum
AUTO_SPREAD_WEIGHT = float(os.getenv("AUTO_SPREAD_WEIGHT", "1.0"))  # penalty per unit of spread
# Allow min_size up to this multiple of trade_size when auto-selecting
AUTO_MIN_SIZE_MULT = float(os.getenv("AUTO_MIN_SIZE_MULT", "2.0"))
# Always rebuild Selected Markets instead of preserving existing rows
AUTO_RESET_SELECTED = os.getenv("AUTO_RESET_SELECTED", "0") == "1"

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
    padded_data = pd.DataFrame('', index=range(max_rows), columns=range(max_cols))

    # Update the padded DataFrame with the original data and its columns
    padded_data.iloc[:num_rows, :num_cols] = data.values
    padded_data.columns = list(data.columns) + [''] * (max_cols - num_cols)

    # Update the sheet with the padded DataFrame, including column headers
    set_with_dataframe(worksheet, padded_data, include_index=False, include_column_header=True, resize=True)

def sort_df(df):
    # Calculate the mean and standard deviation for each column
    mean_gm = df['gm_reward_per_100'].mean()
    std_gm = df['gm_reward_per_100'].std()
    
    mean_volatility = df['volatility_sum'].mean()
    std_volatility = df['volatility_sum'].std()
    
    # Standardize the columns
    df['std_gm_reward_per_100'] = (df['gm_reward_per_100'] - mean_gm) / std_gm
    df['std_volatility_sum'] = (df['volatility_sum'] - mean_volatility) / std_volatility
    
    # Define a custom scoring function for best_bid and best_ask
    def proximity_score(value):
        if 0.1 <= value <= 0.25:
            return (0.25 - value) / 0.15
        elif 0.75 <= value <= 0.9:
            return (value - 0.75) / 0.15
        else:
            return 0
    
    df['bid_score'] = df['best_bid'].apply(proximity_score)
    df['ask_score'] = df['best_ask'].apply(proximity_score)
    
    # Create a composite score (higher is better for rewards, lower is better for volatility, with proximity scores)
    df['composite_score'] = (
        df['std_gm_reward_per_100'] - 
        df['std_volatility_sum'] + 
        df['bid_score'] + 
        df['ask_score']
    )
    
    # Sort by the composite score in descending order
    sorted_df = df.sort_values(by='composite_score', ascending=False)
    
    # Drop the intermediate columns used for calculation
    sorted_df = sorted_df.drop(columns=['std_gm_reward_per_100', 'std_volatility_sum', 'bid_score', 'ask_score', 'composite_score'])
    
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


def _cancel_orders_for_market(client, condition_id):
    try:
        client.cancel_market_orders(market=str(condition_id))
        print(f"Cancelled orders for market {condition_id}")
    except Exception as ex:
        print(f"Warning: could not cancel orders for {condition_id}: {ex}")


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


def auto_manage_selected_markets(new_df, worksheet, client):
    """
    Keep Selected Markets capped, add top candidates, remove stale ones (after cancelling orders).
    """
    try:
        current_sel = pd.DataFrame(worksheet.get_all_records())
    except Exception as ex:
        print(f"Warning: could not read Selected Markets: {ex}")
        return

    # Determine columns to preserve
    sel_columns = list(current_sel.columns) if len(current_sel.columns) > 0 else [
        "question", "answer1", "answer2", "market_slug", "condition_id",
        "token1", "token2", "neg_risk", "param_type", "trade_size",
        "max_size", "min_size", "multiplier"
    ]
    if current_sel.empty:
        current_sel = pd.DataFrame(columns=sel_columns)

    if AUTO_RESET_SELECTED:
        print("AUTO_RESET_SELECTED enabled; clearing Selected Markets and rebuilding.")
        if client is not None and "condition_id" in current_sel.columns:
            for cid in current_sel["condition_id"].astype(str):
                _cancel_orders_for_market(client, cid)
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
    stats_cols = ["condition_id", "gm_reward_per_100", "best_bid", "best_ask"]
    merged = current_sel.merge(new_df[stats_cols], on="condition_id", how="left", suffixes=("", "_stats"))

    stale_ids = []
    for _, row in merged.iterrows():
        cid = str(row.get("condition_id", ""))
        if not cid:
            continue
        last_seen = float(activity_map.get(cid, now))
        time_stale = (now - last_seen) > (AUTO_STALE_HOURS * 3600)

        reward = row.get("gm_reward_per_100", 0)
        try:
            reward_val = float(reward) if reward == reward else 0  # handles NaN
        except Exception:
            reward_val = 0

        no_liquidity = (float(row.get("best_bid", 0)) == 0) and (float(row.get("best_ask", 0)) == 0)
        reward_low = reward_val < AUTO_MIN_GM_REWARD

        if time_stale or no_liquidity or reward_low:
            stale_ids.append(cid)

    if stale_ids:
        print(f"Auto-manage: marking {len(stale_ids)} market(s) as stale: {stale_ids}")
        if client is not None:
            for cid in stale_ids:
                _cancel_orders_for_market(client, cid)
        # Remove stale rows
        current_sel = current_sel[~current_sel["condition_id"].astype(str).isin(stale_ids)]

    # Enforce max markets by trimming lowest reward if still above cap
    current_sel = current_sel.reset_index(drop=True)
    if len(current_sel) > AUTO_MAX_MARKETS:
        trimmed = current_sel.merge(new_df[stats_cols], on="condition_id", how="left")
        trimmed = trimmed.sort_values("gm_reward_per_100", ascending=False)
        keep = trimmed.head(AUTO_MAX_MARKETS)["condition_id"].astype(str)
        drop_ids = set(trimmed["condition_id"].astype(str)) - set(keep)
        if drop_ids:
            print(f"Auto-manage: trimming {len(drop_ids)} market(s) to enforce cap: {drop_ids}")
            if client is not None:
                for cid in drop_ids:
                    _cancel_orders_for_market(client, cid)
            current_sel = current_sel[current_sel["condition_id"].astype(str).isin(keep)]

    # Compute how many slots are open
    slots = max(0, AUTO_MAX_MARKETS - len(current_sel))

    # Build candidate pool not already selected
    selected_ids = set(current_sel["condition_id"].astype(str))
    candidates = new_df[~new_df["condition_id"].astype(str).isin(selected_ids)].copy()

    # Filter candidates by reward/liquidity
    candidates = candidates[
        (candidates["gm_reward_per_100"] >= AUTO_MIN_GM_REWARD) &
        (candidates["best_bid"] > 0) &
        (candidates["best_ask"] > 0)
    ]
    # Skip markets where the minimum size is far above our trade size (configurable multiple)
    if "min_size" in candidates.columns:
        min_size_numeric = pd.to_numeric(candidates["min_size"], errors="coerce")
        candidates = candidates[min_size_numeric <= AUTO_DEFAULT_TRADE_SIZE * AUTO_MIN_SIZE_MULT]

    # Score candidates: reward minus penalties for volatility and spread
    def compute_score(row):
        reward = float(row.get("gm_reward_per_100", 0) or 0)
        vol = float(row.get("volatility_sum", 0) or 0)
        spread = float(row.get("spread", 0) or 0)
        return reward - AUTO_VOL_WEIGHT * vol - AUTO_SPREAD_WEIGHT * spread

    candidates["score"] = candidates.apply(compute_score, axis=1)
    candidates = candidates.sort_values("score", ascending=False)

    additions = []
    if slots > 0 and len(candidates) > 0:
        for _, row in candidates.head(slots).iterrows():
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
    id_cols = ['condition_id', 'token1', 'token2']
    for col in id_cols:
        if col in current_sel.columns:
            current_sel[col] = current_sel[col].astype(str)

    # Normalize sizing columns using defaults
    if "trade_size" in current_sel.columns:
        current_sel["trade_size"] = AUTO_DEFAULT_TRADE_SIZE
    else:
        current_sel["trade_size"] = AUTO_DEFAULT_TRADE_SIZE
    if "max_size" in current_sel.columns:
        current_sel["max_size"] = AUTO_DEFAULT_MAX_SIZE
    else:
        current_sel["max_size"] = AUTO_DEFAULT_MAX_SIZE
    if "min_size" in current_sel.columns and "condition_id" in current_sel.columns:
        min_map = pd.to_numeric(new_df.set_index("condition_id")["min_size"], errors="coerce")
        current_sel["min_size"] = current_sel["condition_id"].map(min_map).fillna(AUTO_DEFAULT_TRADE_SIZE)
    elif "condition_id" in current_sel.columns:
        min_map = pd.to_numeric(new_df.set_index("condition_id")["min_size"], errors="coerce")
        current_sel["min_size"] = current_sel["condition_id"].map(min_map).fillna(AUTO_DEFAULT_TRADE_SIZE)

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


    all_df = get_all_markets(client)
    print("Got all Markets")
    all_results = get_all_results(all_df, client)
    print("Got all Results")
    m_data, all_markets = get_markets(all_results, sel_df, maker_reward=0.75)
    print("Got all orderbook")

    print(f'{pd.to_datetime("now")}: Fetched all markets data of length {len(all_markets)}.')
    new_df = add_volatility_to_df(all_markets)
    new_df['volatility_sum'] =  new_df['24_hour'] + new_df['7_day'] + new_df['14_day']
    
    new_df = new_df.sort_values('volatility_sum', ascending=True)
    new_df['volatilty/reward'] = ((new_df['gm_reward_per_100'] / new_df['volatility_sum']).round(2)).astype(str)

    new_df = new_df[['question', 'answer1', 'answer2', 'spread', 'rewards_daily_rate', 'gm_reward_per_100', 'sm_reward_per_100', 'bid_reward_per_100', 'ask_reward_per_100',  'volatility_sum', 'volatilty/reward', 'min_size', '1_hour', '3_hour', '6_hour', '12_hour', '24_hour', '7_day', '30_day',  
                     'best_bid', 'best_ask', 'volatility_price', 'max_spread', 'tick_size',  
                     'neg_risk',  'market_slug', 'token1', 'token2', 'condition_id']]

    
    volatility_df = new_df.copy()
    volatility_df = volatility_df[new_df['volatility_sum'] < 20]
    # volatility_df = sort_df(volatility_df)
    volatility_df = volatility_df.sort_values('gm_reward_per_100', ascending=False)
   
    new_df = new_df.sort_values('gm_reward_per_100', ascending=False)
    

    print(f'{pd.to_datetime("now")}: Fetched select market of length {len(new_df)}.')

    # Ensure ID columns remain strings (avoid scientific notation in Sheets)
    id_cols = ['token1', 'token2', 'condition_id']
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
            auto_manage_selected_markets(new_df, wk_selected, client)
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
