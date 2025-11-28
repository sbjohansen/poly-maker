import gc  # Garbage collection
import os  # Operating system interface
import json  # JSON handling
import asyncio  # Asynchronous I/O
import traceback  # Exception handling
import pandas as pd  # Data analysis library
import math  # Mathematical functions

import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS

# Import utility functions for trading
from poly_data.trading_utils import (
    get_best_bid_ask_deets,
    get_order_prices,
    get_buy_sell_amount,
    round_down,
    round_up,
)
from poly_data.data_utils import get_position, get_order, set_position

# Create directory for storing position risk information
if not os.path.exists("positions/"):
    os.makedirs("positions/")


def get_dynamic_params(base_params, row):
    """
    Calculate dynamic trading parameters based on market characteristics.

    This adjusts the base hyperparameters (from sheet) based on:
    - Current volatility levels
    - Spread conditions
    - Price position (near edges vs near 0.5)
    - Reward levels

    Args:
        base_params: Base parameters from Hyperparameters sheet
        row: Market row with current data (3_hour, spread, best_bid, best_ask, gm_reward_per_100)

    Returns:
        dict: Adjusted parameters for this specific market
    """
    params = base_params.copy()

    try:
        vol_3h = float(row.get("3_hour", 0) or 0)
        spread = float(row.get("spread", 0) or 0)
        best_bid = float(row.get("best_bid", 0.5) or 0.5)
        best_ask = float(row.get("best_ask", 0.5) or 0.5)
        reward = float(row.get("gm_reward_per_100", 0) or 0)
    except (TypeError, ValueError):
        return params  # Return base params if data is invalid

    mid_price = (best_bid + best_ask) / 2

    # === VOLATILITY-BASED ADJUSTMENTS ===
    # Tighten stop-loss when volatility is higher
    if vol_3h > 30:
        # High volatility: tighter stop-loss, quicker exit
        vol_multiplier = min(vol_3h / 30, 2.0)  # Cap at 2x
        params["stop_loss_threshold"] = max(
            base_params.get("stop_loss_threshold", -3) * (1 / vol_multiplier),
            -1.5,  # Never tighter than -1.5%
        )
        params["take_profit_threshold"] = min(
            base_params.get("take_profit_threshold", 3) * vol_multiplier,
            6.0,  # Cap at 6% take profit
        )
    elif vol_3h < 10:
        # Low volatility: can be more patient
        params["stop_loss_threshold"] = min(
            base_params.get("stop_loss_threshold", -3) * 1.5, -1.0  # Don't go looser than -4.5%
        )
        params["take_profit_threshold"] = max(
            base_params.get("take_profit_threshold", 3) * 0.7, 1.5  # At least 1.5% take profit
        )

    # === PRICE POSITION ADJUSTMENTS ===
    # Near edges (0.1-0.25 or 0.75-0.9) we can be more aggressive
    # Near 0.5 we should be more conservative
    if 0.1 <= mid_price <= 0.25 or 0.75 <= mid_price <= 0.9:
        # Favorable price position - can hold longer
        params["stop_loss_threshold"] = params.get("stop_loss_threshold", -3) * 1.3  # Looser stop
        params["volatility_threshold"] = (
            base_params.get("volatility_threshold", 50) * 1.2
        )  # Allow more vol
    elif 0.4 <= mid_price <= 0.6:
        # Risky price position (near 50/50) - be conservative
        params["stop_loss_threshold"] = max(
            params.get("stop_loss_threshold", -3) * 0.7, -2.0  # Tighter stop
        )
        params["volatility_threshold"] = (
            base_params.get("volatility_threshold", 50) * 0.7
        )  # Less vol tolerance

    # === SPREAD-BASED ADJUSTMENTS ===
    # Wide spreads mean harder exits - need tighter risk management
    if spread > 0.04:
        params["spread_threshold"] = min(
            spread * 1.5, 0.08
        )  # Allow wider spread for stop-loss execution
        params["stop_loss_threshold"] = max(
            params.get("stop_loss_threshold", -3) * 0.8, -2.0  # Tighter when spread is wide
        )

    # === REWARD-BASED ADJUSTMENTS ===
    # Higher reward markets are worth more patience
    if reward > 3.0:
        params["stop_loss_threshold"] = (
            params.get("stop_loss_threshold", -3) * 1.2
        )  # Slightly looser
        params["sleep_period"] = max(base_params.get("sleep_period", 2) - 1, 1)  # Faster re-entry
    elif reward < 1.0:
        params["stop_loss_threshold"] = max(
            params.get("stop_loss_threshold", -3) * 0.8, -1.5  # Tighter for low reward
        )
        params["sleep_period"] = base_params.get("sleep_period", 2) + 1  # Slower re-entry

    # Ensure all params are floats
    for key in [
        "stop_loss_threshold",
        "take_profit_threshold",
        "spread_threshold",
        "volatility_threshold",
        "sleep_period",
    ]:
        if key in params:
            try:
                params[key] = float(params[key])
            except (TypeError, ValueError):
                pass

    return params


def send_buy_order(order):
    """
    Create a BUY order for a specific token.

    This function:
    1. Cancels any existing orders for the token
    2. Checks if the order price is within acceptable range
    3. Creates a new buy order if conditions are met

    Args:
        order (dict): Order details including token, price, size, and market parameters
    """
    client = global_state.client

    # Only cancel existing orders if we need to make significant changes
    existing_buy_size = order["orders"]["buy"]["size"]
    existing_buy_price = order["orders"]["buy"]["price"]

    # Cancel orders if price changed significantly or size needs major adjustment
    price_diff = (
        abs(existing_buy_price - order["price"]) if existing_buy_price > 0 else float("inf")
    )
    size_diff = abs(existing_buy_size - order["size"]) if existing_buy_size > 0 else float("inf")

    should_cancel = (
        price_diff > 0.005  # Cancel if price diff > 0.5 cents
        or size_diff > order["size"] * 0.1  # Cancel if size diff > 10%
        or existing_buy_size == 0  # Cancel if no existing buy order
    )

    if should_cancel and (existing_buy_size > 0 or order["orders"]["sell"]["size"] > 0):
        print(f"Cancelling buy orders - price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        client.cancel_all_asset(order["token"])
    elif not should_cancel:
        print(f"Keeping existing buy orders - minor changes: price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        return  # Don't place new order if existing one is fine

    # Calculate minimum acceptable price based on market spread
    incentive_start = order["mid_price"] - order["max_spread"] / 100

    trade = True

    # Don't place orders that are below incentive threshold
    if order["price"] < incentive_start:
        trade = False

    if trade:
        # Only place orders with prices between 0.1 and 0.9 to avoid extreme positions
        if order["price"] >= 0.1 and order["price"] < 0.9:
            print(f'Creating new order for {order["size"]} at {order["price"]}')
            print(order["token"], "BUY", order["price"], order["size"])
            try:
                resp = client.create_order(
                    order["token"],
                    "BUY",
                    order["price"],
                    order["size"],
                    True if order["neg_risk"] == "TRUE" else False,
                )
                print(f"create_order BUY response: {resp}")
            except Exception as ex:
                print(f"create_order BUY failed: {ex}")
        else:
            print("Not creating buy order because its outside acceptable price range (0.1-0.9)")
    else:
        print(
            f'Not creating new order because order price of {order["price"]} is less than incentive start price of {incentive_start}. Mid price is {order["mid_price"]}'
        )


def send_sell_order(order):
    """
    Create a SELL order for a specific token.

    This function:
    1. Cancels any existing orders for the token
    2. Creates a new sell order with the specified parameters

    Args:
        order (dict): Order details including token, price, size, and market parameters
    """
    client = global_state.client

    # Only cancel existing orders if we need to make significant changes
    existing_sell_size = order["orders"]["sell"]["size"]
    existing_sell_price = order["orders"]["sell"]["price"]

    # Cancel orders if price changed significantly or size needs major adjustment
    price_diff = (
        abs(existing_sell_price - order["price"]) if existing_sell_price > 0 else float("inf")
    )
    size_diff = abs(existing_sell_size - order["size"]) if existing_sell_size > 0 else float("inf")

    should_cancel = (
        price_diff > 0.005  # Cancel if price diff > 0.5 cents
        or size_diff > order["size"] * 0.1  # Cancel if size diff > 10%
        or existing_sell_size == 0  # Cancel if no existing sell order
    )

    if should_cancel and (existing_sell_size > 0 or order["orders"]["buy"]["size"] > 0):
        print(f"Cancelling sell orders - price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        client.cancel_all_asset(order["token"])
    elif not should_cancel:
        print(f"Keeping existing sell orders - minor changes: price diff: {price_diff:.4f}, size diff: {size_diff:.1f}")
        return  # Don't place new order if existing one is fine

    print(f'Creating new order for {order["size"]} at {order["price"]}')
    try:
        resp = client.create_order(
            order["token"],
            "SELL",
            order["price"],
            order["size"],
            True if order["neg_risk"] == "TRUE" else False,
        )
        print(f"create_order SELL response: {resp}")
    except Exception as ex:
        print(f"create_order SELL failed: {ex}")


# Dictionary to store locks for each market to prevent concurrent trading on the same market
market_locks = {}


async def perform_trade(market):
    """
    Main trading function that handles market making for a specific market.

    This function:
    1. Merges positions when possible to free up capital
    2. Analyzes the market to determine optimal bid/ask prices
    3. Manages buy and sell orders based on position size and market conditions
    4. Implements risk management with stop-loss and take-profit logic

    Args:
        market (str): The market ID to trade on
    """
    # Create a lock for this market if it doesn't exist
    if market not in market_locks:
        market_locks[market] = asyncio.Lock()

    # Use lock to prevent concurrent trading on the same market
    async with market_locks[market]:
        try:
            client = global_state.client
            # Get market details from the configuration
            rows = global_state.df[global_state.df["condition_id"] == market]
            if rows.empty:
                print(f"Perform_trade: no config row found for market {market}, skipping.")
                return
            row = rows.iloc[0]
            # Determine decimal precision from tick size (default to 2 if missing)
            tick_size = row.get("tick_size") or row.get("tick_size", 0.01)
            if tick_size is None or tick_size == 0:
                tick_size = 0.01  # Default tick size
            try:
                round_length = len(str(tick_size).split(".")[1])
            except (IndexError, AttributeError):
                round_length = 2  # Default to 2 decimal places

            # Get base trading parameters for this market type
            base_params = global_state.params[row["param_type"]]

            # Calculate dynamic parameters based on market conditions
            params = get_dynamic_params(base_params, row)

            # Log parameter adjustments for visibility
            if params != base_params:
                print(
                    f"Dynamic params adjusted: stop_loss={params.get('stop_loss_threshold', 'N/A'):.2f}%, "
                    f"take_profit={params.get('take_profit_threshold', 'N/A'):.2f}%, "
                    f"vol_threshold={params.get('volatility_threshold', 'N/A'):.1f}"
                )

            # Create a list with both outcomes for the market
            deets = [
                {"name": "token1", "token": row["token1"], "answer": row["answer1"]},
                {"name": "token2", "token": row["token2"], "answer": row["answer2"]},
            ]
            print(f"\n\n{pd.Timestamp.utcnow().tz_localize(None)}: {row['question']}")

            # Get current positions for both outcomes
            pos_1 = get_position(row["token1"])["size"]
            pos_2 = get_position(row["token2"])["size"]

            # ------- POSITION MERGING LOGIC -------
            # Calculate if we have opposing positions that can be merged
            amount_to_merge = min(pos_1, pos_2)

            # Only merge if positions are above minimum threshold
            if float(amount_to_merge) > CONSTANTS.MIN_MERGE_SIZE:
                # Get exact position sizes from blockchain for merging
                pos_1 = client.get_position(row["token1"])[0]
                pos_2 = client.get_position(row["token2"])[0]
                amount_to_merge = min(pos_1, pos_2)
                scaled_amt = amount_to_merge / 10**6

                if scaled_amt > CONSTANTS.MIN_MERGE_SIZE:
                    print(
                        f"Position 1 is of size {pos_1} and Position 2 is of size {pos_2}. Merging positions"
                    )
                    # Execute the merge operation
                    client.merge_positions(amount_to_merge, market, row["neg_risk"] == "TRUE")
                    # Update our local position tracking
                    set_position(row["token1"], "SELL", scaled_amt, 0, "merge")
                    set_position(row["token2"], "SELL", scaled_amt, 0, "merge")

            # ------- TRADING LOGIC FOR EACH OUTCOME -------
            # Loop through both outcomes in the market (YES and NO)
            for detail in deets:
                token = int(detail["token"])

                # Get current orders for this token
                orders = get_order(token)

                # Get market depth and price information
                deets = get_best_bid_ask_deets(market, detail["name"], 100, 0.1)

                # if deet has None for one these values below, call it with min size of 20
                if (
                    deets["best_bid"] is None
                    or deets["best_ask"] is None
                    or deets["best_bid_size"] is None
                    or deets["best_ask_size"] is None
                ):
                    deets = get_best_bid_ask_deets(market, detail["name"], 20, 0.1)

                # Extract all order book details
                best_bid = deets["best_bid"]
                best_bid_size = deets["best_bid_size"]
                second_best_bid = deets["second_best_bid"]
                second_best_bid_size = deets["second_best_bid_size"]
                top_bid = deets["top_bid"]
                best_ask = deets["best_ask"]
                best_ask_size = deets["best_ask_size"]
                second_best_ask = deets["second_best_ask"]
                second_best_ask_size = deets["second_best_ask_size"]
                top_ask = deets["top_ask"]

                # Skip this token if no liquidity (bid or ask is None)
                if best_bid is None or best_ask is None or top_bid is None or top_ask is None:
                    print(
                        f"Skipping {detail['answer']} - no liquidity (bid={best_bid}, ask={best_ask})"
                    )
                    continue

                # Round prices to appropriate precision
                best_bid = round(best_bid, round_length)
                best_ask = round(best_ask, round_length)

                # Calculate ratio of buy vs sell liquidity in the market
                try:
                    overall_ratio = (deets["bid_sum_within_n_percent"]) / (
                        deets["ask_sum_within_n_percent"]
                    )
                except:
                    overall_ratio = 0

                try:
                    second_best_bid = round(second_best_bid, round_length)
                    second_best_ask = round(second_best_ask, round_length)
                except:
                    pass

                top_bid = round(top_bid, round_length)
                top_ask = round(top_ask, round_length)

                # Get our current position and average price
                pos = get_position(token)
                position = pos["size"]
                avgPrice = pos["avgPrice"]

                position = round_down(position, 2)

                # Calculate optimal bid and ask prices based on market conditions
                bid_price, ask_price = get_order_prices(
                    best_bid,
                    best_bid_size,
                    top_bid,
                    best_ask,
                    best_ask_size,
                    top_ask,
                    avgPrice,
                    row,
                )

                bid_price = round(bid_price, round_length)
                ask_price = round(ask_price, round_length)

                # Calculate mid price for reference
                mid_price = (top_bid + top_ask) / 2

                # Log market conditions for this outcome
                print(
                    f"\nFor {detail['answer']}. Orders: {orders} Position: {position}, "
                    f"avgPrice: {avgPrice}, Best Bid: {best_bid}, Best Ask: {best_ask}, "
                    f"Bid Price: {bid_price}, Ask Price: {ask_price}, Mid Price: {mid_price}"
                )

                # Get position for the opposite token to calculate total exposure
                other_token = global_state.REVERSE_TOKENS[str(token)]
                other_position = get_position(other_token)["size"]

                # Calculate how much to buy or sell based on our position
                buy_amount, sell_amount = get_buy_sell_amount(
                    position, bid_price, row, other_position
                )

                # Get max_size for logging (same logic as in get_buy_sell_amount)
                max_size = row.get("max_size", row["trade_size"])

                # Prepare order object with all necessary information
                order = {
                    "token": token,
                    "mid_price": mid_price,
                    "neg_risk": row["neg_risk"],
                    "max_spread": row["max_spread"],
                    "orders": orders,
                    "token_name": detail["name"],
                    "row": row,
                }

                print(
                    f"Position: {position}, Other Position: {other_position}, "
                    f"Trade Size: {row['trade_size']}, Max Size: {max_size}, "
                    f"buy_amount: {buy_amount}, sell_amount: {sell_amount}"
                )

                # File to store risk management information for this market
                fname = "positions/" + str(market) + ".json"

                # ------- OVER-EXPOSURE RISK REDUCTION -------
                # If position exceeds max_size, prioritize reducing it
                if position > max_size:
                    over_exposure = position - max_size
                    print(
                        f"⚠️ OVER-EXPOSURE DETECTED: Position {position:.0f} exceeds max_size {max_size:.0f} "
                        f"by {over_exposure:.0f} units. Prioritizing position reduction."
                    )
                    
                    # Get fresh market data
                    n_deets = get_best_bid_ask_deets(market, detail["name"], 100, 0.1)
                    current_best_bid = n_deets["best_bid"]
                    current_spread = round(n_deets["best_ask"] - n_deets["best_bid"], 2)
                    
                    # Check if we already have a sell order working
                    existing_sell_size = orders["sell"]["size"]
                    existing_sell_price = orders["sell"]["price"]
                    
                    if existing_sell_size > 0:
                        # Check if existing order is stale (market moved significantly in our favor)
                        # If best_bid is now much higher than our sell price, update to capture gains
                        price_improvement = current_best_bid - existing_sell_price
                        
                        if price_improvement > 0.10:  # Market moved 10+ cents in our favor
                            # Update to better price - sell at current best bid
                            order["size"] = existing_sell_size
                            order["price"] = current_best_bid
                            
                            print(
                                f"📈 UPDATING STALE SELL: Market improved! Old: {existing_sell_price:.2f}, "
                                f"New best bid: {current_best_bid:.2f} (+{price_improvement:.2f}). Updating order."
                            )
                            send_sell_order(order)
                            continue
                        else:
                            # Order is at reasonable price - let it work
                            print(
                                f"✓ Existing sell order active: {existing_sell_size:.0f} @ {existing_sell_price:.2f}. "
                                f"Best bid: {current_best_bid:.2f}. Letting it work."
                            )
                            continue  # Skip - let existing order work
                    
                    # No existing sell order - place one
                    reduction_amount = max(row['trade_size'], min(over_exposure, position))
                    
                    # If spread is reasonable, sell aggressively at best bid
                    if current_spread <= params.get("spread_threshold", 0.15):
                        order["size"] = reduction_amount
                        order["price"] = current_best_bid  # Sell at bid for faster execution
                        
                        print(
                            f"🔻 REDUCING POSITION: Selling {reduction_amount:.0f} at {order['price']:.2f} "
                            f"(spread: {current_spread:.2f})"
                        )
                        send_sell_order(order)
                        continue  # Skip normal logic, focus on reducing
                    else:
                        # Spread too wide, place limit sell at ask price
                        order["size"] = reduction_amount
                        order["price"] = ask_price
                        
                        print(
                            f"🔻 REDUCING POSITION (limit): Selling {reduction_amount:.0f} at {order['price']:.2f} "
                            f"(spread too wide: {current_spread:.2f})"
                        )
                        send_sell_order(order)
                        continue  # Skip normal logic, focus on reducing

                # ------- SELL ORDER LOGIC (STOP-LOSS) -------
                # SAFETY CHECK: Only proceed with sell logic if we actually have a position
                if sell_amount > 0 and position > 0 and avgPrice > 0:
                    order["size"] = sell_amount
                    order["price"] = ask_price

                    # Get fresh market data for risk assessment
                    n_deets = get_best_bid_ask_deets(market, detail["name"], 100, 0.1)

                    # Calculate current market price and spread
                    mid_price = round_up(
                        (n_deets["best_bid"] + n_deets["best_ask"]) / 2, round_length
                    )
                    spread = round(n_deets["best_ask"] - n_deets["best_bid"], 2)

                    # Calculate current profit/loss on position
                    pnl = (mid_price - avgPrice) / avgPrice * 100

                    print(f"Mid Price: {mid_price}, Spread: {spread}, PnL: {pnl}")

                    # Prepare risk details for tracking
                    risk_details = {
                        "time": str(pd.Timestamp.utcnow().tz_localize(None)),
                        "question": row["question"],
                    }

                    try:
                        ratio = (n_deets["bid_sum_within_n_percent"]) / (
                            n_deets["ask_sum_within_n_percent"]
                        )
                    except:
                        ratio = 0

                    pos_to_sell = sell_amount  # Amount to sell in risk-off scenario

                    # ------- STOP-LOSS LOGIC -------
                    # Trigger stop-loss if either:
                    # 1. PnL is below threshold and spread is tight enough to exit
                    # 2. Volatility is too high
                    if (
                        pnl < params["stop_loss_threshold"] and spread <= params["spread_threshold"]
                    ) or row["3_hour"] > params["volatility_threshold"]:
                        risk_details["msg"] = (
                            f"Selling {pos_to_sell} because spread is {spread} and pnl is {pnl} "
                            f"and ratio is {ratio} and 3 hour volatility is {row['3_hour']}"
                        )
                        print("Stop loss Triggered: ", risk_details["msg"])

                        # Sell at market best bid to ensure execution
                        order["size"] = pos_to_sell
                        order["price"] = n_deets["best_bid"]

                        # Set period to avoid trading after stop-loss
                        risk_details["sleep_till"] = str(
                            pd.Timestamp.utcnow().tz_localize(None)
                            + pd.Timedelta(hours=params["sleep_period"])
                        )

                        print("Risking off")
                        send_sell_order(order)
                        client.cancel_all_market(market)

                        # Save risk details to file
                        open(fname, "w").write(json.dumps(risk_details))
                        continue

                # ------- BUY ORDER LOGIC -------
                # Get max_size, defaulting to trade_size if not specified
                max_size = row.get("max_size", row["trade_size"])

                # Only buy if:
                # 1. Position is less than max_size (new logic)
                # 2. Position is less than absolute cap (250)
                # 3. Buy amount is above minimum size
                if (
                    position < max_size
                    and position < 250
                    and buy_amount > 0
                    and buy_amount >= row["min_size"]
                ):
                    # Get reference price from LIVE market data (not stale sheet data)
                    # Use the actual best_bid we just fetched from order book
                    sheet_value = best_bid  # Use live data instead of row["best_bid"]

                    sheet_value = round(sheet_value, round_length)
                    order["size"] = buy_amount
                    order["price"] = bid_price

                    # Check if our bid price is reasonable compared to current best bid
                    # Use percentage-based threshold for wide spreads
                    price_change = abs(order["price"] - sheet_value)
                    spread = best_ask - best_bid
                    
                    # For wide spreads, allow more price deviation
                    # Threshold: max of 10 cents OR 50% of spread
                    price_threshold = max(0.10, spread * 0.5)

                    send_buy = True

                    # ------- RISK-OFF PERIOD CHECK -------
                    # If we're in a risk-off period (after stop-loss), don't buy
                    if os.path.isfile(fname):
                        risk_details = json.load(open(fname))

                        start_trading_at = pd.to_datetime(risk_details["sleep_till"])
                        current_time = pd.Timestamp.utcnow().tz_localize(None)

                        print(risk_details, current_time, start_trading_at)
                        if current_time < start_trading_at:
                            send_buy = False
                            print(
                                f"Not sending a buy order because recently risked off. "
                                f"Risked off at {risk_details['time']}"
                            )

                    # Only proceed if we're not in risk-off period
                    if send_buy:
                        # Don't buy if volatility is too high OR price is unreasonably far from best bid
                        if row["3_hour"] > params["volatility_threshold"]:
                            print(
                                f'3 Hour Volatility of {row["3_hour"]} is greater than max volatility of '
                                f'{params["volatility_threshold"]}. Skipping buy but keeping existing orders.'
                            )
                            # Don't cancel orders on high volatility - just don't place new ones
                        elif price_change > price_threshold:
                            print(
                                f'Price of {order["price"]} is more than {price_threshold:.2f} away from '
                                f'best bid {sheet_value}. Skipping this cycle.'
                            )
                            # Don't cancel - the order book data might be stale
                        else:
                            # MARKET MAKING STRATEGY: Allow positions on BOTH sides
                            # Polymarket rewards are maximized by providing liquidity on both bid AND ask
                            # Auto-merge will recover capital if both sides fill
                            
                            # Check for reverse position (holding opposite outcome)
                            rev_token = global_state.REVERSE_TOKENS[str(token)]
                            rev_pos = get_position(rev_token)
                            current_pos = get_position(str(token))

                            # Calculate total exposure across both sides
                            total_exposure = current_pos["size"] + rev_pos["size"]
                            
                            # REWARD FARMING: Always try to place buy orders for two-sided quoting
                            # get_buy_sell_amount() returns min_size at max position (safe for rewards)
                            # Only hard block is extreme total exposure (3x max_size)
                            max_total_exposure = max_size * 3  # Allow more room for reward farming
                            
                            # Original logic: Skip if we have reverse position to avoid over-exposure
                            if rev_pos["size"] > 0:
                                print(
                                    f"Skipping buy order - have reverse position ({rev_pos['size']:.0f})"
                                )
                                continue

                            if total_exposure >= max_total_exposure:
                                print(
                                    f"Skipping buy order - total exposure too high ({total_exposure:.0f}/{max_total_exposure:.0f})"
                                )
                                continue

                            # Check market buy/sell volume ratio
                            if overall_ratio < 0:
                                send_buy = False
                                print(
                                    f"Not sending a buy order because overall ratio is {overall_ratio}"
                                )
                                client.cancel_all_asset(order["token"])
                            else:
                                # Place buy orders based on market conditions
                                # 1. Better price available
                                if best_bid > orders["buy"]["price"]:
                                    print(
                                        f"Sending Buy Order for {token} - better price. "
                                        f"Current: {orders['buy']['price']:.2f}, Best Bid: {best_bid:.2f}"
                                    )
                                    send_buy_order(order)
                                # 2. No existing order or need more size
                                elif orders["buy"]["size"] == 0:
                                    print(
                                        f"Sending Buy Order for {token} - no existing order"
                                    )
                                    send_buy_order(order)
                                # 3. Order size mismatch (need to resize for rewards)
                                elif abs(orders["buy"]["size"] - order["size"]) > order["size"] * 0.1:
                                    print(
                                        f"Sending Buy Order for {token} - size mismatch: "
                                        f"{orders['buy']['size']:.0f} vs {order['size']:.0f}"
                                    )
                                    send_buy_order(order)
                                else:
                                    # Keep existing order
                                    print(
                                        f"Keeping buy order for {token}: price={orders['buy']['price']:.2f}, "
                                        f"size={orders['buy']['size']:.0f}"
                                    )

                # ------- TAKE PROFIT / SELL ORDER MANAGEMENT -------
                # Original logic: Use elif so we don't try to do both buy and sell in same iteration
                # SAFETY CHECK: Only place sell orders if we have a position AND avgPrice
                elif sell_amount > 0 and position > 0 and avgPrice > 0:
                    order["size"] = sell_amount

                    # Calculate take-profit price based on average cost
                    # This is the MINIMUM price we want to sell at
                    tp_price = round_up(
                        avgPrice + (avgPrice * params["take_profit_threshold"] / 100), round_length
                    )
                    
                    # For sell orders, we want the HIGHER of:
                    # 1. Our take-profit price (minimum acceptable)
                    # 2. Current ask price (market rate)
                    # This ensures we don't sell below our profit target
                    target_sell_price = round_up(
                        max(tp_price, ask_price), round_length
                    )
                    order["price"] = target_sell_price

                    existing_sell_price = float(orders["sell"]["price"])
                    existing_sell_size = float(orders["sell"]["size"])

                    # Calculate if we need to update the sell order
                    # Only update if:
                    # 1. No existing sell order, OR
                    # 2. Price changed significantly (> 2 cents), OR
                    # 3. Existing size is significantly different from target size (not position!)
                    
                    needs_update = False
                    update_reason = ""
                    
                    target_size = sell_amount  # This is what we want to place
                    
                    if existing_sell_size == 0:
                        needs_update = True
                        update_reason = "no existing sell order"
                    elif abs(existing_sell_price - target_sell_price) > 0.02:
                        needs_update = True
                        update_reason = f"price changed: {existing_sell_price:.2f} -> {target_sell_price:.2f}"
                    elif abs(existing_sell_size - target_size) > target_size * 0.1:
                        # Only update if existing size differs from TARGET size by >10%
                        needs_update = True
                        update_reason = f"size mismatch: {existing_sell_size:.0f} vs target {target_size:.0f}"
                    
                    if needs_update:
                        print(
                            f"Sending Sell Order for {token}: {update_reason}. "
                            f"Target: {target_sell_price:.2f}, TP: {tp_price:.2f}, Ask: {ask_price:.2f}"
                        )
                        send_sell_order(order)
                    else:
                        pnl_pct = (existing_sell_price - avgPrice) / avgPrice * 100 if avgPrice > 0 else 0
                        print(
                            f"Keeping sell order for {token}: price={existing_sell_price:.2f}, "
                            f"size={existing_sell_size:.0f}/{position:.0f}, PnL={pnl_pct:.1f}%"
                        )

        except Exception as ex:
            print(f"Error performing trade for {market}: {ex}")
            traceback.print_exc()

        # Clean up memory and introduce a small delay
        gc.collect()
        await asyncio.sleep(2)
