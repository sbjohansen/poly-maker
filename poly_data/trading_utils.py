import math 
from poly_data.data_utils import update_positions
import poly_data.global_state as global_state


def calculate_q_score(spread_from_mid, max_spread, size):
    """
    Calculate the Q-score for an order based on Polymarket's reward formula.
    
    Formula: S(v,s) = ((v - s) / v)^2 * size
    
    Args:
        spread_from_mid: Distance from midpoint in absolute terms (e.g., 0.02 = 2 cents)
        max_spread: Maximum qualifying spread for this market (e.g., 0.03 = 3 cents)
        size: Order size in shares
    
    Returns:
        float: Q-score for this order (higher = more rewards)
    """
    if max_spread <= 0 or spread_from_mid >= max_spread:
        return 0
    
    # Quadratic scoring: ((v - s) / v)^2
    score_multiplier = ((max_spread - spread_from_mid) / max_spread) ** 2
    return score_multiplier * size


def find_optimal_price_for_rewards(midpoint, max_spread, tick_size, is_bid=True, min_offset=0.001):
    """
    Find the optimal price that maximizes reward score while maintaining a valid order.
    
    For maximum rewards, we want to be as close to midpoint as possible
    (due to quadratic scoring), but we can't cross it.
    
    Args:
        midpoint: Current market midpoint
        max_spread: Maximum spread from midpoint to qualify for rewards
        tick_size: Minimum price increment
        is_bid: True for bid (buy) orders, False for ask (sell) orders
        min_offset: Minimum distance from midpoint (to avoid crossing)
    
    Returns:
        float: Optimal price for rewards
    """
    if is_bid:
        # For bids, we want to be just below midpoint (as close as possible)
        # At least 1 tick away to be safe
        optimal = midpoint - max(tick_size, min_offset)
    else:
        # For asks, we want to be just above midpoint
        optimal = midpoint + max(tick_size, min_offset)
    
    return round(optimal, 4)


def estimate_reward_per_100(price, midpoint, max_spread, daily_rate, total_market_q=1000):
    """
    Estimate the reward per $100 of orders based on the Q-score formula.
    
    This helps us understand how much we might earn at a given price level.
    
    Args:
        price: Order price
        midpoint: Market midpoint
        max_spread: Max spread for rewards (in decimal, e.g., 0.03)
        daily_rate: Total daily rewards for this market
        total_market_q: Estimated total Q-score of all market makers (for normalization)
    
    Returns:
        float: Estimated daily reward for $100 of orders at this price
    """
    spread_from_mid = abs(price - midpoint)
    
    if spread_from_mid >= max_spread:
        return 0
    
    # For $100 at price P, we get 100/P shares
    shares_per_100 = 100 / price if price > 0 else 0
    
    # Calculate our Q-score
    our_q = calculate_q_score(spread_from_mid, max_spread, shares_per_100)
    
    # Estimate our share of rewards (simplified)
    # Real calculation would need total market Q which we don't have
    reward_share = our_q / (our_q + total_market_q)
    
    return reward_share * daily_rate / 2  # /2 because rewards split bid/ask


# def get_avgPrice(position, assetId):
#     curr_global = global_state.all_positions[global_state.all_positions['asset'] == str(assetId)]
#     api_position_size = 0
#     api_avgPrice = 0

#     if len(curr_global) > 0:
#         c_row = curr_global.iloc[0]
#         api_avgPrice = round(c_row['avgPrice'], 2)
#         api_position_size = c_row['size']

#     if position > 0:
#         if abs((api_position_size - position)/position * 100) > 5:
#             print("Updating global positions")
#             update_positions()

#             try:
#                 c_row = curr_global.iloc[0]
#                 api_avgPrice = round(c_row['avgPrice'], 2)
#                 api_position_size = c_row['size']
#             except:
#                 return 0
#     return api_avgPrice

def get_best_bid_ask_deets(market, name, size, deviation_threshold=0.05):

    best_bid, best_bid_size, second_best_bid, second_best_bid_size, top_bid = find_best_price_with_size(global_state.all_data[market]['bids'], size, reverse=True)
    best_ask, best_ask_size, second_best_ask, second_best_ask_size, top_ask = find_best_price_with_size(global_state.all_data[market]['asks'], size, reverse=False)
    
    # Handle None values in mid_price calculation
    if best_bid is not None and best_ask is not None:
        mid_price = (best_bid + best_ask) / 2
        bid_sum_within_n_percent = sum(size for price, size in global_state.all_data[market]['bids'].items() if best_bid <= price <= mid_price * (1 + deviation_threshold))
        ask_sum_within_n_percent = sum(size for price, size in global_state.all_data[market]['asks'].items() if mid_price * (1 - deviation_threshold) <= price <= best_ask)
    else:
        mid_price = None
        bid_sum_within_n_percent = 0
        ask_sum_within_n_percent = 0

    if name == 'token2':
        # Handle None values before arithmetic operations
        if all(x is not None for x in [best_bid, best_ask, second_best_bid, second_best_ask, top_bid, top_ask]):
            best_bid, second_best_bid, top_bid, best_ask, second_best_ask, top_ask = 1 - best_ask, 1 - second_best_ask, 1 - top_ask, 1 - best_bid, 1 - second_best_bid, 1 - top_bid
            best_bid_size, second_best_bid_size, best_ask_size, second_best_ask_size = best_ask_size, second_best_ask_size, best_bid_size, second_best_bid_size
            bid_sum_within_n_percent, ask_sum_within_n_percent = ask_sum_within_n_percent, bid_sum_within_n_percent
        else:
            # Handle case where some prices are None - use available values or defaults
            if best_bid is not None and best_ask is not None:
                best_bid, best_ask = 1 - best_ask, 1 - best_bid
                best_bid_size, best_ask_size = best_ask_size, best_bid_size
            if second_best_bid is not None:
                second_best_bid = 1 - second_best_bid
            if second_best_ask is not None:
                second_best_ask = 1 - second_best_ask
            if top_bid is not None:
                top_bid = 1 - top_bid
            if top_ask is not None:
                top_ask = 1 - top_ask
            bid_sum_within_n_percent, ask_sum_within_n_percent = ask_sum_within_n_percent, bid_sum_within_n_percent



    #return as dictionary
    return {
        'best_bid': best_bid,
        'best_bid_size': best_bid_size,
        'second_best_bid': second_best_bid,
        'second_best_bid_size': second_best_bid_size,
        'top_bid': top_bid,
        'best_ask': best_ask,
        'best_ask_size': best_ask_size,
        'second_best_ask': second_best_ask,
        'second_best_ask_size': second_best_ask_size,
        'top_ask': top_ask,
        'bid_sum_within_n_percent': bid_sum_within_n_percent,
        'ask_sum_within_n_percent': ask_sum_within_n_percent
    }


def find_best_price_with_size(price_dict, min_size, reverse=False):
    lst = list(price_dict.items())

    if reverse:
        lst.reverse()
    
    best_price, best_size = None, None
    second_best_price, second_best_size = None, None
    top_price = None
    set_best = False

    for price, size in lst:
        if top_price is None:
            top_price = price

        if set_best:
            second_best_price, second_best_size = price, size
            break

        if size > min_size:
            if best_price is None:
                best_price, best_size = price, size
                set_best = True

    return best_price, best_size, second_best_price, second_best_size, top_price

def get_order_prices(best_bid, best_bid_size, top_bid, best_ask, best_ask_size, top_ask, avgPrice, row):
    """
    Calculate bid and ask prices for market making.
    
    Simple strategy that was working well:
    1. Improve best bid/ask by 1 tick to get priority
    2. If best price has low size, just match it
    3. Safety checks to avoid crossing spread
    """
    tick = row['tick_size']
    min_size = row['min_size']
    
    # Default: improve best bid/ask by 1 tick
    bid_price = best_bid + tick
    ask_price = best_ask - tick

    # If the best bid has low size, just match it (don't overpay)
    if best_bid_size < min_size * 1.5:
        bid_price = best_bid

    # If the best ask has low size, just match it
    if best_ask_size < min_size * 1.5:
        ask_price = best_ask

    # === SAFETY: NEVER CROSS THE SPREAD ===
    if bid_price >= top_ask:
        bid_price = top_bid

    if ask_price <= top_bid:
        ask_price = top_ask

    if bid_price >= ask_price:
        bid_price = top_bid
        ask_price = top_ask

    # === LOG Q-SCORE FOR VISIBILITY (but don't change pricing) ===
    midpoint = (best_bid + best_ask) / 2
    max_spread_cents = float(row.get('max_spread', 3))
    max_spread = max_spread_cents / 100
    
    final_bid_spread = abs(midpoint - bid_price)
    final_ask_spread = abs(ask_price - midpoint)
    
    bid_q_score = calculate_q_score(final_bid_spread, max_spread, 100)
    ask_q_score = calculate_q_score(final_ask_spread, max_spread, 100)
    
    if bid_q_score > 0 or ask_q_score > 0:
        print(f"  Reward Q-scores: bid={bid_q_score:.2f} (spread={final_bid_spread:.3f}), "
              f"ask={ask_q_score:.2f} (spread={final_ask_spread:.3f}), max_spread={max_spread:.3f}")

    return bid_price, ask_price




def round_down(number, decimals):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_up(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def get_buy_sell_amount(position, bid_price, row, other_token_position=0):
    """
    Calculate buy and sell amounts based on current position.
    
    Original logic from repo - simple and proven to work:
    - Buy when below max_size
    - Sell when we have position >= trade_size
    """
    buy_amount = 0
    sell_amount = 0

    max_size = row.get('max_size', row['trade_size'])
    trade_size = row['trade_size']
    min_size = row['min_size']
    
    if position < max_size:
        # Below max - calculate buy amount
        remaining_to_max = max_size - position
        buy_amount = min(trade_size, remaining_to_max)
        
        # Only sell if we have enough position
        if position >= trade_size:
            sell_amount = trade_size
        else:
            sell_amount = 0
    else:
        # At or above max position - focus on selling, no buying
        sell_amount = min(position, trade_size)
        buy_amount = 0  # Don't buy when at/over max_size

    # Ensure minimum size compliance
    if buy_amount > 0 and buy_amount < min_size:
        buy_amount = min_size
    
    if sell_amount > 0 and sell_amount < min_size:
        sell_amount = 0  # Don't place sell orders below min_size

    # Apply multiplier for low-priced assets
    if bid_price < 0.1 and buy_amount > 0:
        multiplier = None
        try:
            multiplier = row.get('multiplier', '')
        except Exception:
            pass

        if multiplier not in ['', None]:
            try:
                mult_int = int(multiplier)
                print(f"Multiplying buy amount by {mult_int}")
                buy_amount = buy_amount * mult_int
            except Exception:
                pass

    return buy_amount, sell_amount
