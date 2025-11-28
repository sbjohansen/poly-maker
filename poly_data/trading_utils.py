import math 
from poly_data.data_utils import update_positions
import poly_data.global_state as global_state

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

def get_order_prices(best_bid, best_bid_size, top_bid,  best_ask, best_ask_size, top_ask, avgPrice, row):
    """
    Calculate optimal bid and ask prices for market making.
    
    Strategy:
    - Try to be at the top of the book (best bid + tick or best ask - tick)
    - If there's small size at top, match it to compete
    - Never cross the spread accidentally
    """
    tick = row['tick_size']
    spread = best_ask - best_bid
    
    # === MORE AGGRESSIVE PRICING FOR MORE FILLS ===
    # Default: try to be 1 tick better than current best
    bid_price = best_bid + tick
    ask_price = best_ask - tick
    
    # If spread is very tight (1-2 ticks), don't improve - just match
    if spread <= tick * 2:
        bid_price = best_bid
        ask_price = best_ask
    # If spread is moderate (3-4 ticks), improve by 1 tick
    elif spread <= tick * 4:
        bid_price = best_bid + tick
        ask_price = best_ask - tick
    # If spread is wide, be more aggressive to capture it
    else:
        # Improve by 2 ticks on wide spreads to get filled faster
        bid_price = best_bid + tick * 2
        ask_price = best_ask - tick * 2
    
    # If there's small size at the top, match it to compete
    if best_bid_size < row['min_size'] * 1.5:
        bid_price = max(bid_price, best_bid)  # At least match best
    
    if best_ask_size < row['min_size'] * 1.5:
        ask_price = min(ask_price, best_ask)  # At least match best

    # Safety: never cross the spread
    if bid_price >= top_ask:
        bid_price = top_bid

    if ask_price <= top_bid:
        ask_price = top_ask

    if bid_price >= ask_price:
        bid_price = top_bid
        ask_price = top_ask

    # Ensure sell price is at least our average cost (if we have position)
    if ask_price <= avgPrice and avgPrice > 0:
        ask_price = avgPrice

    return bid_price, ask_price




def round_down(number, decimals):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor

def round_up(number, decimals):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

def get_buy_sell_amount(position, bid_price, row, other_token_position=0):
    buy_amount = 0
    sell_amount = 0

    # Get max_size, defaulting to trade_size if not specified
    max_size = row.get('max_size', row['trade_size'])
    trade_size = row['trade_size']
    min_size = row['min_size']
    
    # Calculate total exposure across both sides
    total_exposure = position + other_token_position
    
    # === AGGRESSIVE ENTRY STRATEGY ===
    # Enter positions faster to capture more rewards
    if position < max_size:
        # Calculate how much room we have
        remaining_to_max = max_size - position
        
        # If we have less than 50% of max_size, be MORE aggressive
        # This helps build positions faster in good markets
        if position < max_size * 0.5:
            # Try to build position with full trade_size
            buy_amount = min(trade_size, remaining_to_max)
        elif position < max_size * 0.8:
            # Moderate position - standard sizing
            buy_amount = min(trade_size, remaining_to_max)
        else:
            # Near max - be more conservative with buys
            buy_amount = min(trade_size * 0.5, remaining_to_max)
        
        # === CONTINUOUS SELL QUOTING ===
        # Always offer to sell when we have a position (for take profit)
        # This ensures we're always providing liquidity on both sides
        if position >= min_size:
            # Quote sells even at smaller positions to capture spreads
            sell_amount = min(position, trade_size)
        else:
            sell_amount = 0
    else:
        # We've reached max_size - focus on exits but maintain presence
        sell_amount = min(position, trade_size)
        
        # Keep a small buy quote to maintain market presence
        # Only if we're not overexposed on both sides
        if total_exposure < max_size * 1.8:
            buy_amount = min(trade_size * 0.5, min_size)
        else:
            buy_amount = 0

    # === ENSURE MINIMUM SIZE COMPLIANCE ===
    # Adjust to meet minimum if we're close
    if buy_amount > 0 and buy_amount < min_size:
        if buy_amount >= min_size * 0.7:
            buy_amount = min_size  # Round up to min
        else:
            buy_amount = 0  # Too small, skip
    
    if sell_amount > 0 and sell_amount < min_size:
        if sell_amount >= min_size * 0.7:
            sell_amount = min_size  # Round up to min
        else:
            sell_amount = 0  # Too small, skip

    # Apply multiplier for low-priced assets
    if bid_price < 0.1 and buy_amount > 0:
        # Multiplier is optional; treat missing/blank as 1x
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
