import poly_data.global_state as global_state
from poly_data.utils import get_sheet_df
import time
import poly_data.global_state as global_state

#sth here seems to be removing the position
def update_positions(avgOnly=False):
    pos_df = global_state.client.get_all_positions()

    for idx, row in pos_df.iterrows():
        asset = str(row['asset'])

        if asset in  global_state.positions:
            position = global_state.positions[asset].copy()
        else:
            position = {'size': 0, 'avgPrice': 0}

        position['avgPrice'] = row['avgPrice']

        if not avgOnly:
            position['size'] = row['size']
        else:
            
            for col in [f"{asset}_sell", f"{asset}_buy"]:
                #need to review this
                if col not in global_state.performing or not isinstance(global_state.performing[col], set) or len(global_state.performing[col]) == 0:
                    try:
                        old_size = position['size']
                    except:
                        old_size = 0

                    if asset in  global_state.last_trade_update:
                        if time.time() - global_state.last_trade_update[asset] < 5:
                            print(f"Skipping update for {asset} because last trade update was less than 5 seconds ago")
                            continue

                    if old_size != row['size']:
                        print(f"No trades are pending. Updating position from {old_size} to {row['size']} and avgPrice to {row['avgPrice']} using API")
    
                    position['size'] = row['size']
                else:
                    print(f"ALERT: Skipping update for {asset} because there are trades pending for {col} looking like {global_state.performing[col]}")
    
        global_state.positions[asset] = position

def get_position(token):
    token = str(token)
    if token in global_state.positions:
        return global_state.positions[token]
    else:
        return {'size': 0, 'avgPrice': 0}

def set_position(token, side, size, price, source='websocket'):
    token = str(token)
    size = float(size)
    price = float(price)

    global_state.last_trade_update[token] = time.time()
    
    if side.lower() == 'sell':
        size *= -1

    if token in global_state.positions:
        
        prev_price = global_state.positions[token]['avgPrice']
        prev_size = global_state.positions[token]['size']


        if size > 0:
            if prev_size == 0:
                # Starting a new position
                avgPrice_new = price
            else:
                # Buying more; update average price
                avgPrice_new = (prev_price * prev_size + price * size) / (prev_size + size)
        elif size < 0:
            # Selling; average price remains the same
            avgPrice_new = prev_price
        else:
            # No change in position
            avgPrice_new = prev_price


        global_state.positions[token]['size'] += size
        global_state.positions[token]['avgPrice'] = avgPrice_new
    else:
        global_state.positions[token] = {'size': size, 'avgPrice': price}

    print(f"Updated position from {source}, set to ", global_state.positions[token])

def update_orders():
    all_orders = global_state.client.get_all_orders()

    orders = {}

    if len(all_orders) > 0:
            for token in all_orders['asset_id'].unique():
                
                if token not in orders:
                    orders[str(token)] = {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}

                curr_orders = all_orders[all_orders['asset_id'] == str(token)]
                
                if len(curr_orders) > 0:
                    sel_orders = {}
                    sel_orders['buy'] = curr_orders[curr_orders['side'] == 'BUY']
                    sel_orders['sell'] = curr_orders[curr_orders['side'] == 'SELL']

                    for type in ['buy', 'sell']:
                        curr = sel_orders[type]

                        if len(curr) > 1:
                            print("Multiple orders found, cancelling")
                            global_state.client.cancel_all_asset(token)
                            orders[str(token)] = {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}
                        elif len(curr) == 1:
                            orders[str(token)][type]['price'] = float(curr.iloc[0]['price'])
                            orders[str(token)][type]['size'] = float(curr.iloc[0]['original_size'] - curr.iloc[0]['size_matched'])

    global_state.orders = orders

def get_order(token):
    token = str(token)
    if token in global_state.orders:

        if 'buy' not in global_state.orders[token]:
            global_state.orders[token]['buy'] = {'price': 0, 'size': 0}

        if 'sell' not in global_state.orders[token]:
            global_state.orders[token]['sell'] = {'price': 0, 'size': 0}

        return global_state.orders[token]
    else:
        return {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}
    
def set_order(token, side, size, price):
    """
    Update a single order (buy or sell) for a token WITHOUT overwriting the other side.
    
    This preserves existing orders on the opposite side when updating one side.
    """
    token = str(token)
    side = side.lower()
    
    # Get existing orders for this token, or create new structure
    if token in global_state.orders:
        curr = global_state.orders[token].copy()
    else:
        curr = {'buy': {'price': 0, 'size': 0}, 'sell': {'price': 0, 'size': 0}}
    
    # Ensure both sides exist
    if 'buy' not in curr:
        curr['buy'] = {'price': 0, 'size': 0}
    if 'sell' not in curr:
        curr['sell'] = {'price': 0, 'size': 0}
    
    # Update only the specified side
    curr[side]['size'] = float(size)
    curr[side]['price'] = float(price)
    
    global_state.orders[token] = curr
    print(f"Updated order for {token[:20]}... {side}: price={price}, size={size}")

    

def update_markets():
    """
    Update market data from Google Sheets.
    
    This function:
    1. Fetches the latest Selected Markets from Google Sheets
    2. Updates global state with new market data
    3. Tracks new tokens for websocket subscription
    4. Cleans up removed markets from tracking
    
    Thread-safe: Uses global_state.lock for atomic updates.
    """
    try:
        received_df, received_params = get_sheet_df()
    except Exception as e:
        print(f"Warning: Failed to fetch sheet data: {e}")
        return

    if received_df is None or len(received_df) == 0:
        print("Warning: Received empty Selected/All Markets merge; skipping update_markets")
        return

    required_cols = {'token1', 'token2', 'question'}
    missing = required_cols - set(received_df.columns)
    if missing:
        print(f"Warning: update_markets missing required columns {missing}; skipping update.")
        return
    
    # Use lock for thread-safe updates
    with global_state.lock:
        # Track which tokens are currently active
        current_tokens = set()
        for _, row in received_df.iterrows():
            current_tokens.add(str(row['token1']))
            current_tokens.add(str(row['token2']))
        
        # Detect removed markets (tokens that were tracked but no longer in sheet)
        previous_tokens = set(global_state.all_tokens)
        removed_tokens = previous_tokens - current_tokens
        if removed_tokens:
            print(f"Markets removed: {len(removed_tokens)} tokens no longer in Selected Markets")
            # Clean up orders dict for removed tokens
            for token in removed_tokens:
                if token in global_state.orders:
                    del global_state.orders[token]
                # Remove from performing tracking
                for suffix in ['_buy', '_sell']:
                    col = f"{token}{suffix}"
                    if col in global_state.performing:
                        del global_state.performing[col]
                    if col in global_state.performing_timestamps:
                        del global_state.performing_timestamps[col]
        
        # Update the dataframe and params
        global_state.df = received_df.copy()
        global_state.params = received_params
        
        new_tokens_added = False

        for _, row in global_state.df.iterrows():
            token1 = str(row['token1'])
            token2 = str(row['token2'])

            if token1 not in global_state.all_tokens:
                global_state.all_tokens.append(token1)
                new_tokens_added = True

            if token1 not in global_state.REVERSE_TOKENS:
                global_state.REVERSE_TOKENS[token1] = token2

            if token2 not in global_state.REVERSE_TOKENS:
                global_state.REVERSE_TOKENS[token2] = token1

            for col2 in [f"{token1}_buy", f"{token1}_sell", f"{token2}_buy", f"{token2}_sell"]:
                if col2 not in global_state.performing:
                    global_state.performing[col2] = set()
                if col2 not in global_state.performing_timestamps:
                    global_state.performing_timestamps[col2] = {}

        # If tokens were removed OR added, bump version to trigger websocket reconnect
        if new_tokens_added or removed_tokens:
            # Rebuild all_tokens to only include current tokens
            global_state.all_tokens = list(current_tokens)
            global_state.market_tokens_version += 1
            print(f"Market subscription version is now {global_state.market_tokens_version}. Total tokens: {len(global_state.all_tokens)}")
