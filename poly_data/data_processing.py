import json
from sortedcontainers import SortedDict
import poly_data.global_state as global_state
import poly_data.CONSTANTS as CONSTANTS

from trading import perform_trade
import time 
import asyncio
from poly_data.data_utils import set_position, set_order, update_positions

def process_book_data(asset, json_data):
    global_state.all_data[asset] = {
        'asset_id': json_data['asset_id'],  # token_id for the Yes token
        'bids': SortedDict(),
        'asks': SortedDict()
    }

    global_state.all_data[asset]['bids'].update({float(entry['price']): float(entry['size']) for entry in json_data['bids']})
    global_state.all_data[asset]['asks'].update({float(entry['price']): float(entry['size']) for entry in json_data['asks']})

def process_price_change(asset, side, price_level, new_size, asset_id=None):
    # Some payloads include asset_id for the specific token. If present and not the
    # primary token we track for this market, skip to avoid duplicate updates.
    if asset_id is not None:
        try:
            if asset_id != global_state.all_data[asset]['asset_id']:
                return
        except Exception:
            pass
    if side == 'bids':
        book = global_state.all_data[asset]['bids']
    else:
        book = global_state.all_data[asset]['asks']

    if new_size == 0:
        if price_level in book:
            del book[price_level]
    else:
        book[price_level] = new_size

def process_data(json_datas, trade=True):
    """
    Process market websocket events (order book updates).
    
    Handles:
    - book: Full order book snapshot
    - price_change: Incremental price level updates
    
    Gracefully handles events for removed markets.
    """
    # Normalize single-event payloads into a list
    if isinstance(json_datas, dict):
        json_datas = [json_datas]

    for json_data in json_datas:
        event_type = json_data.get('event_type')
        asset = json_data.get('market')
        
        if not asset or not event_type:
            continue

        # For 'book' events, always process - this is how we initially populate all_data
        # For 'price_change' events, only process if we have existing book data
        if event_type == 'book':
            process_book_data(asset, json_data)

            if trade:
                asyncio.create_task(perform_trade(asset))
                
        elif event_type == 'price_change':
            # Only process price changes if we have book data for this asset
            if asset not in global_state.all_data:
                continue
                
            for data in json_data.get('price_changes', []):
                side = 'bids' if data.get('side') == 'BUY' else 'asks'
                price_level = float(data.get('price', 0))
                new_size = float(data.get('size', 0))
                process_price_change(asset, side, price_level, new_size, data.get('asset_id'))

                if trade:
                    asyncio.create_task(perform_trade(asset))
        

        # pretty_print(f'Received book update for {asset}:', global_state.all_data[asset])

def add_to_performing(col, id):
    if col not in global_state.performing:
        global_state.performing[col] = set()
    
    if col not in global_state.performing_timestamps:
        global_state.performing_timestamps[col] = {}

    # Add the trade ID and track its timestamp
    global_state.performing[col].add(id)
    global_state.performing_timestamps[col][id] = time.time()

def remove_from_performing(col, id):
    if col in global_state.performing:
        global_state.performing[col].discard(id)

    if col in global_state.performing_timestamps:
        global_state.performing_timestamps[col].pop(id, None)

def process_user_data(rows):
    """
    Process user-specific websocket events (trades and orders).
    
    Handles:
    - Trade events: MATCHED, CONFIRMED, FAILED, MINED statuses
    - Order events: Updates order tracking
    
    Thread-safe with graceful handling of removed markets.
    """
    # Normalize single-event payloads into a list
    if isinstance(rows, dict):
        rows = [rows]

    for row in rows:
        market = row.get('market')
        if not market:
            continue

        side = row.get('side', '').lower()
        token = row.get('asset_id')
        
        if not token:
            continue
            
        # Check if this token is still being tracked (market might have been removed)
        if token not in global_state.REVERSE_TOKENS:
            # Token not in our tracked markets - might be from a removed market
            # Just log and skip to avoid crashes
            print(f"Received event for untracked token {token[:20]}... (market may have been removed)")
            continue
     
        col = token + "_" + side

        if row['event_type'] == 'trade':
            size = 0
            price = 0
            maker_outcome = ""
            taker_outcome = row.get('outcome', '')

            is_user_maker = False
            for maker_order in row.get('maker_orders', []):
                if maker_order.get('maker_address', '').lower() == global_state.client.browser_wallet.lower():
                    print("User is maker")
                    size = float(maker_order.get('matched_amount', 0))
                    price = float(maker_order.get('price', 0))
                    
                    is_user_maker = True
                    maker_outcome = maker_order.get('outcome', '')

                    if maker_outcome == taker_outcome:
                        side = 'buy' if side == 'sell' else 'sell'
                    else:
                        token = global_state.REVERSE_TOKENS.get(token, token)
            
            if not is_user_maker:
                size = float(row.get('size', 0))
                price = float(row.get('price', 0))
                print("User is taker")

            print("TRADE EVENT FOR: ", row['market'], "ID: ", row.get('id'), "STATUS: ", row.get('status'), " SIDE: ", row.get('side'), "  MAKER OUTCOME: ", maker_outcome, " TAKER OUTCOME: ", taker_outcome, " PROCESSED SIDE: ", side, " SIZE: ", size) 

            status = row.get('status', '')
            if status == 'CONFIRMED' or status == 'FAILED':
                if status == 'FAILED':
                    print(f"Trade failed for {token}, decreasing")
                    asyncio.create_task(asyncio.sleep(2))
                    update_positions()
                else:
                    remove_from_performing(col, row.get('id'))
                    print("Confirmed. Performing is ", len(global_state.performing.get(col, set())))
                    print("Last trade update is ", global_state.last_trade_update)
                    print("Performing is ", global_state.performing)
                    print("Performing timestamps is ", global_state.performing_timestamps)
                    
                    asyncio.create_task(perform_trade(market))

            elif status == 'MATCHED':
                add_to_performing(col, row.get('id'))

                print("Matched. Performing is ", len(global_state.performing.get(col, set())))
                set_position(token, side, size, price)
                print("Position after matching is ", global_state.positions.get(str(token), {}))
                print("Last trade update is ", global_state.last_trade_update)
                print("Performing is ", global_state.performing)
                print("Performing timestamps is ", global_state.performing_timestamps)
                asyncio.create_task(perform_trade(market))
            elif status == 'MINED':
                remove_from_performing(col, row.get('id'))

        elif row['event_type'] == 'order':
            print("ORDER EVENT FOR: ", row['market'], " STATUS: ",  row.get('status'), " TYPE: ", row.get('type'), " SIDE: ", side, "  ORIGINAL SIZE: ", row.get('original_size'), " SIZE MATCHED: ", row.get('size_matched'))
            
            set_order(token, side, float(row.get('original_size', 0)) - float(row.get('size_matched', 0)), row.get('price', 0))
            asyncio.create_task(perform_trade(market))
