import asyncio                      # Asynchronous I/O
import json                        # JSON handling
import websockets                  # WebSocket client
import traceback                   # Exception handling
import time                        # For heartbeat timing

from poly_data.data_processing import process_data, process_user_data
import poly_data.global_state as global_state

async def connect_market_websocket(chunk, version=None):
    """
    Connect to Polymarket's market WebSocket API and process market updates.
    
    This function:
    1. Establishes a WebSocket connection to the Polymarket API
    2. Subscribes to updates for a specified list of market tokens
    3. Processes incoming order book and price updates
    
    Args:
        chunk (list): List of token IDs to subscribe to
        version (int): Subscription version to track live updates
        
    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    expected_version = version if version is not None else global_state.market_tokens_version
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare and send subscription message
        message = {"assets_ids": chunk}
        await websocket.send(json.dumps(message))

        print("\n")
        print(f"Sent market subscription message: {message}")

        try:
            # Process incoming market data indefinitely
            msg_count = 0
            last_heartbeat = time.time()
            
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                msg_count += 1
                
                # Heartbeat log every 60 seconds to show bot is alive
                now = time.time()
                if now - last_heartbeat >= 60:
                    print(f"[Heartbeat] Received {msg_count} market updates in last 60s. Tracking {len(global_state.all_tokens)} tokens.")
                    msg_count = 0
                    last_heartbeat = now
                
                # Process order book updates and trigger trading as needed
                process_data(json_data)

                # If subscription version changed (new markets added), drop connection to resubscribe
                if global_state.market_tokens_version != expected_version:
                    print(f"Market subscription version changed ({expected_version} -> {global_state.market_tokens_version}); reconnecting to include new markets.")
                    break
        except websockets.ConnectionClosed:
            print("Connection closed in market websocket")
            print(traceback.format_exc())
        except Exception as e:
            print(f"Exception in market websocket: {e}")
            print(traceback.format_exc())
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)

async def connect_user_websocket():
    """
    Connect to Polymarket's user WebSocket API and process order/trade updates.
    
    This function:
    1. Establishes a WebSocket connection to the Polymarket user API
    2. Authenticates using API credentials
    3. Processes incoming order and trade updates for the user
    
    Notes:
        If the connection is lost, the function will exit and the main loop will
        attempt to reconnect after a short delay.
    """
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as websocket:
        # Prepare authentication message with API credentials
        message = {
            "type": "user",
            "auth": {
                "apiKey": global_state.client.client.creds.api_key, 
                "secret": global_state.client.client.creds.api_secret,  
                "passphrase": global_state.client.client.creds.api_passphrase
            }
        }

        # Send authentication message
        await websocket.send(json.dumps(message))

        print("\n")
        print(f"Sent user subscription message")

        try:
            # Process incoming user data indefinitely
            while True:
                message = await websocket.recv()
                json_data = json.loads(message)
                # Process trade and order updates
                process_user_data(json_data)
        except websockets.ConnectionClosed:
            print("Connection closed in user websocket")
            print(traceback.format_exc())
        except Exception as e:
            print(f"Exception in user websocket: {e}")
            print(traceback.format_exc())
        finally:
            # Brief delay before attempting to reconnect
            await asyncio.sleep(5)
