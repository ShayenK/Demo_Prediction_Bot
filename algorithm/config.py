import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

# Directory
TRADE_LOG_FILEPATH:str = 'trade_logs.csv'
MODEL_FILEPATH:str = 'live_model.pkl'

# API requests
RPC_URL:str = "https://polygon-mainnet.core.chainstack.com/"
KLINES_URL:str = 'https://api.binance.com/api/v3/klines'
MARKET_URL:str = 'https://gamma-api.polymarket.com/markets/slug/'
TRADE_URL:str = "https://clob.polymarket.com"

# Strategy Parameters
MAX_LIST_LEN:int = 0
CHAIN_ID:int = 137
SIGNATURE:int = 0
TRADE_UNITS:float = 0.00
MAX_RETRY_ATTEMPTS:int = 0
POST_REDEMPTION_PERIODS:int = 0
STRATEGY_PERIODS:List[int] = []
UPPER_THRESHOLD:float = 1.00
LOWER_THRESHOLD:float = 0.00

# User Info
PUBLIC_KEY:str = os.getenv("PUBLIC_KEY")
PRIVATE_KEY:str = os.getenv("PRIVATE_KEY")