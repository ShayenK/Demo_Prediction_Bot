import time
import copy
import json
import requests
from typing import Optional
from zoneinfo import ZoneInfo
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from storage.data_attributes import TradePosition
from config import (
    MARKET_URL,
    TRADE_URL,
    PUBLIC_KEY, 
    PRIVATE_KEY,
    CHAIN_ID,
    SIGNATURE,
    TRADE_UNITS,
    MAX_RETRY_ATTEMPTS,
    UPPER_THRESHOLD,
    LOWER_THRESHOLD
)

class TradeEntryAgent:
    def __init__(self):
        self.current_position:Optional[TradePosition] = None
        self._client = self.__client_authentication()

    def __client_authentication(self) -> ClobClient:
        
        # Authentication of polymarket client on init
        client = ClobClient(
            host=TRADE_URL,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID
        )
        user_api_creds = client.create_or_derive_api_creds()
        client = ClobClient(
            host=TRADE_URL,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            creds=user_api_creds,
            signature_type=SIGNATURE,
            funder=PUBLIC_KEY
        )
        
        return client

    def _reset_current_position(self) -> None:
        
        # Reset the current position (only after has been appended in-memory)
        self.current_position = None
        
        return None

    def _get_recent_market_time(self) -> int:

        # Get Recent Unix Timestamp for Market (1 hour)
        now = time.time()
        unix_time = int((now // 3600) * 3600)

        return unix_time
    
    def _get_slug_market_time(self) -> str:
        
        # Get Recent Formated Datetime Slug for Market (month-dom-analouge_hour)
        now_et = datetime.now(ZoneInfo("US/Eastern"))
        next_hour_et = now_et + timedelta(hours=1)
        month_et = next_hour_et.strftime('%B').lower()
        day_of_month_et = next_hour_et.day
        hour_et = next_hour_et.strftime("%I%p").lstrip("0").lower()
        slug_str = f'bitcoin-up-or-down-{month_et}-{day_of_month_et}-{hour_et}-et'

        return slug_str

    def _get_market(self) -> bool:
        
        # Get the Current BTCUSD market (1 hour)
        print("INFO: attempting to fetch market data")
        try:
            unix_time = self._get_recent_market_time()
            slug_time_str = self._get_slug_market_time()
            url = MARKET_URL + slug_time_str
            itr = 0
            while True:
                if itr >= MAX_RETRY_ATTEMPTS: 
                    print("ERROR: could not fetch market data")
                    return False
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_token_ids = data.get('clobTokenIds')
                    token_ids = json.loads(raw_token_ids)
                    condition_id = data.get('conditionId')
                    neg_risk = data.get('negRisk', False)
                    self.current_position = TradePosition(
                        entry_proba=None,
                        entry_time=unix_time,
                        entry_price=None,
                        entry_units=None,
                        direction=None,
                        yes_token_id=str(token_ids[0]),
                        no_token_id=str(token_ids[1]),
                        condition_id=str(condition_id),
                        neg_risk=bool(neg_risk),
                        order_id=None,
                        exit_time=(unix_time+3600),
                        exit_price=None,
                        exit_units=None,
                        position_status=None,
                        outcome=None
                    )
                    print("INFO: collected market data and preparing entry")
                    return True
                itr += 1
                time.sleep(1.333*itr)
        except Exception as e:
            print(f"ERROR: trade entry issue: {e}")

        return False

    def _entry_position(self, prediction:float) -> bool:

        # Enter trade position
        if prediction >= UPPER_THRESHOLD:
            token_id = self.current_position.yes_token_id
            direction = "UP"
        elif prediction <= LOWER_THRESHOLD:
            token_id = self.current_position.no_token_id
            direction = "DOWN"
        else:
            print(f"INFO: no signal -> {prediction}")
            return False
        
        try:
            itr = 0
            while True:
                if itr >= MAX_RETRY_ATTEMPTS:
                    print("ERROR: could not enter trade")
                    return False
                order_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=TRADE_UNITS,
                    side=BUY,
                )
                signed_order = self._client.create_market_order(order_args)
                resp = self._client.post_order(signed_order, orderType=OrderType.FOK)
                if resp.get('orderID'):
                    taking_amt = float(resp.get('takingAmount', 0))
                    making_amt = float(resp.get('makingAmount', 0))
                    execution_price =  making_amt / taking_amt if taking_amt > 0 else 0
                    print(f"INFO: order filled for btcusd 15 minute prediction at entry time {datetime.fromtimestamp(self.current_position.entry_time, tz=timezone.utc)}")
                    new_current_position = {
                        'entry_proba': prediction,
                        'entry_price': execution_price,
                        'entry_units': making_amt,
                        'direction': direction,
                        'order_id': resp.get('orderID'),
                        'position_status': "PENDING"
                    }
                    self.current_position = replace(self.current_position, **new_current_position)
                    return True
                itr += 1
                time.sleep(1.333*itr)
        except Exception as e:
            print(f"ERROR: unable to enter trade {e}")

        return False

    def trade_entry(self, prediction:float) -> Optional[TradePosition]:
        """
        Trade entry function that uses prediction at the current candle to enter a trade

        Args:
            prediction:str -> UP or DOWN signal for betting direction
            candle_data:CandleData -> single candle data object for entry prices
        Returns:
            trade_position:TradePosition -> trade position that can be stored for later redemption
        """

        if not prediction: return None
        self._reset_current_position()
        check_1 = self._get_market()
        if not check_1: return None 
        check_2 = self._entry_position(prediction)
        if not check_2: return None
        trade_position = copy.copy(self.current_position)

        return trade_position