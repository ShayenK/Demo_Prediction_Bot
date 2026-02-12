import time
import copy
import requests
from datetime import datetime
from typing import Optional, List
from storage.data_attributes import CandleData
from config import (
    KLINES_URL,
    MAX_RETRY_ATTEMPTS,
    MAX_LIST_LEN
)

class CollectionAgent:
    def __init__(self):
        self.recent_list_candle_data:List[CandleData] = []
        self.last_hour:int = -1

    def _reset_recent_candle_data(self) -> None:
        
        # Reset recent candle data
        self.recent_list_candle_data = []
        
        return None
    
    def _collection_check(self) -> bool:
        
        # Checks collection time
        now = datetime.now()
        if now.hour != self.last_hour:
            self.last_hour = now.hour
            return True
        
        return False

    def _get_recent_market_time(self) -> int:

        # Get Recent Unix Timestamp for Market (1 hour)
        now = time.time()
        unix_time = int((now // 3600) * 3600)

        return unix_time
    
    def _get_candle_data(self) -> None:

        # Retrieve the most recent candle data (1 hour)
        try:
            unix_time = self._get_recent_market_time()
            itr = 0
            while True:
                if itr >= MAX_RETRY_ATTEMPTS:
                    print("ERROR: exceeded max attempts, cannot retrieve candles right now")
                    return None
                params = {
                    'symbol': 'BTCUSDT',
                    'interval': '1h',
                    'endTime': str((unix_time-1)*1000),
                    'limit': str(MAX_LIST_LEN)
                }
                resp = requests.get(KLINES_URL, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for kline in data:
                        candle_data = CandleData(
                            time=int(kline[0])/1000,
                            open=float(kline[1]),
                            high=float(kline[2]),
                            low=float(kline[3]),
                            close=float(kline[4]),
                            volume=float(kline[5]),
                            qa_volume=float(kline[7]),
                            num_trades=int(kline[8]),
                            taker_buy_ba_volume=float(kline[9]),
                            taker_buy_qa_volume=float(kline[10]),
                            taker_sell_ba_volume=float(kline[5]) - float(kline[9]),
                            taker_sell_qa_volume=float(kline[7]) - float(kline[10])
                        )
                        self.recent_list_candle_data.append(candle_data)
                    print("INFO: retrieved candle data")
                    return None
                else:
                    print(resp.status_code, resp)
                itr += 1
                time.sleep(1.333*itr)
        except Exception as e:
            print(f"ERROR: unable to request candle data {e}")

        return None
    
    def data_collection(self) -> Optional[List[CandleData]]:
        """
        Data collection method to pull most recent historical 1 hour candle data from binance
        
        Args:
            None
        Returns:
            recent_candle_data:List[CandleData] -> returns most recent klines for MAX_LIST_LEN
        """
        
        check_1 = self._collection_check()
        if not check_1: return None
        self._reset_recent_candle_data()
        self._get_candle_data()
        recent_list_candle_data = copy.copy(self.recent_list_candle_data)

        return recent_list_candle_data