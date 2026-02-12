import os
import csv
from types import TracebackType
from typing import Optional, Dict, List, Type, Self
from dataclasses import astuple, fields
from datetime import datetime, timezone
from storage.data_attributes import TradePosition
from config import (
    TRADE_LOG_FILEPATH
)

class Memory:
    def __init__(self):
        self.trade_logs:Dict[str,TradePosition] = {}
        self.trade_log_file_str = f'algorithm/storage/{TRADE_LOG_FILEPATH}'
        self.__initialize_csv_with_headers()
        
    def __initialize_csv_with_headers(self) -> None:
        
        # Initialize the csv with headers
        if not os.path.exists(self.trade_log_file_str):
            try:
                headers = [field.name for field in fields(TradePosition)]
                with open(self.trade_log_file_str, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                print("INFO: Initialized log file with headers")
            except Exception as e:
                print(f"ERROR: could not initialize csv headers: {e}")

        return None

    def add_trade_position(self, trade_position:TradePosition) -> None:
        """
        Adds singular trade position record to memory

        Args:
            trade_position:TradePosition -> singular trade position
        Returns:
            None
        """

        if not trade_position: return None
        self.trade_logs[f'{trade_position.order_id}'] = trade_position

        return None
    
    def return_trade_positions(self) -> Optional[List[TradePosition]]:
        """
        Returns all current trade positions in memory

        Args:
            None
        Returns:
            trade_positions:List[TradePosition] -> List of all trade positions
        """

        if self.trade_logs: 
            trade_positions = [trade_position for order_id, trade_position in self.trade_logs.items()].copy()
            return trade_positions

        return None
    
    def _append_finished_position(self, trade_position:TradePosition) -> None:

        # Append finished trade positions to trade_logs.csv
        try:
            with open(self.trade_log_file_str, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(astuple(trade_position))
                print(f"INFO: added record to csv for position at {datetime.fromtimestamp(trade_position.exit_time, tz=timezone.utc)}")
        except Exception as e:
            print(f"ERROR: unable to append finished records to csv {e}")

        return None
    
    def record_trade_positions(self, modified_trade_positions:List[TradePosition]) -> None:
        """
        Records all finished trade positions to trade_logs.csv

        Args:
            trade_positions:List[TradePositions] -> list of finished trade positions
        Returns:
            None
        """

        if not modified_trade_positions: return None
        for trade_position in modified_trade_positions:
            self.trade_logs.pop(f'{trade_position.order_id}', None)
            self._append_finished_position(trade_position)

        return None
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], 
                 exc_tb: Optional[TracebackType]):
        
        # Auto logging on keyboard interupt
        print("INFO: activating auto position logging")
        if self.trade_logs: 
            self.record_trade_positions(list(self.trade_logs.values()))
        else: print("INFO: no open positions to save")

        return None