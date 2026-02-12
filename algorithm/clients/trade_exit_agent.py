import time
import copy
from dataclasses import replace
from typing import Optional, List
from setup.claim import redeem_position
from datetime import datetime, timezone
from storage.data_attributes import TradePosition
from config import (
    POST_REDEMPTION_PERIODS,
    MAX_RETRY_ATTEMPTS
)

class TradeExitAgent:
    def __init__(self):
        self.redeemed_trade_positions:List[TradePosition] = []

    def _reset_redeemed_trade_position(self) -> None:
        
        # Resets all redeemed trade positions
        self.redeemed_trade_positions = []
        
        return

    def _get_recent_market_time(self) -> int:

        # Get Recent Unix Timestamp for Market (15 minute)
        now = time.time()
        unix_time = int((now // 900) * 900)

        return unix_time
    
    def _exit_positions(self, trade_positions:List[TradePosition]) -> bool:

        # Sequentially Iterate Over Trade Positions and Redeem
        try:
            unix_time = self._get_recent_market_time()
            for trade_position in trade_positions:
                if (unix_time - trade_position.exit_time) >= (900 * POST_REDEMPTION_PERIODS):
                    itr = 0
                    while True:
                        if itr >= MAX_RETRY_ATTEMPTS:
                            modified_position_details = {
                                'exit_price': 0.00,
                                'exit_units': 0.00,
                                'position_status': "UNRESOLVED",
                                'outcome': "LOSS"
                            }
                            redeemed_position = replace(trade_position, **modified_position_details)
                            self.redeemed_trade_positions.append(redeemed_position)
                            print(f"ERROR: could not redeem for position ending at {datetime.fromtimestamp(trade_position.exit_time, tz=timezone.utc)}")
                            break
                        amount = redeem_position(
                            trade_position.condition_id,
                            trade_position.yes_token_id,
                            trade_position.no_token_id,
                            trade_position.neg_risk
                        )
                        if amount is not None:
                            modified_position_details = {
                                'exit_price': 1.00 if float(amount) > 0 else 0.00,
                                'exit_units': float(amount),
                                'position_status': "COMPLETE",
                                'outcome': "WIN" if float(amount) > 0 else "LOSS"
                            }
                            redeemed_position = replace(trade_position, **modified_position_details)
                            self.redeemed_trade_positions.append(redeemed_position)
                            print(f"INFO: position has been collect for market ending at {datetime.fromtimestamp(trade_position.exit_time, tz=timezone.utc)}")
                            break
                        itr += 1
                        time.sleep(1.333*itr)
            return True
        except Exception as e:
            print(f"ERROR: could not redeem positions {e}")

        return False

    def trade_exit(self, trade_positions:List[TradePosition]) -> Optional[TradePosition]:
        """
        Trade exit function that redeems a list of all current claimable positions

        Args:
            trade_positions:List[TradePosition] -> list of current pending trade positions
        Returns:
            trade_positions:List[TradePosition] -> list of redeemed trade positions
        """

        if not trade_positions: return None
        self._reset_redeemed_trade_position()
        check_1 = self._exit_positions(trade_positions)
        if not check_1: return None
        modified_trade_positions = copy.copy(self.redeemed_trade_positions)

        return modified_trade_positions