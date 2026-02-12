import time
from typing import List
from clients.collection_agent import CollectionAgent
from clients.trade_entry_agent import TradeEntryAgent
from clients.trade_exit_agent import TradeExitAgent
from storage.data_attributes import CandleData, TradePosition
from storage.memory import Memory
from strategy.engine import StrategyEngine

def main() -> None:

    try:
        candles = CollectionAgent()
        entry = TradeEntryAgent()
        exit = TradeExitAgent()
        engine = StrategyEngine()

        with Memory() as memory:
            while True:
                
                list_candle_data:List[CandleData] = candles.data_collection()
                prediction:float = engine.model_prediction(list_candle_data)
                trade_position:TradePosition = entry.trade_entry(prediction)

                memory.add_trade_position(trade_position)
                trade_positions:List[TradePosition] = memory.return_trade_positions()

                modified_list_trade_positions:List[TradePosition] = exit.trade_exit(trade_positions)
                memory.record_trade_positions(modified_list_trade_positions)

                time.sleep(1)

    except Exception as e:
        print(f"INFO: error keeping loop alive {e}")
    except KeyboardInterrupt:
        print("INFO: shutting down")

    return None

if __name__ == "__main__":
    main()