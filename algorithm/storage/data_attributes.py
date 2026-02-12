from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class TradePosition:
    entry_proba:float
    entry_time:int
    entry_price:float
    entry_units:float
    direction:str
    yes_token_id:str
    no_token_id:str
    condition_id:str
    order_id:str
    neg_risk:bool
    exit_time:int
    exit_price:Optional[float]
    exit_units:Optional[float]
    position_status:Optional[str]
    outcome:Optional[str]

@dataclass(frozen=True)
class CandleData:
    time:int
    open:float
    high:float
    low:float
    close:float
    volume:float
    qa_volume:float
    num_trades:int
    taker_buy_ba_volume:float
    taker_buy_qa_volume:float
    taker_sell_ba_volume:float
    taker_sell_qa_volume:float