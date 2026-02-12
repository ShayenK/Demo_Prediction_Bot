# Polymarket BTC Prediction Bot

Automated prediction market bot for Polymarket's 1-hour Bitcoin directional markets. Uses XGBoost classification trained on technical indicators derived from Binance orderbook and price data.

*Note: Trade execution logic, position sizing algorithms, and proprietary feature combinations are omitted from this repository as similar strategies are currently deployed.*

## Repository Structure

```
├── algorithm/
|   ├── clients/
|   |   ├── collection_agent.py
|   |   ├── trade_entry_agent.py
|   |   └── trade_exit_agent.py
|   ├── production/
|   |   └── live_model.pkl
|   ├── setup/
|   |   ├── claim.py
|   |   └── set_up_wallet.py
|   ├── storage/
|   |   ├── data_attributes.py
|   |   ├── memory.py
|   |   └── trade_logs.csv
|   ├── strategy/
|   |   └── engine.py
|   ├── config.py
|   └── main.py (run script)
├── analysis/
|   ├── _1data_check.py
|   ├── _2clean_and_engineer.py
|   ├── _3data_split.py
|   ├── _4final_checks.py
|   ├── _5backtest.py
|   ├── _6forward_testing.py
|   ├── _7permutation.py
|   └── _8model_creation.py
├── requirements.txt
└── README.md
```

## Strategy Overview

**Objective**: Predict BTC price direction over 1-hour windows to exploit Polymarket's binary outcome markets with positive expected value.

**Data Pipeline**:
- Real-time ingestion from Binance API (Kline/OHLCV data)
- Single timeframes: 1h
- Feature generation produces 50-144 indicators depending on feature importance analysis (most omited in demo)

**Feature Engineering**:

The model utilizes a comprehensive set of technical indicators across multiple categories:

- **Price Momentum**: price change proxies
- **Volume Analysis**: Proxies for order-flow imbalance
- **Volatility Metrics**: different implied and realized metrics for volatility
- **Temporal Features**: Hour of day (cyclical encoding)

**Model Architecture**:
- **Algorithm**: XGBoost binary classifier
- **Training Window**: Rolling 4-month lookback (~2,880 hourly samples)
- **Retraining Frequency**: 2-4 week recalibration
- **Rationale**: Balances sufficient training data volume with adaptivity to evolving market regimes. Four months provides statistical robustness while monthly updates prevent model drift.

**Feature Selection Process**:
- Initial feature pool: 144 candidates
- Feature importance ranking via SHAP values and built-in XGBoost importance
- Dynamic selection retains 50-80 highest-signal features per training cycle
- Removes multicollinear and low-information indicators

## Model Training Pipeline

1. Data Checking
- check to ensure clean dataset 
- if data is does not meet standards find other data retrieval methods
2. Clean and Feature Enginer
- remove invalid, missing or infinite values
- engineer features into the dataset
3. Split Data
- split data into clear in and out group samples
4. Conduct a Final Check
- check to ensure clean dataset again
6. walkforward
- walkforward testing for hyperparametergeneralisation
- ensure consistancy with results
7. permutation
- conducts random permutations when the target is randomised to ensure statistical robustness
- model should outperform randomness with statistical significance 

## Backtesting Results

**Backtest Period**: 2025-04-01 to 2025-12-31
**Total Predictions**: 3323 (1-hour intervals)  
**Market Conditions**: Tested across out sample period, predominantly choppy, ranging, and bearish

BACKTEST RESULTS: OUT SAMPLE
----------------
TOTAL TRADES: 3323
TOTAL WINS: 1787
WIN RATE: 53.78%
TOTAL RETURNS: 251.00%
MAX CONSEC. WINS: 14
MAX CONSEC. LOSSES: 8
MAX DRAWDOWN: 8.91%
AVG CONSEC. WINS: 2.14
AVG CONSEC LOSSES: 1.83
AVG DRAWDOWN: 2.08%
FULL KELLY FRACTION: 7.55%
1/4 KELLY FRACTION: 1.89%

**Performance Notes**:
- Backtest assumes realistic transaction costs (Polymarket fee structure + 0.3% slippage)
- Edge derived from >50% accuracy requirement in binary markets with ~2% edge over random
- Monthly retraining shows improved performance vs. static model (+8.3% accuracy improvement)
- Performance degrades in low-volatility periods (identified as key limitation)

## Technologies & Dependencies

**Core Stack**:
- Python 3.10+
- XGBoost 2.0.3
- pandas 2.1.0, numpy 1.24.3
- scikit-learn 1.3.0
- ta (Technical Analysis Library) 0.11.0

**Data & API**:
- binance-connector-python
- python-dotenv (API key management)

**Analysis & Visualization**:
- matplotlib, seaborn
- jupyter notebook
- SHAP (model interpretability)

## Installation & Setup
```bash
# Clone repository
git clone https://github.com/ShayenK/Demo_Prediction_Bot
cd demo_prediction_bot

# Install dependencies
pip install -r requirements.txt

# Set up API credentials (create .env file)
PRIVATE_KEY=your_key_here
PUBLIC_KEY=your_secret_here
```

**Important**: The execution engine in `/algorithm` is intentionally non-functional. This repository demonstrates strategy development and backtesting workflows only. Live trading components (order execution, position sizing, risk management) are proprietary.

## Key Insights & Lessons

**What Worked**:
- Monthly retraining significantly improved model adaptivity vs quarterly/static approaches
- Volume-based features showed highest predictive power during trending markets
- Time-series cross-validation prevented overfitting better than random K-fold

**Limitations**:
- Performance degrades in low-volatility, range-bound conditions
- Model requires continuous monitoring for regime changes
- 1-hour prediction window introduces execution timing risk

**Future Improvements** (not implemented in public repo):
- Ensemble methods combining multiple model architectures
- Regime detection layer to adjust strategy based on market conditions
- Alternative data integration (social sentiment, on-chain metrics)

## Project Context

This repository represents the strategy development and research component of a larger automated trading system. The complete production pipeline includes:
- Real-time data ingestion and processing
- Automated model retraining workflows
- Risk management and position sizing algorithms
- API integration with Polymarket
- Monitoring and alerting infrastructure

These components are omitted as they contain proprietary logic currently deployed in live markets.

## Contact

For questions about methodology:
- **Email**: shayen.kesha@gmail.com

---

*This project is for demonstration and educational purposes. Past performance does not guarantee future results. Trading involves substantial risk of loss. STRICTLY FOR DEMONSTRATION PURPOSES ONLY*