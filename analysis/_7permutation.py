import copy
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from _5backtest import TradePosition, BacktestPortfolioManager
from _6walkforward import WalkForwardAnalysis
from warnings import filterwarnings

filterwarnings('ignore')

class PermutationResults:
    def __init__(self, save_directory:str):
        self.save_directory:str = save_directory
        self.base_trade_positions:List[TradePosition] = []
        self.base_prob_var:List[float] = []
        self.base_equity_curve:List[float] = []
        self.permutation_positions:Dict[str,List[TradePosition]] = {}
        self.permutation_prob_var:Dict[str,List[float]] = {}
        self.permutation_equity_curves:Dict[str,List[float]] = {}

    def add_base_results(self, trade_positions:List[TradePosition], probability_variance:List[float], equity_curve:List[float]) -> None:
        
        # Add Base Walk-Forward Results
        self.base_trade_positions = copy.copy(trade_positions)
        self.probability_variance = copy.copy(probability_variance)
        self.base_equity_curve = copy.copy(equity_curve)

        return None

    def add_permutation_run(self, permutation_run:int, trade_positions:List[TradePosition], probability_variance:List[float], 
                                equity_curve:List[float]) -> None:
        
        # Add Single Permutation Results 
        self.permutation_positions[f'{permutation_run}'] = trade_positions
        self.permutation_prob_var[f'{permutation_run}'] = probability_variance
        self.permutation_equity_curves[f'{permutation_run}'] = equity_curve

        return None
    
    def _calculate_statistics(self) -> None:

        # Basic Calculation Statistics
        base_final_return = self.base_equity_curve[-1]
        perm_final_returns = [curve[-1] for curve in self.permutation_equity_curves.values()]
        better_than_base = sum(1 for r in perm_final_returns if r >= base_final_return)
        p_value = (better_than_base + 1) / (len(perm_final_returns) + 1)
        
        self.results_summary = {
            'base_return': base_final_return,
            'avg_perm_return': np.mean(perm_final_returns),
            'max_perm_return': np.max(perm_final_returns),
            'p_value': p_value,
            'n_permutations': len(perm_final_returns)
        }

        return None

    def _print_results(self) -> None:
        
        # Permutation Analysis
        print("\nPERMUTATION ANALYSIS SUMMARY")
        print("----------------------------")
        print(f"Original Strategy Return: {self.results_summary['base_return']:.2f}")
        print(f"Average Permuted Return:  {self.results_summary['avg_perm_return']:.2f}")
        print(f"Max Permuted Return:      {self.results_summary['max_perm_return']:.2f}")
        print(f"STATISTICAL P-VALUE:      {self.results_summary['p_value']:.4f}")
        print("----------------------------")

        return None
    
    def _plot_results(self) -> None:

        # Plot Permutations
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 7))
        for label, curve in self.permutation_equity_curves.items():
            plt.plot(curve, color='gray', alpha=0.2, linewidth=0.8)
        plt.plot(self.base_equity_curve, color='gold', linewidth=2.5, label=f'Original Strategy (P={self.results_summary["p_value"]:.4f})')
        plt.title(f"Permutation Test: Strategy vs. 100 Random Targets")
        plt.xlabel("Number of Trades (Sequential)")
        plt.ylabel("Equity (USD)")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()
        plt.savefig(f"{self.save_directory}/results/permutation_equities.png")

        return None
    
    def return_results(self) -> None:

        # Return Results
        self._calculate_statistics()
        self._print_results()
        self._plot_results()

        return None

class PermutationAnalysis:
    def __init__(self, config:Dict[str,Any], model_parameters:Dict[str,Any]):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.walk_forward:WalkForwardAnalysis = WalkForwardAnalysis(config, model_parameters)
        self.results:PermutationResults = PermutationResults(self.config['save_directory'])
        self.df = pd.read_csv(self.config['df_filepath'], index_col=0, parse_dates=True)

    def _randomise_dataset(self) -> pd.DataFrame:
        randomised_df = self.df.copy()

        # Randomise Target (y) within Dataset
        randomised_df['target'] = np.random.permutation(randomised_df['target'].values)

        return randomised_df

    def _permutation_engine(self) -> None:

        # Permutation Engine
        print("STARTING PERMUTATION ANALYSIS:")
        print("-----------------------------")

        # Baseline Walk-Forward
        print("RUNNING BASELINE WALKFORWARD...")
        trade_posistions, probability_variance, equity_curve = self.walk_forward.run_walkforward_analysis(save=True)
        self.results.add_base_results(trade_posistions, probability_variance, equity_curve)
        permutation_runs = self.config['permutation_runs']
        print("-------------------------------")

        # Indexing Identification
        print("RUNNING PERMUTATION...")
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        training_periods = self.config['training_periods']
        testing_periods = self.config['testing_periods']
        walk_forward_periods = self.config['walkforward_shift_periods']
        first_train_start = start
        first_train_end = first_train_start + pd.DateOffset(months=self.config['training_periods'])
        first_test_start = first_train_end
        first_test_end = first_test_start + pd.DateOffset(months=self.config['testing_periods'])
        total_months = (end.year - start.year) * 12 + (end.month - start.month)
        min_months_needed = training_periods + testing_periods
        if total_months < min_months_needed:
            print(f"ERROR: minimum months required not reached: got ({total_months}), need ({min_months_needed})")
            return None

        # Permutations
        for i in range(1, permutation_runs+1):

            # Initialize variables
            portfolio_manager = BacktestPortfolioManager(self.config['starting_equity'])
            normal_df = self.df.copy()
            randomised_df = self._randomise_dataset()
            non_feature_columns = ['open', 'high', 'low', 'close', 'volume', 'target']
            feature_columns = [col for col in normal_df.columns if col not in non_feature_columns]
            target = 'target'

            # Run Walk-Forward Iteration Logic
            curr_period = 0
            curr_shift = 0
            while True:
                curr_period += 1
                walk_train_start = first_train_start + pd.DateOffset(months=curr_shift)
                walk_train_end = walk_train_start + pd.DateOffset(months=training_periods)
                walk_test_start = walk_train_end + pd.DateOffset(minutes=15)
                walk_test_end = walk_train_end + pd.DateOffset(months=testing_periods)
                if walk_test_end > end:
                    break

                # Prepare Datasets
                training = randomised_df[walk_train_start:walk_train_end]
                if training.empty: break
                X = training[feature_columns].values
                y = training[target].values
                data_len = int(len(X) * 0.8)
                X_train, y_train = X[:data_len], y[:data_len].flatten()
                X_val, y_val = X[data_len:], y[data_len:].flatten()
                testing = normal_df[walk_test_start:walk_test_end]

                # Training
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    **self.model_parameters, # Use the dict directly
                    n_jobs=4,
                    verbosity=0
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )

                #  Testing
                pending_entry = None
                current_position = None
                for index, row in testing.iterrows():
                    current_close = row['close']
                    current_open = row['open']
                    
                    # Entry if Signal
                    if pending_entry:
                        current_position = pending_entry
                        current_position.datetime = str(index)
                        current_position.entry_price = current_open
                        pending_entry = None
                    
                    # Exit if In Trade
                    if current_position:
                        current_position.exit_price = current_close
                        if current_position.direction == "UP" and current_position.entry_price <= current_close:
                            current_position.returns = 1
                        elif current_position.direction == "DOWN" and current_position.entry_price >= current_close:
                            current_position.returns = 1
                        else:
                            current_position.returns = -1
                        portfolio_manager.add_trade_position(copy.copy(current_position))
                        current_position = None
                    
                    # Generate Probability Value
                    features = row[feature_columns].values.reshape(1, -1)
                    y_pred_proba = model.predict_proba(features)[0, 1]
                    
                    # Pending Order Generation
                    if y_pred_proba >= self.config['upper_threshold']:
                        pending_entry = TradePosition(
                            datetime=None,
                            pred_proba=y_pred_proba,
                            direction="UP",
                            entry_price=None,
                            exit_price=None,
                            returns=None
                        )
                    elif y_pred_proba <= self.config['lower_threshold']:
                        pending_entry = TradePosition(
                            datetime=None,
                            pred_proba=y_pred_proba,
                            direction="DOWN",
                            entry_price=None,
                            exit_price=None,
                            returns=None
                        )

                curr_shift += walk_forward_periods

            # Append Walk-Farword Permutation Results
            trade_positions, prob_var, equity = portfolio_manager.return_portfolio_results()
            self.results.add_permutation_run(i, trade_positions, prob_var, equity)
            print(f"COMPLETED: {i}/{permutation_runs}")

        # Results
        self.results.return_results()
        print("----------------------")

        return None
    
    def run_permutation_analysis(self) -> None:

        # Run Full Permutation Analysis
        self._permutation_engine()

        return None

def main() -> None:

    config = {
        'df_filepath': 'analysis/data/btcusd_1h_testing.csv',
        'save_directory': 'analysis',
        'start': '2025-04-01',
        'end': '2025-12-31',
        'starting_equity': 0,
        'training_periods': 0,
        'testing_periods': 0,
        'walkforward_shift_periods': 0,
        'upper_threshold': 1.00,
        'lower_threshold': 0.00,
        'permutation_runs': 100
    }
    model_params = {
        'n_estimators': 0,
        'learning_rate': 0.0,
        'max_depth': 0,
        'min_child_weight': 0,
        'gamma': 0.0,
        'subsample': 0.0,
        'colsample_bytree': 0.0,
        'random_state': 0
    }
    permutation_engine = PermutationAnalysis(config, model_params)
    permutation_engine.run_permutation_analysis()

    return None

if __name__ == "__main__":
    main()