import copy
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from _5backtest import TradePosition, BacktestEngine
from warnings import filterwarnings

filterwarnings('ignore')
np.random.seed(21)

class WalkForwardPortfolioManager:
    def __init__(self, starting_equity:float):
        self.all_trade_positions:List[TradePosition] = []
        self.all_probability_variances:List[float] = []
        self.full_equity_curve:List[float] = [starting_equity]

    def add_monthly_metrics(self, trade_positions:List[TradePosition], probability_variance:List[float], equity_curve:List[float]) -> None:
        
        # Add Single Backtest Instance to Portfolio Manager
        self.all_trade_positions.extend(trade_positions)
        self.all_probability_variances.extend(probability_variance)
        last_equity = self.full_equity_curve[-1]
        new_equity_curve = [value + last_equity for value in equity_curve]
        self.full_equity_curve.extend(new_equity_curve)
        
        return None
    
    def return_all_curves(self) -> Tuple[List[TradePosition],List[float],List[float]]:
        
        # Copy All Metrics
        all_trade_positions = copy.copy(self.all_trade_positions)
        all_probability_variances = copy.copy(self.all_probability_variances)
        full_equity_curve = copy.copy(self.full_equity_curve)

        return all_trade_positions, all_probability_variances, full_equity_curve
    
class WalkForwardResults:
    def __init__(self, save_directory:str):
        self.save_directory:str = save_directory
        self.results_map:Dict[str,Any] = {}
        self.all_trade_positions:List[TradePosition] = []
        self.all_probability_variances:List[float] = []
        self.full_equity_curve:List[float] = []

    def _calculate_statistics(self) -> None:

        # Calculate Statistics
        total_trade_returns = [trade.returns for trade in self.all_trade_positions]
        total_win_returns = [r for r in total_trade_returns if r > 0]
        total_loss_returns = [r for r in total_trade_returns if r < 0]
        total_trades = len(total_trade_returns)
        total_wins = len(total_win_returns)
        winrate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        total_returns = ((self.full_equity_curve[-1] - self.full_equity_curve[0]) / self.full_equity_curve[0]) * 100
        
        win_streaks, loss_streaks = [], []
        curr_win, curr_loss = 0, 0
        for ret in total_trade_returns:
            if ret > 0:
                curr_win += 1
                if curr_loss > 0: loss_streaks.append(curr_loss)
                curr_loss = 0
            elif ret < 0:
                curr_loss += 1
                if curr_win > 0: win_streaks.append(curr_win)
                curr_win = 0
        if curr_win > 0: win_streaks.append(curr_win)
        if curr_loss > 0: loss_streaks.append(curr_loss)
        
        peak = self.full_equity_curve[0]
        drawdowns = []
        for val in self.full_equity_curve:
            if val > peak: peak = val
            drawdown = (peak - val) / peak
            drawdowns.append(drawdown * 100)
        max_drawdown = np.max(drawdowns)
        avg_drawdown = np.mean(drawdowns)
        
        avg_win = np.mean(total_win_returns) if total_win_returns else 0
        avg_loss = abs(np.mean(total_loss_returns)) if total_loss_returns else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        w_decimal = winrate / 100
        kelly = w_decimal - ((1 - w_decimal) / win_loss_ratio) if win_loss_ratio != 0 else 0
        full_kelly = max(0, kelly * 100) 

        # Results Mapping
        self.results_map = {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'win_rate': winrate,
            'total_returns': total_returns,
            'max_consec_wins': np.max(win_streaks) if win_streaks else 0,
            'max_consec_losses': np.max(loss_streaks) if loss_streaks else 0,
            'avg_consec_wins': np.mean(win_streaks) if win_streaks else 0,
            'avg_consec_losses': np.mean(loss_streaks) if loss_streaks else 0,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'full_kelly': full_kelly,
            'quarter_kelly': full_kelly / 4
        }

        return None

    def _print_results(self) -> None:
        
        # Print Results
        print("BACKTEST RESULTS")
        print("----------------")
        print(f"TOTAL TRADES: {self.results_map['total_trades']}")
        print(f"TOTAL WINS: {self.results_map['total_wins']}")
        print(f"WIN RATE: {self.results_map['win_rate']:.2f}%")
        print(f"TOTAL RETURNS: {self.results_map['total_returns']:.2f}%")
        print(f"MAX CONSEC. WINS: {self.results_map['max_consec_wins']}")
        print(f"MAX CONSEC. LOSSES: {self.results_map['max_consec_losses']}")
        print(f"MAX DRAWDOWN: {self.results_map['max_drawdown']:.2f}%")
        print(f"AVG CONSEC. WINS: {self.results_map['avg_consec_wins']:.2f}")
        print(f"AVG CONSEC LOSSES: {self.results_map['avg_consec_losses']:.2f}")
        print(f"AVG DRAWDOWN: {self.results_map['avg_drawdown']:.2f}%")
        print(f"FULL KELLY FRACTION: {self.results_map['full_kelly']:.2f}%")
        print(f"1/4 KELLY FRACTION: {self.results_map['quarter_kelly']:.2f}%")
        print("----------------")

        return None

    def _plot_results(self) -> None:

        # Plot Results
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=False)
        fig.subplots_adjust(hspace=0.4)

        ax1.plot(self.full_equity_curve, color='royalblue', label='Equity')
        ax1.set_title("Strategy Equity Curve")
        ax1.set_ylabel("Price (USD)")
        ax2.set_xlabel("Number of Trades (n)")
        ax1.legend()
        extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{self.save_directory}/results/walkforward_equity.png", bbox_inches=extent1.expanded(1.1, 1.2))

        data = np.array(self.all_probability_variances)
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        q1, q3 = np.percentile(data, [25, 75])
        ax2.hist(data, bins=50, color='royalblue', alpha=0.7, density=True)
        ax2.axvline(mean_val, color='salmon', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax2.axvline(median_val, color='gold', linestyle='-', linewidth=2, label=f'Median: {median_val:.4f}')
        ax2.axvspan(q1, q3, color='gray', alpha=0.2, label='IQR (Q1-Q3)')
        stats_text = f"Std Dev: {std_val:.4f}"
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        ax2.set_title("Probability Conviction Distribution (Variance from 0.5)")
        ax2.set_xlabel("Abs(Proba - 0.5)")
        ax2.set_ylabel("Density")
        ax2.legend()
        extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{self.save_directory}/results/walkforward_probability_variances.png", bbox_inches=extent2.expanded(1.1, 1.2))

        plt.tight_layout()
        plt.show()

        return None
    
    def return_results(self, trade_positions:List[TradePosition], probability_variance:List[float], equity_curve:List[float]) -> None:

        # Copy Metrics
        self.all_trade_positions = copy.copy(trade_positions)
        self.all_probability_variances = copy.copy(probability_variance)
        self.full_equity_curve = copy.copy(equity_curve)

        # Return Results
        self._calculate_statistics()
        self._print_results()
        self._plot_results()

        return None
    
class WalkForwardModelEvaluation:
    def __init__(self, save_directory:str):
        self.save_directory:str = save_directory
        self.model_results:Dict[str,Any] = {}
        self.all_models:List[xgb.XGBClassifier] = []
        self.feature_names:List[str] = []

    def add_model(self, model:Optional[xgb.XGBClassifier]=None, feature_names:Optional[List[str]]=None) -> None:

        # Add Model to List
        if model:
            self.all_models.append(model)
        if feature_names:
            self.feature_names = feature_names

        return None

    def _calculate_model_statistics(self) -> None:

        # Calculate Model Statistics
        fold_importances = []
        for model in self.all_models:
            scores = model.get_booster().get_score(importance_type='gain')
            row = [scores.get(f"f{i}", scores.get(name, 0)) for i, name in enumerate(self.feature_names)]
            fold_importances.append(row)
        df_imp = pd.DataFrame(fold_importances, columns=self.feature_names)
        drift = df_imp.T.corrwith(df_imp.T.shift(1)).dropna()
        self.model_results = {
            'importance_df': df_imp,
            'avg_importance': df_imp.mean().sort_values(ascending=False),
            'importance_std': df_imp.std(),
            'model_drift': drift
        }

        return None
    
    def _print_model_statistics(self) -> None:

        # Print Model Statistics
        print("MODEL EVALUATION")
        print("----------------")
        print(f"TOTAL MODELS EVALUATED: {len(self.all_models)}")
        print()
        print(f"TOP 20 FEATURES (BY AVG GAIN): ")
        print(f"{self.model_results['avg_importance'].head(20)}")
        print()
        print(f"MODEL DRIFT (AVG INTER-FOLD CORRELATION):")
        print(f"{self.model_results['model_drift'].mean():.4f}")

        return None
    
    def _plot_model_statistics(self) -> None:
    
        # Plot Model Statistics
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
        top_features = self.model_results['avg_importance'].head(20).index
        sns.heatmap(self.model_results['importance_df'][top_features].T, ax=ax1, cmap='viridis')
        ax1.set_title("Feature Importance Evolution (Top 20)")
        ax1.set_xlabel("Walk Forward Fold Index")
        extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{self.save_directory}/results/feature_importance.png", bbox_inches=extent1.expanded(1.1, 1.2))

        ax2.plot(self.model_results['model_drift'], marker='o', color='gold', linestyle='--')
        ax2.set_title("Model Stability (Inter-Fold Correlation)")
        ax2.set_ylabel("Correlation Score")
        ax2.set_ylim(-1,1)
        ax2.axhline(0.7, color='white', alpha=0.3)
        extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"{self.save_directory}/results/model_stability.png", bbox_inches=extent2.expanded(1.1, 1.2))

        plt.tight_layout()
        plt.show()

        return None
    
    def save_models(self, save_directory:str) -> None:
        
        # Save All Models
        joblib.dump(self.feature_names, f"{save_directory}/models/model_features.pkl")
        for i, model in enumerate(self.all_models):
            joblib.dump(model, f"{save_directory}/models/model_{i}.pkl")
        
        return None

    def model_evaluation(self) -> None:
        
        # Return Results
        self._calculate_model_statistics()
        self._print_model_statistics()
        self._plot_model_statistics()

        return None

class WalkForwardAnalysis:
    def __init__(self, config:Dict[str,Any], model_parameters:Dict[str,Any]):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.portfolio_manager:WalkForwardPortfolioManager = WalkForwardPortfolioManager(
            self.config['starting_equity']
        )
        self.results:WalkForwardResults = WalkForwardResults(
            self.config['save_directory']
        )
        self.model_evaluation:WalkForwardModelEvaluation = WalkForwardModelEvaluation(
            self.config['save_directory']
        )
        self.df = pd.read_csv(config['df_filepath'], index_col=0, parse_dates=True)

    def _prepare_dataset(self) -> None:

        # Prepare Dataset
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        self.df = self.df[(self.df.index >= start) & (self.df.index <= end)]

        return None

    def _walkforward_engine(self, save:Optional[bool]=False) -> Tuple[List[TradePosition],List[float],List[float]]:
        df = self.df.copy()

        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        training_periods = self.config['training_periods']
        testing_periods = self.config['testing_periods']
        walk_forward_periods = self.config['walkforward_shift_periods']

        # Walk-Forward Engine
        first_train_start = start
        first_train_end = first_train_start + pd.DateOffset(months=self.config['training_periods'])
        first_test_start = first_train_end
        first_test_end = first_test_start + pd.DateOffset(months=self.config['testing_periods'])
        total_months = (end.year - start.year) * 12 + (end.month - start.month)
        min_months_needed = training_periods + testing_periods
        if total_months < min_months_needed:
            print(f"ERROR: minimum months required not reached: got ({total_months}), need ({min_months_needed})")
            return None
        available_months_per_test = total_months - training_periods - testing_periods + 1
        num_walks = (available_months_per_test // walk_forward_periods) + 1
        print("STARTING WALK-FORWARD ANALYSIS:")

        curr_walks = 0
        curr_shift = 0
        while True:
            curr_walks += 1
            walk_train_start = first_train_start + pd.DateOffset(months=curr_shift)
            walk_train_end = walk_train_start + pd.DateOffset(months=training_periods)
            walk_test_start = walk_train_end + pd.DateOffset(minutes=15)
            walk_test_end = walk_train_end + pd.DateOffset(months=testing_periods)
            if walk_test_end > end:
                print(f"INFO: stopping walks num. {curr_walks} | walks exceeded range")
                break

            # Walk-Forward Block
            train_df = df[walk_train_start:walk_train_end]
            test_df = df[walk_test_start:walk_test_end]
            trade_positions, prob_var, equity, model, f_names = BacktestEngine._run_backtest_alt(
                train_df, test_df, self.config, self.model_parameters
            )
            self.portfolio_manager.add_monthly_metrics(trade_positions, prob_var, equity)
            self.model_evaluation.add_model(model=model)
            if not self.model_evaluation.feature_names:
                self.model_evaluation.add_model(feature_names=f_names)

            # Basic Progress Tracking
            print(f"COMPLETED: {curr_walks}/{num_walks}")

            curr_shift += walk_forward_periods

        all_trade_positions, all_probability_variances, full_equity_curve = self.portfolio_manager.return_all_curves()
        self.results.return_results(all_trade_positions, all_probability_variances, full_equity_curve)
        self.model_evaluation.model_evaluation()
        if save:
            self.model_evaluation.save_models(self.config['save_directory'])

        return all_trade_positions, all_probability_variances, full_equity_curve

    def run_walkforward_analysis(self, save:Optional[bool]=False) -> Tuple[List[TradePosition],List[float],List[float]]:

        # Run Walkforward Instance
        self._prepare_dataset()
        all_trade_positions, all_probability_variances, full_equity_curve = self._walkforward_engine(save)

        return all_trade_positions, all_probability_variances, full_equity_curve

def main() -> None:

    # Full Walk-Forward Analysis
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
        'lower_threshold': 0.00
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
    walk_forward_engine = WalkForwardAnalysis(config, model_params)
    walk_forward_engine.run_walkforward_analysis(save=True)

    return None

if __name__ == "__main__":
    main()