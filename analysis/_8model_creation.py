import copy
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from _5backtest import BacktestEngine
from warnings import filterwarnings
filterwarnings('ignore')

class LiveProduction:
    def __init__(self, config:Dict[str,Any], model_parameters:Dict[str,Any]):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.df:pd.DataFrame = pd.read_csv(config['df_filepath'], index_col=0, parse_dates=True)
        self.X_train:Optional[np.ndarray] = None
        self.y_train:Optional[np.ndarray] = None
        self.X_validation:Optional[np.ndarray] = None
        self.y_validation:Optional[np.ndarray] = None

    def _prepare_dataset(self) -> None:
        print("PREPARING DATASET")
        print("-----------------")

        # Prepare Training Dataset
        df = self.df.copy()
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        df = df[(df.index >= start) & (df.index <= end)]

        # Train-Validation Split
        split = int(len(df) * 0.8)
        train = df[:split]
        validation = df[split:]
        non_feature_columns = ['open', 'high', 'low', 'close', 'volume', 'target']
        feature_columns = [col for col in train.columns if col not in non_feature_columns]
        target = 'target'
        self.X_train = train[feature_columns].values
        self.y_train = train[target].values.flatten()
        self.X_validation = validation[feature_columns].values
        self.y_validation = validation[target].values.flatten()

        return None

    def _create_model(self) -> None:
        print("TRAINING MODEL")
        print("--------------")

        # Train Model 
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            **self.model_parameters,
            n_jobs=4,
            verbosity=0
        )
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_validation, self.y_validation)],
            verbose=False
        )
        self.model = model

        return None

    def _basic_model_testing(self) -> None:
        print("BASIC MODEL TESTING")
        print("-------------------")

        # Basic probability evaluation
        y_pred_proba = self.model.predict_proba(self.X_validation)[:, 1]
        y_target = self.y_validation
        upper_threshold = self.config['upper_threshold']
        lower_threshold = self.config['lower_threshold']
        upper_indices = np.where(y_pred_proba >= upper_threshold)[0]
        lower_indices = np.where(y_pred_proba <= lower_threshold)[0]
        if len(upper_indices) > 0:
            upper_preds = np.ones(len(upper_indices))
            upper_actuals = y_target[upper_indices]
            upper_acc = (upper_preds == upper_actuals).mean()
            print(f"UPPER THRESHOLD (>= {upper_threshold})")
            print(f"Signals: {len(upper_indices)} | Accuracy: {upper_acc:.4f}")
        else:
            print("No signals found above upper threshold.")
        if len(lower_indices) > 0:
            lower_preds = np.zeros(len(lower_indices))
            lower_actuals = y_target[lower_indices]
            lower_acc = (lower_preds == lower_actuals).mean()
            print(f"LOWER THRESHOLD (<= {lower_threshold})")
            print(f"Signals: {len(lower_indices)} | Accuracy: {lower_acc:.4f}")
        else:
            print("No signals found below lower threshold.")
        
        return None
    
    def create_live_production_model(self, save:Optional[bool]=False) -> None:
        print("LIVE PRODUCTION MODEL")
        print("---------------------")

        # Build Production Model & Save
        self._prepare_dataset()
        self._create_model()
        self._basic_model_testing()
        if save:
            joblib.dump(self.model, filename=f"{self.config['save_directory']}/production/live_model.pkl")

        return None
    
def main() -> None:

    config = {
        'df_filepath': 'analysis/data/btcusd_1h_testing.csv',
        'save_directory': 'analysis',
        'start': '2025-04-01',
        'end': '2025-12-31',
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
    engine = LiveProduction(config, model_params)
    engine.create_live_production_model(save=True)

    return None

if __name__ == "__main__":
    main()