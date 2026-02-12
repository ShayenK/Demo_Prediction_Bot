import pandas as pd
from typing import Optional

class DataSplit:
    def __init__(self, filepath:str):
        # Read with datetime as index
        self.df:Optional[pd.DataFrame] = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.training_df:Optional[pd.DataFrame] = None
        self.testing_df:Optional[pd.DataFrame] = None

    def split_datasets(self, start:str, end:str, split_point:float) -> None:
        print("SPLITTING DATASET")
        print("-----------------")
        df = self.df.copy()

        # Convert strings to datetime
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # Filter by date range using index
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        len_df = len(df)
        
        if len_df == 0:
            print("ERROR: No data found in the specified date range!")
            print(f"Date range requested: {start} to {end}")
            print(f"Available data range: {self.df.index.min()} to {self.df.index.max()}")
            return None
        
        training_validation_index = int(len_df * split_point)
        training_df = df.iloc[:training_validation_index]
        testing_df = df.iloc[training_validation_index:]
        
        print("\nTotal length df: ", len_df)
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print("\nSplit indicies:\n")
        print(f"Training: {len(training_df)} rows ({training_df.index.min()} to {training_df.index.max()})")
        print(f"Testing: {len(testing_df)} rows ({testing_df.index.min()} to {testing_df.index.max()})")
        print()
        
        self.df = df
        self.training_df = training_df
        self.testing_df = testing_df
        return None
    
    def examine_datasets(self) -> None:
        print("EXAMINING DATASETS")
        print("------------------")
        training_df = self.training_df.copy()
        testing_df = self.testing_df.copy()

        print("\nTraining dataset:\n")
        print(training_df.head())
        print(f"Shape: {training_df.shape}")
        print(f"Date range: {training_df.index.min()} to {training_df.index.max()}")
        print("\nTraining dataset columns:\n",len(training_df.columns)," | ",training_df.columns.tolist())
        print(f"\nTraining dataset NA totals:\n{training_df.isna().sum().sum()}")
        
        # Check target distribution
        if 'target' in training_df.columns:
            target_dist = training_df['target'].value_counts()
            print(f"\nTraining target distribution:\n{target_dist}")
            print(f"Class balance: {target_dist[1] / len(training_df) * 100:.2f}% positive")

        print("\n" + "="*50 + "\n")
        
        print("Testing dataset:\n")
        print(testing_df.head())
        print(f"Shape: {testing_df.shape}")
        print(f"Date range: {testing_df.index.min()} to {testing_df.index.max()}")
        print("\nTesting dataset columns:\n",len(testing_df.columns)," | ",testing_df.columns.tolist())
        print(f"\nTesting dataset NA totals:\n{testing_df.isna().sum().sum()}")
        
        # Check target distribution
        if 'target' in testing_df.columns:
            target_dist = testing_df['target'].value_counts()
            print(f"\nTesting target distribution:\n{target_dist}")
            print(f"Class balance: {target_dist[1] / len(testing_df) * 100:.2f}% positive")

        return None
    
    def output_new_datasets(self, training_output_filepath:str, testing_output_filepath:str) -> None:
        print("\nOUTPUTING DATASETS")
        print("------------------")
        training_df = self.training_df.copy()
        testing_df = self.testing_df.copy()

        training_df.to_csv(training_output_filepath, index=True)
        testing_df.to_csv(testing_output_filepath, index=True)
        print(f"Saved training to: {training_output_filepath}")
        print(f"Saved testing to: {testing_output_filepath}")
        return None

def main() -> None:
    split = DataSplit('analysis/data/btcusd_1h_clean.csv')

    # 1. SPLIT DATASET
    split_point = 0.80
    split.split_datasets(
        '2022-01-01',
        '2026-01-31',
        split_point
    )

    # 2. EXAMINE DATASET
    split.examine_datasets()

    # 3. OUTPUT NEW DATASET
    split.output_new_datasets(
        'analysis/data/btcusd_1h_training.csv',
        'analysis/data/btcusd_1h_testing.csv'
    )

    return None

if __name__ == "__main__":
    main()