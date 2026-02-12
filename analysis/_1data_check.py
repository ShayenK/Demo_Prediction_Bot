import pandas as pd
import numpy as np
import mplfinance as mpf

class DataCheck:
    def __init__(self, filepath:str):
        self.df:pd.DataFrame = pd.read_csv(filepath, skipinitialspace=True)

    def check_data(self) -> None:
        print("CHECKING DATASET")
        print("----------------")
        df = self.df.copy()

        # Check Dataset
        print(df)
        missing_stats = pd.DataFrame({
            'Missing Count': df.isna().sum(),
            'Missing %': round(df.isna().mean() * 100, 2),
            'Total Rows': df.shape[0],
            'Dtype': df.dtypes
        }).sort_values('Missing Count', ascending=False)
        print("\nMissing Data Overview:")
        print(missing_stats[missing_stats['Missing Count'] > 0])

        return None

    def visualize(self, start:str, end:str) -> None:
        print("VISUALISING DATASET")
        print("-------------------")
        df = self.df.copy()

        # Visualise Dataset
        df['Timestamp'] = pd.to_datetime(df['Open time'])
        df = df.set_index('Timestamp')
        df = df[(df.index >= start) & (df.index <= end)]
        ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        mpf.plot(
            ohlcv,
            title='Random Sample of Data Checking for Faulty Data',
            volume=True,
            style='yahoo',
            type='candle'
        )

        return None

def main() -> None:
    clean = DataCheck('analysis/data/btcusd_1h.csv')

    # 1. CHECK THE DATASET FIRST
    clean.check_data()

    # 2. VISUALISE DATASET
    clean.visualize('2026-01-01', '2026-02-28')
    return None

if __name__ == "__main__":
    main()