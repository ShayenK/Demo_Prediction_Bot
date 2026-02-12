import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FinalChecks:
    def __init__(self, training_filepath:str, testing_filepath:str):
        self.training_df:pd.DataFrame = pd.read_csv(training_filepath)
        self.testing_df:pd.DataFrame = pd.read_csv(testing_filepath)

    def final_checks(self) -> None:
        print("FINAL CHECKS")
        print("------------")
        training_df = self.training_df.copy()
        testing_df = self.testing_df.copy()

        # Final Checks
        datasets = [training_df, testing_df]
        names = ['Training', 'Testing']
        for num, set in enumerate(datasets):
            inf_count = np.isinf(set.select_dtypes(include=np.number)).values.sum()
            print(f"\n{names[num]} ranges: \n")
            print(f"Start date: {set['datetime'].iloc[0]}")
            print(f"End date: {set['datetime'].iloc[-1]}")
            print(f"Infinite values found: {inf_count}")
        print()
        print(training_df)
        print(testing_df)

        return None
    
    def visualisation(self) -> None:
        print("VISUALISING ALL DATASETS")
        print("------------------------")
        training_df = self.training_df.copy()
        testing_df = self.testing_df.copy()

        # Visualise Datasets
        datasets = [training_df, testing_df]
        names = ['Training', 'Testing']
        for num, set in enumerate(datasets):
            plt.title(f"{names[num]} Dataset Visualisation")
            plt.plot(set['close'])
            plt.tight_layout()
            plt.show()

        return None
    
def main():
    checking = FinalChecks(
        'analysis/data/btcusd_1h_training.csv',
        'analysis/data/btcusd_1h_testing.csv'
    )

    # 1. FINAL CHECKS
    checking.final_checks()

    # 2. DATASET VISUAL ANALYSIS
    checking.visualisation()

if __name__ == "__main__":
    main()