# generate_dataset.py
# Generates a sample advertising dataset if advertising.csv is not available

import pandas as pd
import numpy as np

def generate_advertising_data(n=200, seed=42):
    np.random.seed(seed)
    TV        = np.random.uniform(0.7, 296.4, n)
    Radio     = np.random.uniform(0.0, 49.6, n)
    Newspaper = np.random.uniform(0.3, 114.0, n)
    Sales     = (0.055 * TV + 0.12 * Radio + 0.004 * Newspaper
                 + np.random.normal(0, 1.2, n))
    Sales     = np.clip(Sales, 1.6, 27.0).round(1)

    df = pd.DataFrame({
        "TV":        TV.round(1),
        "Radio":     Radio.round(1),
        "Newspaper": Newspaper.round(1),
        "Sales":     Sales
    })
    return df

if __name__ == "__main__":
    df = generate_advertising_data()
    df.to_csv("advertising.csv", index=False)
    print(f"Dataset generated: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df.head())
