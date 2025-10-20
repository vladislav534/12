# datastore.py
import os
from typing import Dict
import pandas as pd

from config import DATA_DIR, SAVE_FORMAT, VERBOSE

def save_symbol_data(symbol: str, df: pd.DataFrame):
    safe_name = symbol.replace("/", "_")
    path = os.path.join(DATA_DIR, f"{safe_name}.{ 'parquet' if SAVE_FORMAT=='parquet' else 'csv'}")
    if SAVE_FORMAT == 'parquet':
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    if VERBOSE:
        print(f"Saved {symbol} to {path}")
    return path

def save_all(results: Dict[str, pd.DataFrame]):
    paths = {}
    for sym, df in results.items():
        if df is None or df.empty:
            if VERBOSE:
                print(f"Empty data for {sym}, skipping save")
            continue
        paths[sym] = save_symbol_data(sym, df)
    return paths

def load_symbol_data(path: str):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path, parse_dates=['open_time','close_time'])
