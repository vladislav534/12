# utils.py
import os
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd

from config import DATA_DIR, REPORT_DIR, REQUEST_DELAY_SEC, VERBOSE

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    if VERBOSE:
        print(f"Directories ensured: {DATA_DIR}, {REPORT_DIR}")

def sleep_between_requests():
    time.sleep(REQUEST_DELAY_SEC)

def parse_period_to_start(period_str):
    # period_str examples: "7d", "30d", "1d"
    num = int(period_str[:-1])
    unit = period_str[-1]
    now = dt.datetime.utcnow()
    if unit == 'd':
        start = now - relativedelta(days=num)
    elif unit == 'h':
        start = now - relativedelta(hours=num)
    elif unit == 'm':
        start = now - relativedelta(minutes=num)
    else:
        raise ValueError("Unsupported period unit, use m/h/d")
    return int(start.timestamp() * 1000)  # ms

def kline_to_df(raw_klines):
    cols = ['open_time','open','high','low','close','volume','close_time',
            'quote_asset_volume','trades','taker_base_vol','taker_quote_vol','ignore']
    df = pd.DataFrame(raw_klines, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', errors='coerce')
    numeric_cols = ['open','high','low','close','volume','quote_asset_volume','taker_base_vol','taker_quote_vol']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    return df



def save_dataframe(df, path, fmt='parquet'):
    if fmt == 'parquet':
        df.to_parquet(path, index=False)
    elif fmt == 'csv':
        df.to_csv(path, index=False)
    else:
        raise ValueError("Unsupported format")
    if VERBOSE:
        print(f"Saved {len(df)} rows to {path}")

# utils.py (добавить)
import numpy as np
import pandas as pd

def add_sma(df: pd.DataFrame, window: int, col: str = "close", name: str = None):
    name = name or f"sma{window}"
    df[name] = df[col].rolling(window).mean()
    return df

def add_ema(df: pd.DataFrame, span: int, col: str = "close", name: str = None):
    name = name or f"ema{span}"
    df[name] = df[col].ewm(span=span, adjust=False).mean()
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd - signal_line
    return df

def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close"):
    delta = df[col].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    df[f"rsi_{period}"] = rsi
    return df

def prepare_for_plot(df: pd.DataFrame):
    # гарантируем нужные типы и сортировку
    df = df.sort_values("open_time").reset_index(drop=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
