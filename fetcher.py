# fetcher.py

from tqdm import tqdm
from typing import List
from urllib.parse import urljoin
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BINANCE_API_BASE,
    HTTP_TIMEOUT,
    VERBOSE,
    MANUAL_SYMBOLS,
    TOP_N,
    FILTER_QUOTE
)
from utils import kline_to_df, sleep_between_requests

# -------------------------------------------------------------------
def compute_start_ms(period: str) -> int:
    now = datetime.utcnow()
    unit = period[-1]
    try:
        val = int(period[:-1])
    except ValueError:
        raise ValueError(f"Неправильный формат period: {period}")
    if unit == "m":
        start = now - timedelta(minutes=val)
    elif unit == "h":
        start = now - timedelta(hours=val)
    elif unit == "d":
        start = now - timedelta(days=val)
    elif unit == "M":
        start = now - timedelta(days=30 * val)
    elif unit == "y":
        start = now - timedelta(days=365 * val)
    else:
        raise ValueError(f"Неподдерживаемая единица времени в period: {unit}")
    return int(start.timestamp() * 1000)
# -------------------------------------------------------------------

def get_ticker_24h():
    url = urljoin(BINANCE_API_BASE, "/api/v3/ticker/24hr")
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()

def top_symbols_by_volume(top_n: int = TOP_N) -> List[str]:
    if MANUAL_SYMBOLS:
        if VERBOSE:
            print("Using MANUAL_SYMBOLS from config")
        return MANUAL_SYMBOLS

    tickers = get_ticker_24h()
    if FILTER_QUOTE:
        tickers = [t for t in tickers if t.get('symbol','').endswith(FILTER_QUOTE)]

    sorted_by_quote = sorted(
        tickers,
        key=lambda x: float(x.get('quoteVolume', 0) or 0.0),
        reverse=True
    )
    symbols = [t['symbol'] for t in sorted_by_quote[:top_n]]
    if VERBOSE:
        fq_msg = f" {FILTER_QUOTE}" if FILTER_QUOTE else ""
        print(f"Top {len(symbols)}{fq_msg} symbols selected by quoteVolume")
    return symbols

def fetch_klines_for_symbol(symbol: str, interval: str, start_ms: int):
    url = urljoin(BINANCE_API_BASE, "/api/v3/klines")
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "limit": 1000
    }
    all_klines = []
    while True:
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        chunk = r.json()
        if not chunk:
            break
        all_klines.extend(chunk)

        # если собрали меньше лимита — конец
        if len(chunk) < params["limit"]:
            break

        # пагинация по openTime (chunk[i][0])
        last_open = int(chunk[-1][0])
        params["startTime"] = last_open + 1

        sleep_between_requests()

        # safety break
        if len(all_klines) > 200_000:
            if VERBOSE:
                print(f"[fetcher] safety break for {symbol} at {len(all_klines)} klines")
            break

    return all_klines

def fetch_historical(
    symbols: List[str],
    interval: str,
    period: str,
    max_workers: int = 5
):
    """
    Параллельно скачивает исторические клайны для переданных symbols.
    """
    start_ms = compute_start_ms(period)
    if VERBOSE:
        dt_start = datetime.utcfromtimestamp(start_ms/1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[fetcher] fetching from: {dt_start} UTC ({start_ms} ms)")

    results: dict[str, pd.DataFrame] = {}
    futures = {}

    # запускаем пул потоков
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym in symbols:
            futures[executor.submit(fetch_klines_for_symbol, sym, interval, start_ms)] = sym

        # обрабатываем по мере готовности
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Fetching symbols"):
            sym = futures[future]
            try:
                klines = future.result()
                df = kline_to_df(klines)
                results[sym] = df.copy()
                if VERBOSE:
                    print(f"[fetcher] {sym}: got {len(df)} candles")
            except Exception as e:
                if VERBOSE:
                    print(f"[fetcher] Failed {sym}: {e}")

    return results
