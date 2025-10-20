# diagnostics.py

from config       import TOP_N, HIST_PERIOD, KLINE_INTERVAL
from fetcher      import top_symbols_by_volume, fetch_historical
from plotter      import build_candlestick_figure
import pandas as pd

def run_diagnostics():
    symbols = top_symbols_by_volume(5)
    print("Testing symbols:", symbols)
    # Скачиваем данные за HIST_PERIOD с интервалом KLINE_INTERVAL
    data_map = fetch_historical(symbols, KLINE_INTERVAL, HIST_PERIOD)

    for sym, df in data_map.items():
        print("\n" + "="*40)
        print(f"Symbol: {sym}")
        if df is None or df.empty:
            print("  ⚠️  DataFrame пустой, строк:", 0)
            continue

        # 1) Сводка DataFrame
        print("  DataFrame shape:", df.shape)
        print("  dtypes:\n", df.dtypes[['open_time','open','high','low','close']])
        print("  head:\n", df[['open_time','open','high','low','close']].head(2), sep="")
        print("  tail:\n", df[['open_time','open','high','low','close']].tail(2), sep="")

        # 2) График и инспекция traces
        fig = build_candlestick_figure(df.copy(), sym)
        print("  figure.data count:", len(fig.data))

        for idx, trace in enumerate(fig.data):
            name = getattr(trace, 'name', None) or trace.type
            print(f"    Trace[{idx}] type={trace.type!r} name={name!r}")
            # Общие координаты
            x = getattr(trace, 'x', None)
            y = getattr(trace, 'y', None)
            opens = getattr(trace, 'open', None)
            closes = getattr(trace, 'close', None)

            if x is not None:
                print(f"      x points: {len(x)}  sample:", list(x)[:5])
            if opens is not None:
                print(f"      open points: {len(opens)}  sample:", list(opens)[:5])
            if closes is not None:
                print(f"      close points: {len(closes)}  sample:", list(closes)[:5])
            if y is not None:  # для Scatter
                print(f"      y points: {len(y)}  sample:", list(y)[:5])

    print("\nDiagnostics finished.")

if __name__ == "__main__":
    run_diagnostics()
