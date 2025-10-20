from config import TOP_N, HIST_PERIOD, KLINE_INTERVAL, DATA_DIR, REPORT_DIR, VERBOSE
from utils import ensure_dirs
from fetcher import top_symbols_by_volume, fetch_historical
from datastore import save_all
from plotter import generate_report
from analyzer import generate_multi_period_correlation_report

def main():
    ensure_dirs()

    symbols = top_symbols_by_volume(TOP_N)
    if VERBOSE:
        print(f"Collecting historical klines for {len(symbols)} symbols: first 10 -> {symbols[:10]}")

    results = fetch_historical(symbols, KLINE_INTERVAL, HIST_PERIOD)
    saved = save_all(results)
    if VERBOSE:
        print(f"Saved data for {len(saved)} symbols")

    report_path = generate_report(results, out_filename="candlestick_report.html")
    print(f"Candlestick report created at: {report_path}")

    # ← новый вызов корреляционного отчёта
    corr_path = generate_multi_period_correlation_report(
        periods={"7d":7},  # здесь один период — обычная корреляция за 7 дней
        max_lag=5,
        top_n=20
    )
    print(f"Correlation report (7d) created at: {corr_path}")
    multi_corr = generate_multi_period_correlation_report(
        periods={"7d":7,"30d":30,"90d":90,"365d":365},
        max_lag=5, top_n=20
    )
    print(f"Multi-period correlation report: {multi_corr}")

if __name__ == "__main__":
    main()
