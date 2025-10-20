# main.py

from config import TOP_N, HIST_PERIOD, KLINE_INTERVAL, DATA_DIR, REPORT_DIR, VERBOSE
from utils import ensure_dirs
from fetcher import top_symbols_by_volume, fetch_historical
from datastore import save_all
from plotter import generate_report
from analyzer import generate_multi_period_correlation_report

import os
import glob
import shutil

def clear_cache():
    """
    Удаляем все файлы и папки в DATA_DIR и REPORT_DIR,
    чтобы каждый запуск начинался с нуля.
    """
    # очищаем DATA_DIR
    if os.path.isdir(DATA_DIR):
        for fp in glob.glob(os.path.join(DATA_DIR, "*")):
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)

    # очищаем REPORT_DIR
    if os.path.isdir(REPORT_DIR):
        for fp in glob.glob(os.path.join(REPORT_DIR, "*")):
            if os.path.isfile(fp):
                os.remove(fp)
            else:
                shutil.rmtree(fp)

def main():
    # создаём папки data/ и reports/, если их нет
    ensure_dirs()

    # 1) Загружаем список символов по объёму
    symbols = top_symbols_by_volume(TOP_N)
    if VERBOSE:
        print(f"[main] Collecting historical klines for {len(symbols)} symbols (TOP {TOP_N})")

    # 2) Скачиваем историю (fetch_historical) и сохраняем в DATA_DIR
    results = fetch_historical(symbols, KLINE_INTERVAL, HIST_PERIOD)
    saved = save_all(results)
    if VERBOSE:
        print(f"[main] Saved data for {len(saved)} symbols")

    # 3) Генерируем базовый отчёт со свечами
    report_path = generate_report(results, out_filename="candlestick_report.html")
    print(f"[main] Candlestick report created at: {report_path}")

    # 4) Генерируем корреляционный отчёт за 7 дней
    corr_7d = generate_multi_period_correlation_report(
        periods={"7d": 7},
        max_lag=5,
        top_n=TOP_N
    )
    print(f"[main] Correlation report (7d) created at: {corr_7d}")

    # 5) Генерируем мультипериодный корреляционный отчёт
    corr_multi = generate_multi_period_correlation_report(
        periods={"7d": 7, "30d": 30, "90d": 90, "365d": 365},
        max_lag=5,
        top_n=TOP_N
    )
    print(f"[main] Multi-period correlation report created at: {corr_multi}")

if __name__ == "__main__":
    # перед началом работы очищаем старые данные и отчёты
    clear_cache()
    main()
