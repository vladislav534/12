# config.py
# Глобальные переменные и зависимости — изменяйте значения здесь

# Количество активов по объему (топ N)
TOP_N = 100
# config.py
FILTER_QUOTE = "USDT"  # None или "" для отключения фильтра
# config.py

# config.py (добавить)
TERMINAL_STYLE = True          # True -> интерфейс как в биржевом терминале
INDICATORS = {
    "SMA": [20, 50],
    "EMA": [20, 50],
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "RSI": {"period": 14}
}
RANGE_SELECTOR = True          # показать range selector (1d,1w,1m,all)
SHOW_VOLUME_SUBPLOT = True     # объём вынесен в нижний subplot как на терминале
CANDLE_COLORS = {"increasing": "#1f77b4", "decreasing": "#ff7f0e"}

# Период исторических данных
# Формат: number + unit, unit in 'm' (minutes), 'h' (hours), 'd' (days)
HIST_PERIOD = "365d"   # пример: "1d", "7d", "30d"

# Интервал свечей для исторических данных (Binance kline interval)
# Примеры: '1m', '5m', '15m', '1h', '4h', '1d'
KLINE_INTERVAL = "1h"

# Таймфрейм в читаемом виде (для вывода)
TIMEFRAME_LABEL = f"{HIST_PERIOD} @ {KLINE_INTERVAL}"

# Список символов для явного сбора, если пуст — берём автоматически топ по объёму
MANUAL_SYMBOLS = []  # e.g. ["BTCUSDT","ETHUSDT"]

# Работа с API: публичные endpoints Binance
BINANCE_API_BASE = "https://api.binance.com"

# Папки для хранения данных и отчётов
DATA_DIR = "data"
REPORT_DIR = "reports"

# Формат сохранения: 'csv' или 'parquet'
SAVE_FORMAT = "parquet"

# Ограничение запросов (в миллисекундах) между вызовами к API, чтобы не бить rate limit
REQUEST_DELAY_SEC = 0.15

# Таймаут HTTP-запросов
HTTP_TIMEOUT = 15

# Максимальное число символов в отчёте (контроль)
MAX_SYMBOLS_TO_PLOT = 200

# Лог-уровень простой флаг
VERBOSE = True
