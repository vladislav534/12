# plotter.py

import os
import plotly.graph_objects as go
from config import REPORT_DIR, TIMEFRAME_LABEL, VERBOSE

def build_candlestick_figure(df, symbol: str) -> go.Figure:
    # убедимся, что сортируем и сбрасываем индекс
    df = df.sort_values('open_time').reset_index(drop=True)

    # DEBUG: сколько строк, какие колонки и первые/последние timestamps
    print(f"[build] {symbol}: rows={len(df)}, cols={list(df.columns)}")
    if len(df) > 0:
        print(f"[build] {symbol} head:\n", df[['open_time','open','high','low','close']].head(2))
        print(f"[build] {symbol} tail:\n", df[['open_time','open','high','low','close']].tail(2))
    else:
        print(f"[build] {symbol}: DataFrame пуст, рисовать нечего")

    fig = go.Figure()
    if len(df) >= 2:
        # стандартные свечи
        fig.add_trace(go.Candlestick(
            x=df['open_time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        ))
    else:
        # если данных мало, хотя бы точечный график для проверки
        fig.add_trace(go.Scatter(
            x=df['open_time'],
            y=df['close'],
            mode='markers+lines',
            name=f"{symbol} (points)"
        ))

    fig.update_layout(
        title=f"{symbol} — {TIMEFRAME_LABEL}",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark"
    )
    return fig

def generate_report(data_map: dict, out_filename: str = "report.html") -> str:
    # DEBUG: какие символы и сколько строк у каждого
    print("=== generate_report: symbols & row counts ===")
    for sym, df in data_map.items():
        rows = len(df) if df is not None else 0
        print(f"  {sym}: rows={rows}")

    html_parts = []
    for sym, df in data_map.items():
        if df is None or df.empty:
            # можно выводить предупреждение
            print(f"[report] пропускаем {sym}: пустой DataFrame")
            continue

        fig = build_candlestick_figure(df.copy(), sym)
        div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_parts.append(f"<h2>{sym}</h2>\n{div}\n<hr/>\n")

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<title>Candlestick Report</title>
</head>
<body>
<h1>Candlestick Report</h1>
{''.join(html_parts)}
</body>
</html>"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    if VERBOSE:
        print(f"Report generated at {out_path}")
    return out_path
