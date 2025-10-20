# analyzer.py

import os
import glob
import pandas as pd
import numpy as np           # ← добавить
import warnings              # ← добавить
import plotly.graph_objects as go
from itertools import combinations
from config import DATA_DIR, REPORT_DIR, TIMEFRAME_LABEL, VERBOSE

# отключаем лишние ворнинги от NumPy (divide/invalid)
np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)
import numpy as np
import warnings

np.seterr(divide='ignore', invalid='ignore')
warnings.simplefilter("ignore", category=RuntimeWarning)

def load_all_series() -> dict[str, pd.DataFrame]:
    """
    Загружает из DATA_DIR все .parquet и возвращает {symbol: df},
    где df.index = open_time (DatetimeIndex), df['close'] = float.
    """
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    series_map = {}
    for fp in files:
        sym = os.path.basename(fp).replace(".parquet", "")
        df = pd.read_parquet(fp, columns=["open_time", "close"]).dropna()
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.sort_values("open_time").set_index("open_time")
        series_map[sym] = df[["close"]]
        if VERBOSE:
            print(f"[analyzer] loaded {sym}: {len(df)} rows")
    return series_map




# … остальной ваш импорт …

def pair_lag_corr(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int
) -> tuple[int, float]:
    df = pd.concat([series_a, series_b], axis=1, join="inner", keys=["a", "b"])
    if df["a"].std() == 0 or df["b"].std() == 0:
        return 0, 0.0

    best_lag = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        best_corr = df["a"].corr(df["b"])
    if not np.isfinite(best_corr):
        best_corr = 0.0

    for lag in range(1, max_lag + 1):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = df["a"].corr(df["b"].shift(lag))
        if not np.isfinite(c):
            continue
        if abs(c) > abs(best_corr):
            best_corr, best_lag = c, lag

    return best_lag, float(best_corr)





from plotly.subplots import make_subplots

def build_overlay_figure(df1: pd.DataFrame,
                         df2: pd.DataFrame,
                         s1: str,
                         s2: str) -> go.Figure:
    # делаем inner-join по времени, чтобы X-оси совпадали
    df = pd.concat([df1["close"], df2["close"]],
                   axis=1, join="inner", keys=[s1, s2]).dropna()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[s1],
            mode="lines", name=s1,
            line=dict(width=1, color="cyan")
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[s2],
            mode="lines", name=s2,
            line=dict(width=1, dash="dot", color="magenta")
        ),
        secondary_y=True
    )

    fig.update_layout(
        title=f"{s1} vs {s2} — {TIMEFRAME_LABEL}",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(t=50, b=40, l=60, r=60),
        height=350
    )
    fig.update_xaxes(title_text="Time", rangeslider_visible=False)
    fig.update_yaxes(title_text=f"{s1} price", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text=f"{s2} price", secondary_y=True, showgrid=False)

    return fig



# в файле analyzer.py замените функцию generate_multi_period_correlation_report
# на этот “простой” вариант без фильтрации по числу баров

def generate_multi_period_correlation_report(
    periods: dict[str,int] = {"7d":7, "30d":30, "90d":90, "365d":365},
    max_lag: int = 5,
    top_n: int = 20
) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    data_full = load_all_series()

    head = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css"/>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/js/bootstrap.bundle.min.js"></script>
<meta charset="utf-8">
<title>Multi-Period Correlation Report</title>
"""

    # строим вкладки для каждого периода
    period_tabs = []
    period_contents = []
    for idx, (label, days) in enumerate(periods.items()):
        active = "active" if idx == 0 else ""
        show   = "show active" if idx == 0 else ""
        period_tabs.append(
            f'<li class="nav-item">'
            f'<a class="nav-link {active}" id="tab-{label}" '
            f'data-toggle="tab" href="#content-{label}">{label}</a>'
            f'</li>'
        )
        period_contents.append(
            f'<div class="tab-pane fade {show}" id="content-{label}">'
            f'{{content_{label}}}</div>'
        )

    inner_nav_tpl = """
  <ul class="nav nav-pills my-3">
    <li class="nav-item">
      <a class="nav-link active" id="{pfx}-norm-tab"
         data-toggle="pill" href="#{pfx}-norm">Обычная</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" id="{pfx}-lag-tab"
         data-toggle="pill" href="#{pfx}-lag">С лагом</a>
    </li>
  </ul>
  <div class="tab-content">
    <div class="tab-pane fade show active" id="{pfx}-norm">{norm_html}</div>
    <div class="tab-pane fade"          id="{pfx}-lag" >{lag_html}</div>
  </div>
"""

    # простой render_for_period без фильтрации по минимальному числу баров
    def render_for_period(label: str, days: int) -> str:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)

        # для каждого symbol берём все свечи после cutoff
        sliced = {
            sym: df[df.index >= cutoff]
            for sym, df in data_full.items()
            if not df[df.index >= cutoff].empty
        }

        recs_norm = []
        recs_lag  = []
        for a, b in combinations(sliced.keys(), 2):
            df_ab = pd.concat(
                [sliced[a][["close"]], sliced[b][["close"]]],
                axis=1, join="inner", keys=["a","b"]
            ).dropna()

            series_a = df_ab["a"]["close"]
            series_b = df_ab["b"]["close"]

            c_norm = series_a.corr(series_b)
            lg, c_lag = pair_lag_corr(series_a, series_b, max_lag)

            recs_norm.append((a, b, c_norm))
            recs_lag .append((a, b, lg,   c_lag))

        # отбираем Top N по прямой корреляции
        df_norm = (
            pd.DataFrame(recs_norm, columns=["sym1","sym2","corr"])
              .sort_values("corr", ascending=False)
              .head(top_n)
        )
        # и Top N по лаг-корреляции
        df_lag = (
            pd.DataFrame(recs_lag, columns=["sym1","sym2","lag","corr"])
              .sort_values("lag", ascending=False)
              .head(top_n)
        )

        parts_norm = []
        for _, row in df_norm.iterrows():
            s1, s2, cv = row["sym1"], row["sym2"], row["corr"]

            df_ab = pd.concat(
                [sliced[s1][["close"]], sliced[s2][["close"]]],
                axis=1, join="inner", keys=[s1, s2]
            ).dropna()
            df1 = pd.DataFrame({"close": df_ab[s1]}, index=df_ab.index)
            df2 = pd.DataFrame({"close": df_ab[s2]}, index=df_ab.index)

            div = build_overlay_figure(df1, df2, s1, s2) \
                .to_html(full_html=False, include_plotlyjs="cdn")
            parts_norm.append(f"<h5>{s1}–{s2} (corr={cv:.2f})</h5>{div}")

        parts_lag = []
        for _, row in df_lag.iterrows():
            s1, s2, lag_val, cv = row["sym1"], row["sym2"], row["lag"], row["corr"]

            df_ab = pd.concat(
                [sliced[s1][["close"]], sliced[s2][["close"]]],
                axis=1, join="inner", keys=[s1, s2]
            ).dropna()
            df1 = pd.DataFrame({"close": df_ab[s1]}, index=df_ab.index)
            df2 = pd.DataFrame({"close": df_ab[s2]}, index=df_ab.index)

            div = build_overlay_figure(df1, df2, s1, s2) \
                .to_html(full_html=False, include_plotlyjs="cdn")
            parts_lag.append(f"<h5>{s1}–{s2} (lag={lag_val}, corr={cv:.2f})</h5>{div}")


        return inner_nav_tpl.format(
            pfx       = label.replace("d",""),
            norm_html = "".join(parts_norm),
            lag_html  = "".join(parts_lag)
        )

    # собираем контент каждого таба
    contents = {}
    for label, days in periods.items():
        contents[f"content_{label}"] = render_for_period(label, days)

    # финальный шаблон страницы
    html = (
        "<!DOCTYPE html><html><head>" + head + "</head><body>"
        "<div class='container'><h1 class='my-4'>Multi-Period Correlation Report</h1>"
        "<ul class='nav nav-tabs'>" + "".join(period_tabs) + "</ul>"
        "<div class='tab-content'>"      + "".join(period_contents) + "</div>"
        "</div></body></html>"
    )
    full_html = html.format(**contents)

    out_path = os.path.join(REPORT_DIR, "correlation_multi_period_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    if VERBOSE:
        print(f"[analyzer] report saved to {out_path}")
    return out_path
