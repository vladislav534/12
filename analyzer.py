# analyzer.py

import os
import glob

import pandas as pd
import numpy as np
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

from config import DATA_DIR, REPORT_DIR, TIMEFRAME_LABEL, VERBOSE

# подавляем ворнинги NumPy по делению на 0 и недопустимым операциям
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
def pair_lag_corr(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int
) -> tuple[int, float]:
    # inner-join для общих таймстампов
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

def build_overlay_figure(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    s1: str,
    s2: str
) -> go.Figure:
    # inner-join по общим датам
    df = pd.concat(
        [df1["close"], df2["close"]],
        axis=1, join="inner", keys=[s1, s2]
    ).dropna()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df.index, y=df[s1], mode="lines",
                   name=s1, line=dict(width=1, color="cyan")),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[s2], mode="lines",
                   name=s2, line=dict(width=1, dash="dot", color="magenta")),
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
    # подготовка директорий
    os.makedirs(REPORT_DIR, exist_ok=True)
    plots_dir = os.path.join(REPORT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    data_full = load_all_series()

    # helper: подготовка и запись компактных JSON-файлов для пары (price и spread)
    def _prepare_plot_files(pair_idx: int, s1: str, s2: str, df_ab: pd.DataFrame, timeframe_label: str, lag: int | None):
        import json
        max_points = 1500

        # helper: получить series по символу или по позиции (first/second)
        def _get_close_series(df, sym, pos_fallback):
            # если MultiIndex columns: (sym, 'close') или ('a'/'b', 'close')
            cols = df.columns
            if isinstance(cols, pd.MultiIndex):
                # сначала ищем уровень 0 равный прямому имени symbol
                if sym in cols.get_level_values(0):
                    return df[(sym, "close")].astype(float)
                # иначе пробуем fallback позиции 'a' or 'b' или указанную pos_fallback
                for candidate in (pos_fallback,):
                    if candidate in cols.get_level_values(0):
                        return df[(candidate, "close")].astype(float)
                # как последняя мера попробуем взять первую/вторую пару колонок
                try:
                    lvl0 = list(dict.fromkeys(cols.get_level_values(0)))
                    if pos_fallback == "a":
                        key = lvl0[0]
                    else:
                        key = lvl0[1] if len(lvl0) > 1 else lvl0[0]
                    return df[(key, "close")].astype(float)
                except Exception:
                    raise KeyError(f"cannot find close series for {sym}")
            else:
                # обычные одноуровневые колонки
                if "close" in df.columns and sym in df.columns:
                    # случай, когда df has columns ['sym1','sym2'] with scalar values — unlikely
                    # но стандартный путь: df[sym]["close"] не применим, поэтому ищем 'close' directly
                    try:
                        return df[sym].astype(float)
                    except Exception:
                        pass
                if "close" in df.columns:
                    return df["close"].astype(float)
                # если ничего не найдено — пробуем взять первую колонку
                return df.iloc[:, 0].astype(float)

        # достаём серии y1 и y2
        try:
            series1 = _get_close_series(df_ab, s1, "a")
            series2 = _get_close_series(df_ab, s2, "b")
        except KeyError as e:
            # логируем и пропускаем эту пару
            if VERBOSE:
                print(f"[analyzer] skip pair {s1}-{s2}: {e}")
            return None, None

        # синхронизируем по индексу (на всякий случай)
        df_plot = pd.DataFrame({"y1": series1, "y2": series2}).dropna()
        if df_plot.empty:
            return None, None

        # compactify timestamps и значения
        ts = (df_plot.index.view("int64") // 10**9).astype(int)
        y1 = df_plot["y1"].values
        y2 = df_plot["y2"].values

        n = len(ts)
        if n == 0:
            return None, None

        if n > max_points:
            step = max(1, n // max_points)
            idx = list(range(0, n, step))
            ts_s = ts[idx].tolist()
            y1_s = y1[idx].tolist()
            y2_s = y2[idx].tolist()
        else:
            ts_s = ts.tolist()
            y1_s = y1.tolist()
            y2_s = y2.tolist()

        price_obj = {
            "x": ts_s,
            "y1": y1_s,
            "y2": y2_s,
            "meta": {"s1": s1, "s2": s2, "tf": timeframe_label, "lag": lag}
        }
        price_fname = f"plot_price_{pair_idx}.json"
        price_path = os.path.join(plots_dir, price_fname)
        with open(price_path, "w", encoding="utf-8") as pf:
            json.dump(price_obj, pf, separators=(",", ":"), ensure_ascii=False)

        spread_vals = [a - b for a, b in zip(y1_s, y2_s)]
        spread_obj = {
            "x": ts_s,
            "y": spread_vals,
            "meta": {"s1": s1, "s2": s2, "tf": timeframe_label, "lag": lag}
        }
        spread_fname = f"plot_spread_{pair_idx}.json"
        spread_path = os.path.join(plots_dir, spread_fname)
        with open(spread_path, "w", encoding="utf-8") as sf:
            json.dump(spread_obj, sf, separators=(",", ":"), ensure_ascii=False)

        return os.path.join("plots", price_fname), os.path.join("plots", spread_fname)


    head = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css"/>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<meta charset="utf-8">
<title>Multi-Period Correlation Report (lazy)</title>
<style>
  .pair-row { display:flex; align-items:center; gap:8px; padding:8px; border-bottom:1px solid rgba(255,255,255,0.04); }
  .pair-meta { flex:1; min-width:220px; }
  .fav-on { background:#ffd54f; }
  .controls { display:flex; gap:6px; flex-wrap:wrap; }
  .graph-container { margin-top:10px; }
  #favorites-list { margin-bottom:10px; }
</style>
"""

    # Табы по периодам
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
      <a class="nav-link active" id="{pfx}-norm-tab" data-toggle="pill" href="#{pfx}-norm">Обычная</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" id="{pfx}-lag-tab" data-toggle="pill" href="#{pfx}-lag">С лагом</a>
    </li>
  </ul>
  <div class="tab-content">
    <div class="tab-pane fade show active" id="{pfx}-norm">{norm_html}</div>
    <div class="tab-pane fade"          id="{pfx}-lag" >{lag_html}</div>
  </div>
"""

    # render_for_period: только метаданные и запись JSON-файлов для top_n пар
    def render_for_period(label: str, days: int) -> str:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)

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
            if df_ab.empty:
                continue

            series_a = df_ab["a"]["close"]
            series_b = df_ab["b"]["close"]

            c_norm = float(series_a.corr(series_b))
            lg, c_lag = pair_lag_corr(series_a, series_b, max_lag)

            recs_norm.append((a, b, c_norm, df_ab))
            recs_lag .append((a, b, lg,   c_lag, df_ab))

        df_norm = (
            pd.DataFrame([(r[0], r[1], r[2]) for r in recs_norm], columns=["sym1","sym2","corr"])
              .sort_values("corr", ascending=False)
              .head(top_n)
        )
        df_lag = (
            pd.DataFrame([(r[0], r[1], r[2], r[3]) for r in recs_lag], columns=["sym1","sym2","lag","corr"])
              .sort_values("corr", ascending=False)
              .head(top_n)
        )

        norm_rows = []
        pair_id = 0
        for idx, row in df_norm.iterrows():
            s1, s2, cv = row["sym1"], row["sym2"], row["corr"]
            df_ab = next((r[3] for r in recs_norm if r[0]==s1 and r[1]==s2), None)
            if df_ab is None:
                continue

            price_path, spread_path = _prepare_plot_files(pair_id, s1, s2, df_ab, label, None)
            if price_path is None:
                continue

            html_row = f'''
<div class="pair-row" id="pair-row-{pair_id}">
  <div class="pair-meta">
    <strong>{s1} — {s2}</strong><br/>
    <small>Корреляция: {cv:.4f}</small>
  </div>
  <div class="controls">
    <button class="btn btn-sm btn-primary btn-show" data-pair="{pair_id}" data-price="{price_path}">Показать график</button>
    <button class="btn btn-sm btn-secondary btn-show" data-pair="{pair_id}" data-spread="{spread_path}">Показать спред</button>
    <button class="btn btn-sm btn-outline-warning btn-fav" data-pair="{pair_id}">Избранное</button>
  </div>
</div>
<div class="graph-container" id="graph-{pair_id}" style="display:none;"></div>
'''
            norm_rows.append(html_row)
            pair_id += 1

        lag_rows = []
        for idx, row in df_lag.iterrows():
            s1, s2, lag_val, cv = row["sym1"], row["sym2"], row["lag"], row["corr"]
            df_ab = next((r[4] for r in recs_lag if r[0]==s1 and r[1]==s2), None)
            if df_ab is None:
                continue
            price_path, spread_path = _prepare_plot_files(pair_id, s1, s2, df_ab, label, lag_val)
            if price_path is None:
                continue
            html_row = f'''
<div class="pair-row" id="pair-row-{pair_id}">
  <div class="pair-meta">
    <strong>{s1} — {s2}</strong><br/>
    <small>Lag: {lag_val}; Корр: {cv:.4f}</small>
  </div>
  <div class="controls">
    <button class="btn btn-sm btn-primary btn-show" data-pair="{pair_id}" data-price="{price_path}">Показать график</button>
    <button class="btn btn-sm btn-secondary btn-show" data-pair="{pair_id}" data-spread="{spread_path}">Показать спред</button>
    <button class="btn btn-sm btn-outline-warning btn-fav" data-pair="{pair_id}">Избранное</button>
  </div>
</div>
<div class="graph-container" id="graph-{pair_id}" style="display:none;"></div>
'''
            lag_rows.append(html_row)
            pair_id += 1

        section_html = f'''
<div id="favorites-list" class="mb-2">
  <strong>Избранные (по номерам строк):</strong> <span id="fav-idxs">—</span>
</div>
<div id="pairs-list-norm">
  {"".join(norm_rows)}
</div>
<hr/>
<div id="pairs-list-lag">
  {"".join(lag_rows)}
</div>
'''
        return section_html

    # собираем контент каждого таба
    contents = {}
    for label, days in periods.items():
        contents[f"content_{label}"] = render_for_period(label, days)

    # основной HTML + JS: на click делаем fetch JSON по data-price / data-spread, декодируем ts->Date и рисуем Plotly
    html = (
        "<!DOCTYPE html><html><head>" + head + "</head><body>"
        "<div class='container'><h1 class='my-4'>Multi-Period Correlation Report (lazy)</h1>"
        "<ul class='nav nav-tabs'>" + "".join(period_tabs) + "</ul>"
        "<div class='tab-content'>"      + "".join(period_contents) + "</div>"
        "</div>"
        "<script>"
        "async function loadAndRender(path, containerId, isSpread){"
        "  try{"
        "    const res = await fetch(path);"
        "    if(!res.ok) { console.error('fetch err', path); return; }"
        "    const obj = await res.json();"
        "    const x = obj.x.map(ts=> new Date(ts*1000));"
        "    if(isSpread){"
        "      const data = [{ x: x, y: obj.y, type:'scatter', mode:'lines', name: obj.meta.s1+'-'+obj.meta.s2+' spread', line:{width:1,color:'orange'} }];"
        "      const layout = { title: 'Spread '+obj.meta.s1+' - '+obj.meta.s2, template:'plotly_dark', height:250};"
        "      Plotly.react(containerId, data, layout);"
        "    } else {"
        "      const data = ["
        "        { x: x, y: obj.y1, type:'scatter', mode:'lines', name: obj.meta.s1, line:{width:1,color:'cyan'} },"
        "        { x: x, y: obj.y2, type:'scatter', mode:'lines', name: obj.meta.s2, line:{width:1,dash:'dot',color:'magenta'}, yaxis:'y2' }];"
        "      const layout = { title: obj.meta.s1+' vs '+obj.meta.s2, template:'plotly_dark', height:350, yaxis:{title:obj.meta.s1+' price'}, yaxis2:{title:obj.meta.s2+' price',overlaying:'y',side:'right'}};"
        "      Plotly.react(containerId, data, layout);"
        "    }"
        "  } catch(e){ console.error(e); }"
        "}"
        "document.addEventListener('click', function(e){"
        "  var t = e.target;"
        "  if(t.classList && t.classList.contains('btn-show')){"
        "    var pid = t.getAttribute('data-pair');"
        "    var pricePath = t.getAttribute('data-price');"
        "    var spreadPath = t.getAttribute('data-spread');"
        "    var container = document.getElementById('graph-'+pid);"
        "    if(!container) return;"
        "    if(pricePath && !t.getAttribute('data-type')){"
        "      // show price by default"
        "      container.style.display='block';"
        "      var plotId = 'plot-'+pid;"
        "      container.innerHTML = '<div id=\"'+plotId+'\" style=\"width:100%;height:100%;\"></div>';"
        "      loadAndRender(pricePath, plotId, false);"
        "      container.scrollIntoView({behavior:'smooth', block:'center'});"
        "    } else if(spreadPath){"
        "      container.style.display='block';"
        "      var plotId = 'plot-'+pid;"
        "      container.innerHTML = '<div id=\"'+plotId+'\" style=\"width:100%;height:100%;\"></div>';"
        "      loadAndRender(spreadPath, plotId, true);"
        "      container.scrollIntoView({behavior:'smooth', block:'center'});"
        "    }"
        "  } else if(t.classList && t.classList.contains('btn-fav')){"
        "    var pid = t.getAttribute('data-pair');"
        "    var row = document.getElementById('pair-row-'+pid);"
        "    t.classList.toggle('fav-on');"
        "    if(row) row.classList.toggle('fav-on');"
        "    var favs = document.querySelectorAll('.btn-fav.fav-on');"
        "    var idxs = Array.from(favs).map(function(b){ return b.getAttribute('data-pair'); });"
        "    document.getElementById('fav-idxs').textContent = idxs.length? idxs.join(', '): '—';"
        "  }"
        "});"
        "</script>"
        "</body></html>"
    )
    # аккуратно подставляем только наши content_{label} ключи, не трогая другие фигурные скобки
    full_html = html
    for k, v in contents.items():
        full_html = full_html.replace("{" + k + "}", v)


    out_path = os.path.join(REPORT_DIR, "correlation_multi_period_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    if VERBOSE:
        print(f"[analyzer] report saved to {out_path}")
    return out_path
