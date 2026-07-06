"""
QuantPilot - live trading dashboard.

Design notes:
  - Theme lives in .streamlit/config.toml (near-black base, single
    green accent). This file only adds layout and typography.
  - Charts are Altair (interactive, ships with Streamlit) rather than
    matplotlib PNGs.
  - No decorative icons. State is communicated with text, color, and
    numbers only.

Run locally:
    streamlit run monitoring/dashboard.py

Deploy:
    Push to GitHub, connect the repo on streamlit.io/cloud, and add the
    same secrets as .env in the Streamlit Cloud settings.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config
from data.database import Database
from data.alpaca_feed import AlpacaFeed
from models.registry import ModelRegistry

ACCENT   = "#34D399"   # gains, buys
ACCENT_2 = "#F87171"   # losses, sells
INK_DIM  = "#8B98A5"   # secondary text
GRID     = "#1D2733"

st.set_page_config(
    page_title="QuantPilot",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
    /* Tabular figures everywhere numbers matter */
    [data-testid="stMetricValue"] {{
        font-variant-numeric: tabular-nums;
        font-size: 1.65rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {INK_DIM};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
    }}
    [data-testid="stMetric"] {{
        background: {GRID}40;
        border: 1px solid {GRID};
        border-radius: 10px;
        padding: 14px 18px;
    }}
    .qp-wordmark {{
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0;
    }}
    .qp-wordmark span {{ color: {ACCENT}; }}
    .qp-badge {{
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        vertical-align: middle;
        margin-left: 12px;
    }}
    .qp-badge.paper {{
        color: {ACCENT}; border: 1px solid {ACCENT}66;
        background: {ACCENT}14;
    }}
    .qp-badge.live {{
        color: {ACCENT_2}; border: 1px solid {ACCENT_2}66;
        background: {ACCENT_2}14;
    }}
    .qp-sub {{ color: {INK_DIM}; font-size: 0.85rem; margin-top: 2px; }}
    section[data-testid="stTabs"] button p {{
        letter-spacing: 0.04em;
    }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Cached data access
# ------------------------------------------------------------------

@st.cache_resource
def get_feed():
    return AlpacaFeed(config.ALPACA_API_KEY,
                      config.ALPACA_SECRET_KEY,
                      config.ALPACA_BASE_URL)


@st.cache_resource
def get_db():
    db = Database(config.DB_URL)
    db.create_tables()
    return db


@st.cache_data(ttl=60)
def fetch_account():
    return get_feed().get_account()


@st.cache_data(ttl=300)
def fetch_portfolio_history():
    return get_db().get_portfolio_history()


@st.cache_data(ttl=60)
def fetch_trade_log():
    return get_db().get_trade_log()


@st.cache_data(ttl=60)
def fetch_latest_predictions():
    try:
        return get_db().get_latest_predictions()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_sentiment_history(days: int = 30):
    db    = get_db()
    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=days)).isoformat()
    frames = []
    for ticker in config.WATCHLIST:
        s = db.get_sentiment(ticker, start, end)
        if not s.empty:
            s = s.reset_index()
            s['ticker'] = ticker
            frames.append(s)
    return pd.concat(frames) if frames else pd.DataFrame()


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------

mode_cls  = "live" if config.LIVE_TRADING else "paper"
mode_text = "LIVE CAPITAL" if config.LIVE_TRADING else "PAPER TRADING"

left, right = st.columns([5, 1])
with left:
    st.markdown(
        f'<p class="qp-wordmark">Quant<span>Pilot</span>'
        f'<span class="qp-badge {mode_cls}">{mode_text}</span></p>'
        f'<p class="qp-sub">Daily ensemble signals on '
        f'{" / ".join(config.WATCHLIST)} &middot; next-day return target '
        f'&middot; ATR-sized, drawdown-capped</p>',
        unsafe_allow_html=True,
    )
with right:
    if st.button("Refresh data", width='stretch'):
        st.cache_data.clear()

st.markdown("")

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------

history = fetch_portfolio_history()

initial_capital = 100_000.0
total_return = max_dd = None
if not history.empty:
    values   = history['value'].values
    peak     = np.maximum.accumulate(values)
    drawdown = np.where(peak > 0, (peak - values) / peak, 0.0)
    max_dd   = float(np.max(drawdown)) * 100
    total_return = (values[-1] - initial_capital) / initial_capital * 100

k1, k2, k3, k4 = st.columns(4)
try:
    account = fetch_account()
    k1.metric("Equity", f"${account['equity']:,.2f}")
    k2.metric("Cash",   f"${account['cash']:,.2f}")
except Exception:
    k1.metric("Equity", "-")
    k2.metric("Cash",   "-")
    st.caption("Broker connection unavailable - showing database history only.")

k3.metric("Total return",
          f"{total_return:+.2f}%" if total_return is not None else "-")
k4.metric("Max drawdown",
          f"{max_dd:.2f}%" if max_dd is not None else "-",
          help=f"Trading halts automatically above "
               f"{config.MAX_DRAWDOWN_HALT:.0%}")

st.markdown("")

# ------------------------------------------------------------------
# Equity curve + drawdown
# ------------------------------------------------------------------

if history.empty:
    st.info("No portfolio snapshots yet - the first daily job run will "
            "start the equity curve.")
else:
    eq = history.reset_index().rename(columns={'date': 'Date',
                                               'value': 'Equity'})
    eq['Drawdown'] = -(np.maximum.accumulate(eq['Equity']) - eq['Equity']) \
        / np.maximum.accumulate(eq['Equity']) * 100

    base = alt.Chart(eq).encode(
        x=alt.X('Date:T', axis=alt.Axis(grid=False, title=None)))

    area = base.mark_area(
        line={'color': ACCENT, 'strokeWidth': 2},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color=f'{ACCENT}00', offset=0),
                   alt.GradientStop(color=f'{ACCENT}55', offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        y=alt.Y('Equity:Q',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format='$,.0f', title=None,
                              gridColor=GRID)),
        tooltip=[alt.Tooltip('Date:T'),
                 alt.Tooltip('Equity:Q', format='$,.2f')],
    ).properties(height=280)

    baseline = alt.Chart(pd.DataFrame({'y': [initial_capital]})) \
        .mark_rule(strokeDash=[4, 4], color=INK_DIM, opacity=0.6) \
        .encode(y='y:Q')

    dd = base.mark_area(color=f'{ACCENT_2}66',
                        line={'color': ACCENT_2, 'strokeWidth': 1}) \
        .encode(
            y=alt.Y('Drawdown:Q',
                    axis=alt.Axis(format='.1f', title='Drawdown %',
                                  gridColor=GRID)),
            tooltip=[alt.Tooltip('Date:T'),
                     alt.Tooltip('Drawdown:Q', format='.2f')],
        ).properties(height=90)

    st.altair_chart(
        alt.vconcat(area + baseline, dd).resolve_scale(x='shared')
           .configure_view(strokeOpacity=0),
        width='stretch',
    )

# ------------------------------------------------------------------
# Detail tabs
# ------------------------------------------------------------------

tab_signals, tab_trades, tab_sentiment, tab_models = st.tabs(
    ["Signals", "Trades", "Sentiment", "Models"])

with tab_signals:
    preds = fetch_latest_predictions()
    if preds.empty:
        st.info("No predictions logged yet - they appear after the next "
                "daily job run.")
    else:
        preds = preds.copy()
        preds['date'] = pd.to_datetime(preds['date']).dt.date
        st.dataframe(
            preds.rename(columns={
                'ticker': 'Ticker', 'date': 'Date',
                'signal': 'Signal', 'confidence': 'Confidence',
                'predicted_price': 'Predicted close',
                'model_version': 'Model',
            }),
            width='stretch', hide_index=True,
            column_config={
                'Confidence': st.column_config.ProgressColumn(
                    'Confidence', min_value=0.0, max_value=1.0,
                    format='%.2f'),
                'Predicted close': st.column_config.NumberColumn(
                    format='$%.2f'),
            },
        )
        st.caption(f"BUY/SELL requires a predicted move beyond "
                   f"{config.SIGNAL_THRESHOLD:.1%} and confidence of at "
                   f"least {config.CONFIDENCE_THRESHOLD:.2f} "
                   f"(two of three base models agreeing on direction).")

with tab_trades:
    trades = fetch_trade_log()
    if trades.empty:
        st.info("No trades recorded yet.")
    else:
        show = trades.head(25).copy()
        st.dataframe(
            show.rename(columns={
                'order_id': 'Order', 'ticker': 'Ticker', 'side': 'Side',
                'qty': 'Qty', 'price': 'Price', 'status': 'Status',
                'timestamp': 'Time (UTC)',
            }),
            width='stretch', hide_index=True,
            column_config={
                'Price': st.column_config.NumberColumn(format='$%.2f'),
                'Order': st.column_config.TextColumn(width='small'),
            },
        )

with tab_sentiment:
    senti = fetch_sentiment_history()
    if senti.empty:
        st.info("No sentiment data in the last 30 days.")
    else:
        latest = senti.sort_values('date').groupby('ticker').tail(1)
        cols = st.columns(len(config.WATCHLIST))
        for col, ticker in zip(cols, config.WATCHLIST):
            row = latest[latest['ticker'] == ticker]
            score = float(row['score'].iloc[0]) if not row.empty else 0.0
            label = ("bullish" if score > 0.05
                     else "bearish" if score < -0.05 else "neutral")
            col.metric(f"{ticker} news tone", f"{score:+.3f}", label,
                       delta_color=("normal" if score > 0.05
                                    else "inverse" if score < -0.05
                                    else "off"))

        chart = alt.Chart(senti).mark_line(
            interpolate='monotone', strokeWidth=1.6,
        ).encode(
            x=alt.X('date:T', axis=alt.Axis(grid=False, title=None)),
            y=alt.Y('score:Q',
                    axis=alt.Axis(title='VADER compound', gridColor=GRID)),
            color=alt.Color('ticker:N',
                            scale=alt.Scale(range=[ACCENT, '#60A5FA',
                                                   '#FBBF24']),
                            legend=alt.Legend(title=None, orient='top')),
            tooltip=['ticker:N', 'date:T',
                     alt.Tooltip('score:Q', format='+.3f')],
        ).properties(height=240)
        st.altair_chart(chart.configure_view(strokeOpacity=0),
                        width='stretch')

with tab_models:
    try:
        # Models live in the DB (durable across runners); read them from
        # there so the deployed dashboard sees the same registry the jobs do.
        versions = ModelRegistry(db=get_db()).list_versions()
    except Exception:
        versions = []
    if not versions:
        st.info("No models in the registry yet - run the train job first.")
    else:
        rows = []
        for v in versions:
            m    = v.get('metrics', {})
            meta = v.get('meta', {})
            c80  = meta.get('conformal_80')
            rows.append({
                'Model':     v['name'],
                'Version':   v['version_id'],
                'Target':    meta.get('target', 'price (legacy)'),
                'RMSE':      m.get('rmse'),
                'Dir. acc':  m.get('dir_acc'),
                '+/-80% (bps)': round(c80 * 10000, 1) if c80 is not None else None,
                'Saved':     v['timestamp'][:19].replace('T', ' '),
            })
        st.dataframe(
            pd.DataFrame(rows), width='stretch', hide_index=True,
            column_config={
                'RMSE': st.column_config.NumberColumn(format='%.5f'),
                'Dir. acc': st.column_config.NumberColumn(format='%.3f'),
            },
        )
        st.caption("RMSE is in next-day return units. A model is promoted "
                   "only when it beats the incumbent on the same held-out "
                   "window; directional accuracy of 0.500 is a coin flip. "
                   "+/-80% is the conformal half-interval - a signal only trades "
                   "when the predicted move exceeds it.")

st.markdown("")
st.caption("Data: Alpaca Markets (IEX), NewsAPI + Claude/VADER sentiment, "
           "Neon PostgreSQL. Jobs run via GitHub Actions - daily after the "
           "close, retraining on Sundays.")
