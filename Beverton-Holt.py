# ====================================================
# Beverton–Holt vs Baseline Trading Backtest
# Price-based, RSI-dependent M and K
# - R0 = 1 + rolling geometric mean daily return (decimal)
# - M depends on price, RSI, and geo_mean_daily
# - K = (R0 - 1) * M
# - BH multi-step prediction equivalent to:
#   n_{t+1} = R0 * n_t / (1 + n_t / M) for H = 1
# - Long if predicted > current, short if predicted < current
# - Rebalances every REBALANCE_FREQ days
# - Handles partial histories: uses only available tickers per day
# - Equal-weight long/short across tickers with signals
# - Outputs: equity curve plot + performance table
# ====================================================

!pip install yfinance --quiet

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------
# USER PARAMETERS (EDIT HERE)
# -----------------------

TICKERS = ["A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES",
    "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
    "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP", "APTV", "ARE", "ATO", "AVB",
    "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BX", "BXP", "C", "CAG", "CAH",
    "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR",
    "CI", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COIN", "COO", "COP",
    "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA",
    "CTSH", "CTVA", "CVS", "CVX", "D", "DAL", "DASH", "DAY", "DD", "DDOG", "DE", "DECK", "DELL", "DG", "DGX",
    "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EME", "EMN", "EMR", "EOG", "EPAM", "EQIX", "EQR",
    "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FOX", "FOXA", "FRT", "FSLR", "FTNT",
    "FTV", "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL",
    "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON",
    "HOOD", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE", "IDXX", "IEX",
    "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT",
    "JBL", "JCI", "JKHY", "JNJ", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KMX",
    "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX",
    "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ",
    "MDT", "MET", "META", "MGM", "MHK", "MKC", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR",
    "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM",
    "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PEG",
    "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL",
    "PPG", "PPL", "PRU", "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "QCOM", "RCL", "REG", "REGN", "RF", "RJF", "RL",
    "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI",
    "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS",
    "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS",
    "TPL", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN", "TXT", "TYL",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC",
    "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC",
    "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH", "ZBRA", "ZTS"
]

START_DATE = "2000-01-01"
END_DATE   = "2025-01-01"

ROLL_WINDOW_DAYS   = 14
HORIZON_DAYS       = 14
RSI_PERIOD         = 14
RSI_SHORT_BOUNDARY = 100     # RSI threshold for M regime
WEIGHT_POWER       = 1.0    # no longer used for position sizing

REBALANCE_FREQ        = 14
INITIAL_CAPITAL       = 100000.0
RISK_FREE_RATE_ANNUAL = 0.02

VERBOSE = False

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def rolling_geo_mean_return(returns: pd.Series, window: int) -> pd.Series:
    gross = (1.0 + returns).rolling(window).apply(np.prod, raw=True)
    return gross ** (1.0 / window) - 1.0

def compute_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def beverton_holt_multi_step(n0, R0, K, H):
    return K * n0 / (n0 + (K - n0) * (R0 ** (-H)))

def calc_metrics(returns: pd.Series, benchmark_returns: pd.Series, rf_annual: float):
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.loc[returns.index].dropna()
    returns = returns.loc[benchmark_returns.index]

    if len(returns) == 0:
        return {k: np.nan for k in ["annual_geom_return","cumulative_return","alpha_annual","beta","sharpe_ratio"]}

    trading_days = 252
    years = len(returns) / trading_days
    cum_return = (1 + returns).prod() - 1
    annual_geom = (1 + cum_return)**(1/years) - 1 if years > 0 else np.nan

    rf_daily   = rf_annual / trading_days
    mean_ret   = returns.mean()
    mean_bench = benchmark_returns.mean()
    std_ret    = returns.std(ddof=0)

    var_bench = benchmark_returns.var(ddof=0)
    beta = returns.cov(benchmark_returns) / var_bench if var_bench > 0 else np.nan
    alpha_annual = (((mean_ret - rf_daily) - beta*(mean_bench-rf_daily)) * trading_days
                    if not np.isnan(beta) else np.nan)
    sharpe = ((mean_ret - rf_daily)/std_ret*np.sqrt(trading_days) if std_ret > 0 else np.nan)

    return {
        "annual_geom_return": annual_geom,
        "cumulative_return":  cum_return,
        "alpha_annual":       alpha_annual,
        "beta":               beta,
        "sharpe_ratio":       sharpe,
    }

# -----------------------
# DATA DOWNLOAD (BULK)
# -----------------------

data = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
    threads=True,
)

# Use adjusted Close if available, else Close
if isinstance(data.columns, pd.MultiIndex):
    if "Adj Close" in data.columns.get_level_values(0):
        close_px = data["Adj Close"]
    else:
        close_px = data["Close"]
else:
    close_px = data.get("Adj Close", data["Close"])

if isinstance(close_px, pd.Series):
    close_px = close_px.to_frame(name=TICKERS[0])

# -----------------------
# BUILD PER-TICKER PREDICTIONS (M & K from RSI and geo_mean)
# -----------------------

per_ticker = {}
EPS = 1e-8

for ticker in TICKERS:
    if ticker not in close_px.columns:
        continue

    prices = close_px[ticker].dropna()
    if prices.empty:
        continue

    daily_ret      = prices.pct_change()
    geo_mean_daily = rolling_geo_mean_return(daily_ret, ROLL_WINDOW_DAYS)
    rsi            = compute_rsi(prices, RSI_PERIOD)

    factor_low  = (100.0 - rsi) / 100.0
    factor_high = rsi / 100.0
    factor = np.where(rsi <= RSI_SHORT_BOUNDARY, factor_low, factor_high)

    geo_safe = geo_mean_daily.copy()
    geo_safe[np.abs(geo_safe) < EPS] = np.nan

    M_raw    = prices * factor / geo_safe
    M_series = pd.Series(M_raw, index=prices.index)

    R0_series = 1.0 + geo_mean_daily
    K_series  = (R0_series - 1.0) * M_series

    bh_pred_price  = beverton_holt_multi_step(prices, R0_series, K_series, HORIZON_DAYS)
    baseline_price = prices * (R0_series ** HORIZON_DAYS)
    next_ret       = daily_ret.shift(-1)

    df_ticker = pd.DataFrame({
        "price":               prices,
        "next_ret":            next_ret,
        "bh_pred_price":       bh_pred_price,
        "baseline_pred_price": baseline_price,
    }).dropna()

    if not df_ticker.empty:
        per_ticker[ticker] = df_ticker

if not per_ticker:
    raise ValueError("No valid tickers with data and predictions.")

# -----------------------
# CROSS-SECTIONAL BACKTEST (EQUAL-WEIGHT LONG/SHORT)
# -----------------------

all_dates = list(close_px.index)

bh_rets   = []
base_rets = []
eqw_rets  = []
bh_weights_prev   = None
base_weights_prev = None

for idx, dt in enumerate(all_dates):
    diffs_bh   = {}
    diffs_base = {}
    next_rets  = {}

    for ticker, df in per_ticker.items():
        if dt not in df.index:
            continue

        row = df.loc[dt]
        price_curr      = row["price"]
        price_pred_bh   = row["bh_pred_price"]
        price_pred_base = row["baseline_pred_price"]
        r_next          = row["next_ret"]

        if any(pd.isna(x) for x in [price_curr, price_pred_bh, price_pred_base, r_next]):
            continue

        next_rets[ticker] = r_next

        diff_bh   = price_pred_bh   - price_curr
        diff_base = price_pred_base - price_curr

        if diff_bh == 0 and diff_base == 0:
            continue

        diffs_bh[ticker]   = diff_bh
        diffs_base[ticker] = diff_base

    if len(next_rets) == 0:
        bh_rets.append(0.0)
        base_rets.append(0.0)
        eqw_rets.append(0.0)
        continue

    # equal-weight benchmark: long-only across whatever has data
    eqw_ret = np.mean(list(next_rets.values()))

    if len(diffs_bh) == 0:
        bh_rets.append(0.0)
        base_rets.append(0.0)
        eqw_rets.append(eqw_ret)
        continue

    rebalance_today = (idx == 0) or (idx % REBALANCE_FREQ == 0)

    # ----- Beverton–Holt equal-weight long/short -----
    if rebalance_today:
        n_bh = len(diffs_bh)
        if n_bh > 0:
            bh_weights = {
                tic: (1.0 / n_bh) * (1.0 if diffs_bh[tic] > 0 else -1.0)
                for tic in diffs_bh.keys()
            }
        else:
            bh_weights = {}
        bh_weights_prev = bh_weights
    else:
        bh_weights = bh_weights_prev or {}

    # ----- Baseline equal-weight long/short -----
    if rebalance_today:
        n_base = len(diffs_base)
        if n_base > 0:
            base_weights = {
                tic: (1.0 / n_base) * (1.0 if diffs_base[tic] > 0 else -1.0)
                for tic in diffs_base.keys()
            }
        else:
            base_weights = {}
        base_weights_prev = base_weights
    else:
        base_weights = base_weights_prev or {}

    bh_ret   = sum(bh_weights.get(tic, 0.0)   * next_rets[tic] for tic in next_rets)
    base_ret = sum(base_weights.get(tic, 0.0) * next_rets[tic] for tic in next_rets)

    bh_rets.append(bh_ret)
    base_rets.append(base_ret)
    eqw_rets.append(eqw_ret)

bh_rets   = pd.Series(bh_rets,   index=all_dates)
base_rets = pd.Series(base_rets, index=all_dates)
eqw_rets  = pd.Series(eqw_rets,  index=all_dates)

# -----------------------
# EQUITY CURVES
# -----------------------

bh_equity   = (1 + bh_rets).cumprod()   * INITIAL_CAPITAL
base_equity = (1 + base_rets).cumprod() * INITIAL_CAPITAL
eqw_equity  = (1 + eqw_rets).cumprod()  * INITIAL_CAPITAL

equity_df = pd.DataFrame({
    "Beverton-Holt Strategy": bh_equity,
    "Baseline Strategy":      base_equity,
    "Equal-Weight Benchmark": eqw_equity
})

# -----------------------
# PERFORMANCE TABLE
# -----------------------

metrics_bh   = calc_metrics(bh_rets,   eqw_rets, RISK_FREE_RATE_ANNUAL)
metrics_base = calc_metrics(base_rets, eqw_rets, RISK_FREE_RATE_ANNUAL)
metrics_eqw  = calc_metrics(eqw_rets,  eqw_rets, RISK_FREE_RATE_ANNUAL)

perf = pd.DataFrame.from_dict({
    "Beverton-Holt Strategy": metrics_bh,
    "Baseline Strategy":      metrics_base,
    "Equal-Weight Benchmark": metrics_eqw,
}, orient="index")[[
    "annual_geom_return","cumulative_return","alpha_annual","beta","sharpe_ratio"
]]

print("\n=== Performance Summary ===")
print(perf.to_string(float_format=lambda x: f"{x:0.4f}"))

# -----------------------
# PLOT EQUITY CURVES
# -----------------------

plt.figure(figsize=(10,5))
plt.plot(equity_df.index, equity_df["Beverton-Holt Strategy"], label="Beverton–Holt Strategy")
plt.plot(equity_df.index, equity_df["Baseline Strategy"],      label="Baseline Strategy")
plt.plot(equity_df.index, equity_df["Equal-Weight Benchmark"], label="Equal-Weight Benchmark")
plt.title(f"Equity Curves (Initial capital = {INITIAL_CAPITAL:,.0f})")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------
# SETTINGS RECAP
# -----------------------

print("\n=== Backtest Settings ===")
print(f"Tickers (count):         {len(TICKERS)}")
print(f"Sample tickers:          {TICKERS[:10]}")
print(f"Start date:              {START_DATE}")
print(f"End date:                {END_DATE}")
print(f"Roll window (days):      {ROLL_WINDOW_DAYS}")
print(f"Horizon (days):          {HORIZON_DAYS}")
print(f"RSI period (days):       {RSI_PERIOD}")
print(f"RSI short boundary:      {RSI_SHORT_BOUNDARY}")
print("M_t (RSI <= boundary):   Price_t * ((100 - RSI_t)/100) / geo_mean_daily_t")
print("M_t (RSI > boundary):    Price_t * (RSI_t/100) / geo_mean_daily_t")
print("K_t:                     (1 + geo_mean_daily_t - 1) * M_t = geo_mean_daily_t * M_t")
print("Position sizing:         Equal-weight long/short across tickers with signals")
print("                         w_i = sign(pred - price) / N_signals")
print(f"Rebalance frequency:     Every {REBALANCE_FREQ} trading days")
print(f"Initial capital:         {INITIAL_CAPITAL:,.2f}")
print(f"Risk-free (annual):      {RISK_FREE_RATE_ANNUAL:0.4%}")

