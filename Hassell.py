# ====================================================
# Hassell vs Baseline Trading Backtest
# Price-based, RSI-dependent M
# - R0 = 1 + rolling geometric mean daily return (decimal)
# - M depends on price, RSI, and geo_mean_daily
# - Hassell single-step map:
#     n_{t+1} = R0 * n_t / (1 + n_t / M)^c_i
#   where c_i (Hassell weighting index) depends on the chosen strategy:
#     * Inflation based Hassell Indice: c = 1 - inflation_rate (from CPI)
#     * Constant Hassell Indice:       c = CONSTANT_HASSELL_C
#     * Market Cap Indice:             c_i = (MC_i - sum_j MC_j) / sum_j MC_j
#     * buffet Indice:                 c = 1 - buffett_indicator_level
# - Multi-step prediction: iterate the above H times
# - Long if predicted > current, short if predicted < current
# - Rebalances every REBALANCE_FREQ days
# - Handles partial histories: uses only available tickers per day
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

START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"

ROLL_WINDOW_DAYS   = 60
HORIZON_DAYS       = 15
RSI_PERIOD         = 14
RSI_SHORT_BOUNDARY = 60
WEIGHT_POWER       = 1.0    # not used in Hassell weighting, kept for completeness

REBALANCE_FREQ        = 15
INITIAL_CAPITAL       = 100000.0
RISK_FREE_RATE_ANNUAL = 0.03

VERBOSE = False

# -----------------------
# HASELL INDICE STRATEGY SWITCHES
# (Make exactly ONE of these True)
# -----------------------

INFLATION_BASED_HASSELL = False      # "Inflation based Hassell Indice"
CONSTANT_HASSELL_INDICE = False      # "Constant Hassell Indice"
MARKET_CAP_INDICE       = False      # "Market Cap Indice"
BUFFET_INDICE           = False       # "buffet Indice"

# Constant used when CONSTANT_HASSELL_INDICE = True
CONSTANT_HASSELL_C = 0.5

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

def hassell_multi_step(n0, R0, M, H, c):
    """
    Multi-step Hassell / Beverton–Holt-style map:

      Single-step:
        n_{t+1} = R0 * n_t / (1 + n_t / M)^c

      c is scalar per ticker.
    """
    n = n0.astype(float).copy()
    for _ in range(H):
        denom = 1.0 + n / M
        denom = denom.where(denom > 0, np.nan)
        n = R0 * n / (denom ** c)
    return n

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

def _extract_series_from_yf(df: pd.DataFrame) -> pd.Series:
    """Helper: get a single price/level series from a yfinance download."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            return df["Adj Close"].iloc[:, 0]
        elif "Close" in lvl0:
            return df["Close"].iloc[:, 0]
        else:
            # Just take first top-level field
            return df.xs(df.columns.levels[0][0], axis=1, level=0).iloc[:, 0]
    else:
        if "Adj Close" in df.columns:
            return df["Adj Close"]
        elif "Close" in df.columns:
            return df["Close"]
        else:
            # take the first column as fallback
            return df.iloc[:, 0]

def compute_latest_inflation_rate() -> float:
    """
    Compute latest YoY inflation from CPI (CPIAUCSL @ FRED via yfinance).
    Returns decimal (e.g. 0.03 for 3%).
    """
    try:
        cpi_raw = yf.download("CPIAUCSL", start="1990-01-01", progress=False)
        if cpi_raw.empty:
            raise ValueError("Empty CPI data")
        cpi = _extract_series_from_yf(cpi_raw)
        # Monthly or higher frequency: resample to month-end
        cpi_m = cpi.resample("M").last()
        if len(cpi_m) < 13:
            raise ValueError("Not enough CPI history for YoY calculation")
        last = cpi_m.iloc[-1]
        prev_year = cpi_m.iloc[-13]
        inflation_yoy = last / prev_year - 1.0
        return float(inflation_yoy)
    except Exception as e:
        print(f"WARNING: Could not compute inflation from CPIAUCSL: {e}")
        print("Using fallback inflation = 2%")
        return 0.02

def compute_latest_buffett_indicator() -> float:
    """
    Compute a simple Buffett indicator:
      buffett_indicator_level = Wilshire 5000 index / GDP
    Using:
      ^W5000 (Wilshire 5000 Total Market Index)
      GDP (US GDP, FRED)
    Returns a ratio (e.g. 1.5 ~ 150%).
    """
    try:
        w_raw = yf.download("^W5000", start="1990-01-01", progress=False)
        gdp_raw = yf.download("GDP", start="1990-01-01", progress=False)

        if w_raw.empty or gdp_raw.empty:
            raise ValueError("Empty Wilshire or GDP data")

        w = _extract_series_from_yf(w_raw)
        g = _extract_series_from_yf(gdp_raw)

        # GDP is quarterly, Wilshire is daily; align by taking last values
        w_last = w.dropna().iloc[-1]
        g_last = g.dropna().iloc[-1]

        if g_last == 0:
            raise ValueError("GDP last value is zero")

        level = float(w_last / g_last)
        return level
    except Exception as e:
        print(f"WARNING: Could not compute Buffett indicator (^W5000/GDP): {e}")
        print("Using fallback buffett_indicator_level = 1.5 (150%)")
        return 1.5

# -----------------------
# VALIDATE STRATEGY SELECTION
# -----------------------

strategy_flags = [
    INFLATION_BASED_HASSELL,
    CONSTANT_HASSELL_INDICE,
    MARKET_CAP_INDICE,
    BUFFET_INDICE,
]
if sum(strategy_flags) != 1:
    raise ValueError("Exactly ONE of INFLATION_BASED_HASSELL, CONSTANT_HASSELL_INDICE, "
                     "MARKET_CAP_INDICE, BUFFET_INDICE must be True.")

if INFLATION_BASED_HASSELL:
    active_strategy_name = "Inflation based Hassell Indice"
elif CONSTANT_HASSELL_INDICE:
    active_strategy_name = "Constant Hassell Indice"
elif MARKET_CAP_INDICE:
    active_strategy_name = "Market Cap Indice"
else:
    active_strategy_name = "buffet Indice"

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

if VERBOSE:
    print("Downloaded close data for tickers:", close_px.columns.tolist())
    print(close_px.head())

# -----------------------
# HASELL INDICE PER TICKER (c_i)
# -----------------------

hassell_indice_by_ticker = {}

if MARKET_CAP_INDICE:
    # Market Cap Indice: c_i = (MC_i - sum_j MC_j) / sum_j MC_j
    market_caps = {}
    tickers_yf = {tic: yf.Ticker(tic) for tic in TICKERS}

    for tic, obj in tickers_yf.items():
        mc = None
        # Try fast_info first
        try:
            fi = getattr(obj, "fast_info", {})
            mc = fi.get("market_cap", None)
        except Exception:
            mc = None
        # Fallback to .info if needed
        if mc is None:
            try:
                info = obj.info
                mc = info.get("marketCap", None)
            except Exception:
                mc = None
        if mc is not None:
            market_caps[tic] = mc

    total_mc = sum(market_caps.values()) if len(market_caps) > 0 else 0.0

    for tic in TICKERS:
        mc = market_caps.get(tic, None)
        if mc is None or total_mc == 0:
            c_i = 0.0  # fallback if we don't have MC
        else:
            c_i = (mc - total_mc) / total_mc
        hassell_indice_by_ticker[tic] = c_i

else:
    # Global scalar c for all tickers, from inflation / constant / Buffett
    if INFLATION_BASED_HASSELL:
        inflation_rate = compute_latest_inflation_rate()
        c_global = 1.0 - inflation_rate
        print(f"Computed inflation_rate (YoY CPI): {inflation_rate:.4%} -> c = {c_global:.4f}")
    elif CONSTANT_HASSELL_INDICE:
        c_global = CONSTANT_HASSELL_C
        print(f"Using constant Hassell c = {c_global:.4f}")
    elif BUFFET_INDICE:
        buffett_level = compute_latest_buffett_indicator()
        c_global = 1.0 - buffett_level
        print(f"Computed Buffett indicator level: {buffett_level:.4f} -> c = {c_global:.4f}")
    else:
        c_global = 0.0  # should never hit because of earlier validation

    for tic in TICKERS:
        hassell_indice_by_ticker[tic] = c_global

if VERBOSE:
    print(f"Active Hassell strategy: {active_strategy_name}")
    print("Sample c_i values (ticker: c_i):")
    for tic in list(hassell_indice_by_ticker.keys())[:10]:
        print(tic, hassell_indice_by_ticker[tic])

# -----------------------
# BUILD PER-TICKER PREDICTIONS (M from RSI and geo_mean)
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

    M_raw = prices * factor / geo_safe
    M_series = pd.Series(M_raw, index=prices.index)

    R0_series = 1.0 + geo_mean_daily

    # Optional BH-style K (exact only if c = 1); kept for reference
    K_series = (R0_series - 1.0) * M_series

    c_ticker = hassell_indice_by_ticker.get(ticker, 0.0)

    hassell_pred_price = hassell_multi_step(
        n0=prices,
        R0=R0_series,
        M=M_series,
        H=HORIZON_DAYS,
        c=c_ticker
    )

    baseline_price = prices * (R0_series ** HORIZON_DAYS)
    next_ret = daily_ret.shift(-1)

    df_ticker = pd.DataFrame({
        "price":               prices,
        "next_ret":            next_ret,
        "hassell_pred_price":  hassell_pred_price,
        "baseline_pred_price": baseline_price,
    }).dropna()

    if not df_ticker.empty:
        per_ticker[ticker] = df_ticker

if not per_ticker:
    raise ValueError("No valid tickers with data and predictions.")

# -----------------------
# CROSS-SECTIONAL BACKTEST
# -----------------------

all_dates = list(close_px.index)

hassell_rets = []
base_rets    = []
eqw_rets     = []
hassell_weights_prev = None
base_weights_prev    = None

for idx, dt in enumerate(all_dates):
    diffs_hassell = {}
    diffs_base    = {}
    next_rets     = {}

    for ticker, df in per_ticker.items():
        if dt not in df.index:
            continue

        row = df.loc[dt]
        price_curr         = row["price"]
        price_pred_hassell = row["hassell_pred_price"]
        price_pred_base    = row["baseline_pred_price"]
        r_next             = row["next_ret"]

        if any(pd.isna(x) for x in [price_curr, price_pred_hassell, price_pred_base, r_next]):
            continue

        next_rets[ticker] = r_next

        diff_hassell = price_pred_hassell - price_curr
        diff_base    = price_pred_base    - price_curr

        if diff_hassell == 0 and diff_base == 0:
            continue

        diffs_hassell[ticker] = diff_hassell
        diffs_base[ticker]    = diff_base

    if len(next_rets) == 0:
        hassell_rets.append(0.0)
        base_rets.append(0.0)
        eqw_rets.append(0.0)
        continue

    eqw_ret = np.mean(list(next_rets.values()))

    if len(diffs_hassell) == 0:
        hassell_rets.append(0.0)
        base_rets.append(0.0)
        eqw_rets.append(eqw_ret)
        continue

    rebalance_today = (idx == 0) or (idx % REBALANCE_FREQ == 0)

    # Hassell-based weights (equal-weight long/short)
    if rebalance_today:
        active_hassell = [tic for tic, diff in diffs_hassell.items() if diff != 0]
        n_hass = len(active_hassell)
        if n_hass > 0:
            hassell_weights = {
                tic: np.sign(diffs_hassell[tic]) / n_hass
                for tic in active_hassell
            }
        else:
            hassell_weights = {tic: 0.0 for tic in diffs_hassell.keys()}
        hassell_weights_prev = hassell_weights
    else:
        hassell_weights = hassell_weights_prev or {tic: 0.0 for tic in diffs_hassell.keys()}

    # Baseline weights (equal-weight long/short)
    if rebalance_today:
        active_base = [tic for tic, diff in diffs_base.items() if diff != 0]
        n_base = len(active_base)
        if n_base > 0:
            base_weights = {
                tic: np.sign(diffs_base[tic]) / n_base
                for tic in active_base
            }
        else:
            base_weights = {tic: 0.0 for tic in diffs_base.keys()}
        base_weights_prev = base_weights
    else:
        base_weights = base_weights_prev or {tic: 0.0 for tic in diffs_base.keys()}

    hassell_ret = sum(hassell_weights.get(tic, 0.0) * next_rets[tic] for tic in next_rets)
    base_ret    = sum(base_weights.get(tic, 0.0)    * next_rets[tic] for tic in next_rets)

    hassell_rets.append(hassell_ret)
    base_rets.append(base_ret)
    eqw_rets.append(eqw_ret)

hassell_rets = pd.Series(hassell_rets, index=all_dates)
base_rets    = pd.Series(base_rets,    index=all_dates)
eqw_rets     = pd.Series(eqw_rets,     index=all_dates)

# -----------------------
# EQUITY CURVES
# -----------------------

hassell_equity = (1 + hassell_rets).cumprod() * INITIAL_CAPITAL
base_equity    = (1 + base_rets).cumprod()    * INITIAL_CAPITAL
eqw_equity     = (1 + eqw_rets).cumprod()     * INITIAL_CAPITAL

equity_df = pd.DataFrame({
    "Hassell Strategy":       hassell_equity,
    "Baseline Strategy":      base_equity,
    "Equal-Weight Benchmark": eqw_equity
})

# -----------------------
# PERFORMANCE TABLE
# -----------------------

metrics_hassell = calc_metrics(hassell_rets, eqw_rets, RISK_FREE_RATE_ANNUAL)
metrics_base    = calc_metrics(base_rets,    eqw_rets, RISK_FREE_RATE_ANNUAL)
metrics_eqw     = calc_metrics(eqw_rets,     eqw_rets, RISK_FREE_RATE_ANNUAL)

perf = pd.DataFrame.from_dict({
    "Hassell Strategy":       metrics_hassell,
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
plt.plot(equity_df.index, equity_df["Hassell Strategy"],       label="Hassell Strategy")
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
print(f"Active Hassell strategy: {active_strategy_name}")
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
print("K_t (BH-style):          (R0_t - 1) * M_t  (exact carrying capacity only if c = 1)")
print("Single-step Hassell:     n_{t+1} = R0_t * n_t / (1 + n_t / M_t)^c_i")
print("Inflation based Hassell: c = 1 - inflation_rate (CPI YoY)")
print("Constant Hassell Indice: c = CONSTANT_HASSELL_C")
print("Market Cap Indice:       c_i = (MC_i - sum_j MC_j) / sum_j MC_j")
print("buffet Indice:           c = 1 - buffett_indicator_level (^W5000 / GDP)")
print("Portfolio weights:       Equal-weight among active long/short signals")
print(f"Rebalance frequency:     Every {REBALANCE_FREQ} trading days")
print(f"Initial capital:         {INITIAL_CAPITAL:,.2f}")
print(f"Risk-free (annual):      {RISK_FREE_RATE_ANNUAL:0.4%}")

