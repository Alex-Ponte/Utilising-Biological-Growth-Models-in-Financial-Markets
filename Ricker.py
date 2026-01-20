# ====================================================
# Ricker Model vs Baseline Trading Backtest
# Price-based, RSI-dependent M and K
# - r_t = 1 + rolling geometric mean daily return
# - Carrying capacity K same as Beverton–Holt version
# - Ricker model:
#   n_{t+1} = n_t * exp( r_t * (1 - n_t / K_t) )
# - Multi-step forecast via iteration
# ====================================================

!pip install yfinance --quiet

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------
# USER PARAMETERS
# -----------------------

START_DATE = "2015-01-01"
END_DATE   = "2025-01-01"

ROLL_WINDOW_DAYS   = 60
HORIZON_DAYS       = 15
RSI_PERIOD         = 14
RSI_SHORT_BOUNDARY = 90

REBALANCE_FREQ        = 15
INITIAL_CAPITAL       = 100000.0
RISK_FREE_RATE_ANNUAL = 0.03

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
    return 100 - (100 / (1 + rs))

def ricker_multi_step(n0, r_series, K_series, H):
    """
    Multi-step Ricker forecast:
        n_{t+1} = n_t * exp( r_t * (1 - n_t / K_t) )
    """
    n = n0.copy()
    for _ in range(H):
        n = n * np.exp(r_series * (1.0 - n / K_series))
    return n

def calc_metrics(returns, benchmark_returns, rf_annual):
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.loc[returns.index].dropna()
    returns = returns.loc[benchmark_returns.index]

    if len(returns) == 0:
        return {k: np.nan for k in
                ["annual_geom_return","cumulative_return","alpha_annual","beta","sharpe_ratio"]}

    trading_days = 252
    years = len(returns) / trading_days

    cumulative_return = (1 + returns).prod() - 1
    annual_geom = (1 + cumulative_return)**(1/years) - 1

    rf_daily = rf_annual / trading_days

    mean_ret   = returns.mean()
    mean_bench = benchmark_returns.mean()
    std_ret    = returns.std(ddof=0)

    beta = returns.cov(benchmark_returns) / benchmark_returns.var(ddof=0)
    alpha_annual = (mean_ret - rf_daily - beta*(mean_bench - rf_daily)) * trading_days
    sharpe = (mean_ret - rf_daily) / std_ret * np.sqrt(trading_days)

    return {
        "annual_geom_return": annual_geom,
        "cumulative_return": cumulative_return,
        "alpha_annual": alpha_annual,
        "beta": beta,
        "sharpe_ratio": sharpe,
    }

# -----------------------
# DATA DOWNLOAD
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

data = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
)

prices_df = data["Close"]

# -----------------------
# BUILD PER-TICKER RICKER FORECASTS
# -----------------------

per_ticker = {}
EPS = 1e-8

for ticker in prices_df.columns:

    prices = prices_df[ticker].dropna()
    daily_ret = prices.pct_change()

    geo_mean_daily = rolling_geo_mean_return(daily_ret, ROLL_WINDOW_DAYS)
    rsi = compute_rsi(prices, RSI_PERIOD)

    factor_low  = (100 - rsi) / 100
    factor_high = rsi / 100
    factor = np.where(rsi <= RSI_SHORT_BOUNDARY, factor_low, factor_high)

    geo_safe = geo_mean_daily.copy()
    geo_safe[np.abs(geo_safe) < EPS] = np.nan

    M_series = prices * factor / geo_safe
    R0_series = 1 + geo_mean_daily
    K_series = (R0_series - 1) * M_series  # SAME capacity definition

    r_series = 1 + geo_mean_daily

    ricker_pred = ricker_multi_step(prices, r_series, K_series, HORIZON_DAYS)
    baseline_pred = prices * (R0_series ** HORIZON_DAYS)

    df = pd.DataFrame({
        "price": prices,
        "next_ret": daily_ret.shift(-1),
        "ricker_pred": ricker_pred,
        "baseline_pred": baseline_pred
    }).dropna()

    if not df.empty:
        per_ticker[ticker] = df

# -----------------------
# CROSS-SECTIONAL BACKTEST
# -----------------------

all_dates = prices_df.index

ricker_rets = []
baseline_rets = []
eqw_rets = []

weights_prev_ricker = None
weights_prev_base = None

for i, dt in enumerate(all_dates):

    diffs_ricker = {}
    diffs_base = {}
    next_rets = {}

    for ticker, df in per_ticker.items():
        if dt not in df.index:
            continue

        row = df.loc[dt]
        next_rets[ticker] = row["next_ret"]
        diffs_ricker[ticker] = row["ricker_pred"] - row["price"]
        diffs_base[ticker]   = row["baseline_pred"] - row["price"]

    if not next_rets:
        ricker_rets.append(0)
        baseline_rets.append(0)
        eqw_rets.append(0)
        continue

    eqw_rets.append(np.mean(list(next_rets.values())))

    rebalance = (i == 0) or (i % REBALANCE_FREQ == 0)

    if rebalance:
        n = len(diffs_ricker)
        weights_prev_ricker = {
            t: np.sign(diffs_ricker[t]) / n for t in diffs_ricker
        }
        weights_prev_base = {
            t: np.sign(diffs_base[t]) / n for t in diffs_base
        }

    ricker_rets.append(
        sum(weights_prev_ricker.get(t, 0) * next_rets[t] for t in next_rets)
    )
    baseline_rets.append(
        sum(weights_prev_base.get(t, 0) * next_rets[t] for t in next_rets)
    )

ricker_rets = pd.Series(ricker_rets, index=all_dates)
baseline_rets = pd.Series(baseline_rets, index=all_dates)
eqw_rets = pd.Series(eqw_rets, index=all_dates)

# -----------------------
# EQUITY CURVES
# -----------------------

equity = pd.DataFrame({
    "Ricker Strategy": (1 + ricker_rets).cumprod() * INITIAL_CAPITAL,
    "Baseline Strategy": (1 + baseline_rets).cumprod() * INITIAL_CAPITAL,
    "Equal-Weight": (1 + eqw_rets).cumprod() * INITIAL_CAPITAL,
})

# -----------------------
# PERFORMANCE
# -----------------------

perf = pd.DataFrame.from_dict({
    "Ricker Strategy": calc_metrics(ricker_rets, eqw_rets, RISK_FREE_RATE_ANNUAL),
    "Baseline Strategy": calc_metrics(baseline_rets, eqw_rets, RISK_FREE_RATE_ANNUAL),
    "Equal-Weight": calc_metrics(eqw_rets, eqw_rets, RISK_FREE_RATE_ANNUAL),
}, orient="index")

print("\n=== PERFORMANCE SUMMARY ===")
print(perf.to_string(float_format=lambda x: f"{x:0.4f}"))

# -----------------------
# PLOT
# -----------------------

plt.figure(figsize=(10,5))
plt.plot(equity, linewidth=2)
plt.title("Ricker Model Trading Strategy")
plt.ylabel("Equity")
plt.grid(True)
plt.tight_layout()
plt.show()

