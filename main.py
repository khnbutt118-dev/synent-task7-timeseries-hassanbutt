"""Synent Data Science Internship - Task 7: Time Series Analysis (Stock Prices)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'stock': {'files': ['stock.csv', 'all_stocks_5yr.csv'],
           'kaggle': 'camnugent/sandp500',
           'name': 'S&P 500 daily stock prices (5 years)',
           'page': 'https://www.kaggle.com/datasets/camnugent/sandp500'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def _load_stock_series(repo_dir):
    try:
        df = get_data("stock", repo_dir)
        df.columns = [str(c).lower() for c in df.columns]
        if "name" in df.columns:
            tick = "AAPL" if "AAPL" in set(df["name"]) else df["name"].iloc[0]
            df = df[df["name"] == tick]
        else:
            tick = "STOCK"
        date_col = next(c for c in df.columns if "date" in c)
        close_col = "close" if "close" in df.columns else next(c for c in df.columns if "close" in c)
        s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values, index=pd.to_datetime(df[date_col]), name="close")
        return s.dropna().sort_index(), tick
    except FileNotFoundError as e:
        print("  [data] stock CSV not found; trying yfinance ...")
        try:
            import yfinance as yf
            d = yf.download("AAPL", period="5y", progress=False, auto_adjust=True)
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna().rename("close"), "AAPL"
        except Exception:
            raise e


def task7_run(repo_dir):
    s, tick = _load_stock_series(repo_dir)
    s = s[~s.index.duplicated()]
    shape = [int(len(s)), 1]
    ret = s.pct_change().dropna()
    roll30, roll90 = s.rolling(30).mean(), s.rolling(90).mean()
    vol30 = ret.rolling(30).std() * np.sqrt(252) * 100

    # Trend
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1 if years > 0 else float("nan")
    slope = np.polyfit(np.arange(len(s)), s.values, 1)[0]
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax[0].plot(s.index, s.values, label="Close price", color="#2c3e50", lw=1)
    ax[0].plot(roll30.index, roll30.values, label="30-day average", color="#e67e22")
    ax[0].plot(roll90.index, roll90.values, label="90-day average", color="#c0392b")
    ax[0].set_title(f"{tick}: price trend with moving averages")
    ax[0].legend()
    ax[1].plot(vol30.index, vol30.values, color="#8e44ad")
    ax[1].set_title("30-day rolling volatility (annualised %)")
    fig.tight_layout()
    save_fig(fig, repo_dir, "trend_and_volatility.png")

    # Seasonality
    monthly = s.resample("MS").mean().dropna()
    decomposed = False
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        if len(monthly) >= 24:
            dec = seasonal_decompose(monthly, model="additive", period=12)
            fig = dec.plot()
            fig.set_size_inches(11, 8)
            save_fig(fig, repo_dir, "seasonal_decomposition.png")
            decomposed = True
    except Exception:
        pass
    mret = ret.groupby(ret.index.month).mean() * 100
    dret = ret.groupby(ret.index.dayofweek).mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    mret.plot(kind="bar", ax=ax[0], color=["#27ae60" if v >= 0 else "#c0392b" for v in mret])
    ax[0].set_title("Average daily return by calendar month (%)")
    ax[0].set_xlabel("Month")
    dret.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(dret)] if len(dret) <= 7 else dret.index
    dret.plot(kind="bar", ax=ax[1], color="#2980b9")
    ax[1].set_title("Average daily return by weekday (%)")
    save_fig(fig, repo_dir, "seasonality_patterns.png")

    # Optional forecasting with honest evaluation
    test_len = 60
    train, test = s.iloc[:-test_len], s.iloc[-test_len:]
    res = {}
    res["Naive (last value)"] = np.repeat(train.iloc[-1], test_len)
    win = min(252, len(train))
    coef = np.polyfit(np.arange(win), train.iloc[-win:].values, 1)
    res["Linear trend (last year)"] = np.polyval(coef, np.arange(win, win + test_len))
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw = ExponentialSmoothing(train.reset_index(drop=True), trend="add", damped_trend=True).fit()
        res["Holt-Winters (damped trend)"] = np.asarray(hw.forecast(test_len))
    except Exception:
        pass
    rows = []
    for name, pred in res.items():
        rmse = float(np.sqrt(np.mean((test.values - pred) ** 2)))
        mape = float(np.mean(np.abs((test.values - pred) / test.values)) * 100)
        rows.append({"model": name, "RMSE": round(rmse, 2), "MAPE_%": round(mape, 2)})
    metrics = pd.DataFrame(rows).sort_values("RMSE")
    metrics.to_csv(_outdir(repo_dir) / "forecast_model_comparison.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    h = 30
    if best_name.startswith("Holt"):
        hw_full = ExponentialSmoothing(s.reset_index(drop=True), trend="add", damped_trend=True).fit()
        future = np.asarray(hw_full.forecast(h))
    elif best_name.startswith("Linear"):
        w = min(252, len(s))
        c2 = np.polyfit(np.arange(w), s.iloc[-w:].values, 1)
        future = np.polyval(c2, np.arange(w, w + h))
    else:
        future = np.repeat(s.iloc[-1], h)
    fidx = pd.bdate_range(s.index[-1] + pd.Timedelta(days=1), periods=h)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(s.index[-250:], s.values[-250:], label="History", color="#2c3e50")
    ax.plot(test.index, res[best_name], label=f"Backtest ({best_name})", color="#e67e22", ls="--")
    ax.plot(fidx, future, label=f"30-day forecast ({best_name})", color="#c0392b")
    ax.set_title(f"{tick}: backtest and 30-business-day forecast (educational, not financial advice)")
    ax.legend()
    save_fig(fig, repo_dir, "forecast.png")

    best_month = int(mret.idxmax())
    worst_month = int(mret.idxmin())
    insights = [
        f"{tick} moved from {s.iloc[0]:.2f} to {s.iloc[-1]:.2f} over {years:.1f} years (CAGR about {cagr * 100:.1f}% per year).",
        f"Average annualised volatility (30-day rolling) was about {float(vol30.mean()):.0f}%, ranging from {float(vol30.min()):.0f}% to {float(vol30.max()):.0f}%.",
        f"Calendar month {best_month} had the highest and month {worst_month} the lowest average daily return, but with only about {years:.0f} years of data this seasonality is weak evidence.",
        f"In the backtest on the last {test_len} trading days the best model was '{best_name}' (MAPE {metrics.iloc[0]['MAPE_%']:.1f}%); forecasts of prices are uncertain and are shown for learning only.",
    ]
    return {"headline": {"Ticker": tick, "Observations": shape[0], "CAGR": f"{cagr * 100:.1f}%",
                         "Best forecast model": best_name, "Backtest MAPE": f"{metrics.iloc[0]['MAPE_%']:.1f}%"},
            "insights": insights, "dataset_shape": shape,
            "files": ["trend_and_volatility.png", "seasonality_patterns.png", "forecast.png", "forecast_model_comparison.csv"]
            + (["seasonal_decomposition.png"] if decomposed else [])}



if __name__ == "__main__":
    res = task7_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
