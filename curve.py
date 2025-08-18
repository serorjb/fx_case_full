import os

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("ignore", OptimizeWarning)

DATA_DIR = Path('data/input/FRED')
RESULTS_DIR = Path('results')
PARQUET_PATH = Path('data/output/curve.parquet')

TARGET_TENORS = {
    '1W': 1 / 52, '2W': 2 / 52, '3W': 3 / 52,
    '1M': 1 / 12, '2M': 2 / 12, '3M': 3 / 12,
    '4M': 4 / 12, '6M': 6 / 12, '9M': 9 / 12, '1Y': 1.0
}


def load_fred_series(filename, label):
    df = pd.read_csv(filename)
    df.columns = ['date', label]
    df['date'] = pd.to_datetime(df['date'])
    df[label] = pd.to_numeric(df[label], errors='coerce') / 100
    return df.set_index('date')


def nelson_siegel(T, beta0, beta1, beta2, tau):
    """Nelson-Siegel yield curve model."""
    T = np.maximum(T, 1e-6)
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-T / tau)) / (T / tau)
    term3 = beta2 * ((1 - np.exp(-T / tau)) / (T / tau) - np.exp(-T / tau))
    return term1 + term2 + term3


def fit_nelson_siegel(tenors, rates):
    # Initial guess: [level, slope, curvature, tau]
    guess = [rates[-1], rates[0] - rates[-1], 0.0, 0.5]
    try:
        popt, _ = curve_fit(nelson_siegel, tenors, rates, p0=guess, bounds=([-1, -1, -1, 0.01], [1, 1, 1, 10]))
        return popt
    except Exception:
        return None


def build_discount_curve(df_rates):
    records = []
    for date, row in df_rates.iterrows():
        tenors, rates = [], []
        for label, rate in row.dropna().items():
            T = int(label.strip('MY')) / 12 if 'M' in label else float(label.strip('Y'))
            tenors.append(T)
            rates.append(rate)
        if len(tenors) < 2:
            continue
        params = fit_nelson_siegel(np.array(tenors), np.array(rates))
        if params is None:
            continue
        for label, T in TARGET_TENORS.items():
            r = nelson_siegel(T, *params)
            df_val = np.exp(-r * T)
            records.append({
                'date': date, 'tenor': label, 'tenor_years': T,
                'discount_factor': df_val, 'interpolated_rate': r
            })
    return pd.DataFrame(records)


def save_parquet(df, path):
    df.to_parquet(path, index=False)
    print(f"✅ Discount curve saved to {path}")


def plot_3d_matplotlib(df):
    pivot = df.pivot_table(index='date', columns='tenor_years', values='interpolated_rate')
    dates = mdates.date2num(pivot.index)
    tenors = pivot.columns.values
    rates = pivot.values
    X, Y = np.meshgrid(tenors, dates)
    Z = rates
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Date")
    ax.set_zlabel("Interpolated Rate")
    ax.set_title("Curve Over Time")
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.colorbar(surf, shrink=0.5, aspect=10, label="Interpolated Rate")
    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "curve.png")
    plt.close()
    print(f"✅ PNG plot saved to {RESULTS_DIR / 'curve.png'}")


def plot_yield_curves_and_spread(df_discount):
    os.makedirs("results", exist_ok=True)
    pivot = df_discount.pivot_table(index="date", columns="tenor_years", values="interpolated_rate")

    # Plot 1Y-3M spread to show inversion
    if 1.0 in pivot.columns and 0.25 in pivot.columns:
        spread = pivot[1.0] - pivot[0.25]
        plt.figure(figsize=(16, 4))
        plt.plot(pivot.index, spread, label="1Y - 3M Spread")
        plt.axhline(0, color="red", linestyle="--", label="Inversion Threshold")
        plt.xlabel("Date")
        plt.ylabel("Spread (1Y - 3M)")
        plt.title("Curve Inversion (1Y - 3M Spread)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/curve_inversion.png")
        plt.close()


def main():
    df_1m = load_fred_series(DATA_DIR / "DTB1.csv", "1M")
    df_3m = load_fred_series(DATA_DIR / "DTB3.csv", "3M")
    df_6m = load_fred_series(DATA_DIR / "DTB6.csv", "6M")
    df_1y = load_fred_series(DATA_DIR / "DTB12.csv", "1Y")
    df_rates = df_1m.join([df_3m, df_6m, df_1y], how='outer').sort_index()
    df_discount = build_discount_curve(df_rates)
    df_discount = df_discount.sort_values(['date', 'tenor'])
    save_parquet(df_discount, PARQUET_PATH)
    plot_yield_curves_and_spread(df_discount)
    plot_3d_matplotlib(df_discount)


if __name__ == "__main__":
    main()
