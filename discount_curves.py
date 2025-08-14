import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_fred(filename, label):
    try:
        df = pd.read_csv(filename)
        df.columns = ['date', label]
        df['date'] = pd.to_datetime(df['date'])
        df[label] = pd.to_numeric(df[label], errors='coerce') / 100
        return df.set_index('date')
    except Exception as e:
        logging.error(f"Failed to load {filename}: {e}")
        return pd.DataFrame(columns=['date', label]).set_index('date')


def build_discount_curve(df_rates, label_to_years, target_tenors):
    records = []
    for date, row in df_rates.iterrows():
        available = row.dropna()
        if len(available) < 2:
            continue

        tenors = []
        dfs = []
        for label, r in available.items():
            T = label_to_years[label]
            tenors.append(T)
            dfs.append(np.exp(-r * T))

        try:
            interpolator = PchipInterpolator(tenors, dfs, extrapolate=True)
        except Exception as e:
            logging.warning(f"Interpolation failed on {date}: {e}")
            continue

        for label, T in target_tenors.items():
            df_val = float(interpolator(T))
            df_val = np.clip(df_val, 1e-6, 1.0)
            if T < 0.025:
                r = 1.0 - df_val
            else:
                r = -np.log(df_val) / T
            if T <= 0.1:
                r = np.clip(r, -0.05, 0.25)
            records.append({
                'date': date,
                'tenor': label,
                'tenor_years': T,
                'discount_factor': df_val,
                'interpolated_rate': r
            })
    return pd.DataFrame(records)


def plot_3d_surface(df_discount):
    os.makedirs("results", exist_ok=True)
    pivot = df_discount.pivot_table(index="date", columns="tenor_years", values="interpolated_rate")
    dates = mdates.date2num(pivot.index)
    tenors = pivot.columns.values
    rates = pivot.values

    # Matplotlib 3D surface
    X, Y = np.meshgrid(tenors, dates)
    Z = rates
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Date")
    ax.set_zlabel("Interpolated Rate")
    ax.set_title("Yield Curves Over Time")
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.colorbar(surf, shrink=0.5, aspect=10, label="Interpolated Rate")
    plt.tight_layout()
    plt.savefig("results/discount_curves_3d.png")
    plt.close()
    logging.info("✅ PNG plot saved to results/discount_curves_3d.png")


def plot_yield_curve_inversion(df_discount):
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
        plt.title("Yield Curve Inversion (1Y - 3M Spread)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/yield_curve_inversion.png")
        plt.close()


def main():
    data_dir = Path("data/input/FRED")
    label_to_years = {
        "1M": 1 / 12,
        "3M": 3 / 12,
        "6M": 6 / 12,
        "12M": 1.0
    }
    target_tenors = {
        '1W': 1 / 52,
        '2W': 2 / 52,
        '3W': 3 / 52,
        '1M': 1 / 12,
        '2M': 2 / 12,
        '3M': 3 / 12,
        '4M': 4 / 12,
        '6M': 6 / 12,
        '9M': 9 / 12,
        '1Y': 1.0
    }

    # Check data directory and files
    if not data_dir.exists():
        logging.error(f"Data directory {data_dir} does not exist.")
        return

    df_1m = load_fred(data_dir / "DTB4WK.csv", "1M")
    df_3m = load_fred(data_dir / "DTB3.csv", "3M")
    df_6m = load_fred(data_dir / "DTB6.csv", "6M")
    df_1y = load_fred(data_dir / "DTB1YR.csv", "12M")

    df_rates = df_1m.join([df_3m, df_6m, df_1y], how='outer').sort_index()
    df_discount = build_discount_curve(df_rates, label_to_years, target_tenors)
    df_discount = df_discount.sort_values(['date', 'tenor'])
    os.makedirs("data", exist_ok=True)
    df_discount.to_parquet("data/discount_curves.parquet", index=False)
    logging.info("✅ Discount curve saved to data/discount_curves.parquet")
    plot_3d_surface(df_discount)
    plot_yield_curve_inversion(df_discount)


if __name__ == "__main__":
    main()
