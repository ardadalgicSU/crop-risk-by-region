"""
Generate core report figures for P-band, Y-band, and Monte Carlo results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")
sns.set_style("whitegrid")

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
PRED_DIR = REPORTS_DIR / "predictions"

CROPS = ["wheat", "barley", "maize"]


def load_price_monthly(crop: str) -> pd.DataFrame:
    path = DATA_DIR / f"P-{crop}-monthly.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
    )
    return df.sort_values("date").reset_index(drop=True)


def load_yield_features(crop: str) -> pd.DataFrame:
    path = DATA_DIR / f"Y-{crop}-highfreq.csv"
    df = pd.read_csv(path)
    return df.sort_values("year").reset_index(drop=True)


def load_mc_simulations(crop: str) -> pd.DataFrame:
    path = PRED_DIR / f"monte_carlo_{crop}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_mc_summary() -> pd.DataFrame:
    path = PRED_DIR / "monte_carlo_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_price_real_all() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for crop in CROPS:
        df = load_price_monthly(crop)
        ax.plot(df["date"], df["price_real"], label=crop)
    ax.set_title("Monthly Real Prices by Crop")
    ax.set_xlabel("Date")
    ax.set_ylabel("Real price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_real_monthly_all_crops.png", dpi=150)
    plt.close(fig)


def plot_nominal_vs_real_wheat() -> None:
    df = load_price_monthly("wheat")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["price_nominal"], label="nominal")
    ax.plot(df["date"], df["price_real"], label="real")
    ax.set_title("Wheat: Nominal vs Real Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_nominal_vs_real_wheat.png", dpi=150)
    plt.close(fig)


def plot_price_seasonality() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, crop in zip(axes, CROPS):
        df = load_price_monthly(crop)
        df["month"] = df["date"].dt.month
        sns.boxplot(x="month", y="price_real", data=df, ax=ax, color="#5B8FF9")
        ax.set_title(f"{crop.title()} Monthly Seasonality")
        ax.set_xlabel("Month")
        ax.set_ylabel("Real price")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_seasonality_boxplot.png", dpi=150)
    plt.close(fig)


def plot_yield_timeseries() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for crop in CROPS:
        df = load_yield_features(crop)
        ax.plot(df["year"], df["yield_t_ha"], marker="o", label=crop)
    ax.set_title("Annual Yield (t/ha) by Crop")
    ax.set_xlabel("Year")
    ax.set_ylabel("Yield (t/ha)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "yield_timeseries_all_crops.png", dpi=150)
    plt.close(fig)


def plot_climate_yield_scatter() -> None:
    df = load_yield_features("wheat")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x="gs_precip_sum", y="yield_t_ha", data=df, ax=ax, scatter_kws={"s": 40})
    ax.set_title("Wheat: Yield vs GS Precipitation")
    ax.set_xlabel("GS precipitation sum")
    ax.set_ylabel("Yield (t/ha)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "yield_vs_gs_precip_wheat.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.regplot(x="gs_t2m_mean", y="yield_t_ha", data=df, ax=ax, scatter_kws={"s": 40})
    ax.set_title("Wheat: Yield vs GS Mean Temperature")
    ax.set_xlabel("GS mean temperature")
    ax.set_ylabel("Yield (t/ha)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "yield_vs_gs_temp_wheat.png", dpi=150)
    plt.close(fig)


def plot_hazard_yield_wheat() -> None:
    df = load_yield_features("wheat")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.regplot(
        x="gs_heatwave_35_days",
        y="yield_t_ha",
        data=df,
        ax=axes[0],
        scatter_kws={"s": 40},
    )
    axes[0].set_title("Wheat: Heatwave Days vs Yield")
    axes[0].set_xlabel("GS heatwave_35 days")
    axes[0].set_ylabel("Yield (t/ha)")

    sns.regplot(
        x="gs_dry_spell_max",
        y="yield_t_ha",
        data=df,
        ax=axes[1],
        scatter_kws={"s": 40},
    )
    axes[1].set_title("Wheat: Dry Spell Max vs Yield")
    axes[1].set_xlabel("GS dry spell max")
    axes[1].set_ylabel("Yield (t/ha)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hazard_vs_yield_wheat.png", dpi=150)
    plt.close(fig)


def plot_hazard_price_wheat() -> None:
    price = load_price_monthly("wheat")
    price_yearly = price.groupby("year", as_index=False).agg(price_mean=("price_real", "mean"))
    climate = load_yield_features("wheat")[["year", "gs_heatwave_35_days", "gs_dry_spell_max"]]
    df = price_yearly.merge(climate, on="year", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.regplot(
        x="gs_heatwave_35_days",
        y="price_mean",
        data=df,
        ax=axes[0],
        scatter_kws={"s": 40},
    )
    axes[0].set_title("Wheat: Heatwave Days vs Price")
    axes[0].set_xlabel("GS heatwave_35 days")
    axes[0].set_ylabel("Price mean")

    sns.regplot(
        x="gs_dry_spell_max",
        y="price_mean",
        data=df,
        ax=axes[1],
        scatter_kws={"s": 40},
    )
    axes[1].set_title("Wheat: Dry Spell Max vs Price")
    axes[1].set_xlabel("GS dry spell max")
    axes[1].set_ylabel("Price mean")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "hazard_vs_price_wheat.png", dpi=150)
    plt.close(fig)


def plot_earnings_timeseries() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for crop in ["wheat", "barley"]:
        price = load_price_monthly(crop)
        price_yearly = price.groupby("year", as_index=False).agg(price_mean=("price_real", "mean"))
        y = load_yield_features(crop)
        df = price_yearly.merge(
            y[["year", "production_mass_t"]],
            on="year",
            how="inner",
        )
        df["earnings"] = df["price_mean"] * df["production_mass_t"]
        ax.plot(df["year"], df["earnings"], marker="o", label=crop)
    ax.set_title("Annual Earnings (Price Mean x Production Mass)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Earnings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "earnings_timeseries_wheat_barley.png", dpi=150)
    plt.close(fig)


def plot_monte_carlo_distributions() -> None:
    for crop in ["wheat", "barley"]:
        sims = load_mc_simulations(crop)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sims["earnings"], bins=50, alpha=0.85, color="#6E9EEC")
        ax.set_title(f"Earnings Distribution (Monte Carlo) - {crop.title()}")
        ax.set_xlabel("Earnings")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"monte_carlo_distribution_{crop}.png", dpi=150)
        plt.close(fig)


def plot_risk_return() -> None:
    summary = load_mc_summary()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        summary["expected_earnings"],
        summary["earnings_std"],
        s=80,
        color="#E07A5F",
    )
    for _, row in summary.iterrows():
        ax.text(
            row["expected_earnings"],
            row["earnings_std"],
            row["crop"],
            fontsize=10,
            ha="left",
            va="bottom",
        )
    ax.set_title("Risk-Return (Monte Carlo)")
    ax.set_xlabel("Expected earnings")
    ax.set_ylabel("Earnings std")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monte_carlo_risk_return.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_price_real_all()
    plot_nominal_vs_real_wheat()
    plot_price_seasonality()
    plot_yield_timeseries()
    plot_climate_yield_scatter()
    plot_hazard_yield_wheat()
    plot_hazard_price_wheat()
    plot_earnings_timeseries()
    plot_monte_carlo_distributions()
    plot_risk_return()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
