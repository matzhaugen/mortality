"""
Python translation of deaths_over_time.ipynb.

Question: How many deaths can be predicted by vaccination rates?
Strategy: regress weekly state-level "centered" deaths onto current and lagged
weekly vaccination counts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")

STATE_CODE_TO_NAME = {
    "AK": "alaska", "AL": "alabama", "AR": "arkansas", "AZ": "arizona",
    "CA": "california", "CO": "colorado", "CT": "connecticut",
    "DC": "district of columbia", "DE": "delaware", "FL": "florida",
    "GA": "georgia", "HI": "hawaii", "IA": "iowa", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "KS": "kansas", "KY": "kentucky",
    "LA": "louisiana", "MA": "massachusetts", "MD": "maryland", "ME": "maine",
    "MI": "michigan", "MN": "minnesota", "MO": "missouri", "MS": "mississippi",
    "MT": "montana", "NC": "north carolina", "ND": "north dakota",
    "NE": "nebraska", "NH": "new hampshire", "NJ": "new jersey",
    "NM": "new mexico", "NV": "nevada", "NY": "new york", "OH": "ohio",
    "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "PR": "puerto rico", "RI": "rhode island", "SC": "south carolina",
    "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VA": "virginia", "VT": "vermont", "WA": "washington", "WI": "wisconsin",
    "WV": "west virginia", "WY": "wyoming",
}


def load_deaths() -> pd.DataFrame:
    """Read centered deaths data (see the 'statewide' notebook for derivation)."""
    deaths = pd.read_csv(DATA_DIR / "state_centered.csv")
    deaths["Week.Ending.Date"] = pd.to_datetime(deaths["Week.Ending.Date"])
    deaths["State"] = deaths["State"].str.lower()
    deaths["Date"] = deaths["Week.Ending.Date"]
    return deaths


def load_vaccinations() -> pd.DataFrame:
    """Read raw CDC vaccination data and aggregate to weekly state totals."""
    raw = pd.read_csv(
        DATA_DIR / "COVID-19_Vaccinations_in_the_United_States_Jurisdiction.csv"
    )
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw["year"] = raw["Date"].dt.year

    weekly = (
        raw.groupby(["MMWR_week", "year", "Location"], as_index=False)
        .agg(
            Administered_12Plus=("Administered_12Plus", "sum"),
            Admin_Per_100k_12Plus=("Admin_Per_100k_12Plus", "sum"),
            Administered=("Administered", "sum"),
            Admin_Per_100K=("Admin_Per_100K", "sum"),
            Date=("Date", "max"),
        )
    )
    weekly["State"] = weekly["Location"].map(STATE_CODE_TO_NAME)
    weekly = weekly.dropna(subset=["State"]).copy()
    return weekly


def add_lag_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add current and lagged week-over-week vaccination deltas, per state."""
    weekly = weekly.sort_values(["Location", "Date"]).reset_index(drop=True)

    grouped = weekly.groupby("Location", group_keys=False)
    weekly["l_Admin_Per_100K"] = grouped["Admin_Per_100K"].diff().fillna(0)
    weekly["l_Administered_12Plus"] = grouped["Administered_12Plus"].diff().fillna(0)
    weekly["l_Administered"] = grouped["Administered"].diff().fillna(0)

    # Lagged versions of the weekly delta (l2 = delta two weeks ago, etc.)
    for lag in range(2, 6):
        weekly[f"l{lag}_Administered"] = (
            grouped["l_Administered"].shift(lag - 1).fillna(0)
        )

    weekly = weekly[weekly["Date"] < pd.Timestamp("2022-12-15")]
    nonneg_cols = [
        "l_Administered", "l2_Administered",
        "l3_Administered", "l4_Administered",
    ]
    mask = (weekly[nonneg_cols] >= 0).all(axis=1)
    return weekly[mask].copy()


def plot_vaccinations_with_deaths(
    vacc_weekly: pd.DataFrame, deaths: pd.DataFrame, big_states: list[str],
    name: str = "vaccinations_with_deaths",
) -> None:
    """Vaccination weekly delta with Total Deaths overlaid on a second y-axis."""
    plot_df = vacc_weekly[vacc_weekly["State"].isin(big_states)].merge(
        deaths[["Date", "State", "Total.Deaths"]], on=["Date", "State"]
    )

    n = len(big_states)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    for ax, state in zip(axes, big_states):
        sub = plot_df[plot_df["State"] == state].sort_values("Date")
        ax.plot(sub["Date"], sub["l_Administered"], color="tab:blue",
                label="Vaccinations Administered (weekly delta)")
        ax.set_ylabel("Vaccinations", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_title(state)

        ax2 = ax.twinx()
        ax2.plot(sub["Date"], sub["Total.Deaths"], color="tab:red",
                 linestyle="--", label="Total Deaths")
        ax2.set_ylabel("Total Deaths", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Weekly vaccinations + Total Deaths by state")
    fig.tight_layout()
    _save(fig, name)


def fit_models(df: pd.DataFrame) -> dict:
    """Fit OLS and three quantile regressions of centered deaths on lagged vaccinations."""
    lag_cols = [
        "l_Administered", "l2_Administered",
        "l3_Administered", "l4_Administered", "l5_Administered",
    ]

    # Full-lag fits
    X_full = df[lag_cols].to_numpy()
    y = df["centered"].to_numpy()
    full_ols = sm.OLS(y, X_full).fit()
    full_qr = QuantReg(y, X_full).fit(q=0.5)
    print("=== Full-lag OLS ===")
    print(full_ols.summary())
    print("=== Full-lag Quantile (median) ===")
    print(full_qr.summary())

    # Best simplified model: only the current week's vaccination delta
    X = df[["l_Administered"]].to_numpy()
    ols = sm.OLS(y, X).fit()
    qr_med = QuantReg(y, X).fit(q=0.5)
    qr_low = QuantReg(y, X).fit(q=0.25)
    qr_vlow = QuantReg(y, X).fit(q=0.05)
    qr_high = QuantReg(y, X).fit(q=0.75)

    print("=== Single-lag OLS ===")
    print(ols.summary())
    print("=== Single-lag Quantile (median) ===")
    print(qr_med.summary())

    return {
        "ols": ols, "qr_med": qr_med,
        "qr_low": qr_low, "qr_vlow": qr_vlow, "qr_high": qr_high,
    }


def attach_predictions(df: pd.DataFrame, fits: dict) -> pd.DataFrame:
    X = df[["l_Administered"]].to_numpy()
    df = df.copy()
    df["death_estimate"] = fits["ols"].predict(X)
    df["robust_death_estimate"] = fits["qr_med"].predict(X)
    df["robust_death_estimate_low"] = fits["qr_low"].predict(X)
    df["robust_death_estimate_high"] = fits["qr_high"].predict(X)
    df["residuals"] = df["centered"].to_numpy() - df["death_estimate"]
    df["non_flu_like"] = df["centered"] - df["centered_flu_like"]
    return df


def plot_estimate_lines(df: pd.DataFrame, states: list[str], y_col: str,
                        title: str, name: str) -> None:
    sub = df[df["State"].isin(states)].sort_values("Date")
    fig, ax = plt.subplots(figsize=(15, 8))
    for state in states:
        s = sub[sub["State"] == state]
        ax.plot(s["Date"], s[y_col], label=state)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, name)


def plot_l1_vs_l2(df: pd.DataFrame, states: list[str], title: str, name: str) -> None:
    sub = df[df["State"].isin(states)].sort_values("Date")
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.get_cmap("tab10")
    for i, state in enumerate(states):
        s = sub[sub["State"] == state]
        ax.plot(s["Date"], s["robust_death_estimate"],
                color=colors(i), linestyle="-", label=f"{state} (L1)")
        ax.plot(s["Date"], s["death_estimate"],
                color=colors(i), linestyle="--", label=f"{state} (L2)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, name)


def plot_estimate_with_band(df: pd.DataFrame, states: list[str], title: str,
                            name: str) -> None:
    n = len(states)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax, state in zip(axes, states):
        s = df[df["State"] == state].sort_values("Date")
        ax.plot(s["Date"], s["robust_death_estimate"], label="median")
        ax.fill_between(
            s["Date"],
            s["robust_death_estimate_low"],
            s["robust_death_estimate_high"],
            alpha=0.3,
            label="25–75%",
        )
        ax.set_title(state)
        ax.set_xlabel("Date")
        ax.set_ylabel("Estimated deaths")
        ax.legend()
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, name)


def main() -> None:
    deaths = load_deaths()
    vacc_weekly = add_lag_features(load_vaccinations())

    # Pick big and small states by total deaths.
    sum_deaths = (
        deaths.groupby("State", as_index=False)["Total.Deaths"].sum()
        .rename(columns={"Total.Deaths": "sum"})
        .sort_values("sum")
    )
    big_states = sum_deaths.tail(7)["State"].tolist()
    small_states = sum_deaths.head(7)["State"].tolist()
    print("big_states:", big_states)

    plot_vaccinations_with_deaths(vacc_weekly, deaths, big_states)

    df = deaths.merge(vacc_weekly, on=["Date", "State"])
    print("merged shape:", df.shape)

    fits = fit_models(df)
    df = attach_predictions(df, fits)

    plot_estimate_lines(df, big_states, "death_estimate",
                        "OLS death estimate by state (big)",
                        name="ols_death_estimate_big")
    plot_l1_vs_l2(df, big_states[:3], "L1 vs L2 fits — big states",
                  name="l1_vs_l2_big")
    plot_l1_vs_l2(df, small_states[:3], "L1 vs L2 fits — small states",
                  name="l1_vs_l2_small")

    print(f"L2-norm overall estimated deaths: {round(fits['ols'].fittedvalues.sum())}")
    print(f"L1-norm overall estimated deaths: {round(fits['qr_med'].fittedvalues.sum())}")
    print(
        "25th/75th IQR: "
        f"[{round(fits['qr_low'].fittedvalues.sum())}, "
        f"{round(fits['qr_high'].fittedvalues.sum())}]"
    )
    print(
        "NOT SIGNIFICANT at 5th quantile: "
        f"{round(fits['qr_vlow'].fittedvalues.sum())}"
    )

    plot_estimate_with_band(df, big_states,
                            "Robust estimate with IQR band — big states",
                            name="robust_estimate_band_big")
    plot_estimate_with_band(df, small_states,
                            "Robust estimate with IQR band — small states",
                            name="robust_estimate_band_small")


if __name__ == "__main__":
    main()
