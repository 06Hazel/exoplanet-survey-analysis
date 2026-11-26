"""Reproduce the three insights and the figures used in the README.

Input : data/exoplanet_cleaned.csv
Output: figures/01_discovery_trend.png
        figures/02_mass_orbit.png
        figures/03_thermal_zones.png
        data/insight_tables.xlsx   (summary tables for inspection in Excel)

Run: python src/analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exoplanet_cleaned.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# palette shared with the slide deck
BLUE, DEEP, MID, LIGHT = "#1E4FA8", "#0E3F8C", "#3D7BD9", "#A8C2E8"
GREY, AMBER, INK = "#8B97A8", "#FFC107", "#1A2230"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#D6DCE5",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------------- #
# Insight 1 — how the detection method mix changed over time
# --------------------------------------------------------------------------- #
def insight_1(df: pd.DataFrame) -> pd.DataFrame:
    methods = ["Transit", "Radial Velocity", "Microlensing", "Imaging"]
    sub = df[df["discoverymethod"].isin(methods)]
    yearly = (sub.pivot_table(index="disc_year", columns="discoverymethod",
                              values="pl_name", aggfunc="count")
                 .reindex(columns=methods).fillna(0).astype(int))

    early = sub[sub.disc_year <= 2008]
    late = sub[sub.disc_year >= 2010]
    transit_share_late = (late.assign(t=late.discoverymethod.eq("Transit"))
                          .groupby("disc_year")["t"].mean().mean() * 100)

    print("\n== Insight 1 ==")
    print(f"planets across four methods : {len(sub):,}")
    print(f"1992-2008 : {len(early)} planets, Radial Velocity "
          f"{(early.discoverymethod == 'Radial Velocity').sum()}, "
          f"Transit {(early.discoverymethod == 'Transit').sum()} "
          f"({(early.discoverymethod == 'Transit').mean() * 100:.1f}%)")
    print(f"2010-2026 : Transit averages {transit_share_late:.1f}% of annual discoveries")
    print(f"peak years: {yearly.sum(axis=1).sort_values(ascending=False).head(3).to_dict()}")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(yearly))
    colors = [BLUE, MID, AMBER, GREY]
    for method, color in zip(methods, colors):
        ax.bar(yearly.index, yearly[method], bottom=bottom, width=0.78,
               label=method, color=color, edgecolor="white", linewidth=0.6)
        bottom += yearly[method].values

    ax.axvline(2009, color="#D9534F", linestyle="--", linewidth=1.2)
    ax.annotate("Kepler launch (2009)\ntransit surveys take over",
                xy=(2009, bottom.max() * 0.62), xytext=(1992.5, bottom.max() * 0.86),
                fontsize=10, color="#D9534F", va="center",
                arrowprops=dict(arrowstyle="->", color="#D9534F", linewidth=1.1,
                                shrinkA=4, shrinkB=2))
    ax.set_title("Annual confirmed exoplanets by detection method, 1992-2026",
                 fontsize=14, weight="bold", loc="left", pad=14)
    ax.set_xlabel("Year of discovery")
    ax.set_ylabel("Planets confirmed per year")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_xlim(1991, 2027)
    ax.grid(axis="y", color="#EEF1F5", linewidth=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "01_discovery_trend.png", dpi=180)
    plt.close(fig)
    return yearly


# --------------------------------------------------------------------------- #
# Insight 2 — mass versus orbital distance
# --------------------------------------------------------------------------- #
def insight_2(df: pd.DataFrame) -> pd.DataFrame:
    pairs = df.dropna(subset=["pl_bmasse", "pl_orbsmax"])
    edges = [0, 0.1, 0.5, 2, 10, np.inf]
    labels = ["<0.1", "0.1-0.5", "0.5-2", "2-10", ">10"]
    pairs = pairs.assign(bin=pd.cut(pairs.pl_orbsmax, edges, labels=labels, right=False))

    table = (pairs.groupby("bin", observed=True)
                  .agg(planets=("pl_name", "size"),
                       median_mass=("pl_bmasse", "median"),
                       median_period=("pl_orbper", "median")))
    share_close = len(pairs[pairs.pl_orbsmax < 0.5]) / len(pairs) * 100
    ratio = table.loc[">10", "median_mass"] / table.loc["<0.1", "median_mass"]
    hot_jup = pairs[(pairs.pl_orbsmax < 0.1) & (pairs.pl_bmasse > 100)]

    print("\n== Insight 2 ==")
    print(f"planets with both mass and orbit : {len(pairs):,}")
    print(table.round(1).to_string())
    print(f"within 0.5 au : {share_close:.1f}%")
    print(f"median mass ratio (>10 au vs <0.1 au) : {ratio:.0f}x")
    print(f"hot Jupiters (>100 Me, <0.1 au) : {len(hot_jup)}, "
          f"median period {hot_jup.pl_orbper.median():.1f} d")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(table))
    bars = ax.bar(x, table["median_mass"], width=0.62, color=BLUE,
                  edgecolor="white", linewidth=0.8)
    bars[0].set_color(LIGHT)
    ax.set_yscale("log")
    for xi, (n, mass) in enumerate(zip(table["planets"], table["median_mass"])):
        label = f"{mass:.1f} M$_\\oplus$" if mass < 10 else f"{mass:,.0f} M$_\\oplus$"
        ax.annotate(f"{label}\n(n={n:,})", (xi, mass),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x, table.index, fontsize=10)
    ax.set_title("Median planet mass rises steeply with orbital distance",
                 fontsize=14, weight="bold", loc="left", pad=14)
    ax.set_xlabel("Orbital semi-major axis (au)")
    ax.set_ylabel("Median mass (Earth masses, log scale)")
    ax.set_ylim(4, 12_000)
    ax.grid(axis="y", color="#EEF1F5", linewidth=0.9)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "02_mass_orbit.png", dpi=180)
    plt.close(fig)
    return table


# --------------------------------------------------------------------------- #
# Insight 3 — thermal zones and host-star temperature
# --------------------------------------------------------------------------- #
def insight_3(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.dropna(subset=["pl_eqt"])
    zones = pd.Series(np.where(temp.pl_eqt > 320, "Hot",
                      np.where(temp.pl_eqt >= 200, "Temperate", "Cold")),
                      index=temp.index)
    table = (temp.assign(zone=zones)
                 .groupby("zone")
                 .agg(planets=("pl_name", "size"),
                      mean_host_teff=("st_teff", "mean"))
                 .reindex(["Hot", "Temperate", "Cold"]))
    table["share"] = table["planets"] / len(temp) * 100
    gap = table.loc["Hot", "mean_host_teff"] - table.loc["Temperate", "mean_host_teff"]

    print("\n== Insight 3 ==")
    print(f"planets with measured Teq : {len(temp):,}")
    print(table.round(1).to_string())
    print(f"hot / temperate ratio : {table.loc['Hot','planets'] / table.loc['Temperate','planets']:.1f}")
    print(f"temperate hosts are {gap:.0f} K cooler than hot-planet hosts")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ["#C9463F", AMBER, MID]
    bars = ax.bar(table.index, table["planets"], width=0.55,
                  color=colors, edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, table.itertuples()):
        ax.annotate(f"{row.planets:,}  ({row.share:.1f}%)\nhost star {row.mean_host_teff:,.0f} K",
                    (bar.get_x() + bar.get_width() / 2, row.planets),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color=INK)
    ax.set_title("88.5% of measured planets are hotter than 320 K",
                 fontsize=14, weight="bold", loc="left", pad=14)
    ax.set_ylabel("Planets (equilibrium temperature measured)")
    ax.set_ylim(0, table["planets"].max() * 1.28)
    ax.grid(axis="y", color="#EEF1F5", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.text(0.99, 0.94,
            "Hot zones outnumber temperate ones 14.5 : 1\n"
            "Temperate planets orbit cooler stars by ~465 K",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            color="#4A5568", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F7F9FC",
                                       edgecolor="#D6DCE5"))
    fig.tight_layout()
    fig.savefig(FIG / "03_thermal_zones.png", dpi=180)
    plt.close(fig)
    return table


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"loaded {len(df):,} planets x {df.shape[1]} columns")

    t1 = insight_1(df)
    t2 = insight_2(df)
    t3 = insight_3(df)

    with pd.ExcelWriter(ROOT / "data" / "insight_tables.xlsx") as xl:
        t1.to_excel(xl, sheet_name="1_annual_by_method")
        t2.to_excel(xl, sheet_name="2_mass_by_orbit")
        t3.to_excel(xl, sheet_name="3_thermal_zones")
    print("\nsummary tables -> data/insight_tables.xlsx")


if __name__ == "__main__":
    main()
