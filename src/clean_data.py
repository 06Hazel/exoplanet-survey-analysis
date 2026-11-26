"""Clean the NASA Exoplanet Archive PSCompPars export into an analysis-ready table.

Input : data/exoplanet_raw.csv   (raw archive download, '#' comment header)
Output: data/exoplanet_cleaned.csv

Steps
-----
1. Parse the raw file, skipping the '#' metadata block written by the archive.
2. Drop the 58 records flagged as disputed (pl_controv_flag == 1).
3. Keep 14 analysis-relevant columns out of the original 84, so that uncertainty
   and reference columns do not dilute the table.
4. Derive an equilibrium-temperature zone label (keeps rows with missing Teq).

Run: python src/clean_data.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "exoplanet_raw.csv"
OUT = ROOT / "data" / "exoplanet_cleaned.csv"

KEEP = [
    "pl_name", "hostname", "sy_snum", "sy_pnum", "discoverymethod",
    "disc_year", "disc_facility", "pl_orbper", "pl_orbsmax", "pl_rade",
    "pl_bmasse", "pl_eqt", "st_teff", "sy_dist",
]


def thermal_zone(teq: float) -> str:
    """Three-zone split on equilibrium temperature (K)."""
    if pd.isna(teq):
        return "Unknown"
    if teq > 320:
        return "Hot"
    if teq >= 200:
        return "Temperate"
    return "Cold"


def main() -> None:
    raw = pd.read_csv(RAW, comment="#")
    print(f"raw      : {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    disputed = int((raw["pl_controv_flag"] == 1).sum())
    clean = raw.loc[raw["pl_controv_flag"] == 0, KEEP].copy()
    clean["Temp_Zone"] = clean["pl_eqt"].map(thermal_zone)

    dup = int(clean["pl_name"].duplicated().sum())
    print(f"disputed : {disputed} rows removed")
    print(f"cleaned  : {clean.shape[0]:,} rows x {clean.shape[1]} columns")
    print(f"duplicate planet names: {dup}")
    print("missing values:", {c: int(clean[c].isna().sum())
                              for c in ["pl_eqt", "pl_bmasse", "pl_orbsmax", "pl_rade"]})

    clean.to_csv(OUT, index=False, encoding="utf-8")
    print(f"written  : {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
