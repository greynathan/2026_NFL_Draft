"""
QB (Quarterback) data cleaning: combine QBs + PFF Passing + RAS + arm length.

- Training 2015–2023 from nfl_combine_2010_to_2023.csv (Pos == 'QB').
- Testing 2024 from data/raw/2024 Draft - Public - QB.csv.
- Testing 2025 from data/raw/GabrielGTB 2025 NFL Combine - Master List.csv (Position == 'QB')
  + 2025_draft_picks.csv for Round/Pick.
- Testing 2026 from data/raw/2026_Combine/NFL Combine 2026 @JordanSportGuy Twitter - QBs.csv.
- PFF: passing stats from data/raw/pff/Passing/*_passing_summary.csv matched by Player + School + Year.
- RAS: from data/raw/ras.csv matched by player/school/year.
- Arm length: from data/raw/mockdraftable_qb_arm_length.csv if available; falls back to
  arm length parsed from the 2024/2026 combine files.

Outputs (written to data/processed/):
- qb_training.csv  (2015–2023)
- qb_testing.csv   (2024–2026)

Run from project root:
    python QB/data_cleaning.py
"""

import os
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# ─── Height / arm / broad conversion helpers ─────────────────────────────────

def _ht_4digit_to_inches(ht):
    """Convert PFF/combine 4-digit height (e.g. 6030) to decimal inches."""
    if pd.isna(ht) or str(ht).strip() == "":
        return np.nan
    try:
        s = str(int(float(ht))).zfill(4)
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 4:
        return np.nan
    ft = int(s[0])
    inch = int(s[1:3])
    eighth = int(s[3]) if len(s) > 3 else 0
    return ft * 12 + inch + eighth / 8.0


def _arm_4digit_to_inches(arm):
    """Convert 4-digit arm-length code (e.g. 3138 → 31.38") to decimal inches."""
    if pd.isna(arm) or str(arm).strip() == "":
        return np.nan
    try:
        s = str(arm).strip().replace(".", "")
        if not s.isdigit():
            return np.nan
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 4:
        return np.nan
    return int(s[:2]) + int(s[2:]) / 100.0


def _broad_ffii_to_inches(broad):
    """Convert feet-fraction-inches code (e.g. 906 → 9'6" = 114") to inches."""
    if pd.isna(broad) or str(broad).strip() == "":
        return np.nan
    try:
        s = str(int(float(broad)))
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 3:
        return np.nan
    return int(s[:-2]) * 12 + int(s[-2:])


def _pick_to_round(pick_taken):
    """Convert overall pick number to draft round."""
    if pd.isna(pick_taken) or str(pick_taken).strip().upper() == "UDFA":
        return 8
    try:
        p = int(float(str(pick_taken).replace(",", "")))
        if 1 <= p <= 32:   return 1
        if p <= 64:        return 2
        if p <= 96:        return 3
        if p <= 128:       return 4
        if p <= 160:       return 5
        if p <= 192:       return 6
        if p <= 257:       return 7
    except (ValueError, TypeError):
        pass
    return 8


def _ht_combine_to_inches(ht):
    """Convert 'feet-inches' string (e.g. '6-4') or numeric to decimal inches."""
    if pd.isna(ht):
        return np.nan
    s = str(ht).strip()
    if "-" in s:
        parts = s.split("-")
        try:
            return int(parts[0]) * 12 + int(parts[1])
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ─── Normalization helpers (shared across all years) ─────────────────────────

def normalize_player_name(name: str) -> str:
    s = str(name).strip().upper()
    s = re.sub(r"\s+(III|II|IV|JR|SR|JR\.|SR\.)$", "", s)
    s = re.sub(r"[.\',\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_pff_school(name):
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    mapping = {
        "OHIO STATE": "Ohio State", "FLORIDA ST": "Florida State", "PENN STATE": "Penn State",
        "NOTRE DAME": "Notre Dame", "MICHIGAN": "Michigan", "GEORGIA": "Georgia",
        "ALABAMA": "Alabama", "OREGON ST": "Oregon State", "OREGON": "Oregon",
        "WASHINGTON": "Washington", "DUKE": "Duke", "NORTH CAROLINA": "North Carolina",
        "N CAROLINA": "North Carolina", "NC STATE": "North Carolina State",
        "VIRGINIA TECH": "Virginia Tech", "VA TECH": "Virginia Tech",
        "UCLA": "UCLA", "USC": "USC", "CAL": "California", "STANFORD": "Stanford",
        "OKLAHOMA": "Oklahoma", "TEXAS": "Texas", "LSU": "LSU", "AUBURN": "Auburn",
        "TENNESSEE": "Tennessee", "BYU": "BYU", "HOUSTON": "Houston", "YALE": "Yale",
        "ARIZONA": "Arizona", "ARIZONA ST": "Arizona State", "KENTUCKY": "Kentucky",
        "CONNECTICUT": "Connecticut", "UCONN": "Connecticut",
        "E KENTUCKY": "Eastern Kentucky", "GA STATE": "Georgia State",
        "GA TECH": "Georgia Tech", "WASH STATE": "Washington State",
        "S CAROLINA": "South Carolina", "COLO STATE": "Colorado State",
        "MICH STATE": "Michigan State", "MISS STATE": "Mississippi State",
        "OLE MISS": "Mississippi", "BOISE ST": "Boise State",
        "BOSTON COL": "Boston College", "KANSAS ST": "Kansas State",
        "S DIEGO ST": "San Diego State", "W MICHIGAN": "Western Michigan",
        "W VIRGINIA": "West Virginia", "TEXAS A&M": "Texas A&M",
        "NWESTERN": "Northwestern", "MIAMI FL": "Miami", "MIAMI OH": "Miami (OH)",
        "W KENTUCKY": "Western Kentucky", "UTEP": "Texas-El Paso",
        "UTSA": "Texas-San Antonio", "APP STATE": "Appalachian State",
        "C MICHIGAN": "Central Michigan", "E MICHIGAN": "Eastern Michigan",
        "DOMINION": "Old Dominion", "TCU": "TCU", "UNLV": "UNLV", "FLORIDA": "Florida",
        "ARKANSAS": "Arkansas", "VANDERBILT": "Vanderbilt", "IOWA": "Iowa",
        "ILLINOIS": "Illinois", "INDIANA": "Indiana", "KANSAS": "Kansas",
        "MARYLAND": "Maryland", "MINNESOTA": "Minnesota", "MISSOURI": "Missouri",
        "NEBRASKA": "Nebraska", "RUTGERS": "Rutgers", "TEXAS TECH": "Texas Tech",
        "PURDUE": "Purdue", "WAKE FOREST": "Wake Forest", "CLEMSON": "Clemson",
        "PITTSBURGH": "Pittsburgh", "SMU": "SMU", "COLORADO": "Colorado",
        "OKLAHOMA ST": "Oklahoma State", "IOWA ST": "Iowa State",
        "BAYLOR": "Baylor", "CINCINNATI": "Cincinnati", "UCF": "UCF",
        "MEMPHIS": "Memphis", "TULANE": "Tulane", "TULSA": "Tulsa",
        "N ILLINOIS": "Northern Illinois", "NAVY": "Navy", "ARMY": "Army",
        "AIR FORCE": "Air Force", "UTAH": "Utah", "COLORADO ST": "Colorado State",
        "UTAH ST": "Utah State", "WYOMING": "Wyoming", "NEW MEXICO": "New Mexico",
        "BOISE STATE": "Boise State", "HAWAII": "Hawaii", "FRESNO ST": "Fresno State",
        "SAN JOSE ST": "San Jose State", "NEVADA": "Nevada",
        "FLORIDA INTL": "Florida International", "FIU": "Florida International",
        "IDAHO": "Idaho", "EASTERN WASH": "Eastern Washington",
        "N DAKOTA ST": "North Dakota State", "S DAKOTA ST": "South Dakota State",
        "NDSU": "North Dakota State", "SDSU": "South Dakota State",
        "LIBERTY": "Liberty", "JAMES MADISON": "James Madison",
        "LAMAR": "Lamar", "SAM HOUSTON": "Sam Houston State",
        "SE LOUISIANA": "Southeastern Louisiana",
        "S CAROLINA ST": "South Carolina State",
        "GEORGIA SOUTHERN": "Georgia Southern", "GA SOUTHRN": "Georgia Southern",
        "ARK STATE": "Arkansas State",
    }
    return mapping.get(name, name.title())


def normalize_combine_school(name):
    if pd.isna(name):
        return name
    x = str(name).strip()
    upper_mapping = {
        "NOTRE DAME": "Notre Dame", "ALABAMA": "Alabama", "PENN STATE": "Penn State",
        "WASHINGTON": "Washington", "OREGON ST": "Oregon State", "OREGON": "Oregon",
        "GEORGIA": "Georgia", "HOUSTON": "Houston", "YALE": "Yale",
        "OKLAHOMA": "Oklahoma", "MARYLAND": "Maryland", "KANSAS": "Kansas",
        "TCU": "TCU", "MISSOURI": "Missouri", "TEXAS": "Texas",
        "KANSAS ST": "Kansas State", "LA LAFAYET": "Louisiana",
        "EASTERN KENTUCKY": "Eastern Kentucky", "UTAH": "Utah",
        "PITTSBURGH": "Pittsburgh", "GA STATE": "Georgia State",
        "MICHIGAN": "Michigan", "CONNECTICUT": "Connecticut", "UCONN": "Connecticut",
        "UCF": "UCF", "OHIO STATE": "Ohio State", "FLORIDA STATE": "Florida State",
        "WEST. MICHIGAN": "Western Michigan", "MISSISSIPPI ST.": "Mississippi State",
        "ARIZONA ST.": "Arizona State", "BOISE ST.": "Boise State",
        "NORTH DAKOTA ST.": "North Dakota State", "BOSTON COL.": "Boston College",
        "WASH STATE": "Washington State", "S CAROLINA": "South Carolina",
        "COLO STATE": "Colorado State", "MICH STATE": "Michigan State",
        "MISS STATE": "Mississippi State", "OLE MISS": "Mississippi",
        "BOISE ST": "Boise State", "ARIZONA ST": "Arizona State",
        "OHIO ST.": "Ohio State", "VIRGINIA TECH": "Virginia Tech",
        "VA TECH": "Virginia Tech", "S DIEGO ST": "San Diego State",
        "W MICHIGAN": "Western Michigan", "W VIRGINIA": "West Virginia",
        "NC STATE": "North Carolina State", "N CAROLINA": "North Carolina",
        "NORTH CAROLINA": "North Carolina", "TEXAS A&M": "Texas A&M",
        "NWESTERN": "Northwestern", "NORTHWESTERN": "Northwestern",
        "ILLINOIS": "Illinois", "WAKE FOREST": "Wake Forest",
        "TEXAS TECH": "Texas Tech", "SAN DIEGO ST.": "San Diego State",
        "SAN DIEGO ST": "San Diego State", "TEXAS-EL PASO": "Texas-El Paso",
        "WESTERN KENTUCKY": "Western Kentucky", "APPALACHIAN STATE": "Appalachian State",
        "CENTRAL MICHIGAN": "Central Michigan", "OLD DOMINION": "Old Dominion",
        "EASTERN MICHIGAN": "Eastern Michigan", "NORTH DAKOTA ST": "North Dakota State",
        "SMU": "SMU", "CLEMSON": "Clemson", "ARKANSAS": "Arkansas",
        "VANDERBILT": "Vanderbilt", "LSU": "LSU", "IOWA": "Iowa",
        "INDIANA": "Indiana", "TENNESSEE": "Tennessee", "FLORIDA": "Florida",
        "MINNESOTA": "Minnesota", "NEBRASKA": "Nebraska",
        "BAYLOR": "Baylor", "COLORADO": "Colorado", "CINCINNATI": "Cincinnati",
        "MEMPHIS": "Memphis", "TULANE": "Tulane", "TULSA": "Tulsa",
        "UTAH STATE": "Utah State", "WYOMING": "Wyoming",
        "LIBERTY": "Liberty",
    }
    if x.upper() in upper_mapping:
        return upper_mapping[x.upper()]
    alias = {
        "Ole Miss": "Mississippi", "Miami (FL)": "Miami", "Southern California": "USC",
        "Ohio St.": "Ohio State", "Florida St.": "Florida State",
        "Penn St.": "Penn State", "North Carolina St.": "North Carolina State",
        "NC State": "North Carolina State", "Oregon St.": "Oregon State",
        "Washington St.": "Washington State", "Cal": "California",
        "Brigham Young": "BYU", "Central Florida": "UCF",
        "LA-Lafayette": "Louisiana", "West. Michigan": "Western Michigan",
        "Mississippi St.": "Mississippi State", "Arizona St.": "Arizona State",
        "Boise St.": "Boise State", "North Dakota St.": "North Dakota State",
        "Boston Col.": "Boston College", "San Diego St.": "San Diego State",
        "Lousiville": "Louisville", "Syracruse": "Syracuse",
        "Georgia Tech": "Georgia Tech",
    }
    return alias.get(x, x)


# ─── Load PFF Passing data ───────────────────────────────────────────────────

PFF_PASSING_DIR = os.path.join(DATA_RAW, "pff", "Passing")

passing_cols = [
    "player", "team_name", "position",
    "player_game_count",
    "btt_rate",
    "twp_rate",
    "ypa",
    "qb_rating",
    "pressure_to_sack_rate",
    "sack_percent",
    "epa",
    "positive_epa_percent",
    "avg_depth_of_target",
    "attempts",
    "dropbacks",
]

passing_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_PASSING_DIR, f"{year}_passing_summary.csv")
    if not os.path.exists(path):
        continue
    df_p = pd.read_csv(path)
    sub = df_p[[c for c in passing_cols if c in df_p.columns]].copy()
    if "player" not in sub.columns:
        continue
    # Filter to QB position
    if "position" in sub.columns:
        sub = sub[sub["position"].astype(str).str.strip().str.upper() == "QB"].copy()
    sub["Year"] = year
    sub = sub.rename(columns={"player": "Player", "team_name": "School"})
    for c in ["player_game_count", "btt_rate", "twp_rate", "ypa", "qb_rating",
              "pressure_to_sack_rate", "sack_percent", "epa", "positive_epa_percent",
              "avg_depth_of_target", "attempts", "dropbacks"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    passing_files.append(sub.drop(columns=["position"], errors="ignore"))
    print(f"Loaded PFF passing {year}: {len(sub)} QBs")

if passing_files:
    passing_data = pd.concat(passing_files, ignore_index=True)
    # Deduplicate: keep highest-attempt season per player/school/year
    passing_data = (
        passing_data.sort_values("attempts", ascending=False)
        .drop_duplicates(subset=["Player", "School", "Year"], keep="first")
        .reset_index(drop=True)
    )
    print(f"PFF passing records (deduped): {len(passing_data)}")
else:
    passing_data = pd.DataFrame(
        columns=["Player", "School", "Year", "player_game_count", "btt_rate", "twp_rate",
                 "ypa", "qb_rating", "pressure_to_sack_rate", "sack_percent",
                 "epa", "positive_epa_percent", "avg_depth_of_target", "attempts"]
    )
    print("No PFF passing files found.")


# ─── PFF merge function ───────────────────────────────────────────────────────

_pff_value_cols = [c for c in passing_data.columns
                   if c not in ("Player", "School", "Year", "attempts", "dropbacks")]


def add_pff_passing(combine_df: pd.DataFrame, pff_df: pd.DataFrame) -> pd.DataFrame:
    """Merge PFF passing stats into combine_df on normalized Player + School + Year."""
    if pff_df.empty:
        for c in _pff_value_cols:
            combine_df[c] = np.nan
        return combine_df

    combine_df = combine_df.copy()
    pff_n = pff_df.copy()
    pff_n["School_n"] = pff_n["School"].apply(normalize_pff_school)
    pff_n["Player_n"] = pff_n["Player"].apply(normalize_player_name)
    combine_df["School_n"] = combine_df["School"].apply(normalize_combine_school)

    value_cols = [c for c in pff_df.columns if c not in ("Player", "School", "Year")]

    def lookup(row):
        season = int(row["Year"]) - 1
        player = normalize_player_name(row["Player"])
        school = row["School_n"]
        mask = (
            (pff_n["Player_n"] == player)
            & (pff_n["School_n"] == school)
            & (pff_n["Year"] == season)
        )
        match = pff_n.loc[mask]
        if match.empty:
            # Try school-agnostic match (player only, same season) — useful for late transfers
            mask2 = (pff_n["Player_n"] == player) & (pff_n["Year"] == season)
            match = pff_n.loc[mask2]
        if match.empty:
            return pd.Series({c: np.nan for c in value_cols})
        return pd.Series({c: match.iloc[0][c] for c in value_cols})

    res = combine_df.apply(lookup, axis=1)
    for c in res.columns:
        combine_df[c] = res[c]
    return combine_df.drop(columns=["School_n"], errors="ignore")


# ─── RAS merge ────────────────────────────────────────────────────────────────

def add_ras_data(combine_df: pd.DataFrame, ras_path: str) -> pd.DataFrame:
    combine_df = combine_df.copy()
    if not os.path.exists(ras_path):
        if "RAS" not in combine_df.columns:
            combine_df["RAS"] = np.nan
        return combine_df
    ras = pd.read_csv(ras_path)
    if not {"Name", "Year", "College", "RAS"}.issubset(ras.columns):
        if "RAS" not in combine_df.columns:
            combine_df["RAS"] = np.nan
        return combine_df
    ras["Year"] = ras["Year"].astype(int)
    ras["RAS"] = pd.to_numeric(ras["RAS"], errors="coerce")
    ras_n = ras.copy()
    ras_n["Name_n"] = ras_n["Name"].apply(normalize_player_name)
    ras_school = {
        "Miami (FL)": "Miami", "Southern California": "USC",
        "Ole Miss": "Mississippi", "Ohio St.": "Ohio State",
        "Florida St.": "Florida State", "Oklahoma St.": "Oklahoma State",
        "Penn St.": "Penn State", "Michigan St.": "Michigan State",
        "NC State": "North Carolina State", "Virginia Tech": "Virginia Tech",
        "Louisiana State": "LSU", "Boston Col.": "Boston College",
        "San Diego St.": "San Diego State", "Kansas St.": "Kansas State",
        "Iowa St.": "Iowa State", "Arizona St.": "Arizona State",
        "Mississippi St.": "Mississippi State", "West Virginia": "West Virginia",
        "Texas A&M": "Texas A&M", "North Carolina": "North Carolina",
        "South Carolina": "South Carolina", "Oregon St.": "Oregon State",
        "Washington St.": "Washington State", "North Carolina St.": "North Carolina State",
        "BYU": "BYU", "Brigham Young": "BYU", "UCF": "UCF",
        "Central Florida": "UCF", "Georgia Tech": "Georgia Tech",
        "Texas State": "Texas State", "Colorado State": "Colorado State",
        "Montana St.": "Montana State",
    }
    ras_n["College_n"] = ras_n["College"].apply(
        lambda x: ras_school.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x
    )

    def lookup_ras(row):
        player = normalize_player_name(row["Player"])
        school = normalize_combine_school(row["School"])
        year = int(row["Year"])
        m = (ras_n["Name_n"] == player) & (ras_n["College_n"] == school) & (ras_n["Year"] == year)
        hit = ras_n.loc[m]
        if hit.empty:
            return np.nan
        return float(hit.iloc[0]["RAS"])

    combine_df["RAS"] = combine_df.apply(lookup_ras, axis=1)
    return combine_df


# ─── Arm length merge ─────────────────────────────────────────────────────────

def add_arm_length(combine_df: pd.DataFrame, arm_df: pd.DataFrame) -> pd.DataFrame:
    combine_df = combine_df.copy()
    has_existing = "arm_length_inches" in combine_df.columns
    if has_existing:
        combine_df = combine_df.rename(columns={"arm_length_inches": "_arm_backup"})
    if arm_df.empty or "arm_length_inches" not in arm_df.columns:
        combine_df["arm_length_inches"] = combine_df["_arm_backup"] if has_existing else np.nan
        if has_existing:
            combine_df = combine_df.drop(columns=["_arm_backup"])
        return combine_df
    combine_df = combine_df.merge(
        arm_df[["Player", "Year", "arm_length_inches"]], on=["Player", "Year"], how="left"
    )
    if has_existing:
        combine_df["arm_length_inches"] = combine_df["arm_length_inches"].fillna(
            combine_df["_arm_backup"]
        )
        combine_df = combine_df.drop(columns=["_arm_backup"])
    return combine_df


# ─── Load training data (2015–2023 from nfl_combine) ─────────────────────────

nfl_combine_path = os.path.join(DATA_RAW, "nfl_combine_2010_to_2023.csv")
nfl_combine_data = pd.read_csv(nfl_combine_path)
QB_POSITIONS = ["QB"]
training_raw = nfl_combine_data[
    (nfl_combine_data["Pos"].isin(QB_POSITIONS)) & (nfl_combine_data["Year"].between(2015, 2023))
].copy()

# Convert height from "6-4" string format to inches
training_raw["Height"] = training_raw["Height"].apply(_ht_combine_to_inches)
for c in ["Weight", "40yd", "Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"]:
    training_raw[c] = pd.to_numeric(training_raw[c], errors="coerce")

print(f"QB training rows (2015–2023): {len(training_raw)}")


# ─── Load 2024 QBs from public draft file ────────────────────────────────────

draft_2024_path = os.path.join(DATA_RAW, "2024 Draft - Public - QB.csv")
qb_2024_list = []
if os.path.exists(draft_2024_path):
    d24 = pd.read_csv(draft_2024_path)
    d24 = d24[d24["Name"].notna() & (d24["Name"].astype(str).str.strip() != "")].copy()
    d24 = d24[d24["Pos"].astype(str).str.strip().str.upper() == "QB"].copy()
    for _, row in d24.iterrows():
        pick_taken = row.get("Pick Taken", row.iloc[0] if len(row) > 0 else None)
        pick_int = np.nan
        try:
            if str(pick_taken).strip() not in ("", "UDFA"):
                pick_int = int(float(str(pick_taken).replace(",", "")))
        except (ValueError, TypeError):
            pass
        qb_2024_list.append({
            "Year": 2024, "Player": str(row["Name"]).strip(), "Pos": "QB",
            "School": str(row["School"]).strip(),
            "Height": _ht_4digit_to_inches(row.get("HT")),
            "Weight": pd.to_numeric(row.get("WT"), errors="coerce"),
            "40yd": pd.to_numeric(str(row.get("40", "")).replace("DNP", "").strip() or np.nan, errors="coerce"),
            "Vertical": pd.to_numeric(str(row.get("VJ", "")).replace("DNP", "").strip() or np.nan, errors="coerce"),
            "Bench": pd.to_numeric(str(row.get("BP", "")).replace("DNP", "").strip() or np.nan, errors="coerce"),
            "Broad Jump": _broad_ffii_to_inches(str(row.get("BJ", "")).replace("DNP", "").strip() or np.nan),
            "3Cone": pd.to_numeric(str(row.get("3c", "")).replace("DNP", "").strip() or np.nan, errors="coerce"),
            "Shuttle": pd.to_numeric(str(row.get("Shuttle", "")).replace("DNP", "").strip() or np.nan, errors="coerce"),
            "Drafted": True, "Round": _pick_to_round(pick_taken), "Pick": pick_int,
            "RAS": pd.to_numeric(str(row.get("RAS", "")).replace("DNP", "").strip() or np.nan, errors="coerce"),
            "arm_length_inches": _arm_4digit_to_inches(row.get("Arm")),
        })
    print(f"Loaded 2024 QBs: {len(qb_2024_list)} from 2024 Draft - Public - QB.csv")
else:
    print("2024 Draft - Public - QB.csv not found.")

qb_2024 = pd.DataFrame(qb_2024_list)


# ─── Load 2025 QBs from GabrielGTB + draft picks ─────────────────────────────

combine_2025_path = os.path.join(DATA_RAW, "GabrielGTB 2025 NFL Combine - Master List.csv")
draft_picks_2025_path = os.path.join(DATA_RAW, "2025_draft_picks.csv")

qb_2025_list = []
if os.path.exists(combine_2025_path):
    c25 = pd.read_csv(combine_2025_path)
    c25_qb = c25[c25["Position"].astype(str).str.strip().str.upper() == "QB"].copy()
    for _, row in c25_qb.iterrows():
        bj_raw = row.get("Broad Jump (FFII)")
        qb_2025_list.append({
            "Year": 2025, "Player": str(row["Name"]).strip(), "Pos": "QB",
            "School": normalize_combine_school(str(row.get("School", "")).strip()),
            "Height": _ht_4digit_to_inches(row.get("Height (FIIE)")),
            "Weight": pd.to_numeric(row.get("Weight (lbs.)"), errors="coerce"),
            "40yd": pd.to_numeric(row.get("40-yard Dash (seconds)"), errors="coerce"),
            "Vertical": pd.to_numeric(row.get("Vertical Jump (inches)"), errors="coerce"),
            "Bench": pd.to_numeric(row.get("Bench Press (reps)"), errors="coerce"),
            "Broad Jump": _broad_ffii_to_inches(bj_raw),
            "3Cone": pd.to_numeric(row.get("Three-cone Drill (seconds)"), errors="coerce"),
            "Shuttle": pd.to_numeric(row.get("20-yard Shuttle (seconds)"), errors="coerce"),
            "Drafted": True, "Round": np.nan, "Pick": np.nan,
            "RAS": pd.to_numeric(row.get("RAS"), errors="coerce"),
            "arm_length_inches": pd.to_numeric(row.get("Arm Length (inches)"), errors="coerce"),
        })
    print(f"Loaded 2025 QBs from GabrielGTB: {len(qb_2025_list)}")
else:
    print("GabrielGTB 2025 NFL Combine - Master List.csv not found.")

qb_2025 = pd.DataFrame(qb_2025_list)

# Merge Round/Pick from 2025 draft picks
if not qb_2025.empty and os.path.exists(draft_picks_2025_path):
    dp25 = pd.read_csv(draft_picks_2025_path)
    dp_qb = dp25[dp25["Pos"].astype(str).str.upper().isin(["QB"])].copy()

    def _norm_name(n):
        return re.sub(r"\s+Jr\.?$|\s+III$|\s+II$|\s+IV$", "", str(n).strip(),
                      flags=re.IGNORECASE).strip() if pd.notna(n) else ""

    def _norm_school(s):
        aliases = {"Penn St.": "Penn State", "Ohio St.": "Ohio State",
                   "Florida St.": "Florida State", "Ole Miss": "Mississippi",
                   "Syracruse": "Syracuse", "Lousiville": "Louisville"}
        x = str(s).strip() if pd.notna(s) else ""
        return aliases.get(x, x)

    dp_qb = dp_qb.rename(columns={"Rnd": "Round"})
    dp_qb["Player_n"] = dp_qb["Player"].map(_norm_name)
    dp_qb["School_n"] = dp_qb["School"].map(_norm_school)
    qb_2025 = qb_2025.drop(columns=["Round", "Pick"], errors="ignore")
    qb_2025["Player_n"] = qb_2025["Player"].map(_norm_name)
    qb_2025["School_n"] = qb_2025["School"].map(_norm_school)
    qb_2025 = qb_2025.merge(
        dp_qb[["Player_n", "School_n", "Round", "Pick"]], on=["Player_n", "School_n"], how="left"
    ).drop(columns=["Player_n", "School_n"], errors="ignore")
    # Players not in draft picks → undrafted (round 8)
    qb_2025["Round"] = qb_2025["Round"].fillna(8)
    drafted_mask = qb_2025["Round"] < 8
    qb_2025 = qb_2025[drafted_mask].copy()
    print(f"2025 QBs after draft-pick filter (drafted only): {len(qb_2025)}")


# ─── Load 2026 QBs from JordanSportGuy combine file ──────────────────────────

combine_2026_path = os.path.join(
    DATA_RAW, "2026_Combine",
    "NFL Combine 2026 @JordanSportGuy Twitter - QBs.csv"
)
qb_2026_list = []
if os.path.exists(combine_2026_path):
    c26 = pd.read_csv(combine_2026_path)
    # Column names have trailing spaces/colons
    c26.columns = [col.strip().rstrip(":").strip() for col in c26.columns]
    # Rename to standard names
    col_map = {
        "NAME": "Player", "SCHOOL": "School",
        "HEIGHT": "_ht_raw", "WEIGHT": "Weight",
        "Arm Length": "_arm_raw", "40 Yard Dash": "40yd",
        "Vertical": "Vertical", "Broad": "_broad_raw",
        "3 Cone": "3Cone", "Shuttle": "Shuttle", "Bench": "Bench",
    }
    for old, new in col_map.items():
        if old in c26.columns:
            c26 = c26.rename(columns={old: new})

    for _, row in c26.iterrows():
        player = str(row.get("Player", "")).strip()
        school = str(row.get("School", "")).strip()
        if not player or player.lower() in ("nan", ""):
            continue
        qb_2026_list.append({
            "Year": 2026, "Player": player, "Pos": "QB", "School": school,
            "Height": _ht_4digit_to_inches(row.get("_ht_raw")),
            "Weight": pd.to_numeric(row.get("Weight"), errors="coerce"),
            "40yd": pd.to_numeric(str(row.get("40yd", "")).strip() or np.nan, errors="coerce"),
            "Vertical": pd.to_numeric(str(row.get("Vertical", "")).strip() or np.nan, errors="coerce"),
            "Bench": pd.to_numeric(str(row.get("Bench", "")).strip() or np.nan, errors="coerce"),
            "Broad Jump": _broad_ffii_to_inches(str(row.get("_broad_raw", "")).strip() or np.nan),
            "3Cone": pd.to_numeric(str(row.get("3Cone", "")).strip() or np.nan, errors="coerce"),
            "Shuttle": pd.to_numeric(str(row.get("Shuttle", "")).strip() or np.nan, errors="coerce"),
            "Drafted": False, "Round": np.nan, "Pick": np.nan,
            "arm_length_inches": _arm_4digit_to_inches(row.get("_arm_raw")),
        })
    print(f"Loaded 2026 QBs from JordanSportGuy: {len(qb_2026_list)}")
else:
    print(f"2026 QB combine file not found at {combine_2026_path}")

qb_2026 = pd.DataFrame(qb_2026_list)


# ─── Assemble full testing frame ──────────────────────────────────────────────

testing_frames = [f for f in [qb_2024, qb_2025, qb_2026] if not f.empty]
if testing_frames:
    qb_testing_data = pd.concat(testing_frames, ignore_index=True)
else:
    qb_testing_data = pd.DataFrame()
print(f"Total testing rows: {len(qb_testing_data)}")


# ─── Merge PFF passing ────────────────────────────────────────────────────────

training_raw = add_pff_passing(training_raw, passing_data)
if not qb_testing_data.empty:
    qb_testing_data = add_pff_passing(qb_testing_data, passing_data)


# ─── Merge RAS ───────────────────────────────────────────────────────────────

RAS_PATH = os.path.join(DATA_RAW, "ras.csv")
training_raw = add_ras_data(training_raw, RAS_PATH)
if not qb_testing_data.empty:
    qb_testing_data = add_ras_data(qb_testing_data, RAS_PATH)


# ─── Merge arm length ─────────────────────────────────────────────────────────

arm_path = os.path.join(DATA_RAW, "mockdraftable_qb_arm_length.csv")
if os.path.exists(arm_path):
    arm_df = pd.read_csv(arm_path)
    if {"Player", "Year", "arm_length_inches"}.issubset(arm_df.columns):
        arm_df["Year"] = arm_df["Year"].astype(int)
        arm_df["arm_length_inches"] = pd.to_numeric(arm_df["arm_length_inches"], errors="coerce")
        print(f"Arm length QB: {len(arm_df)} records")
    else:
        arm_df = pd.DataFrame(columns=["Player", "Year", "arm_length_inches"])
else:
    arm_df = pd.DataFrame(columns=["Player", "Year", "arm_length_inches"])
    print("No mockdraftable_qb_arm_length.csv; arm_length_inches from combine files only.")

training_raw = add_arm_length(training_raw, arm_df)
if not qb_testing_data.empty:
    qb_testing_data = add_arm_length(qb_testing_data, arm_df)


# ─── Final column ordering ────────────────────────────────────────────────────

COL_ORDER = [
    "Year", "Player", "Pos", "School",
    "Height", "Weight", "40yd", "Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle",
    "Drafted", "Round", "Pick",
    "RAS", "arm_length_inches",
    "player_game_count",
    "btt_rate", "twp_rate",
    "ypa", "qb_rating",
    "pressure_to_sack_rate", "sack_percent",
    "epa", "positive_epa_percent",
    "avg_depth_of_target",
]

def reorder_cols(df, col_order):
    present = [c for c in col_order if c in df.columns]
    extra = [c for c in df.columns if c not in col_order]
    return df[present + extra]

training_raw = reorder_cols(training_raw, COL_ORDER)
if not qb_testing_data.empty:
    qb_testing_data = reorder_cols(qb_testing_data, COL_ORDER)


# ─── PFF match stats ──────────────────────────────────────────────────────────

pff_cols = ["btt_rate", "twp_rate", "ypa", "qb_rating", "pressure_to_sack_rate",
            "sack_percent", "epa", "positive_epa_percent", "avg_depth_of_target",
            "player_game_count"]
for col in pff_cols:
    if col in training_raw.columns:
        n = training_raw[col].notna().sum()
        print(f"Training QBs with {col}: {n}/{len(training_raw)}")


# ─── Write outputs ────────────────────────────────────────────────────────────

os.makedirs(DATA_PROCESSED, exist_ok=True)

training_raw.to_csv(os.path.join(DATA_PROCESSED, "qb_training.csv"), index=False)
print(f"Saved qb_training.csv: {len(training_raw)} rows (2015–2023)")

if not qb_testing_data.empty:
    qb_testing_data.to_csv(os.path.join(DATA_PROCESSED, "qb_testing.csv"), index=False)
    print(f"Saved qb_testing.csv: {len(qb_testing_data)} rows (2024–2026)")
else:
    pd.DataFrame(columns=COL_ORDER).to_csv(
        os.path.join(DATA_PROCESSED, "qb_testing.csv"), index=False
    )
    print("Saved qb_testing.csv (empty)")
