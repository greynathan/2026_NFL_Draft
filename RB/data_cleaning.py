"""
RB (Running Back) data cleaning: combine RBs + PFF Rushing + RAS + optional arm length.

- Training 2015–2023 from nfl_combine_2010_to_2023.csv (Pos in ['RB', 'HB']).
- Testing 2024–2026 from RB/rb_drafted_2024.csv, RB/rb_drafted_2025.csv, RB/rb_drafted_2026.csv (if present).
- PFF: rushing stats from data/raw/pff/Rushing/*_rushing_summary.csv matched by Player + School + Year.
- RAS: from data/raw/ras.csv matched by player/school/year.
- Arm length: optional, from data/raw/mockdraftable_rb_arm_length.csv (same pattern as LB/S).

Outputs (written to data/processed/):
- rb_training.csv (2015–2023)
- rb_testing.csv (2024–2026; drafted only)
- rb_drafted_2026.csv (rewritten with normalized columns if any 2026 RBs are present)

Run from project root:
    python RB/data_cleaning.py
"""

import os
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")


# --- Load combine RBs (2015–2023) ---

nfl_combine_path = os.path.join(DATA_RAW, "nfl_combine_2010_to_2023.csv")
if not os.path.exists(nfl_combine_path):
    raise FileNotFoundError(f"Missing {nfl_combine_path}. RB training data cannot be built.")

nfl_combine_data = pd.read_csv(nfl_combine_path)
RB_POSITIONS = ["RB", "HB"]
nfl_combine_data_rb = nfl_combine_data[nfl_combine_data["Pos"].isin(RB_POSITIONS)].copy()
print(f"RB combine rows: {len(nfl_combine_data_rb)} (Pos in {RB_POSITIONS})")


# --- PFF Rushing ---

PFF_RUSHING_DIR = os.path.join(DATA_RAW, "pff", "Rushing")

rushing_cols = [
    "player",
    "team_name",
    "position",
    "attempts",
    "yards",
    "yards_after_contact",
    "yco_attempt",
    "ypa",
    "touchdowns",
    "total_touches",
    "explosive",
    "first_downs",
    "fumbles",
    "avoided_tackles",
    "breakaway_attempts",
    "breakaway_percent",
    "breakaway_yards",
    "elu_rush_mtf",
    "elusive_rating",
    "yprr",
    "routes",
    "targets",
]

rushing_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_RUSHING_DIR, f"{year}_rushing_summary.csv")
    if not os.path.exists(path):
        continue
    df_r = pd.read_csv(path)
    sub = df_r[[c for c in rushing_cols if c in df_r.columns]].copy()
    if "attempts" not in sub.columns or "yards" not in sub.columns:
        # If key rushing stats are missing, skip this year
        continue
    if "position" not in sub.columns:
        sub["position"] = None
    sub["Year"] = year
    sub = sub.rename(columns={"player": "Player", "team_name": "School"})
    # Numeric conversions
    for c in [
        "attempts",
        "yards",
        "yards_after_contact",
        "yco_attempt",
        "ypa",
        "touchdowns",
        "total_touches",
        "explosive",
        "first_downs",
        "fumbles",
        "avoided_tackles",
        "breakaway_attempts",
        "breakaway_percent",
        "breakaway_yards",
        "elu_rush_mtf",
        "elusive_rating",
        "yprr",
        "routes",
        "targets",
    ]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    rushing_files.append(sub)
    print(f"Loaded PFF rushing {year}: {len(sub)} players")

if rushing_files:
    rushing_data = pd.concat(rushing_files, ignore_index=True)
    # RBs and similar positions from PFF (HB/RB/FB etc.); keep all for now.
    rushing_data = rushing_data.drop(columns=["position"], errors="ignore")
    print(f"PFF rushing records: {len(rushing_data)}")
else:
    rushing_data = pd.DataFrame(
        columns=[
            "Player",
            "School",
            "Year",
            "attempts",
            "yards",
            "yards_after_contact",
            "yco_attempt",
            "ypa",
            "touchdowns",
            "total_touches",
            "explosive",
            "first_downs",
            "fumbles",
            "avoided_tackles",
            "breakaway_attempts",
            "breakaway_percent",
            "breakaway_yards",
            "elusive_rating",
            "yprr",
        ]
    )
    print("No PFF rushing files found; rushing features will be empty.")


# --- Normalize player / school names (reused pattern from S/LB/IOL/Edges) ---


def normalize_player_name(name: str) -> str:
    s = str(name).strip().upper()
    # Drop common suffixes and punctuation
    s = re.sub(r"\s+(III|II|IV|JR|SR|JR\.|SR\.)$", "", s)
    s = re.sub(r"[.\',\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_pff_school(name):
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    mapping = {
        "OHIO STATE": "Ohio State",
        "FLORIDA ST": "Florida State",
        "PENN STATE": "Penn State",
        "NOTRE DAME": "Notre Dame",
        "MICHIGAN": "Michigan",
        "GEORGIA": "Georgia",
        "ALABAMA": "Alabama",
        "OREGON ST": "Oregon State",
        "OREGON": "Oregon",
        "WASHINGTON": "Washington",
        "DUKE": "Duke",
        "NORTH CAROLINA": "North Carolina",
        "N CAROLINA": "North Carolina",
        "NC STATE": "North Carolina State",
        "VIRGINIA TECH": "Virginia Tech",
        "VA TECH": "Virginia Tech",
        "UCLA": "UCLA",
        "USC": "USC",
        "CAL": "California",
        "STANFORD": "Stanford",
        "OKLAHOMA": "Oklahoma",
        "TEXAS": "Texas",
        "LSU": "LSU",
        "AUBURN": "Auburn",
        "TENNESSEE": "Tennessee",
        "BYU": "BYU",
        "HOUSTON": "Houston",
        "YALE": "Yale",
        "ARIZONA": "Arizona",
        "ARIZONA ST": "Arizona State",
        "KENTUCKY": "Kentucky",
        "CONNECTICUT": "Connecticut",
        "UCONN": "Connecticut",
        "E KENTUCKY": "Eastern Kentucky",
        "GA STATE": "Georgia State",
        "WASH STATE": "Washington State",
        "S CAROLINA": "South Carolina",
        "COLO STATE": "Colorado State",
        "MICH STATE": "Michigan State",
        "MISS STATE": "Mississippi State",
        "OLE MISS": "Mississippi",
        "BOISE ST": "Boise State",
        "BOSTON COL": "Boston College",
        "KANSAS ST": "Kansas State",
        "S DIEGO ST": "San Diego State",
        "W MICHIGAN": "Western Michigan",
        "W VIRGINIA": "West Virginia",
        "TEXAS A&M": "Texas A&M",
        "NWESTERN": "Northwestern",
        "MIAMI FL": "Miami",
        "MIAMI OH": "Miami (OH)",
        "W KENTUCKY": "Western Kentucky",
        "UTEP": "Texas-El Paso",
        "UTSA": "Texas-San Antonio",
        "APP STATE": "Appalachian State",
        "C MICHIGAN": "Central Michigan",
        "E MICHIGAN": "Eastern Michigan",
        "DOMINION": "Old Dominion",
        "TCU": "TCU",
        "UNLV": "UNLV",
        "FLORIDA": "Florida",
    }
    return mapping.get(name, name.title())


def normalize_combine_school(name):
    if pd.isna(name):
        return name
    x = str(name).strip()
    upper_mapping = {
        "NOTRE DAME": "Notre Dame",
        "ALABAMA": "Alabama",
        "PENN STATE": "Penn State",
        "WASHINGTON": "Washington",
        "OREGON ST": "Oregon State",
        "OREGON": "Oregon",
        "GEORGIA": "Georgia",
        "HOUSTON": "Houston",
        "YALE": "Yale",
        "OKLAHOMA": "Oklahoma",
        "MARYLAND": "Maryland",
        "KANSAS": "Kansas",
        "TCU": "TCU",
        "MISSOURI": "Missouri",
        "TEXAS": "Texas",
        "KANSAS ST": "Kansas State",
        "LA LAFAYET": "Louisiana",
        "EASTERN KENTUCKY": "Eastern Kentucky",
        "UTAH": "Utah",
        "PITTSBURGH": "Pittsburgh",
        "GA STATE": "Georgia State",
        "MICHIGAN": "Michigan",
        "CONNECTICUT": "Connecticut",
        "UCONN": "Connecticut",
        "UCF": "UCF",
        "OHIO STATE": "Ohio State",
        "FLORIDA STATE": "Florida State",
        "FINDLAY": "Findlay",
        "WEST. MICHIGAN": "Western Michigan",
        "MISSISSIPPI ST.": "Mississippi State",
        "ARIZONA ST.": "Arizona State",
        "BOISE ST.": "Boise State",
        "NORTH DAKOTA ST.": "North Dakota State",
        "BOSTON COL.": "Boston College",
        "WASH STATE": "Washington State",
        "S CAROLINA": "South Carolina",
        "COLO STATE": "Colorado State",
        "MICH STATE": "Michigan State",
        "MISS STATE": "Mississippi State",
        "OLE MISS": "Mississippi",
        "BOISE ST": "Boise State",
        "ARIZONA ST": "Arizona State",
        "OHIO ST.": "Ohio State",
        "VIRGINIA TECH": "Virginia Tech",
        "VA TECH": "Virginia Tech",
        "S DIEGO ST": "San Diego State",
        "W MICHIGAN": "Western Michigan",
        "W VIRGINIA": "West Virginia",
        "NC STATE": "North Carolina State",
        "N CAROLINA": "North Carolina",
        "NORTH CAROLINA": "North Carolina",
        "TEXAS A&M": "Texas A&M",
        "NWESTERN": "Northwestern",
        "NORTHWESTERN": "Northwestern",
        "ILLINOIS": "Illinois",
        "WAKE FOREST": "Wake Forest",
        "TEXAS TECH": "Texas Tech",
        "SAN DIEGO ST.": "San Diego State",
        "SAN DIEGO ST": "San Diego State",
        "TEXAS-EL PASO": "Texas-El Paso",
        "TEXAS-SAN ANTONIO": "Texas-San Antonio",
        "WESTERN KENTUCKY": "Western Kentucky",
        "APPALACHIAN STATE": "Appalachian State",
        "CENTRAL MICHIGAN": "Central Michigan",
        "OLD DOMINION": "Old Dominion",
        "EASTERN MICHIGAN": "Eastern Michigan",
        "NORTH DAKOTA ST.": "North Dakota State",
        "NORTH DAKOTA ST": "North Dakota State",
    }
    if x.upper() in upper_mapping:
        return upper_mapping[x.upper()]
    alias = {
        "Ole Miss": "Mississippi",
        "Miami (FL)": "Miami",
        "Southern California": "USC",
        "Ohio St.": "Ohio State",
        "Florida St.": "Florida State",
        "Penn St.": "Penn State",
        "North Carolina St.": "North Carolina State",
        "NC State": "North Carolina State",
        "Oregon St.": "Oregon State",
        "Washington St.": "Washington State",
        "Cal": "California",
        "Brigham Young": "BYU",
        "Central Florida": "UCF",
        "LA-Lafayette": "Louisiana",
        "West. Michigan": "Western Michigan",
        "Mississippi St.": "Mississippi State",
        "Arizona St.": "Arizona State",
        "Boise St.": "Boise State",
        "North Dakota St.": "North Dakota State",
        "Boston Col.": "Boston College",
        "Southern Utah St.": "Southern Utah",
        "San Diego St.": "San Diego State",
        "Texas-El Paso": "Texas-El Paso",
        "Texas-San Antonio": "Texas-San Antonio",
        "Tenn-Chattanooga": "Tennessee-Chattanooga",
    }
    return alias.get(x, x)


def add_pff_data(combine_df: pd.DataFrame, rush_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge PFF rushing stats into combine_df on normalized Player + School + Year.
    """
    combine_df = combine_df.copy()
    rush_n = rush_df.copy()
    rush_n["School_normalized"] = rush_n["School"].apply(normalize_pff_school)
    rush_n["Player_normalized"] = rush_n["Player"].apply(normalize_player_name)
    combine_df["School_normalized"] = combine_df["School"].apply(normalize_combine_school)

    value_cols = [c for c in rush_df.columns if c not in ("Player", "School", "Year")]

    def lookup(row):
        draft_year = int(row["Year"])
        season = draft_year - 1
        player = normalize_player_name(row["Player"])
        school = row["School_normalized"]
        mask = (
            (rush_n["Player_normalized"] == player)
            & (rush_n["School_normalized"] == school)
            & (rush_n["Year"] == season)
        )
        match = rush_n.loc[mask]
        if match.empty:
            return pd.Series({c: None for c in value_cols})
        return pd.Series({c: match.iloc[0][c] for c in value_cols})

    res = combine_df.apply(lookup, axis=1)
    for c in res.columns:
        combine_df[c] = res[c]
    return combine_df.drop(columns=["School_normalized"], errors="ignore")


def add_ras_data(combine_df: pd.DataFrame, ras_path: str) -> pd.DataFrame:
    """
    Add RAS value by matching player/school/year against data/raw/ras.csv.
    Reuses the normalization logic from LB/S.
    """
    combine_df = combine_df.copy()
    if not os.path.exists(ras_path):
        combine_df["RAS"] = None
        print(f"No RAS file at {ras_path}; RAS will be empty for RBs.")
        return combine_df

    ras = pd.read_csv(ras_path)
    if not {"Name", "Year", "College", "RAS"}.issubset(ras.columns):
        combine_df["RAS"] = None
        print("RAS file missing required columns; RAS will be empty for RBs.")
        return combine_df

    ras["Year"] = ras["Year"].astype(int)
    ras_n = ras.copy()
    ras_n["Name_n"] = ras_n["Name"].apply(normalize_player_name)

    ras_school = {
        "Miami (FL)": "Miami",
        "Miami": "Miami",
        "Miami (Ohio)": "Miami (OH)",
        "Southern California": "USC",
        "USC": "USC",
        "UCLA": "UCLA",
        "Central Florida": "UCF",
        "UCF": "UCF",
        "Brigham Young": "BYU",
        "BYU": "BYU",
        "Ole Miss": "Mississippi",
        "Mississippi": "Mississippi",
        "Ohio St.": "Ohio State",
        "Ohio State": "Ohio State",
        "Florida St.": "Florida State",
        "Florida State": "Florida State",
        "Oklahoma St.": "Oklahoma State",
        "Oklahoma State": "Oklahoma State",
        "Oklahoma": "Oklahoma",
        "Penn St.": "Penn State",
        "Penn State": "Penn State",
        "Michigan St.": "Michigan State",
        "Michigan State": "Michigan State",
        "North Carolina State": "North Carolina State",
        "NC State": "North Carolina State",
        "Virginia Tech": "Virginia Tech",
        "Texas State": "Texas State",
        "Louisiana Tech": "Louisiana Tech",
        "Appalachian State": "Appalachian State",
        "Florida Atlantic": "Florida Atlantic",
        "Texas-San Antonio": "Texas-San Antonio",
        "UTSA": "Texas-San Antonio",
        "Toledo": "Toledo",
        "Georgia Southern": "Georgia Southern",
        "Kentucky": "Kentucky",
        "TCU": "TCU",
        "Texas Christian": "TCU",
        "Louisiana State": "LSU",
        "LSU": "LSU",
        "Boston Col.": "Boston College",
        "Boston College": "Boston College",
        "San Diego St.": "San Diego State",
        "San Diego State": "San Diego State",
        "San Jose St.": "San Jose State",
        "San Jose State": "San Jose State",
        "Kansas St.": "Kansas State",
        "Kansas State": "Kansas State",
        "Iowa St.": "Iowa State",
        "Iowa State": "Iowa State",
        "Alabama-Birmingham": "UAB",
        "Tenn-Chattanooga": "Chattanooga",
        "Washington State": "Washington State",
        "Colorado State": "Colorado State",
        "Northwestern": "Northwestern",
        "Arizona St.": "Arizona State",
        "Arizona State": "Arizona State",
        "Mississippi St.": "Mississippi State",
        "Mississippi State": "Mississippi State",
        "West Virginia": "West Virginia",
        "Texas A&M": "Texas A&M",
        "Georgia Tech": "Georgia Tech",
        "North Carolina": "North Carolina",
        "South Carolina": "South Carolina",
        "Montana St.": "Montana State",
        "Montana State": "Montana State",
        "Oregon St.": "Oregon State",
        "Oregon State": "Oregon State",
        "Washington St.": "Washington State",
        "North Carolina St.": "North Carolina State",
        "Ala-Birmingham": "UAB",
        "Texas AM": "Texas A&M",
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
            return pd.Series({"RAS": None})
        return pd.Series({"RAS": hit.iloc[0]["RAS"]})

    ras_cols = combine_df.apply(lookup_ras, axis=1)
    combine_df["RAS"] = ras_cols["RAS"]
    return combine_df


def add_arm_length(combine_df: pd.DataFrame, arm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add arm_length_inches by left merge on Player + Year.
    Falls back to any pre-existing arm_length_inches value (e.g. from official combine).
    """
    combine_df = combine_df.copy()
    has_existing = "arm_length_inches" in combine_df.columns
    if has_existing:
        combine_df = combine_df.rename(columns={"arm_length_inches": "_arm_backup"})
    if arm_df.empty or "arm_length_inches" not in arm_df.columns:
        combine_df["arm_length_inches"] = combine_df["_arm_backup"] if has_existing else None
        if has_existing:
            combine_df = combine_df.drop(columns=["_arm_backup"])
        return combine_df
    combine_df = combine_df.merge(
        arm_df[["Player", "Year", "arm_length_inches"]], on=["Player", "Year"], how="left"
    )
    if has_existing:
        combine_df["arm_length_inches"] = combine_df["arm_length_inches"].fillna(combine_df["_arm_backup"])
        combine_df = combine_df.drop(columns=["_arm_backup"])
    return combine_df


# --- Build training and testing sets ---

# Training 2015–2023 from combine RBs
rb_training_data = nfl_combine_data_rb[nfl_combine_data_rb["Year"].between(2015, 2023)].copy()

# Testing 2024–2026 from rb_drafted_20xx.csv (if present), mapped explicitly into
# the base testing schema to avoid any column misalignment.
rb_testing_frames = []
for year, fname in [
    (2024, "rb_drafted_2024.csv"),
    (2025, "rb_drafted_2025.csv"),
    (2026, "rb_drafted_2026.csv"),
]:
    path = os.path.join(SCRIPT_DIR, fname)
    if not os.path.exists(path):
        print(f"Missing {path}; RB {year} testing rows will be empty.")
        continue

    df_y = pd.read_csv(path).copy()
    df_y["Year"] = year

    base_cols = [
        "Year",
        "Player",
        "Pos",
        "School",
        "Height",
        "Weight",
        "40yd",
        "Vertical",
        "Bench",
        "Broad Jump",
        "3Cone",
        "Shuttle",
        "Drafted",
        "Round",
        "Pick",
    ]

    out = pd.DataFrame(columns=base_cols)
    out["Year"] = df_y["Year"]
    out["Player"] = df_y["Player"]
    out["Pos"] = df_y["Pos"]
    out["School"] = df_y["School"]
    out["Height"] = df_y["Height"]
    out["Weight"] = df_y["Weight"]
    out["40yd"] = df_y.get("40yd")
    out["Vertical"] = df_y.get("Vertical")
    out["Bench"] = df_y.get("Bench")
    out["Broad Jump"] = df_y.get("Broad Jump")
    out["3Cone"] = df_y.get("3Cone")
    out["Shuttle"] = df_y.get("Shuttle")
    out["Round"] = df_y.get("Round")
    out["Pick"] = df_y.get("Pick")
    out["Drafted"] = True

    rb_testing_frames.append(out)

if rb_testing_frames:
    rb_testing_data = pd.concat(rb_testing_frames, ignore_index=True)
else:
    rb_testing_data = pd.DataFrame(columns=rb_training_data.columns)


# Normalize columns to expected schema before PFF/RAS/arm merges

def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    base_cols = [
        "Year",
        "Player",
        "Pos",
        "School",
        "Height",
        "Weight",
        "40yd",
        "Vertical",
        "Bench",
        "Broad Jump",
        "3Cone",
        "Shuttle",
        "Drafted",
        "Round",
        "Pick",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan
    # Coerce numerics
    for c in ["Height", "Weight", "40yd", "Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Drafted"] = df.get("Drafted", False).fillna(False).astype(bool)
    return df[base_cols]


rb_training_data = ensure_base_columns(rb_training_data)
rb_testing_data = ensure_base_columns(rb_testing_data) if not rb_testing_data.empty else rb_testing_data


# Merge PFF rushing

pff_data = rushing_data.copy()
if not pff_data.empty:
    pff_keys = pff_data[["Player", "School", "Year"]].drop_duplicates()
    pff_data = pff_keys.merge(pff_data, on=["Player", "School", "Year"], how="left")
    print(f"PFF merged rushing records: {len(pff_data)}")
else:
    print("No PFF rushing data to merge.")

if not pff_data.empty:
    rb_training_data = add_pff_data(rb_training_data, pff_data)
    if not rb_testing_data.empty:
        rb_testing_data = add_pff_data(rb_testing_data, pff_data)

    # Derived rushing metrics from PFF
    def add_derived_rushing_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        atts = pd.to_numeric(df.get("attempts"), errors="coerce")
        routes = pd.to_numeric(df.get("routes"), errors="coerce")
        touches = pd.to_numeric(df.get("total_touches"), errors="coerce")

        elu_mtf = pd.to_numeric(df.get("elu_rush_mtf"), errors="coerce")
        explosive = pd.to_numeric(df.get("explosive"), errors="coerce")
        fumbles = pd.to_numeric(df.get("fumbles"), errors="coerce")
        targets = pd.to_numeric(df.get("targets"), errors="coerce")

        # mtf_per_attempt = elu_rush_mtf / attempts
        df["mtf_per_attempt"] = np.where(
            (atts > 0) & (~elu_mtf.isna()), elu_mtf / atts, np.nan
        )
        # explosive_rate = explosive / attempts
        df["explosive_rate"] = np.where(
            (atts > 0) & (~explosive.isna()), explosive / atts, np.nan
        )
        # targets_per_route = targets / routes
        df["targets_per_route"] = np.where(
            (routes > 0) & (~targets.isna()), targets / routes, np.nan
        )
        # fumble_rate = fumbles / total_touches
        df["fumble_rate"] = np.where(
            (touches > 0) & (~fumbles.isna()), fumbles / touches, np.nan
        )
        return df

    rb_training_data = add_derived_rushing_features(rb_training_data)
    if not rb_testing_data.empty:
        rb_testing_data = add_derived_rushing_features(rb_testing_data)


# Add RAS

RAS_PATH = os.path.join(DATA_RAW, "ras.csv")
rb_training_data = add_ras_data(rb_training_data, RAS_PATH)
if not rb_testing_data.empty:
    rb_testing_data = add_ras_data(rb_testing_data, RAS_PATH)


# Add arm length (optional)

arm_path = os.path.join(DATA_RAW, "mockdraftable_rb_arm_length.csv")
if os.path.exists(arm_path):
    arm_df = pd.read_csv(arm_path)
    if {"Player", "Year", "arm_length_inches"}.issubset(arm_df.columns):
        arm_df["Year"] = arm_df["Year"].astype(int)
        arm_df["arm_length_inches"] = pd.to_numeric(arm_df["arm_length_inches"], errors="coerce")
        print(
            f"Arm length RB: {len(arm_df)} records "
            f"({arm_df['arm_length_inches'].notna().sum()} with values)"
        )
    else:
        arm_df = pd.DataFrame(columns=["Player", "Year", "arm_length_inches"])
        print("mockdraftable_rb_arm_length.csv missing required columns; arm length will be empty.")
else:
    arm_df = pd.DataFrame(columns=["Player", "Year", "arm_length_inches"])
    print("No mockdraftable_rb_arm_length.csv; arm_length_inches will be empty.")

rb_training_data = add_arm_length(rb_training_data, arm_df)
if not rb_testing_data.empty:
    rb_testing_data = add_arm_length(rb_testing_data, arm_df)


# Rebuild 2026 testing rows directly from rb_drafted_2026.csv using the same
# processing pipeline (ensure_base_columns + PFF + derived metrics + RAS + arm).
rb_2026_input = os.path.join(SCRIPT_DIR, "rb_drafted_2026.csv")
if not rb_testing_data.empty and os.path.exists(rb_2026_input):
    raw_2026 = pd.read_csv(rb_2026_input)
    if not raw_2026.empty:
        raw_2026 = raw_2026.copy()
        raw_2026["Year"] = 2026
        # Map drafted CSV into base schema
        base_2026 = pd.DataFrame(
            {
                "Year": raw_2026["Year"],
                "Player": raw_2026["Player"],
                "Pos": raw_2026["Pos"],
                "School": raw_2026["School"],
                "Height": raw_2026["Height"],
                "Weight": raw_2026["Weight"],
                "40yd": raw_2026.get("40yd"),
                "Vertical": raw_2026.get("Vertical"),
                "Bench": raw_2026.get("Bench"),
                "Broad Jump": raw_2026.get("Broad Jump"),
                "3Cone": raw_2026.get("3Cone"),
                "Shuttle": raw_2026.get("Shuttle"),
                "arm_length_inches": raw_2026.get("arm_length_inches"),
                "Drafted": True,
                "Round": raw_2026.get("Round"),
                "Pick": raw_2026.get("Pick"),
            }
        )
        # Preserve arm_length_inches through ensure_base_columns (which only keeps base cols)
        arm_backup_2026 = base_2026.get("arm_length_inches", pd.Series(dtype=float)).reset_index(drop=True)
        # Run through same normalization / merge steps as main testing data
        base_2026 = ensure_base_columns(base_2026)
        base_2026["arm_length_inches"] = arm_backup_2026
        if not pff_data.empty:
            base_2026 = add_pff_data(base_2026, pff_data)
            base_2026 = add_derived_rushing_features(base_2026)
        base_2026 = add_ras_data(base_2026, RAS_PATH)
        base_2026 = add_arm_length(base_2026, arm_df)
        # Align columns to current testing frame
        aligned_cols = list(rb_testing_data.columns)
        base_2026 = base_2026.reindex(columns=aligned_cols)
        # Replace any existing 2026 rows with the rebuilt ones
        rb_testing_data = pd.concat(
            [rb_testing_data[rb_testing_data["Year"] != 2026], base_2026],
            ignore_index=True,
        )


# Final column ordering (similar shape to other position training files)

training_cols_order = [
    "Year",
    "Player",
    "Pos",
    "School",
    "Height",
    "Weight",
    "40yd",
    "Vertical",
    "Bench",
    "Broad Jump",
    "3Cone",
    "Shuttle",
    "Drafted",
    "Round",
    "Pick",
    "RAS",
    "arm_length_inches",
    "attempts",
    "yards",
    "yards_after_contact",
    "yco_attempt",
    "ypa",
    "touchdowns",
    "total_touches",
    "explosive",
    "first_downs",
    "fumbles",
    "avoided_tackles",
    "breakaway_attempts",
    "breakaway_percent",
    "breakaway_yards",
    "elusive_rating",
    "elu_rush_mtf",
    "mtf_per_attempt",
    "explosive_rate",
    "yprr",
    "targets",
    "routes",
    "targets_per_route",
    "fumble_rate",
]

training_cols_order = [c for c in training_cols_order if c in rb_training_data.columns]
rb_training_data = rb_training_data[training_cols_order].copy()

if not rb_testing_data.empty:
    testing_cols_order = [c for c in training_cols_order]  # same ordering
    testing_cols_order = [c for c in testing_cols_order if c in rb_testing_data.columns]
    rb_testing_data = rb_testing_data[testing_cols_order].copy()


# --- Write outputs ---

os.makedirs(DATA_PROCESSED, exist_ok=True)

rb_training_path = os.path.join(DATA_PROCESSED, "rb_training.csv")
rb_training_data.to_csv(rb_training_path, index=False)
print(f"Saved rb_training.csv: {len(rb_training_data)} (2015–2023)")

rb_testing_path = os.path.join(DATA_PROCESSED, "rb_testing.csv")
rb_testing_data.to_csv(rb_testing_path, index=False)
print(f"Saved rb_testing.csv: {len(rb_testing_data)} (2024–2026, drafted only)")

# Ensure a header-only rb_drafted_2026.csv exists if it has not been created yet.
# If the user has already created and formatted rb_drafted_2026.csv, do not modify it here.
rb_2026_path = os.path.join(SCRIPT_DIR, "rb_drafted_2026.csv")
if not os.path.exists(rb_2026_path):
    pd.DataFrame(
        columns=[
            "Round",
            "Pick",
            "Player",
            "Pos",
            "School",
            "Year",
            "Height",
            "Weight",
            "40yd",
            "Vertical",
            "Bench",
            "Broad Jump",
            "3Cone",
            "Shuttle",
            "RAS",
            "arm_length_inches",
        ]
    ).to_csv(rb_2026_path, index=False)
    print("Saved rb_drafted_2026.csv (empty header)")

