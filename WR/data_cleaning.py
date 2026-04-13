"""
WR (Wide Receiver) data cleaning: combine WRs + PFF Receiving + RAS + arm length.

- Training 2015–2023 from nfl_combine_2010_to_2023.csv (Pos == 'WR').
- Testing 2024 from data/raw/2024 Draft - Public - WR.csv.
- Testing 2025 from data/raw/GabrielGTB 2025 NFL Combine - Master List.csv (Position == 'WR')
  + 2025_draft_picks.csv for Round/Pick.
- Testing 2026 from data/raw/2026_Combine/NFL Combine 2026 @JordanSportGuy Twitter - WRs.csv.
- PFF: receiving stats from data/raw/pff/Receiving/*_receiving_summary.csv matched by
  Player + School + Year (filter to WR position).
- RAS: from data/raw/ras.csv.
- Arm length: from data/raw/mockdraftable_wr_arm_length.csv if available.

Outputs (written to data/processed/):
- wr_training.csv  (2015–2023)
- wr_testing.csv   (2024–2026)

Run from project root:
    python WR/data_cleaning.py
"""

import os
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# ─── Conversion helpers ───────────────────────────────────────────────────────

def _ht_4digit_to_inches(ht):
    if pd.isna(ht) or str(ht).strip() == "":
        return np.nan
    try:
        s = str(int(float(ht))).zfill(4)
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 4:
        return np.nan
    return int(s[0]) * 12 + int(s[1:3]) + int(s[3]) / 8.0


def _arm_4digit_to_inches(arm):
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


def _safe_num(val, dnp_val=np.nan):
    """Convert to float, treating empty/DNP as dnp_val."""
    if pd.isna(val):
        return dnp_val
    s = str(val).strip().upper()
    if s in ("DNP", "", "NA", "N/A"):
        return dnp_val
    try:
        return float(s)
    except ValueError:
        return dnp_val


# ─── Normalization ────────────────────────────────────────────────────────────

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
        "HAWAII": "Hawaii", "FRESNO ST": "Fresno State",
        "SAN JOSE ST": "San Jose State", "NEVADA": "Nevada",
        "FIU": "Florida International", "FLORIDA INTL": "Florida International",
        "GA SOUTHRN": "Georgia Southern", "GEORGIA SOUTHERN": "Georgia Southern",
        "ARK STATE": "Arkansas State", "LIBERTY": "Liberty",
        "JAMES MADISON": "James Madison", "S JOSE ST": "San Jose State",
    }
    return mapping.get(name, name.title())


def normalize_combine_school(name):
    if pd.isna(name):
        return name
    x = str(name).strip()
    upper_mapping = {
        "NOTRE DAME": "Notre Dame", "ALABAMA": "Alabama", "PENN STATE": "Penn State",
        "WASHINGTON": "Washington", "GEORGIA": "Georgia", "HOUSTON": "Houston",
        "OKLAHOMA": "Oklahoma", "MARYLAND": "Maryland", "KANSAS": "Kansas",
        "TCU": "TCU", "MISSOURI": "Missouri", "TEXAS": "Texas",
        "UTAH": "Utah", "PITTSBURGH": "Pittsburgh", "GA STATE": "Georgia State",
        "MICHIGAN": "Michigan", "CONNECTICUT": "Connecticut", "UCONN": "Connecticut",
        "UCF": "UCF", "OHIO STATE": "Ohio State", "FLORIDA STATE": "Florida State",
        "OHIO ST.": "Ohio State", "VIRGINIA TECH": "Virginia Tech",
        "S CAROLINA": "South Carolina", "COLO STATE": "Colorado State",
        "MICH STATE": "Michigan State", "MISS STATE": "Mississippi State",
        "OLE MISS": "Mississippi", "BOISE ST": "Boise State",
        "ARIZONA ST": "Arizona State", "NC STATE": "North Carolina State",
        "N CAROLINA": "North Carolina", "NORTH CAROLINA": "North Carolina",
        "TEXAS A&M": "Texas A&M", "NORTHWESTERN": "Northwestern",
        "ILLINOIS": "Illinois", "WAKE FOREST": "Wake Forest",
        "TEXAS TECH": "Texas Tech", "SAN DIEGO ST": "San Diego State",
        "SMU": "SMU", "CLEMSON": "Clemson", "ARKANSAS": "Arkansas",
        "VANDERBILT": "Vanderbilt", "LSU": "LSU", "IOWA": "Iowa",
        "INDIANA": "Indiana", "TENNESSEE": "Tennessee", "FLORIDA": "Florida",
        "MINNESOTA": "Minnesota", "NEBRASKA": "Nebraska",
        "BAYLOR": "Baylor", "COLORADO": "Colorado", "CINCINNATI": "Cincinnati",
        "MEMPHIS": "Memphis", "TULANE": "Tulane", "UTAH STATE": "Utah State",
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
        "Brigham Young": "BYU", "Central Florida": "UCF", "LA-Lafayette": "Louisiana",
        "West. Michigan": "Western Michigan", "Mississippi St.": "Mississippi State",
        "Arizona St.": "Arizona State", "Boise St.": "Boise State",
        "North Dakota St.": "North Dakota State", "Boston Col.": "Boston College",
        "San Diego St.": "San Diego State", "Lousiville": "Louisville",
        "Syracruse": "Syracuse", "Georgia Tech": "Georgia Tech",
        "Louisiana State": "LSU",
    }
    return alias.get(x, x)


# ─── Load PFF Receiving data ──────────────────────────────────────────────────

PFF_RECEIVING_DIR = os.path.join(DATA_RAW, "pff", "Receiving")

WR_PFF_POSITIONS = {"WR", "FL", "SE", "SL"}

receiving_cols = [
    "player", "team_name", "position",
    "player_game_count",
    "yprr", "yards_per_reception", "caught_percent",
    "avg_depth_of_target", "targeted_qb_rating",
    "contested_catch_rate", "drop_rate",
    "yards_after_catch_per_reception", "avoided_tackles",
    "slot_rate", "wide_rate", "inline_rate", "route_rate",
    "routes", "targets", "receptions",
]

receiving_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_RECEIVING_DIR, f"{year}_receiving_summary.csv")
    if not os.path.exists(path):
        continue
    df_r = pd.read_csv(path)
    sub = df_r[[c for c in receiving_cols if c in df_r.columns]].copy()
    if "player" not in sub.columns:
        continue
    # Filter to WR positions
    if "position" in sub.columns:
        sub = sub[sub["position"].astype(str).str.strip().str.upper().isin(WR_PFF_POSITIONS)].copy()
    sub["Year"] = year
    sub = sub.rename(columns={"player": "Player", "team_name": "School"})
    for c in receiving_cols[3:]:   # skip player/team_name/position
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    receiving_files.append(sub.drop(columns=["position"], errors="ignore"))
    print(f"Loaded PFF receiving (WR) {year}: {len(sub)} players")

if receiving_files:
    receiving_data = pd.concat(receiving_files, ignore_index=True)
    receiving_data = (
        receiving_data.sort_values("routes", ascending=False)
        .drop_duplicates(subset=["Player", "School", "Year"], keep="first")
        .reset_index(drop=True)
    )
    print(f"PFF WR receiving records (deduped): {len(receiving_data)}")
else:
    receiving_data = pd.DataFrame(
        columns=["Player", "School", "Year", "player_game_count", "yprr",
                 "yards_per_reception", "caught_percent", "avg_depth_of_target",
                 "targeted_qb_rating", "contested_catch_rate", "drop_rate",
                 "yards_after_catch_per_reception", "avoided_tackles",
                 "slot_rate", "wide_rate", "inline_rate", "route_rate"]
    )
    print("No PFF receiving files found.")


# ─── PFF merge ────────────────────────────────────────────────────────────────

_recv_value_cols = [c for c in receiving_data.columns
                    if c not in ("Player", "School", "Year", "routes", "targets", "receptions")]


def add_pff_receiving(combine_df: pd.DataFrame, pff_df: pd.DataFrame) -> pd.DataFrame:
    if pff_df.empty:
        for c in _recv_value_cols:
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
            # Try player-only match (handles school transfer edge cases)
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
        "Florida St.": "Florida State", "Penn St.": "Penn State",
        "Michigan St.": "Michigan State", "NC State": "North Carolina State",
        "Louisiana State": "LSU", "Boston Col.": "Boston College",
        "San Diego St.": "San Diego State", "Kansas St.": "Kansas State",
        "Iowa St.": "Iowa State", "Arizona St.": "Arizona State",
        "Mississippi St.": "Mississippi State", "West Virginia": "West Virginia",
        "Texas A&M": "Texas A&M", "Georgia Tech": "Georgia Tech",
        "North Carolina": "North Carolina", "South Carolina": "South Carolina",
        "Oregon St.": "Oregon State", "Washington St.": "Washington State",
        "BYU": "BYU", "Brigham Young": "BYU", "UCF": "UCF", "Central Florida": "UCF",
        "Montana St.": "Montana State", "Virginia Tech": "Virginia Tech",
        "Colorado State": "Colorado State",
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
WR_POSITIONS = ["WR"]
training_raw = nfl_combine_data[
    (nfl_combine_data["Pos"].isin(WR_POSITIONS)) & (nfl_combine_data["Year"].between(2015, 2023))
].copy()

training_raw["Height"] = training_raw["Height"].apply(_ht_combine_to_inches)
for c in ["Weight", "40yd", "Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle"]:
    training_raw[c] = pd.to_numeric(training_raw[c], errors="coerce")
print(f"WR training rows (2015–2023): {len(training_raw)}")


# ─── Load 2024 WRs from public draft file ────────────────────────────────────

draft_2024_path = os.path.join(DATA_RAW, "2024 Draft - Public - WR.csv")
wr_2024_list = []
if os.path.exists(draft_2024_path):
    d24 = pd.read_csv(draft_2024_path)
    d24 = d24[d24["Name"].notna() & (d24["Name"].astype(str).str.strip() != "")].copy()
    d24 = d24[d24["Pos"].astype(str).str.strip().str.upper() == "WR"].copy()
    for _, row in d24.iterrows():
        pick_taken = row.get("Pick Taken", row.iloc[0])
        pick_int = np.nan
        try:
            if str(pick_taken).strip() not in ("", "UDFA"):
                pick_int = int(float(str(pick_taken).replace(",", "")))
        except (ValueError, TypeError):
            pass
        wr_2024_list.append({
            "Year": 2024, "Player": str(row["Name"]).strip(), "Pos": "WR",
            "School": str(row["School"]).strip(),
            "Height": _ht_4digit_to_inches(row.get("HT")),
            "Weight": _safe_num(row.get("WT")),
            "40yd": _safe_num(row.get("40")),
            "Vertical": _safe_num(row.get("VJ")),
            "Bench": _safe_num(row.get("BP")),
            "Broad Jump": _broad_ffii_to_inches(str(row.get("BJ", "")).replace("DNP", "").strip() or np.nan),
            "3Cone": _safe_num(row.get("3c")),
            "Shuttle": _safe_num(row.get("Shuttle")),
            "Drafted": True, "Round": _pick_to_round(pick_taken), "Pick": pick_int,
            "RAS": _safe_num(row.get("RAS")),
            "arm_length_inches": _arm_4digit_to_inches(row.get("Arm")),
        })
    print(f"Loaded 2024 WRs: {len(wr_2024_list)}")
else:
    print("2024 Draft - Public - WR.csv not found.")

wr_2024 = pd.DataFrame(wr_2024_list)


# ─── Load 2025 WRs from GabrielGTB + draft picks ─────────────────────────────

combine_2025_path = os.path.join(DATA_RAW, "GabrielGTB 2025 NFL Combine - Master List.csv")
draft_picks_2025_path = os.path.join(DATA_RAW, "2025_draft_picks.csv")

wr_2025_list = []
if os.path.exists(combine_2025_path):
    c25 = pd.read_csv(combine_2025_path)
    c25_wr = c25[c25["Position"].astype(str).str.strip().str.upper() == "WR"].copy()
    for _, row in c25_wr.iterrows():
        wr_2025_list.append({
            "Year": 2025, "Player": str(row["Name"]).strip(), "Pos": "WR",
            "School": normalize_combine_school(str(row.get("School", "")).strip()),
            "Height": _ht_4digit_to_inches(row.get("Height (FIIE)")),
            "Weight": pd.to_numeric(row.get("Weight (lbs.)"), errors="coerce"),
            "40yd": pd.to_numeric(row.get("40-yard Dash (seconds)"), errors="coerce"),
            "Vertical": pd.to_numeric(row.get("Vertical Jump (inches)"), errors="coerce"),
            "Bench": pd.to_numeric(row.get("Bench Press (reps)"), errors="coerce"),
            "Broad Jump": _broad_ffii_to_inches(row.get("Broad Jump (FFII)")),
            "3Cone": pd.to_numeric(row.get("Three-cone Drill (seconds)"), errors="coerce"),
            "Shuttle": pd.to_numeric(row.get("20-yard Shuttle (seconds)"), errors="coerce"),
            "Drafted": True, "Round": np.nan, "Pick": np.nan,
            "RAS": pd.to_numeric(row.get("RAS"), errors="coerce"),
            "arm_length_inches": pd.to_numeric(row.get("Arm Length (inches)"), errors="coerce"),
        })
    print(f"Loaded 2025 WRs from GabrielGTB: {len(wr_2025_list)}")
else:
    print("GabrielGTB 2025 NFL Combine - Master List.csv not found.")

wr_2025 = pd.DataFrame(wr_2025_list)

if not wr_2025.empty and os.path.exists(draft_picks_2025_path):
    dp25 = pd.read_csv(draft_picks_2025_path)
    dp_wr = dp25[dp25["Pos"].astype(str).str.upper().isin(["WR"])].copy()

    def _norm_name(n):
        return re.sub(r"\s+Jr\.?$|\s+III$|\s+II$|\s+IV$", "", str(n).strip(),
                      flags=re.IGNORECASE).strip() if pd.notna(n) else ""

    def _norm_school(s):
        aliases = {"Penn St.": "Penn State", "Ohio St.": "Ohio State",
                   "Florida St.": "Florida State", "Ole Miss": "Mississippi",
                   "Syracruse": "Syracuse", "Lousiville": "Louisville",
                   "Washington St.": "Washington State", "Iowa St.": "Iowa State",
                   "Utah St.": "Utah State", "Virginia Tech": "Virginia Tech"}
        x = str(s).strip() if pd.notna(s) else ""
        return aliases.get(x, x)

    dp_wr = dp_wr.rename(columns={"Rnd": "Round"})
    dp_wr["Player_n"] = dp_wr["Player"].map(_norm_name)
    dp_wr["School_n"] = dp_wr["School"].map(_norm_school)
    wr_2025 = wr_2025.drop(columns=["Round", "Pick"], errors="ignore")
    wr_2025["Player_n"] = wr_2025["Player"].map(_norm_name)
    wr_2025["School_n"] = wr_2025["School"].map(_norm_school)
    wr_2025 = wr_2025.merge(
        dp_wr[["Player_n", "School_n", "Round", "Pick"]], on=["Player_n", "School_n"], how="left"
    ).drop(columns=["Player_n", "School_n"], errors="ignore")
    wr_2025["Round"] = wr_2025["Round"].fillna(8)
    wr_2025 = wr_2025[wr_2025["Round"] < 8].copy()
    print(f"2025 WRs after draft-pick filter (drafted only): {len(wr_2025)}")


# ─── Load 2026 WRs from JordanSportGuy combine file ──────────────────────────

combine_2026_path = os.path.join(
    DATA_RAW, "2026_Combine",
    "NFL Combine 2026 @JordanSportGuy Twitter - WRs.csv"
)
wr_2026_list = []
if os.path.exists(combine_2026_path):
    c26 = pd.read_csv(combine_2026_path)
    c26.columns = [col.strip().rstrip(":").strip() for col in c26.columns]
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
        wr_2026_list.append({
            "Year": 2026, "Player": player, "Pos": "WR", "School": school,
            "Height": _ht_4digit_to_inches(row.get("_ht_raw")),
            "Weight": pd.to_numeric(row.get("Weight"), errors="coerce"),
            "40yd": _safe_num(row.get("40yd")),
            "Vertical": _safe_num(row.get("Vertical")),
            "Bench": _safe_num(row.get("Bench")),
            "Broad Jump": _broad_ffii_to_inches(str(row.get("_broad_raw", "")).strip() or np.nan),
            "3Cone": _safe_num(row.get("3Cone")),
            "Shuttle": _safe_num(row.get("Shuttle")),
            "Drafted": False, "Round": np.nan, "Pick": np.nan,
            "arm_length_inches": _arm_4digit_to_inches(row.get("_arm_raw")),
        })
    print(f"Loaded 2026 WRs from JordanSportGuy: {len(wr_2026_list)}")
else:
    print(f"2026 WR combine file not found at {combine_2026_path}")

wr_2026 = pd.DataFrame(wr_2026_list)


# ─── Assemble testing frame ───────────────────────────────────────────────────

testing_frames = [f for f in [wr_2024, wr_2025, wr_2026] if not f.empty]
if testing_frames:
    wr_testing_data = pd.concat(testing_frames, ignore_index=True)
else:
    wr_testing_data = pd.DataFrame()
print(f"Total WR testing rows: {len(wr_testing_data)}")


# ─── Merge PFF, RAS, arm length ───────────────────────────────────────────────

training_raw = add_pff_receiving(training_raw, receiving_data)
if not wr_testing_data.empty:
    wr_testing_data = add_pff_receiving(wr_testing_data, receiving_data)

RAS_PATH = os.path.join(DATA_RAW, "ras.csv")
training_raw = add_ras_data(training_raw, RAS_PATH)
if not wr_testing_data.empty:
    wr_testing_data = add_ras_data(wr_testing_data, RAS_PATH)

arm_path = os.path.join(DATA_RAW, "mockdraftable_wr_arm_length.csv")
if os.path.exists(arm_path):
    arm_df = pd.read_csv(arm_path)
    if {"Player", "Year", "arm_length_inches"}.issubset(arm_df.columns):
        arm_df["Year"] = arm_df["Year"].astype(int)
        arm_df["arm_length_inches"] = pd.to_numeric(arm_df["arm_length_inches"], errors="coerce")
        print(f"Arm length WR: {len(arm_df)} records")
    else:
        arm_df = pd.DataFrame(columns=["Player", "Year", "arm_length_inches"])
else:
    arm_df = pd.DataFrame(columns=["Player", "Year", "arm_length_inches"])
    print("No mockdraftable_wr_arm_length.csv; arm_length_inches from combine files only.")

training_raw = add_arm_length(training_raw, arm_df)
if not wr_testing_data.empty:
    wr_testing_data = add_arm_length(wr_testing_data, arm_df)


# ─── Final column ordering ────────────────────────────────────────────────────

COL_ORDER = [
    "Year", "Player", "Pos", "School",
    "Height", "Weight", "40yd", "Vertical", "Broad Jump",
    "Bench", "3Cone", "Shuttle",
    "Drafted", "Round", "Pick",
    "RAS", "arm_length_inches",
    "player_game_count",
    "yprr", "yards_per_reception", "caught_percent",
    "avg_depth_of_target", "targeted_qb_rating",
    "contested_catch_rate", "drop_rate",
    "yards_after_catch_per_reception", "avoided_tackles",
    "slot_rate", "wide_rate", "inline_rate", "route_rate",
]


def reorder_cols(df, col_order):
    present = [c for c in col_order if c in df.columns]
    extra = [c for c in df.columns if c not in col_order]
    return df[present + extra]


training_raw = reorder_cols(training_raw, COL_ORDER)
if not wr_testing_data.empty:
    wr_testing_data = reorder_cols(wr_testing_data, COL_ORDER)


# ─── PFF match stats ──────────────────────────────────────────────────────────

pff_cols = ["player_game_count", "yprr", "yards_per_reception", "caught_percent",
            "avg_depth_of_target", "targeted_qb_rating", "contested_catch_rate",
            "drop_rate", "yards_after_catch_per_reception", "avoided_tackles",
            "slot_rate", "wide_rate", "inline_rate", "route_rate"]
for col in pff_cols:
    if col in training_raw.columns:
        n = training_raw[col].notna().sum()
        print(f"Training WRs with {col}: {n}/{len(training_raw)}")


# ─── Write outputs ────────────────────────────────────────────────────────────

os.makedirs(DATA_PROCESSED, exist_ok=True)

training_raw.to_csv(os.path.join(DATA_PROCESSED, "wr_training.csv"), index=False)
print(f"Saved wr_training.csv: {len(training_raw)} rows (2015–2023)")

if not wr_testing_data.empty:
    wr_testing_data.to_csv(os.path.join(DATA_PROCESSED, "wr_testing.csv"), index=False)
    print(f"Saved wr_testing.csv: {len(wr_testing_data)} rows (2024–2026)")
else:
    pd.DataFrame(columns=COL_ORDER).to_csv(
        os.path.join(DATA_PROCESSED, "wr_testing.csv"), index=False
    )
    print("Saved wr_testing.csv (empty)")
