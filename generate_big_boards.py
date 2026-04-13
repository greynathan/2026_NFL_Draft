#!/usr/bin/env python3
"""
generate_big_boards.py

Re-trains each position's Ridge regression and produces:
  data/processed/big_board_2026_detailed.csv  – 2026 prospects + all model inputs
  data/processed/big_board_2024_detailed.csv  – 2024 drafted + actual Round/Pick
  data/processed/big_board_2025_detailed.csv  – 2025 drafted + actual Round/Pick
  model_performance.txt                        – per-position metrics
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "processed")

# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def height_to_inches(h):
    if pd.isna(h): return np.nan
    if isinstance(h, float) and not np.isnan(h): return h
    if isinstance(h, int): return float(h)
    s = str(h).strip()
    if "-" in s:
        parts = s.split("-")
        try: return int(parts[0]) * 12 + int(parts[1])
        except: return np.nan
    try: return float(s)
    except: return np.nan

SCHOOL_ALIAS = {
    "Ole Miss": "Mississippi", "Miami (FL)": "Miami",
    "Southern California": "USC", "Central Florida": "UCF",
    "Brigham Young": "BYU", "Ohio St.": "Ohio State",
    "Florida St.": "Florida State", "Kansas St.": "Kansas State",
    "Iowa St.": "Iowa State", "Oklahoma St.": "Oklahoma State",
    "Penn St.": "Penn State", "NC State": "North Carolina State",
    "Oregon St.": "Oregon State", "Boston Col.": "Boston College",
}
SEC = {"Alabama","Arkansas","Auburn","Florida","Georgia","Kentucky","LSU","Mississippi",
       "Mississippi State","Missouri","South Carolina","Tennessee","Texas A&M","Vanderbilt",
       "Oklahoma","Texas"}
BIG_TEN = {"Illinois","Indiana","Iowa","Maryland","Michigan","Michigan State","Minnesota",
           "Nebraska","Northwestern","Ohio State","Penn State","Purdue","Rutgers","Wisconsin",
           "UCLA","USC","Oregon","Washington"}
BIG_12 = {"Baylor","Iowa State","Kansas","Kansas State","Oklahoma State","TCU","Texas Tech",
          "West Virginia","BYU","UCF","Cincinnati","Houston","Arizona","Arizona State",
          "Colorado","Utah","SMU"}
ACC = {"Boston College","Clemson","Duke","Florida State","Georgia Tech","Louisville","Miami",
       "North Carolina","North Carolina State","Pittsburgh","Syracuse","Virginia","Virginia Tech",
       "Wake Forest","California","Stanford","SMU"}
PAC12 = {"Arizona","Arizona State","California","Colorado","Oregon","Oregon State","Stanford",
         "UCLA","USC","Utah","Washington","Washington State"}
P4_PRE2024 = SEC | BIG_TEN | BIG_12 | ACC | PAC12
P4_2024PLUS = SEC | BIG_TEN | BIG_12 | ACC  # excludes Pac-12 remnants

def p4_conf(school, year):
    s = SCHOOL_ALIAS.get(str(school).strip(), str(school).strip())
    return 1 if s in (P4_PRE2024 if int(year) < 2024 else P4_2024PLUS) else 0

def metrics(actual, pred):
    mae  = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2   = r2_score(actual, pred)
    exact = float((np.round(pred) == actual).mean())
    w1    = float((np.abs(np.round(pred) - actual) <= 1).mean())
    return dict(n=len(actual), mae=mae, rmse=rmse, r2=r2, exact=exact, w1=w1)

def train_model(df, features_all, target_col="Round", drafted_col="Drafted"):
    y = np.where(df[drafted_col].astype(bool),
                 np.clip(df[target_col].fillna(1).astype(int), 1, 7), 8)
    X_raw = df[features_all].copy()
    for c in features_all:
        if c not in X_raw.columns: X_raw[c] = np.nan
    # Track which features are all-NaN in training (KNNImputer will drop them)
    all_nan_mask = X_raw.isna().all()
    kept_features = [f for f, drop in zip(features_all, all_nan_mask) if not drop]
    imp = KNNImputer(n_neighbors=10)
    X   = imp.fit_transform(X_raw)
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    mdl = Ridge(alpha=1.0, random_state=42)
    mdl.fit(Xs, y)
    return mdl, imp, sc, y, kept_features

def predict(df, features_all, mdl, imp, sc):
    X_raw = df[features_all].copy()
    for c in features_all:
        if c not in X_raw.columns: X_raw[c] = np.nan
    return np.clip(mdl.predict(sc.transform(imp.transform(X_raw))), 1, 8)

def read(relpath):
    return pd.read_csv(os.path.join(ROOT, relpath))

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _speed(df):
    return np.where(
        df["40yd"].notna() & (df["40yd"] > 0),
        df["Weight"] * 200 / (df["40yd"] ** 4), np.nan)

def eng_standard(df, stats):
    """CB / S / LB / ED / DT: speed_score + explosive_score(Vert + BroadJump)."""
    df = df.copy()
    df["Height"] = df["Height"].apply(height_to_inches)
    for c in ["Weight","40yd","Vertical","Broad Jump","RAS","arm_length_inches"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["speed_score"] = _speed(df)
    mv, sv, mb, sb = stats["mean_v"], stats["std_v"], stats["mean_b"], stats["std_b"]
    df["explosive_score"] = ((df["Vertical"] - mv).fillna(0) / sv
                           + (df["Broad Jump"] - mb).fillna(0) / sb)
    df["p4_conference"] = df.apply(lambda r: p4_conf(r.get("School",""), r.get("Year",2020)), axis=1)
    return df

def eng_oline(df, stats):
    """OT / IOL: speed_score + agility_score(3Cone + Shuttle)."""
    df = df.copy()
    df["Height"] = df["Height"].apply(height_to_inches)
    for c in ["Weight","40yd","arm_length_inches","3Cone","Shuttle","RAS"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["speed_score"] = _speed(df)
    m3, s3, ms, ss = stats["mean_3c"], stats["std_3c"], stats["mean_sh"], stats["std_sh"]
    z3 = (df.get("3Cone", np.nan) - m3) / s3
    zs = (df.get("Shuttle", np.nan) - ms) / ss
    df["agility_score"] = (-z3.fillna(0)) + (-zs.fillna(0))
    df["arm_33_plus"] = (df["arm_length_inches"] >= 33).fillna(False).astype(int) if "arm_length_inches" in df.columns else 0
    df["arm_34_plus"] = (df["arm_length_inches"] >= 34).fillna(False).astype(int) if "arm_length_inches" in df.columns else 0
    df["p4_conference"] = df.apply(lambda r: p4_conf(r.get("School",""), r.get("Year",2020)), axis=1)
    return df

def eng_skill_vert(df, stats):
    """WR / TE: speed_score + explosive_score(Vertical only)."""
    df = df.copy()
    df["Height"] = df["Height"].apply(height_to_inches)
    for c in ["Weight","40yd","Vertical","Broad Jump","RAS","arm_length_inches"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["speed_score"] = _speed(df)
    mv, sv = stats["mean_v"], stats["std_v"]
    df["explosive_score"] = np.where(df["Vertical"].notna(),
                                     (df["Vertical"] - mv) / (sv + 1e-8), np.nan)
    df["p4_conference"] = df.apply(lambda r: p4_conf(r.get("School",""), r.get("Year",2020)), axis=1)
    return df

def eng_qb(df, stats):
    """QB: speed_score only."""
    df = df.copy()
    df["Height"] = df["Height"].apply(height_to_inches)
    for c in ["Weight","40yd","RAS"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["speed_score"] = _speed(df)
    df["p4_conference"] = df.apply(lambda r: p4_conf(r.get("School",""), r.get("Year",2020)), axis=1)
    return df

def eng_rb(df, stats):
    """RB: height conversion only (no speed_score/p4 in feature list)."""
    df = df.copy()
    df["Height"] = df["Height"].apply(height_to_inches)
    for c in ["Weight","40yd","Vertical","Bench","Broad Jump","RAS","arm_length_inches"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_contains(df, raw_features):
    """Add contains_* flags for all raw features."""
    for feat in raw_features:
        flag = f"contains_{feat}"
        if flag not in df.columns:
            df[flag] = df[feat].notna().astype(int) if feat in df.columns else 0
    if "contains_p4_conference" not in df.columns:
        df["contains_p4_conference"] = df["School"].notna().astype(int) if "School" in df.columns else 1
    return df

def ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Output row builder
# ─────────────────────────────────────────────────────────────────────────────

ALL_PFF_COLS = [
    # Defensive coverage
    "yards_per_coverage_snap","forced_incompletion_rate","snap_counts_coverage",
    "coverage_percent","coverage_snaps_per_target","INT_rate","PBU_rate",
    "qb_rating_against","catch_rate",
    # Defensive run
    "stop_percent","missed_tackle_rate","avg_depth_of_tackle","snap_counts_run","forced_fumbles",
    "interceptions","pass_break_ups",
    # Pass rush
    "true_pass_set_pass_rush_win_rate","pass_rush_win_rate","snap_counts_pass_rush",
    # OLine
    "true_pass_set_pressure_rate","true_pass_set_sack_rate","snap_counts_pass_block",
    "snap_counts_run_block","grades_run_block","gap_rate","zone_rate","penalty_rate",
    "arm_33_plus","arm_34_plus","is_center",
    # QB
    "btt_rate","twp_rate","ypa","qb_rating","pressure_to_sack_rate","sack_percent",
    "epa","positive_epa_percent","player_game_count",
    # WR/TE receiving
    "yprr","yards_per_reception","caught_percent","avg_depth_of_target","targeted_qb_rating",
    "contested_catch_rate","drop_rate","yards_after_catch_per_reception","avoided_tackles",
    "slot_rate","wide_rate","inline_rate","route_rate",
    # RB
    "yards_after_contact","yco_attempt","elusive_rating","mtf_per_attempt",
    "breakaway_percent","explosive_rate","targets_per_route","fumble_rate",
]

ATHLETIC_COLS = ["Height","Weight","arm_length_inches","40yd","Vertical","Broad Jump",
                 "Bench","3Cone","Shuttle","RAS","speed_score","explosive_score",
                 "agility_score","p4_conference"]

def collect_rows(df, preds, pos, year, include_actual=False):
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        r = {
            "Player": row.get("Player",""),
            "School": row.get("School",""),
            "Pos": pos,
            "Year": year,
            "predicted_round": round(float(preds[i]), 4),
        }
        if include_actual:
            raw_round = row.get("Round", np.nan)
            raw_pick  = row.get("Pick", np.nan)
            try: r["actual_round"] = int(float(raw_round)) if pd.notna(raw_round) else np.nan
            except: r["actual_round"] = np.nan
            try: r["actual_pick"] = int(float(raw_pick)) if pd.notna(raw_pick) else np.nan
            except: r["actual_pick"] = np.nan
        for c in ATHLETIC_COLS:
            r[c] = row.get(c, np.nan)
        for c in ALL_PFF_COLS:
            r[c] = row.get(c, np.nan)
        rows.append(r)
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# Position runners
# ─────────────────────────────────────────────────────────────────────────────

def run_standard(label, pos, train_csv, test_csv, test_2026_path,
                 features_main, features_all, pos_pff_cols):
    """CB, S, LB, ED, DT."""
    print(f"\n{'='*60}\n{pos} ({label})")
    df = read(f"data/processed/{train_csv}")
    df = df[df["Year"].between(2015, 2023)].copy()
    for c in pos_pff_cols + ["Vertical","Broad Jump","RAS","arm_length_inches"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    mean_v = df["Vertical"].mean(); std_v = max(df["Vertical"].std(), 1e-8)
    mean_b = df["Broad Jump"].mean(); std_b = max(df["Broad Jump"].std(), 1e-8)
    stats  = dict(mean_v=mean_v, std_v=std_v, mean_b=mean_b, std_b=std_b)
    df = eng_standard(df, stats)
    df = ensure_cols(df, features_all)
    df = add_contains(df, features_main)
    mdl, imp, sc, y_train, kept = train_model(df, features_all)
    y_pred_train = np.clip(mdl.predict(sc.transform(imp.transform(df[features_all]))), 1, 8)
    train_m = metrics(y_train, y_pred_train)
    print(f"  Train n={train_m['n']}  MAE={train_m['mae']:.4f}  R²={train_m['r2']:.4f}")

    # 2024 / 2025
    results = {2024:[], 2025:[], 2026:[]}
    holdout_m = {}
    test_df = read(f"data/processed/{test_csv}")
    for yr in [2024, 2025]:
        sub = test_df[test_df["Year"] == yr].copy()
        if sub.empty: continue
        sub["Year"] = yr
        sub = eng_standard(sub, stats)
        sub = ensure_cols(sub, features_all)
        sub = add_contains(sub, features_main)
        preds = predict(sub, features_all, mdl, imp, sc)
        actual = pd.to_numeric(sub["Round"], errors="coerce").fillna(8).astype(int).values
        holdout_m[yr] = metrics(actual, preds)
        m = holdout_m[yr]
        print(f"  {yr}: n={m['n']}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
              f"Exact={m['exact']:.1%}  W1={m['w1']:.1%}")
        results[yr] = collect_rows(sub, preds, pos, yr, include_actual=True)

    # 2026
    if test_2026_path:
        df26 = read(test_2026_path)
    else:
        df26 = test_df[test_df["Year"] == 2026].copy()
    df26["Year"] = 2026
    df26 = eng_standard(df26, stats)
    df26 = ensure_cols(df26, features_all)
    df26 = add_contains(df26, features_main)
    p26 = predict(df26, features_all, mdl, imp, sc)
    print(f"  2026: n={len(p26)}")
    results[2026] = collect_rows(df26, p26, pos, 2026, include_actual=False)

    perf_lines = _perf_block(label, pos, kept, mdl, train_m, holdout_m)
    return results, perf_lines


def run_oline(label, pos, train_csv, test_csv, test_2026_path,
              features_main, features_all, pos_pff_cols):
    """OT, IOL."""
    print(f"\n{'='*60}\n{pos} ({label})")
    df = read(f"data/processed/{train_csv}")
    df = df[df["Year"].between(2015, 2023)].copy()
    for c in ["3Cone","Shuttle","arm_length_inches"] + pos_pff_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    mean_3c = df["3Cone"].mean(); std_3c = max(df["3Cone"].std(), 1e-8)
    mean_sh = df["Shuttle"].mean(); std_sh = max(df["Shuttle"].std(), 1e-8)
    stats = dict(mean_3c=mean_3c, std_3c=std_3c, mean_sh=mean_sh, std_sh=std_sh)
    df = eng_oline(df, stats)
    if pos == "IOL" and "Pos" in df.columns:
        df["is_center"] = (df["Pos"].astype(str).str.strip().str.upper() == "C").astype(int)
    df = ensure_cols(df, features_all)
    df = add_contains(df, features_main)
    mdl, imp, sc, y_train, kept = train_model(df, features_all)
    y_pred_train = np.clip(mdl.predict(sc.transform(imp.transform(df[features_all]))), 1, 8)
    train_m = metrics(y_train, y_pred_train)
    print(f"  Train n={train_m['n']}  MAE={train_m['mae']:.4f}  R²={train_m['r2']:.4f}")

    results = {2024:[], 2025:[], 2026:[]}
    holdout_m = {}
    test_df = read(f"data/processed/{test_csv}")
    for yr in [2024, 2025]:
        sub = test_df[test_df["Year"] == yr].copy()
        if sub.empty: continue
        sub["Year"] = yr
        sub = eng_oline(sub, stats)
        if pos == "IOL" and "Pos" in sub.columns:
            sub["is_center"] = (sub["Pos"].astype(str).str.strip().str.upper() == "C").astype(int)
        sub = ensure_cols(sub, features_all)
        sub = add_contains(sub, features_main)
        preds = predict(sub, features_all, mdl, imp, sc)
        actual = pd.to_numeric(sub["Round"], errors="coerce").fillna(8).astype(int).values
        holdout_m[yr] = metrics(actual, preds)
        m = holdout_m[yr]
        print(f"  {yr}: n={m['n']}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
              f"Exact={m['exact']:.1%}  W1={m['w1']:.1%}")
        results[yr] = collect_rows(sub, preds, pos, yr, include_actual=True)

    df26 = read(test_2026_path)
    df26["Year"] = 2026
    df26 = eng_oline(df26, stats)
    if pos == "IOL" and "Pos" in df26.columns:
        df26["is_center"] = (df26["Pos"].astype(str).str.strip().str.upper() == "C").astype(int)
    df26 = ensure_cols(df26, features_all)
    df26 = add_contains(df26, features_main)
    p26 = predict(df26, features_all, mdl, imp, sc)
    print(f"  2026: n={len(p26)}")
    results[2026] = collect_rows(df26, p26, pos, 2026, include_actual=False)

    perf_lines = _perf_block(label, pos, kept, mdl, train_m, holdout_m)
    return results, perf_lines


def run_skill(label, pos, train_csv, test_csv, features_main, features_all,
              eng_fn, eng_stats_fn, pos_pff_cols):
    """QB, WR, TE, RB."""
    print(f"\n{'='*60}\n{pos} ({label})")
    df = read(f"data/processed/{train_csv}")
    df = df[df["Year"].between(2015, 2023)].copy()
    for c in pos_pff_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    stats = eng_stats_fn(df)
    df = eng_fn(df, stats)
    df = ensure_cols(df, features_all)
    df = add_contains(df, features_main)
    mdl, imp, sc, y_train, kept = train_model(df, features_all)
    y_pred_train = np.clip(mdl.predict(sc.transform(imp.transform(df[features_all]))), 1, 8)
    train_m = metrics(y_train, y_pred_train)
    print(f"  Train n={train_m['n']}  MAE={train_m['mae']:.4f}  R²={train_m['r2']:.4f}")

    results = {2024:[], 2025:[], 2026:[]}
    holdout_m = {}
    test_df = read(f"data/processed/{test_csv}")
    for yr in [2024, 2025]:
        drafted = test_df[(test_df["Year"] == yr) &
                          (pd.to_numeric(test_df["Round"], errors="coerce") < 8)].copy()
        if drafted.empty: continue
        drafted["Year"] = yr
        drafted = eng_fn(drafted, stats)
        drafted = ensure_cols(drafted, features_all)
        drafted = add_contains(drafted, features_main)
        preds = predict(drafted, features_all, mdl, imp, sc)
        actual = pd.to_numeric(drafted["Round"], errors="coerce").fillna(8).astype(int).values
        holdout_m[yr] = metrics(actual, preds)
        m = holdout_m[yr]
        print(f"  {yr}: n={m['n']}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  "
              f"Exact={m['exact']:.1%}  W1={m['w1']:.1%}")
        results[yr] = collect_rows(drafted, preds, pos, yr, include_actual=True)

    prospects = test_df[test_df["Year"] == 2026].copy()
    prospects["Year"] = 2026
    prospects = eng_fn(prospects, stats)
    prospects = ensure_cols(prospects, features_all)
    prospects = add_contains(prospects, features_main)
    p26 = predict(prospects, features_all, mdl, imp, sc)
    print(f"  2026: n={len(p26)}")
    results[2026] = collect_rows(prospects, p26, pos, 2026, include_actual=False)

    perf_lines = _perf_block(label, pos, kept, mdl, train_m, holdout_m)
    return results, perf_lines


def _perf_block(label, pos, features_all, mdl, train_m, holdout_m):
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"{pos} – {label}")
    lines.append(f"{'='*60}")
    lines.append(f"Features ({len(features_all)}): {', '.join(features_all)}")
    lines.append(f"\nTraining set (2015–2023):")
    lines.append(f"  n={train_m['n']}  MAE={train_m['mae']:.4f}  RMSE={train_m['rmse']:.4f}  R²={train_m['r2']:.4f}")
    for yr, m in sorted(holdout_m.items()):
        lines.append(f"\n{yr} holdout:")
        lines.append(f"  n={m['n']}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  "
                     f"Exact={m['exact']:.1%}  Within-1={m['w1']:.1%}")
    coef_df = pd.DataFrame({"feature": features_all, "coef": mdl.coef_})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top5 = coef_df.nlargest(5, "abs_coef")
    lines.append("\nTop 5 features by |coefficient|:")
    for _, row in top5.iterrows():
        lines.append(f"  {row['feature']:40s}  {row['coef']:+.4f}")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Position definitions
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_rows = {2024:[], 2025:[], 2026:[]}
    perf_all = []

    # ── CB ────────────────────────────────────────────────────────────────────
    CB_MAIN = ["Broad Jump","Vertical","40yd","Height","Weight",
               "speed_score","explosive_score","RAS","arm_length_inches",
               "missed_tackle_rate","forced_fumbles",
               "yards_per_coverage_snap","forced_incompletion_rate",
               "snap_counts_coverage","coverage_percent","coverage_snaps_per_target",
               "INT_rate","PBU_rate","qb_rating_against","catch_rate","avg_depth_of_target",
               "p4_conference"]
    CB_CONTAINS = [f"contains_{f}" for f in CB_MAIN]
    CB_ALL = CB_MAIN + CB_CONTAINS
    rows, lines = run_standard("Corner Back","CB","cb_training.csv","cb_testing.csv",
                               "CB/cb_drafted_2026.csv",CB_MAIN,CB_ALL,
                               ["missed_tackle_rate","forced_fumbles","yards_per_coverage_snap",
                                "forced_incompletion_rate","snap_counts_coverage","coverage_percent",
                                "coverage_snaps_per_target","INT_rate","PBU_rate",
                                "qb_rating_against","catch_rate","avg_depth_of_target"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── S ─────────────────────────────────────────────────────────────────────
    S_MAIN = ["Broad Jump","Vertical","40yd","Height","Weight",
              "speed_score","explosive_score","RAS","arm_length_inches",
              "true_pass_set_pass_rush_win_rate","pass_rush_win_rate","snap_counts_pass_rush",
              "stop_percent","missed_tackle_rate","avg_depth_of_tackle","snap_counts_run","forced_fumbles",
              "yards_per_coverage_snap","forced_incompletion_rate","snap_counts_coverage","coverage_percent",
              "coverage_snaps_per_target","INT_rate","PBU_rate",
              "qb_rating_against","catch_rate","avg_depth_of_target",
              "p4_conference"]
    S_CONTAINS = [f"contains_{f}" for f in S_MAIN]
    S_ALL = S_MAIN + S_CONTAINS
    rows, lines = run_standard("Safety","S","s_training.csv","s_testing.csv",
                               "S/s_drafted_2026.csv",S_MAIN,S_ALL,
                               ["true_pass_set_pass_rush_win_rate","pass_rush_win_rate",
                                "snap_counts_pass_rush","stop_percent","missed_tackle_rate",
                                "avg_depth_of_tackle","snap_counts_run","forced_fumbles",
                                "yards_per_coverage_snap","forced_incompletion_rate",
                                "snap_counts_coverage","coverage_percent","coverage_snaps_per_target",
                                "INT_rate","PBU_rate","qb_rating_against","catch_rate","avg_depth_of_target"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── LB ────────────────────────────────────────────────────────────────────
    LB_MAIN = ["Broad Jump","Vertical","40yd","Height","Weight",
               "speed_score","explosive_score","RAS","arm_length_inches",
               "true_pass_set_pass_rush_win_rate","pass_rush_win_rate","snap_counts_pass_rush","stop_percent",
               "missed_tackle_rate","avg_depth_of_tackle","snap_counts_run","forced_fumbles",
               "yards_per_coverage_snap","forced_incompletion_rate","snap_counts_coverage","coverage_percent",
               "interceptions","pass_break_ups","coverage_snaps_per_target","INT_rate","PBU_rate",
               "p4_conference"]
    LB_CONTAINS = [f"contains_{f}" for f in LB_MAIN]
    LB_ALL = LB_MAIN + LB_CONTAINS
    rows, lines = run_standard("Linebacker","LB","lb_training.csv","lb_testing.csv",
                               None,LB_MAIN,LB_ALL,
                               ["true_pass_set_pass_rush_win_rate","pass_rush_win_rate",
                                "snap_counts_pass_rush","stop_percent","missed_tackle_rate",
                                "avg_depth_of_tackle","snap_counts_run","forced_fumbles",
                                "yards_per_coverage_snap","forced_incompletion_rate",
                                "snap_counts_coverage","coverage_percent","interceptions",
                                "pass_break_ups","coverage_snaps_per_target","INT_rate","PBU_rate"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── ED ────────────────────────────────────────────────────────────────────
    ED_MAIN = ["Broad Jump","Vertical","40yd","Height","Weight",
               "speed_score","explosive_score","RAS","arm_length_inches",
               "true_pass_set_pass_rush_win_rate","pass_rush_win_rate","snap_counts_pass_rush",
               "stop_percent","p4_conference"]
    ED_CONTAINS = [f"contains_{f}" for f in ED_MAIN]
    ED_ALL = ED_MAIN + ED_CONTAINS
    rows, lines = run_standard("Edge Rusher","ED","edge_training.csv","edge_testing.csv",
                               None,ED_MAIN,ED_ALL,
                               ["true_pass_set_pass_rush_win_rate","pass_rush_win_rate",
                                "snap_counts_pass_rush","stop_percent"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── DT ────────────────────────────────────────────────────────────────────
    DT_MAIN = ["Broad Jump","Vertical","40yd","Height","Weight",
               "speed_score","explosive_score","RAS","arm_length_inches",
               "true_pass_set_pass_rush_win_rate","pass_rush_win_rate","snap_counts_pass_rush",
               "stop_percent","p4_conference"]
    DT_CONTAINS = [f"contains_{f}" for f in DT_MAIN]
    DT_ALL = DT_MAIN + DT_CONTAINS
    rows, lines = run_standard("Defensive Tackle","DT","dt_training.csv","dt_testing.csv",
                               None,DT_MAIN,DT_ALL,
                               ["true_pass_set_pass_rush_win_rate","pass_rush_win_rate",
                                "snap_counts_pass_rush","stop_percent"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── OT ────────────────────────────────────────────────────────────────────
    OT_MAIN = ["Height","Weight","arm_length_inches","arm_33_plus","arm_34_plus",
               "speed_score","agility_score",
               "true_pass_set_pressure_rate","true_pass_set_sack_rate","snap_counts_pass_block",
               "snap_counts_run_block","grades_run_block","gap_rate","zone_rate","penalty_rate",
               "p4_conference"]
    OT_CONTAINS = ["contains_height","contains_weight","contains_arm_length_inches",
                   "contains_speed_score","contains_agility_score",
                   "contains_true_pass_set_pressure_rate","contains_true_pass_set_sack_rate",
                   "contains_snap_counts_pass_block","contains_snap_counts_run_block",
                   "contains_grades_run_block","contains_gap_rate","contains_zone_rate",
                   "contains_penalty_rate","contains_p4_conference"]
    OT_ALL = OT_MAIN + OT_CONTAINS
    rows, lines = run_oline("Offensive Tackle","OT","ot_training.csv","ot_testing.csv",
                            "OT/ot_drafted_2026.csv",OT_MAIN,OT_ALL,
                            ["true_pass_set_pressure_rate","true_pass_set_sack_rate",
                             "snap_counts_pass_block","snap_counts_run_block",
                             "grades_run_block","gap_rate","zone_rate","penalty_rate"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── IOL ───────────────────────────────────────────────────────────────────
    IOL_MAIN = ["Height","Weight","arm_length_inches","speed_score","RAS","is_center",
                "true_pass_set_pressure_rate","true_pass_set_sack_rate","snap_counts_pass_block",
                "snap_counts_run_block","grades_run_block","gap_rate","zone_rate","penalty_rate",
                "p4_conference"]
    IOL_CONTAINS = ["contains_height","contains_weight","contains_arm_length_inches",
                    "contains_speed_score","contains_agility_score","contains_ras",
                    "contains_3cone","contains_shuttle",
                    "contains_true_pass_set_pressure_rate","contains_true_pass_set_sack_rate",
                    "contains_snap_counts_pass_block","contains_snap_counts_run_block",
                    "contains_grades_run_block","contains_gap_rate","contains_zone_rate",
                    "contains_penalty_rate","contains_p4_conference"]
    IOL_ALL = IOL_MAIN + IOL_CONTAINS
    rows, lines = run_oline("Interior Offensive Lineman","IOL","iol_training.csv","iol_testing.csv",
                            "IOL/iol_drafted_2026.csv",IOL_MAIN,IOL_ALL,
                            ["true_pass_set_pressure_rate","true_pass_set_sack_rate",
                             "snap_counts_pass_block","snap_counts_run_block",
                             "grades_run_block","gap_rate","zone_rate","penalty_rate"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── QB ────────────────────────────────────────────────────────────────────
    QB_MAIN = ["Height","Weight","40yd","speed_score","RAS",
               "btt_rate","twp_rate","ypa","qb_rating",
               "pressure_to_sack_rate","sack_percent",
               "epa","positive_epa_percent","avg_depth_of_target",
               "player_game_count","p4_conference"]
    QB_CONTAINS = [f"contains_{f}" for f in QB_MAIN]
    QB_ALL = QB_MAIN + QB_CONTAINS

    def qb_stats(df): return {}

    rows, lines = run_skill("Quarterback","QB","qb_training.csv","qb_testing.csv",
                            QB_MAIN,QB_ALL,eng_qb,qb_stats,
                            ["btt_rate","twp_rate","ypa","qb_rating",
                             "pressure_to_sack_rate","sack_percent",
                             "epa","positive_epa_percent","avg_depth_of_target","player_game_count"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── WR ────────────────────────────────────────────────────────────────────
    WR_MAIN = ["Height","Weight","40yd","Vertical","Broad Jump",
               "speed_score","explosive_score","RAS","arm_length_inches",
               "yprr","yards_per_reception","caught_percent","avg_depth_of_target",
               "targeted_qb_rating","contested_catch_rate","drop_rate",
               "yards_after_catch_per_reception","avoided_tackles",
               "slot_rate","wide_rate","inline_rate","route_rate",
               "player_game_count","p4_conference"]
    WR_CONTAINS = [f"contains_{f}" for f in WR_MAIN]
    WR_ALL = WR_MAIN + WR_CONTAINS

    def wr_stats(df):
        mv = df["Vertical"].mean(); sv = max(df["Vertical"].std(), 1e-8)
        return dict(mean_v=mv, std_v=sv)

    rows, lines = run_skill("Wide Receiver","WR","wr_training.csv","wr_testing.csv",
                            WR_MAIN,WR_ALL,eng_skill_vert,wr_stats,
                            ["yprr","yards_per_reception","caught_percent","avg_depth_of_target",
                             "targeted_qb_rating","contested_catch_rate","drop_rate",
                             "yards_after_catch_per_reception","avoided_tackles",
                             "slot_rate","wide_rate","inline_rate","route_rate","player_game_count"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── TE ────────────────────────────────────────────────────────────────────
    TE_MAIN = ["Height","Weight","40yd","Vertical","Broad Jump",
               "speed_score","explosive_score","RAS","arm_length_inches",
               "yprr","yards_per_reception","caught_percent","avg_depth_of_target",
               "targeted_qb_rating","contested_catch_rate","drop_rate",
               "yards_after_catch_per_reception","avoided_tackles",
               "slot_rate","wide_rate","inline_rate","route_rate",
               "grades_run_block",
               "player_game_count","p4_conference"]
    TE_CONTAINS = [f"contains_{f}" for f in TE_MAIN]
    TE_ALL = TE_MAIN + TE_CONTAINS

    def te_stats(df):
        mv = df["Vertical"].mean(); sv = max(df["Vertical"].std(), 1e-8)
        return dict(mean_v=mv, std_v=sv)

    rows, lines = run_skill("Tight End","TE","te_training.csv","te_testing.csv",
                            TE_MAIN,TE_ALL,eng_skill_vert,te_stats,
                            ["yprr","yards_per_reception","caught_percent","avg_depth_of_target",
                             "targeted_qb_rating","contested_catch_rate","drop_rate",
                             "yards_after_catch_per_reception","avoided_tackles",
                             "slot_rate","wide_rate","inline_rate","route_rate",
                             "grades_run_block","player_game_count"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ── RB ────────────────────────────────────────────────────────────────────
    RB_MAIN = ["Height","Weight","40yd","Vertical","Bench","Broad Jump",
               "RAS","arm_length_inches",
               "yards_after_contact","yco_attempt","ypa",
               "elusive_rating","mtf_per_attempt","breakaway_percent",
               "explosive_rate","yprr","targets_per_route","fumble_rate"]
    RB_CONTAINS = [f"contains_{f}" for f in RB_MAIN]
    RB_ALL = RB_MAIN + RB_CONTAINS

    def rb_stats(df): return {}

    rows, lines = run_skill("Running Back","RB","rb_training.csv","rb_testing.csv",
                            RB_MAIN,RB_ALL,eng_rb,rb_stats,
                            ["yards_after_contact","yco_attempt","ypa","elusive_rating",
                             "mtf_per_attempt","breakaway_percent","explosive_rate",
                             "yprr","targets_per_route","fumble_rate"])
    for yr in rows: all_rows[yr].extend(rows[yr])
    perf_all.extend(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Build and save CSVs
    # ─────────────────────────────────────────────────────────────────────────
    id_cols_26 = ["Player","School","Pos","Year","predicted_round"]
    id_cols_hist = ["Player","School","Pos","Year","actual_round","actual_pick","predicted_round"]

    def build_df(rows_list, id_cols):
        df = pd.DataFrame(rows_list)
        df = df.sort_values("predicted_round").reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df)+1))
        out_cols = ["rank"] + id_cols + ATHLETIC_COLS + ALL_PFF_COLS
        for c in out_cols:
            if c not in df.columns: df[c] = np.nan
        return df[out_cols]

    df26 = build_df(all_rows[2026], id_cols_26)
    df24 = build_df(all_rows[2024], id_cols_hist)
    df25 = build_df(all_rows[2025], id_cols_hist)

    out26 = os.path.join(DATA, "big_board_2026_detailed.csv")
    out24 = os.path.join(DATA, "big_board_2024_detailed.csv")
    out25 = os.path.join(DATA, "big_board_2025_detailed.csv")
    df26.to_csv(out26, index=False)
    df24.to_csv(out24, index=False)
    df25.to_csv(out25, index=False)
    print(f"\n{'='*60}")
    print(f"Saved {len(df26)} players → {out26}")
    print(f"Saved {len(df24)} players → {out24}")
    print(f"Saved {len(df25)} players → {out25}")

    # ─────────────────────────────────────────────────────────────────────────
    # Write model_performance.txt
    # ─────────────────────────────────────────────────────────────────────────
    perf_path = os.path.join(ROOT, "model_performance.txt")
    header = [
        "NFL Draft Round Regression – Model Performance",
        "=" * 60,
        "Ridge Regression (alpha=1.0) | KNNImputer (k=10) | StandardScaler",
        "Target: round 1–7 if drafted, 8 if undrafted",
        "Train: 2015–2023 | Holdout: 2024 and 2025 drafted players",
        "",
    ]
    with open(perf_path, "w") as f:
        f.write("\n".join(header + perf_all) + "\n")
    print(f"Saved model performance → {perf_path}")


if __name__ == "__main__":
    main()
