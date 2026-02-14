"""
Cornerback (CB) data cleaning: combine CBs + PFF Pass_Rush, Run_Defense, Pass_Coverage.
- Training 2015-2023 from nfl_combine_2010_to_2023.csv (Pos == 'CB').
- 2024 from data/raw/2024 Draft - Public - CB.csv if present (combine + pick/round).
- 2025 from data/raw/GabrielGTB 2025 NFL Combine - Master List.csv (Position == 'CB') + 2025_draft_picks.csv for Round/Pick.
- PFF: match by Player + School + Year; prefer CB/DB when dedup.
- RAS for CB. Optional arm length (mockdraftable_cb_arm_length.csv).
- Output: cb_training.csv (2015-2023), cb_testing.csv (2024-2025), CB/cb_drafted_2026.csv.
Run from CB/ directory.
"""
import os
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Load combine, CB only (2015-2023)
nfl_combine_data = pd.read_csv(os.path.join(DATA_RAW, 'nfl_combine_2010_to_2023.csv'))
CB_POSITIONS = ['CB']
nfl_combine_data_cb = nfl_combine_data[nfl_combine_data['Pos'].isin(CB_POSITIONS)].copy()
print(f"CB combine rows: {len(nfl_combine_data_cb)} (Pos in {CB_POSITIONS})")

# PFF: prefer CB then DB when same player/school/year
POSITION_PRIORITY = {'CB': 0, 'DB': 1}

PFF_PASS_RUSH_DIR = os.path.join(DATA_RAW, 'pff', 'Pass_Rush')
PFF_RUN_DEFENSE_DIR = os.path.join(DATA_RAW, 'pff', 'Run_Defense')
PFF_PASS_COVERAGE_DIR = os.path.join(DATA_RAW, 'pff', 'Pass_Coverage')

# --- Pass Rush ---
pff_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_PASS_RUSH_DIR, f'{year}_pass_rush_summary.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        cols = ['player', 'team_name', 'position', 'true_pass_set_pass_rush_win_rate',
                'pass_rush_win_rate', 'snap_counts_pass_rush']
        sub = df[[c for c in cols if c in df.columns]].copy()
        if 'player' not in sub.columns or 'true_pass_set_pass_rush_win_rate' not in sub.columns:
            continue
        if 'position' not in sub.columns:
            sub['position'] = None
        sub['Year'] = year
        sub = sub.rename(columns={'player': 'Player', 'team_name': 'School'})
        pff_files.append(sub)
        print(f'Loaded PFF pass rush {year}: {len(sub)} players')

if not pff_files:
    pass_rush_data = pd.DataFrame(columns=['Player', 'School', 'Year', 'true_pass_set_pass_rush_win_rate',
                                          'pass_rush_win_rate', 'snap_counts_pass_rush'])
    print('No PFF pass rush files found.')
else:
    pass_rush_data = pd.concat(pff_files, ignore_index=True)
    pass_rush_data['_pos_order'] = pass_rush_data['position'].map(POSITION_PRIORITY).fillna(99)
    pass_rush_data = pass_rush_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    pass_rush_data = pass_rush_data.drop(columns=['_pos_order', 'position'], errors='ignore')
    print(f'PFF pass rush records (after dedup): {len(pass_rush_data)}')

# --- Run Defense ---
run_defense_cols = ['player', 'team_name', 'position', 'stop_percent', 'missed_tackle_rate',
                    'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles']
run_defense_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_RUN_DEFENSE_DIR, f'{year}_run_defense_summary.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        sub = df[[c for c in run_defense_cols if c in df.columns]].copy()
        if 'stop_percent' not in sub.columns:
            continue
        if 'position' not in sub.columns:
            sub['position'] = None
        sub['Year'] = year
        sub = sub.rename(columns={'player': 'Player', 'team_name': 'School'})
        for c in ['stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles']:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
        run_defense_files.append(sub)
        print(f'Loaded PFF run defense {year}: {len(sub)} players')

if run_defense_files:
    run_defense_data = pd.concat(run_defense_files, ignore_index=True)
    run_defense_data['_pos_order'] = run_defense_data['position'].map(POSITION_PRIORITY).fillna(99)
    run_defense_data = run_defense_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    run_defense_data = run_defense_data.drop(columns=['_pos_order', 'position'], errors='ignore')
else:
    run_defense_data = None

# --- Pass Coverage ---
coverage_cols = ['player', 'team_name', 'position', 'yards_per_coverage_snap', 'forced_incompletion_rate',
                 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target',
                 'qb_rating_against', 'catch_rate', 'avg_depth_of_target']
coverage_numeric = ['yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent',
                    'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'qb_rating_against', 'catch_rate', 'avg_depth_of_target']
coverage_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_PASS_COVERAGE_DIR, f'{year}_defense_coverage_summary.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        sub = df[[c for c in coverage_cols if c in df.columns]].copy()
        if 'snap_counts_coverage' not in sub.columns:
            continue
        if 'position' not in sub.columns:
            sub['position'] = None
        sub['Year'] = year
        sub = sub.rename(columns={'player': 'Player', 'team_name': 'School'})
        for c in coverage_numeric:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
        coverage_files.append(sub)
        print(f'Loaded PFF pass coverage {year}: {len(sub)} players')

if coverage_files:
    coverage_data = pd.concat(coverage_files, ignore_index=True)
    coverage_data['_pos_order'] = coverage_data['position'].map(POSITION_PRIORITY).fillna(99)
    coverage_data = coverage_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    coverage_data = coverage_data.drop(columns=['_pos_order', 'position'], errors='ignore')
else:
    coverage_data = None

# Build pff_data from union of (Player, School, Year) so coverage-only / run-only players get rows
all_keys = pass_rush_data[['Player', 'School', 'Year']].drop_duplicates()
if run_defense_data is not None:
    all_keys = pd.concat([all_keys, run_defense_data[['Player', 'School', 'Year']]], ignore_index=True).drop_duplicates()
if coverage_data is not None:
    all_keys = pd.concat([all_keys, coverage_data[['Player', 'School', 'Year']]], ignore_index=True).drop_duplicates()
pff_data = all_keys.merge(pass_rush_data, on=['Player', 'School', 'Year'], how='left')
if run_defense_data is not None:
    merge_cols = ['Player', 'School', 'Year'] + [c for c in ['stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles'] if c in run_defense_data.columns]
    pff_data = pff_data.merge(run_defense_data[merge_cols], on=['Player', 'School', 'Year'], how='left')
else:
    for c in ['stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles']:
        pff_data[c] = None
    print('No run defense files found.')
if coverage_data is not None:
    merge_cov = ['Player', 'School', 'Year'] + [c for c in ['yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'qb_rating_against', 'catch_rate', 'avg_depth_of_target'] if c in coverage_data.columns]
    pff_data = pff_data.merge(coverage_data[merge_cov], on=['Player', 'School', 'Year'], how='left')
    print(f'Merged pass coverage; columns: {list(pff_data.columns)}')
else:
    for c in ['yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'qb_rating_against', 'catch_rate', 'avg_depth_of_target']:
        pff_data[c] = None
    print('No pass coverage files found.')
print(f'PFF records (union of pass rush/run D/coverage): {len(pff_data)}')

# RAS for CB (ras.football uses CB, DB)
ras_df = pd.read_csv(os.path.join(DATA_RAW, 'ras.csv'))
RAS_CB_POSITIONS = ['CB', 'DB']
ras_df = ras_df[ras_df['Pos'].isin(RAS_CB_POSITIONS)].copy()
ras_df['RAS'] = pd.to_numeric(ras_df['RAS'], errors='coerce')
ras_df['Year'] = ras_df['Year'].astype(int)
ras_cb = ras_df[['Name', 'Year', 'RAS', 'College']].drop_duplicates(subset=['Name', 'Year'])
print(f'RAS CB records: {len(ras_cb)}')

# Arm length (optional; from mockdraftable_cb_arm_length.csv)
arm_path = os.path.join(DATA_RAW, 'mockdraftable_cb_arm_length.csv')
if os.path.exists(arm_path):
    arm_length_df = pd.read_csv(arm_path)
    arm_length_df['Year'] = arm_length_df['Year'].astype(int)
    arm_length_df = arm_length_df.drop_duplicates(subset=['Player', 'Year'], keep='first')
    arm_length_df = arm_length_df[['Player', 'Year', 'arm_length_inches']].copy()
    arm_length_df['arm_length_inches'] = pd.to_numeric(arm_length_df['arm_length_inches'], errors='coerce')
    print(f'Arm length CB: {len(arm_length_df)} records ({arm_length_df["arm_length_inches"].notna().sum()} with values)')
else:
    arm_length_df = pd.DataFrame(columns=['Player', 'Year', 'arm_length_inches'])
    print('No mockdraftable_cb_arm_length.csv; arm_length_inches will be empty.')

# --- Helpers for 2024/2025 draft CSVs ---
def _ht_2024_to_inches(ht):
    """2024 format 5104 = 5'10.5\" -> 70.5"""
    if pd.isna(ht) or str(ht).strip() == '':
        return np.nan
    s = str(int(float(ht))).zfill(4)
    if len(s) < 4:
        return np.nan
    ft = int(s[0])
    inch = int(s[1:3])
    eighth = int(s[3]) if len(s) > 3 else 0
    return ft * 12 + inch + eighth / 8.0

def _broad_2024_to_inches(broad):
    """2024 format 1002 = 10'02\" -> 122"""
    if pd.isna(broad) or str(broad).strip() in ('', '--'):
        return np.nan
    try:
        s = str(int(float(broad)))
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 3:
        return np.nan
    ft = int(s[:-2])
    inch = int(s[-2:])
    return ft * 12 + inch

def _arm_2024_to_inches(arm):
    """2024 format 3068 = 30.68 inches"""
    if pd.isna(arm) or str(arm).strip() == '':
        return np.nan
    s = str(int(float(arm)))
    if len(s) < 4:
        return np.nan
    return int(s[:2]) + int(s[2:]) / 100.0

def _pick_to_round(pick_taken):
    """Derive draft round from pick number; UDFA -> 8."""
    if pd.isna(pick_taken) or str(pick_taken).strip().upper() == 'UDFA':
        return 8
    try:
        p = int(float(str(pick_taken).replace(',', '')))
        if 1 <= p <= 32:
            return 1
        if p <= 64:
            return 2
        if p <= 96:
            return 3
        if p <= 128:
            return 4
        if p <= 160:
            return 5
        if p <= 192:
            return 6
        if p <= 257:
            return 7
    except (ValueError, TypeError):
        pass
    return 8

# --- 2024: from 2024 Draft - Public - CB.csv (if present) ---
draft_2024_path = os.path.join(DATA_RAW, '2024 Draft - Public - CB.csv')
cb_2024_list = []
if os.path.exists(draft_2024_path):
    d24 = pd.read_csv(draft_2024_path)
    d24 = d24[d24['Name'].notna() & (d24['Name'].astype(str).str.strip() != '')].copy()
    for _, row in d24.iterrows():
        # 2024 CB CSV uses 'ov' (overall pick); S uses 'Pick Taken'
        pick_taken = row.get('Pick Taken', row.get('ov', row.iloc[0] if len(row) > 0 else None))
        try:
            pick_int = int(float(str(pick_taken).replace(',', ''))) if str(pick_taken).strip() not in ('', 'UDFA') else np.nan
        except (ValueError, TypeError):
            pick_int = np.nan
        round_num = _pick_to_round(pick_taken)
        ht = _ht_2024_to_inches(row.get('HT'))
        # CB file has Arm as decimal inches (31.625); S file uses 3068 format
        arm_val = row.get('Arm')
        arm = pd.to_numeric(arm_val, errors='coerce')
        if pd.isna(arm) or arm < 20 or arm > 45:
            arm = _arm_2024_to_inches(arm_val)
        broad = _broad_2024_to_inches(row.get('Broad', row.get('BJ')))
        cb_2024_list.append({
            'Year': 2024,
            'Player': row['Name'],
            'Pos': 'CB',
            'School': row['School'],
            'Height': ht,
            'Weight': pd.to_numeric(row.get('WT'), errors='coerce'),
            '40yd': pd.to_numeric(row.get('40'), errors='coerce'),
            'Vertical': pd.to_numeric(row.get('Vert', row.get('VJ')), errors='coerce'),
            'Bench': pd.to_numeric(row.get('Bench', row.get('BP')), errors='coerce'),
            'Broad Jump': broad,
            '3Cone': pd.to_numeric(row.get('3Cone', row.get('3C')), errors='coerce'),
            'Shuttle': pd.to_numeric(row.get('SS'), errors='coerce'),
            'Drafted': True,
            'Round': round_num,
            'Pick': pick_int,
            'RAS': pd.to_numeric(row.get('RAS'), errors='coerce'),
            'arm_length_inches': arm,
        })
    cb_2024 = pd.DataFrame(cb_2024_list)
    print(f'Loaded 2024 corners: {len(cb_2024)} from 2024 Draft - Public - CB.csv')
else:
    cb_2024 = pd.DataFrame(columns=['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS'])
    print('2024 Draft - Public - CB.csv not found.')

# --- 2025: from GabrielGTB 2025 NFL Combine - Master List.csv (Position == 'CB') ---
combine_2025_path = os.path.join(DATA_RAW, 'GabrielGTB 2025 NFL Combine - Master List.csv')
def _ht_2025_to_inches(ht):
    """2025 FIIE 6031 = 6 ft, 03 in, 1 eighth -> 75.125"""
    if pd.isna(ht) or str(ht).strip() == '':
        return np.nan
    s = str(int(float(ht))).zfill(4)
    if len(s) < 4:
        return np.nan
    ft = int(s[0])
    inch = int(s[1:3])
    eighth = int(s[3]) if len(s) > 3 else 0
    return ft * 12 + inch + eighth / 8.0

def _broad_2025_to_inches(broad):
    """2025 FFII 1106 = 11'06\" -> 138"""
    if pd.isna(broad) or str(broad).strip() == '':
        return np.nan
    s = str(int(float(broad)))
    if len(s) < 3:
        return np.nan
    ft = int(s[:-2])
    inch = int(s[-2:])
    return ft * 12 + inch

cb_2025_list = []
if os.path.exists(combine_2025_path):
    c25 = pd.read_csv(combine_2025_path)
    c25_cb = c25[c25['Position'].astype(str).str.strip().str.upper() == 'CB'].copy()
    for _, row in c25_cb.iterrows():
        _school = str(row['School']).replace('Syracruse', 'Syracuse').strip() if pd.notna(row.get('School')) else row.get('School')
        cb_2025_list.append({
            'Year': 2025,
            'Player': row['Name'],
            'Pos': 'CB',
            'School': _school,
            'Height': _ht_2025_to_inches(row.get('Height (FIIE)')),
            'Weight': pd.to_numeric(row.get('Weight (lbs.)'), errors='coerce'),
            '40yd': pd.to_numeric(row.get('40-yard Dash (seconds)'), errors='coerce'),
            'Vertical': pd.to_numeric(row.get('Vertical Jump (inches)'), errors='coerce'),
            'Bench': pd.to_numeric(row.get('Bench Press (reps)'), errors='coerce'),
            'Broad Jump': _broad_2025_to_inches(row.get('Broad Jump (FFII)')),
            '3Cone': pd.to_numeric(row.get('Three-cone Drill (seconds)'), errors='coerce'),
            'Shuttle': pd.to_numeric(row.get('20-yard Shuttle (seconds)'), errors='coerce'),
            'Drafted': True,
            'Round': np.nan,
            'Pick': np.nan,
            'RAS': pd.to_numeric(row.get('RAS'), errors='coerce'),
            'arm_length_inches': pd.to_numeric(row.get('Arm Length (inches)'), errors='coerce'),
        })
    cb_2025_from_combine = pd.DataFrame(cb_2025_list)
    # 2025 Round/Pick: prefer data/raw/2025_draft_picks.csv (PFR), then cb_drafted_2025.csv
    draft_picks_2025_path = os.path.join(DATA_RAW, '2025_draft_picks.csv')
    cb_drafted_2025_path = os.path.join(SCRIPT_DIR, 'cb_drafted_2025.csv')

    def _norm_name(name):
        if pd.isna(name):
            return ''
        return re.sub(r'\s+Jr\.?$|\s+III$|\s+II$|\s+IV$', '', str(name).strip(), flags=re.IGNORECASE).strip()

    def _norm_school(s):
        if pd.isna(s) or str(s).strip() == '':
            return ''
        x = str(s).strip()
        aliases = {
            'Penn St.': 'Penn State', 'Ohio St.': 'Ohio State', 'Kansas St.': 'Kansas State',
            'Boston Col.': 'Boston College', 'North Carolina St.': 'North Carolina State',
            'Florida St.': 'Florida State', 'Washington St.': 'Washington State',
            'Iowa St.': 'Iowa State', 'Ole Miss': 'Mississippi', 'Syracruse': 'Syracuse',
        }
        return aliases.get(x, x)

    round_pick_2025 = None
    if os.path.exists(draft_picks_2025_path):
        draft_all = pd.read_csv(draft_picks_2025_path)
        draft_cbs = draft_all[draft_all['Pos'].astype(str).str.upper().isin(['CB', 'DB'])].copy()
        if not draft_cbs.empty:
            draft_cbs = draft_cbs.rename(columns={'Rnd': 'Round'})
            draft_cbs['Player_norm'] = draft_cbs['Player'].astype(str).map(_norm_name)
            draft_cbs['School_norm'] = draft_cbs['School'].astype(str).map(_norm_school)
            round_pick_2025 = draft_cbs[['Player_norm', 'School_norm', 'Round', 'Pick']].drop_duplicates()
            print(f'Loaded 2025 draft CBs: {len(round_pick_2025)} from {draft_picks_2025_path}')

    cb_2025_from_combine = cb_2025_from_combine.drop(columns=['Round', 'Pick'], errors='ignore')
    cb_2025_from_combine['Player_norm'] = cb_2025_from_combine['Player'].astype(str).map(_norm_name)
    cb_2025_from_combine['School_norm'] = cb_2025_from_combine['School'].astype(str).map(_norm_school)

    if round_pick_2025 is not None and not round_pick_2025.empty:
        cb_2025_from_combine = cb_2025_from_combine.merge(
            round_pick_2025,
            on=['Player_norm', 'School_norm'],
            how='left',
            suffixes=('', '_draft')
        )
        cb_2025_from_combine = cb_2025_from_combine.drop(columns=['Player_norm', 'School_norm'], errors='ignore')
        # If any Round/Pick still missing, try cb_drafted_2025.csv
        missing = cb_2025_from_combine['Round'].isna()
        if missing.any() and os.path.exists(cb_drafted_2025_path):
            draft25 = pd.read_csv(cb_drafted_2025_path)
            if 'Player' in draft25.columns and 'Round' in draft25.columns:
                fill = draft25[['Player', 'Round', 'Pick']].drop_duplicates()
                fill = fill.rename(columns={'Round': 'Round_fill', 'Pick': 'Pick_fill'})
                cb_2025_from_combine = cb_2025_from_combine.merge(
                    fill,
                    left_on='Player',
                    right_on='Player',
                    how='left'
                )
                cb_2025_from_combine['Round'] = cb_2025_from_combine['Round'].fillna(cb_2025_from_combine['Round_fill'])
                cb_2025_from_combine['Pick'] = cb_2025_from_combine['Pick'].fillna(cb_2025_from_combine['Pick_fill'])
                cb_2025_from_combine = cb_2025_from_combine.drop(columns=['Round_fill', 'Pick_fill'], errors='ignore')
    else:
        cb_2025_from_combine = cb_2025_from_combine.drop(columns=['Player_norm', 'School_norm'], errors='ignore')
        if os.path.exists(cb_drafted_2025_path):
            draft25 = pd.read_csv(cb_drafted_2025_path)
            if 'Player' in draft25.columns and 'Round' in draft25.columns:
                round_pick = draft25[['Player', 'Year', 'Round', 'Pick']].drop_duplicates()
                cb_2025_from_combine['Year'] = 2025
                cb_2025_from_combine = cb_2025_from_combine.merge(
                    round_pick,
                    on=['Player', 'Year'],
                    how='left'
                )

    cb_2025 = cb_2025_from_combine
    if not os.path.exists(cb_drafted_2025_path) or cb_2025['Round'].notna().any():
        out_cols = [c for c in ['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Round', 'Pick', 'RAS', 'arm_length_inches'] if c in cb_2025.columns]
        cb_2025[out_cols].to_csv(cb_drafted_2025_path, index=False)
    print(f'Loaded 2025 corners: {len(cb_2025)} from GabrielGTB 2025 combine')
else:
    cb_2025 = pd.DataFrame(columns=['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS'])
    print('GabrielGTB 2025 NFL Combine - Master List.csv not found.')

# Splits: training 2015-2023, testing 2024-2025
cb_training_data = nfl_combine_data_cb[nfl_combine_data_cb['Year'].between(2015, 2023)].copy()

if cb_2024.empty and cb_2025.empty:
    cb_testing_data = pd.DataFrame()
elif cb_2024.empty:
    cb_testing_data = cb_2025.copy()
elif cb_2025.empty:
    cb_testing_data = cb_2024.copy()
else:
    cb_testing_data = pd.concat([cb_2024, cb_2025], ignore_index=True)
if not cb_testing_data.empty:
    cb_testing_data['Year'] = cb_testing_data['Year'].astype(int)
    cb_testing_data['Drafted'] = True
    cb_testing_data = cb_testing_data[cb_testing_data['Round'].notna() & (cb_testing_data['Round'] != 8)].copy()
if 'arm_length_inches' not in cb_2024.columns and not cb_2024.empty:
    cb_2024['arm_length_inches'] = np.nan
if 'arm_length_inches' not in cb_testing_data.columns and not cb_testing_data.empty:
    cb_testing_data['arm_length_inches'] = np.nan

cols_drop = ['speed_score', 'explosive_score', 'agility_score']
cb_testing_data = cb_testing_data.drop(columns=cols_drop, errors='ignore')

# 2026: load from cb_drafted_2026.csv (prospects from nfldraftbuzz; PFF and RAS merged below)
cb_drafted_2026_path = os.path.join(SCRIPT_DIR, 'cb_drafted_2026.csv')
if os.path.exists(cb_drafted_2026_path):
    cb_2026_raw = pd.read_csv(cb_drafted_2026_path)
    if 'Player' in cb_2026_raw.columns and 'School' in cb_2026_raw.columns:
        cb_2026 = cb_2026_raw.copy()
        cb_2026['Year'] = 2026
        cb_2026['Pos'] = 'CB'
        cb_2026['Drafted'] = True
        cb_2026['Round'] = np.nan
        cb_2026['Pick'] = np.nan
        cb_2026['RAS'] = np.nan
        for col in ['Height', 'Weight', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'arm_length_inches']:
            if col not in cb_2026.columns:
                cb_2026[col] = np.nan
        cb_2026['40yd'] = pd.to_numeric(cb_2026.get('40yd', cb_2026.get('40', np.nan)), errors='coerce')
        cb_2026_processed = cb_2026.drop(columns=cols_drop, errors='ignore')
        print(f'Loaded 2026 corners: {len(cb_2026)} from cb_drafted_2026.csv (PFF/RAS will be merged)')
    else:
        cb_2026 = pd.DataFrame(columns=['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS'])
        cb_2026['Year'] = 2026
        cb_2026['Drafted'] = True
        cb_2026_processed = cb_2026.drop(columns=cols_drop, errors='ignore')
        print('cb_drafted_2026.csv missing Player/School; using empty 2026.')
else:
    cb_2026 = pd.DataFrame(columns=['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS'])
    cb_2026['Year'] = 2026
    cb_2026['Drafted'] = True
    cb_2026_processed = cb_2026.drop(columns=cols_drop, errors='ignore')
    print('No cb_drafted_2026.csv; using empty 2026.')


def normalize_player_name(name):
    s = str(name).strip().upper()
    s = re.sub(r'\s+(IV|III|II|JR|SR|JR\.|SR\.)$', '', s)
    s = re.sub(r'[.\',\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_pff_school(name):
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    mapping = {
        'OHIO STATE': 'Ohio State', 'OHIO ST': 'Ohio State',
        'FLORIDA ST': 'Florida State', 'FLORIDA STATE': 'Florida State',
        'KANSAS ST': 'Kansas State', 'KANSAS STATE': 'Kansas State',
        'IOWA ST': 'Iowa State', 'IOWA STATE': 'Iowa State',
        'OKLAHOMA ST': 'Oklahoma State', 'OKLA STATE': 'Oklahoma State', 'OKLAHOMA STATE': 'Oklahoma State', 'OKLAHOMA': 'Oklahoma',
        'PENN ST': 'Penn State', 'PENN STATE': 'Penn State',
        'MISSISSIPPI ST': 'Mississippi State', 'MISS STATE': 'Mississippi State', 'MISSISSIPPI STATE': 'Mississippi State',
        'MICHIGAN ST': 'Michigan State', 'MICHIGAN STATE': 'Michigan State', 'MICH STATE': 'Michigan State',
        'NORTH CAROLINA ST': 'North Carolina State', 'NC STATE': 'North Carolina State', 'NORTH CAROLINA': 'North Carolina',
        'S CAROLINA': 'South Carolina', 'SOUTH CAROLINA': 'South Carolina',
        'SOUTHERN CAL': 'USC', 'SOUTHERN CALIFORNIA': 'USC', 'USC': 'USC', 'UCLA': 'UCLA',
        'CENTRAL FLORIDA': 'UCF', 'UCF': 'UCF', 'TCU': 'TCU', 'LSU': 'LSU',
        'BRIGHAM YOUNG': 'BYU', 'MIAMI (FL)': 'Miami', 'MIAMI FL': 'Miami', 'MIAMI': 'Miami', 'MIAMI OH': 'Miami (OH)',
        'OLE MISS': 'Mississippi', 'ALABAMA-BIRMINGHAM': 'UAB', 'UAB': 'UAB', 'TENN-CHATTANOOGA': 'Chattanooga',
        'WASHINGTON STATE': 'Washington State', 'WSU': 'Washington State', 'WASH STATE': 'Washington State',
        'WAKE': 'Wake Forest', 'CAL': 'California',
        'NOTRE DAME': 'Notre Dame', 'OREGON': 'Oregon', 'OREGON ST': 'Oregon State', 'MISSOURI': 'Missouri', 'SMU': 'SMU',
        'GEORGIA': 'Georgia', 'SYRACUSE': 'Syracuse',
        'VA TECH': 'Virginia Tech', 'VIRGINIA TECH': 'Virginia Tech',
        'GA TECH': 'Georgia Tech', 'GEORGIA TECH': 'Georgia Tech',
        'ARIZONA ST': 'Arizona State', 'ARIZONA STATE': 'Arizona State',
        'BOSTON COL': 'Boston College', 'BOSTON COLLEGE': 'Boston College',
        'BOISE ST': 'Boise State', 'BOISE STATE': 'Boise State',
        'FRESNO ST': 'Fresno State', 'FRESNO STATE': 'Fresno State',
        'LA TECH': 'Louisiana Tech', 'LOUISIANA TECH': 'Louisiana Tech',
        'LA LAFAYET': 'Louisiana', 'LOUISIANA': 'Louisiana',
        'NWESTERN': 'Northwestern', 'NORTHWESTERN': 'Northwestern',
        'W VIRGINIA': 'West Virginia', 'WEST VIRGINIA': 'West Virginia',
        'ARK STATE': 'Arkansas State', 'ARKANSAS STATE': 'Arkansas State', 'ARKANSAS ST': 'Arkansas State',
        'C MICHIGAN': 'Central Michigan', 'CENTRAL MICHIGAN': 'Central Michigan',
        'CINCINNATI': 'Cincinnati', 'DELAWARE': 'Delaware', 'FLORIDA': 'Florida',
        'S DIEGO ST': 'San Diego State', 'SAN DIEGO STATE': 'San Diego State',
        'UCONN': 'Connecticut', 'CONNECTICUT': 'Connecticut',
        'NEBRASKA': 'Nebraska', 'AUBURN': 'Auburn',
        'W KENTUCKY': 'Western Kentucky', 'WESTERN KENTUCKY': 'Western Kentucky',
        'W MICHIGAN': 'Western Michigan', 'WESTERN MICHIGAN': 'Western Michigan',
        'E CAROLINA': 'East Carolina', 'EAST CAROLINA': 'East Carolina',
        'N CAROLINA': 'North Carolina',
        'COLORADO ST': 'Colorado State', 'COLORADO STATE': 'Colorado State', 'COLO STATE': 'Colorado State',
        'SAN JOSE ST': 'San Jose State', 'SAN JOSE STATE': 'San Jose State',
        'GA STATE': 'Georgia State', 'GEORGIA STATE': 'Georgia State',
        'FLA ATLANTIC': 'Florida Atlantic', 'FLORIDA ATLANTIC': 'Florida Atlantic', 'FAU': 'Florida Atlantic',
        'FIU': 'Florida International', 'FLORIDA INTERNATIONAL': 'Florida International',
        'GA SOUTHERN': 'Georgia Southern', 'GEORGIA SOUTHERN': 'Georgia Southern',
        'LA MONROE': 'Louisiana-Monroe', 'ULM': 'Louisiana-Monroe',
        'BALL ST': 'Ball State', 'BALL STATE': 'Ball State',
        'S ALABAMA': 'South Alabama', 'SOUTH ALABAMA': 'South Alabama',
        'STANFORD': 'Stanford', 'WASHINGTON': 'Washington', 'MARYLAND': 'Maryland',
        'LOUISIANA ST': 'LSU', 'LOUISIANA STATE': 'LSU',
        'VA TECH': 'Virginia Tech', 'VIRGINIA TECH': 'Virginia Tech',
        'CAL': 'California', 'CALIFORNIA': 'California',
        'SO MISS': 'Southern Mississippi', 'SOUTHERN MISS': 'Southern Mississippi', 'SOUTHERN MISSISSIPPI': 'Southern Mississippi',
        'INDIANA': 'Indiana', 'TENNESSEE': 'Tennessee', 'ALABAMA': 'Alabama', 'ARKANSAS': 'Arkansas',
    }
    return mapping.get(name, name.title())


def normalize_combine_school(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    alias = {
        'Ole Miss': 'Mississippi', 'Miami (FL)': 'Miami', 'Miami': 'Miami',
        'Southern California': 'USC', 'USC': 'USC', 'UCLA': 'UCLA',
        'Central Florida': 'UCF', 'UCF': 'UCF', 'Brigham Young': 'BYU',
        'Ohio St.': 'Ohio State', 'Ohio State': 'Ohio State',
        'Florida St.': 'Florida State', 'Florida State': 'Florida State',
        'Kansas St.': 'Kansas State', 'Kansas State': 'Kansas State',
        'Iowa St.': 'Iowa State', 'Iowa State': 'Iowa State',
        'Oklahoma St.': 'Oklahoma State', 'Oklahoma State': 'Oklahoma State', 'Oklahoma': 'Oklahoma',
        'Penn St.': 'Penn State', 'Penn State': 'Penn State',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State', 'North Carolina St.': 'North Carolina State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
        'West Virginia': 'West Virginia', 'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
        'Washington St.': 'Washington State', 'Washington State': 'Washington State',
        'Montana St.': 'Montana State', 'Montana State': 'Montana State',
        'Cal': 'California', 'California': 'California', 'Syracruse': 'Syracuse', 'Syracuse': 'Syracuse',
        'Boston Col.': 'Boston College', 'Boston College': 'Boston College',
        'Boise St.': 'Boise State', 'Boise State': 'Boise State',
        'Arkansas St.': 'Arkansas State', 'Arkansas State': 'Arkansas State',
        'Louisiana-Lafayette': 'Louisiana', 'Louisiana': 'Louisiana',
        'Virginia Tech': 'Virginia Tech', 'Georgia Tech': 'Georgia Tech',
        'Arizona State': 'Arizona State', 'Fresno State': 'Fresno State',
        'Northwestern': 'Northwestern', 'Cincinnati': 'Cincinnati',
        'Central Florida': 'UCF', 'East. Washington': 'Eastern Washington',
        'William & Mary': 'William & Mary', 'James Madison': 'James Madison',
        'Southern Utah': 'Southern Utah', 'Jacksonville State': 'Jacksonville State',
        'San Diego State': 'San Diego State', 'Indiana (PA)': 'Indiana (PA)',
        'Connecticut': 'Connecticut', 'Nebraska': 'Nebraska', 'Auburn': 'Auburn',
        'Western Kentucky': 'Western Kentucky', 'Western Michigan': 'Western Michigan', 'East Carolina': 'East Carolina',
        'West. Michigan': 'Western Michigan',
        'Louisiana St': 'LSU', 'Louisiana St.': 'LSU',
        'Colorado State': 'Colorado State', 'San Jose State': 'San Jose State', 'North Carolina': 'North Carolina',
        'Georgia State': 'Georgia State', 'Central Michigan': 'Central Michigan',
        'Florida Atlantic': 'Florida Atlantic', 'Florida International': 'Florida International',
        'Georgia Southern': 'Georgia Southern',
        'San Diego St.': 'San Diego State',
        'La-Monroe': 'Louisiana-Monroe', 'Louisiana-Monroe': 'Louisiana-Monroe', 'ULM': 'Louisiana-Monroe',
        'Arizona St.': 'Arizona State', 'Kansas St.': 'Kansas State',
        'Ball St.': 'Ball State', 'Ball State': 'Ball State',
        'South Alabama': 'South Alabama', 'Stanford': 'Stanford', 'Washington': 'Washington', 'Maryland': 'Maryland',
        'Southern Mississippi': 'Southern Mississippi',
    }
    return alias.get(name, name)


def add_pff_data(combine_df, pff_df):
    combine_df = combine_df.copy()
    pff_n = pff_df.copy()
    pff_n['School_normalized'] = pff_n['School'].apply(normalize_pff_school)
    pff_n['Player_normalized'] = pff_n['Player'].apply(normalize_player_name)
    combine_df['School_normalized'] = combine_df['School'].apply(normalize_combine_school)
    pff_value_cols = [c for c in pff_df.columns if c not in ('Player', 'School', 'Year')]

    # Overrides for PFF matching (player_normalized, school_normalized) -> PFF school to use (2025/2026 transfers)
    player_nickname_map = {}
    player_school_pff_override = {
        # 2026 CB prospects: draft CSV school -> PFF 2025 school (transfers or PFF listing)
        ('MANSOOR DELANE', 'Virginia Tech'): 'LSU',
        ('DOMANI JACKSON', 'USC'): 'Alabama',
        ('DAVISON IGBINOSUN', 'Mississippi'): 'Ohio State',
        ('JOSH MOTEN', 'Texas A&M'): 'Southern Mississippi',
        ('THADDEUS DIXON', 'Washington'): 'North Carolina',
        ('TACARIO DAVIS', 'Arizona'): 'Washington',
        ('HEZEKIAH MASSES', 'Florida International'): 'California',
        ('DANGELO PONDS', 'James Madison'): 'Indiana',
        ('COLTON HOOD', 'Auburn'): 'Tennessee',
        ('BRANDON CISSE', 'North Carolina State'): 'South Carolina',
        ('JULIAN NEAL', 'Fresno State'): 'Arkansas',
        ('KEIONTE SCOTT', 'Auburn'): 'Miami',
        ('JERMOD MCCOY', 'Oregon State'): 'Tennessee',  # 2024 at Tennessee, then transferred back to OSU; injured 2025
    }
    # (player_normalized, school_to_use, draft_year) -> PFF year to use (e.g. injury/opt-out)
    player_school_year_override = {
        ('SHAVON REVEL JR', 'East Carolina', 2025): 2023,  # injured senior year; use season before
        ('JERMOD MCCOY', 'Tennessee', 2026): 2024,  # injured 2025 at OSU; use 2024 (Tennessee)
    }

    def lookup(row):
        draft_year = int(row['Year'])
        final_season = draft_year - 1
        player = normalize_player_name(row['Player'])
        school = row['School_normalized']
        player_to_search = player_nickname_map.get(player, player)
        school_to_use = player_school_pff_override.get((player_to_search, school), school)
        pff_year = player_school_year_override.get((player_to_search, school_to_use, draft_year), final_season)
        mask = (
            (pff_n['Player_normalized'] == player_to_search) &
            (pff_n['School_normalized'] == school_to_use) &
            (pff_n['Year'] == pff_year)
        )
        if not mask.any() and player_to_search != player:
            mask = (
                (pff_n['Player_normalized'] == player) &
                (pff_n['School_normalized'] == school_to_use) &
                (pff_n['Year'] == pff_year)
            )
        match = pff_n.loc[mask]
        if match.empty:
            return pd.Series({c: None for c in pff_value_cols})
        r = match.iloc[0]
        return pd.Series({c: r[c] for c in pff_value_cols})

    res = combine_df.apply(lookup, axis=1)
    for c in res.columns:
        combine_df[c] = res[c]
    return combine_df.drop(columns=['School_normalized'], errors='ignore')


def add_ras_data(combine_df, ras_subset):
    combine_df = combine_df.copy()
    ras_n = ras_subset.copy()
    ras_n['Year'] = ras_n['Year'].astype(int)
    ras_n['Name_n'] = ras_n['Name'].apply(normalize_player_name)
    ras_school = {
        'Miami (FL)': 'Miami', 'Miami': 'Miami', 'Southern California': 'USC', 'USC': 'USC', 'UCLA': 'UCLA',
        'Central Florida': 'UCF', 'UCF': 'UCF', 'Brigham Young': 'BYU', 'BYU': 'BYU',
        'Ole Miss': 'Mississippi', 'Mississippi': 'Mississippi', 'Ohio St.': 'Ohio State', 'Ohio State': 'Ohio State',
        'Florida St.': 'Florida State', 'Florida State': 'Florida State',
        'Oklahoma St.': 'Oklahoma State', 'Oklahoma State': 'Oklahoma State', 'Oklahoma': 'Oklahoma',
        'Penn St.': 'Penn State', 'Penn State': 'Penn State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
        'Virginia Tech': 'Virginia Tech', 'Texas State': 'Texas State', 'Louisiana Tech': 'Louisiana Tech',
        'TCU': 'TCU', 'Texas Christian': 'TCU', 'Louisiana State': 'LSU', 'LSU': 'LSU',
        'Boston Col.': 'Boston College', 'Boston College': 'Boston College',
        'Washington State': 'Washington State', 'Washington St.': 'Washington State',
        'North Carolina St.': 'North Carolina State', 'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
        'Texas AM': 'Texas A&M',
    }
    ras_n['College_n'] = ras_n['College'].apply(lambda x: ras_school.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    ras_name_alias = {}

    def lookup_ras(row):
        player = normalize_player_name(row['Player'])
        player_ras = ras_name_alias.get(player, player)
        school = normalize_combine_school(row['School'])
        year = int(row['Year'])
        m = (ras_n['Name_n'] == player_ras) & (ras_n['College_n'] == school) & (ras_n['Year'] == year)
        if not m.any() and player_ras != player:
            m = (ras_n['Name_n'] == player) & (ras_n['College_n'] == school) & (ras_n['Year'] == year)
        hit = ras_n.loc[m]
        if hit.empty:
            return pd.Series({'RAS': None})
        return pd.Series({'RAS': hit.iloc[0]['RAS']})

    ras_cols = combine_df.apply(lookup_ras, axis=1)
    combine_df['RAS'] = ras_cols['RAS']
    return combine_df


def add_arm_length(combine_df, arm_df):
    """
    Add arm_length_inches by left merge on Player + Year (same as LB).
    """
    combine_df = combine_df.copy()
    combine_df = combine_df.drop(columns=['arm_length_inches'], errors='ignore')
    if arm_df.empty or 'arm_length_inches' not in arm_df.columns:
        combine_df['arm_length_inches'] = np.nan
        return combine_df
    combine_df = combine_df.merge(
        arm_df[['Player', 'Year', 'arm_length_inches']],
        on=['Player', 'Year'],
        how='left'
    )
    return combine_df


# Apply PFF, then RAS
cb_training_data = add_pff_data(cb_training_data, pff_data)
cb_testing_data = add_pff_data(cb_testing_data, pff_data)
cb_2026_processed = add_pff_data(cb_2026_processed, pff_data)

for _df in (cb_training_data, cb_testing_data, cb_2026_processed):
    snap = _df.get('snap_counts_coverage')
    interceptions = _df.get('interceptions')
    pbu = _df.get('pass_break_ups')
    _df['INT_rate'] = np.nan
    _df['PBU_rate'] = np.nan
    if snap is not None and interceptions is not None:
        _df['INT_rate'] = np.where(pd.notna(snap) & (snap > 0), pd.to_numeric(interceptions, errors='coerce') / snap, np.nan)
    if snap is not None and pbu is not None:
        _df['PBU_rate'] = np.where(pd.notna(snap) & (snap > 0), pd.to_numeric(pbu, errors='coerce') / snap, np.nan)

cb_training_data = add_ras_data(cb_training_data, ras_cb)
cb_testing_data = add_ras_data(cb_testing_data, ras_cb)
cb_2026_processed = add_ras_data(cb_2026_processed, ras_cb)

# Preserve 2025 arm from combine list before MockDraftable merge
arm_backup_2025 = cb_testing_data.loc[cb_testing_data['Year'] == 2025, 'arm_length_inches'].copy() if not cb_testing_data.empty and 'arm_length_inches' in cb_testing_data.columns else pd.Series(dtype=float)

cb_training_data = add_arm_length(cb_training_data, arm_length_df)
cb_testing_data = add_arm_length(cb_testing_data, arm_length_df)
cb_2026_processed = add_arm_length(cb_2026_processed, arm_length_df)

if not arm_backup_2025.empty and (cb_testing_data['Year'] == 2025).any():
    idx_2025 = cb_testing_data['Year'] == 2025
    cb_testing_data.loc[idx_2025, 'arm_length_inches'] = cb_testing_data.loc[idx_2025, 'arm_length_inches'].fillna(arm_backup_2025)

cb_training_data = cb_training_data.drop(columns=cols_drop, errors='ignore')
cb_testing_data = cb_testing_data.drop(columns=cols_drop, errors='ignore')

training_cols_order = ['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical',
                       'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick',
                       'RAS', 'arm_length_inches',
                       'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate', 'snap_counts_pass_rush',
                       'stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles',
                       'yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent',
                       'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'INT_rate', 'PBU_rate',
                       'qb_rating_against', 'catch_rate', 'avg_depth_of_target']
cb_training_data = cb_training_data[[c for c in training_cols_order if c in cb_training_data.columns]]
cb_testing_data = cb_testing_data[[c for c in training_cols_order if c in cb_testing_data.columns]]

cb_training_data.to_csv(os.path.join(DATA_PROCESSED, 'cb_training.csv'), index=False)
cb_testing_data.to_csv(os.path.join(DATA_PROCESSED, 'cb_testing.csv'), index=False)

cb_2026_cols = ['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight',
                '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
                'RAS', 'arm_length_inches',
                'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate', 'snap_counts_pass_rush',
                'stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles',
                'yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent',
                'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'INT_rate', 'PBU_rate',
                'qb_rating_against', 'catch_rate', 'avg_depth_of_target']
cb_2026_final = cb_2026_processed[[c for c in cb_2026_cols if c in cb_2026_processed.columns]]
cb_2026_final.to_csv(os.path.join(SCRIPT_DIR, 'cb_drafted_2026.csv'), index=False)

arm_train = cb_training_data['arm_length_inches'].notna().sum()
arm_test = cb_testing_data['arm_length_inches'].notna().sum()
arm_2026 = cb_2026_final['arm_length_inches'].notna().sum()
print(f'\nSaved cb_training.csv: {len(cb_training_data)} (2015-2023)')
print(f'Saved cb_testing.csv: {len(cb_testing_data)} (2024-2025)')
print(f'Saved cb_drafted_2026.csv: {len(cb_2026_final)}')
print(f'Arm length coverage: Training {arm_train}/{len(cb_training_data)}, Testing {arm_test}/{len(cb_testing_data)}, 2026 {arm_2026}/{len(cb_2026_final)}')
print(f'Columns: {list(cb_training_data.columns)}')
