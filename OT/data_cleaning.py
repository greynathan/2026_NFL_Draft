"""
Offensive Tackle (OT) data cleaning: combine OTs + PFF Pass_Blocking, Run_Blocking.
- Training 2015-2023 from nfl_combine_2010_to_2023.csv (Pos == 'OT').
- 2024 from data/raw/2024 Draft - Public - OT.csv (combine + pick/round).
- 2025 from data/raw/GabrielGTB 2025 NFL Combine - Master List.csv (Position == 'OT') + 2025_draft_picks.csv for Round/Pick.
- PFF: match by Player + School + Year; filter position T/OT. Pass block: pressure_rate, sack_rate, snap_counts_pass_block. Run block: snap_counts_run_block, gap_rate, zone_rate. penalty_rate from penalties/snaps.
- RAS for OT. Optional arm length (mockdraftable_ot_arm_length.csv).
- Output: ot_training.csv (2015-2023), ot_testing.csv (2024-2025), OT/ot_drafted_2026.csv.
Run from project root: python OT/data_cleaning.py
"""
import os
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Load combine, OT only (2015-2023)
nfl_combine_data = pd.read_csv(os.path.join(DATA_RAW, 'nfl_combine_2010_to_2023.csv'))
OT_POSITIONS = ['OT']
nfl_combine_data_ot = nfl_combine_data[nfl_combine_data['Pos'].isin(OT_POSITIONS)].copy()
print(f"OT combine rows: {len(nfl_combine_data_ot)} (Pos in {OT_POSITIONS})")

# PFF Pass_Blocking: filter T/OT
PFF_PASS_BLOCKING_DIR = os.path.join(DATA_RAW, 'pff', 'Pass_Blocking')
PFF_RUN_BLOCKING_DIR = os.path.join(DATA_RAW, 'pff', 'Run_Blocking')
OT_PFF_POSITIONS = ['T', 'OT']

pass_block_cols = ['player', 'team_name', 'position', 'true_pass_set_pressures_allowed', 'true_pass_set_sacks_allowed',
                   'true_pass_set_snap_counts_pass_block', 'penalties']
pass_block_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_PASS_BLOCKING_DIR, f'{year}_offense_pass_blocking.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        sub = df[[c for c in pass_block_cols if c in df.columns]].copy()
        if 'player' not in sub.columns or 'true_pass_set_snap_counts_pass_block' not in sub.columns:
            continue
        sub = sub[sub['position'].astype(str).str.strip().str.upper().isin(OT_PFF_POSITIONS)]
        sub['Year'] = year
        sub = sub.rename(columns={'player': 'Player', 'team_name': 'School'})
        for c in ['true_pass_set_pressures_allowed', 'true_pass_set_sacks_allowed', 'true_pass_set_snap_counts_pass_block', 'penalties', 'snap_counts_pass_block']:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
        # Derived rates (PFF raw -> rates):
        #   true_pass_set_pressure_rate = true_pass_set_pressures_allowed / true_pass_set_snap_counts_pass_block
        #   true_pass_set_sack_rate = true_pass_set_sacks_allowed / true_pass_set_snap_counts_pass_block
        snaps = sub['true_pass_set_snap_counts_pass_block']
        sub['true_pass_set_pressure_rate'] = np.where(snaps > 0, sub['true_pass_set_pressures_allowed'] / snaps, np.nan)
        sub['true_pass_set_sack_rate'] = np.where(snaps > 0, sub['true_pass_set_sacks_allowed'] / snaps, np.nan)
        sub = sub.rename(columns={'true_pass_set_snap_counts_pass_block': 'snap_counts_pass_block'})
        pass_block_files.append(sub.drop(columns=['true_pass_set_pressures_allowed', 'true_pass_set_sacks_allowed', 'position'], errors='ignore'))
        print(f'Loaded PFF pass blocking {year}: {len(sub)} OT/T')
if not pass_block_files:
    pass_block_data = pd.DataFrame(columns=['Player', 'School', 'Year', 'true_pass_set_pressure_rate', 'true_pass_set_sack_rate', 'snap_counts_pass_block', 'penalties'])
    print('No PFF pass blocking files found.')
else:
    pass_block_data = pd.concat(pass_block_files, ignore_index=True).drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    pass_block_data = pass_block_data.rename(columns={'penalties': 'pass_block_penalties'})
    print(f'PFF pass blocking records: {len(pass_block_data)}')

# PFF Run_Blocking (filename has typo: run_blockng)
run_block_cols = ['player', 'team_name', 'position', 'grades_run_block', 'snap_counts_run_block', 'gap_snap_counts_run_block_percent', 'zone_snap_counts_run_block_percent', 'penalties']
run_block_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_RUN_BLOCKING_DIR, f'{year}_offense_run_blockng.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        sub = df[[c for c in run_block_cols if c in df.columns]].copy()
        if 'snap_counts_run_block' not in sub.columns:
            continue
        sub = sub[sub['position'].astype(str).str.strip().str.upper().isin(OT_PFF_POSITIONS)]
        sub['Year'] = year
        sub = sub.rename(columns={'player': 'Player', 'team_name': 'School'})
        for c in ['grades_run_block', 'snap_counts_run_block', 'gap_snap_counts_run_block_percent', 'zone_snap_counts_run_block_percent', 'penalties']:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
        # gap_rate, zone_rate: PFF stores as 0-100, convert to 0-1
        if 'gap_snap_counts_run_block_percent' in sub.columns:
            sub['gap_rate'] = sub['gap_snap_counts_run_block_percent'] / 100.0
        if 'zone_snap_counts_run_block_percent' in sub.columns:
            sub['zone_rate'] = sub['zone_snap_counts_run_block_percent'] / 100.0
        sub = sub.drop(columns=['gap_snap_counts_run_block_percent', 'zone_snap_counts_run_block_percent', 'position'], errors='ignore')
        sub = sub.rename(columns={'penalties': 'run_block_penalties'})
        run_block_files.append(sub)
        print(f'Loaded PFF run blocking {year}: {len(sub)} OT/T')
if not run_block_files:
    run_block_data = pd.DataFrame(columns=['Player', 'School', 'Year', 'grades_run_block', 'snap_counts_run_block', 'gap_rate', 'zone_rate', 'run_block_penalties'])
else:
    run_block_data = pd.concat(run_block_files, ignore_index=True).drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    print(f'PFF run blocking records: {len(run_block_data)}')

# Merge PFF: all keys from both
pff_keys = pass_block_data[['Player', 'School', 'Year']].drop_duplicates()
if not run_block_data.empty:
    pff_keys = pd.concat([pff_keys, run_block_data[['Player', 'School', 'Year']]], ignore_index=True).drop_duplicates()
pff_data = pff_keys.merge(pass_block_data, on=['Player', 'School', 'Year'], how='left')
pff_data = pff_data.merge(run_block_data, on=['Player', 'School', 'Year'], how='left')
# If merge created duplicate column names (e.g. _x, _y), keep first
pff_data = pff_data.loc[:, ~pff_data.columns.duplicated(keep='first')]
pff_data = pff_data.reset_index(drop=True)
# penalty_rate: total penalties / total snaps (use .values to avoid index alignment issues)
pb_pen = np.asarray(pff_data['pass_block_penalties'].fillna(0)).ravel() if 'pass_block_penalties' in pff_data.columns else np.zeros(len(pff_data))
rb_pen = np.asarray(pff_data['run_block_penalties'].fillna(0)).ravel() if 'run_block_penalties' in pff_data.columns else np.zeros(len(pff_data))
pb_snaps = np.asarray(pff_data['snap_counts_pass_block'].fillna(0)).ravel() if 'snap_counts_pass_block' in pff_data.columns else np.zeros(len(pff_data))
rb_snaps = np.asarray(pff_data['snap_counts_run_block'].fillna(0)).ravel() if 'snap_counts_run_block' in pff_data.columns else np.zeros(len(pff_data))
total_snaps = pb_snaps + rb_snaps
total_pen = pb_pen + rb_pen
with np.errstate(divide='ignore', invalid='ignore'):
    pff_data['penalty_rate'] = np.where(total_snaps > 0, total_pen.astype(float) / total_snaps.astype(float), np.nan)
pff_data = pff_data.drop(columns=['pass_block_penalties', 'run_block_penalties'], errors='ignore')
print(f'PFF merged records: {len(pff_data)}')

# RAS for OT
ras_df = pd.read_csv(os.path.join(DATA_RAW, 'ras.csv'))
RAS_OT_POSITIONS = ['OT', 'T']
ras_df = ras_df[ras_df['Pos'].astype(str).str.strip().str.upper().isin(RAS_OT_POSITIONS)].copy()
ras_df['RAS'] = pd.to_numeric(ras_df['RAS'], errors='coerce')
ras_df['Year'] = ras_df['Year'].astype(int)
ras_ot = ras_df[['Name', 'Year', 'RAS', 'College']].drop_duplicates(subset=['Name', 'Year'])
print(f'RAS OT records: {len(ras_ot)}')

# Arm length
arm_path = os.path.join(DATA_RAW, 'mockdraftable_ot_arm_length.csv')
if os.path.exists(arm_path):
    arm_length_df = pd.read_csv(arm_path)
    arm_length_df['Year'] = arm_length_df['Year'].astype(int)
    arm_length_df = arm_length_df.drop_duplicates(subset=['Player', 'Year'], keep='first')
    arm_length_df = arm_length_df[['Player', 'Year', 'arm_length_inches']].copy()
    arm_length_df['arm_length_inches'] = pd.to_numeric(arm_length_df['arm_length_inches'], errors='coerce')
    print(f'Arm length OT: {len(arm_length_df)} records')
else:
    arm_length_df = pd.DataFrame(columns=['Player', 'Year', 'arm_length_inches'])
    print('No mockdraftable_ot_arm_length.csv; arm_length_inches will be empty.')

# --- 2024 from 2024 Draft - Public - OT.csv ---
def _ht_2024_to_inches(ht):
    if pd.isna(ht) or str(ht).strip() == '':
        return np.nan
    try:
        s = str(int(float(ht))).zfill(4)
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 4:
        return np.nan
    ft, inch = int(s[0]), int(s[1:3])
    eighth = int(s[3]) if len(s) > 3 else 0
    return ft * 12 + inch + eighth / 8.0

def _arm_2024_to_inches(arm):
    if pd.isna(arm) or str(arm).strip() == '':
        return np.nan
    try:
        s = str(arm).strip().replace('.', '')
        if not s.isdigit():
            return np.nan
    except (ValueError, TypeError):
        return np.nan
    if len(s) < 4:
        return np.nan
    return int(s[:2]) + int(s[2:]) / 100.0

def _broad_2024_to_inches(broad):
    if pd.isna(broad) or str(broad).strip() == '':
        return np.nan
    try:
        s = str(int(float(broad)))
    except (ValueError, TypeError):
        return np.nan  # e.g. DNP
    if len(s) < 3:
        return np.nan
    return int(s[:-2]) * 12 + int(s[-2:])

def _pick_to_round(pick_taken):
    if pd.isna(pick_taken) or str(pick_taken).strip().upper() == 'UDFA':
        return 8
    try:
        p = int(float(str(pick_taken).replace(',', '')))
        if 1 <= p <= 32: return 1
        if p <= 64: return 2
        if p <= 96: return 3
        if p <= 128: return 4
        if p <= 160: return 5
        if p <= 192: return 6
        if p <= 257: return 7
    except (ValueError, TypeError):
        pass
    return 8

draft_2024_path = os.path.join(DATA_RAW, '2024 Draft - Public - OT.csv')
ot_2024_list = []
if os.path.exists(draft_2024_path):
    d24 = pd.read_csv(draft_2024_path)
    d24 = d24[d24['Name'].notna() & (d24['Name'].astype(str).str.strip() != '')].copy()
    # Only rows where Pos is OT (exclude C/OG listed in same file)
    d24 = d24[d24['Pos'].astype(str).str.strip().str.upper() == 'OT'].copy()
    for _, row in d24.iterrows():
        pick_taken = row.get('Pick Taken', row.iloc[0] if len(row) > 0 else None)
        pick_int = np.nan
        try:
            if str(pick_taken).strip() not in ('', 'UDFA'):
                pick_int = int(float(str(pick_taken).replace(',', '')))
        except (ValueError, TypeError):
            pass
        ot_2024_list.append({
            'Year': 2024, 'Player': row['Name'], 'Pos': 'OT', 'School': row['School'],
            'Height': _ht_2024_to_inches(row.get('HT')), 'Weight': pd.to_numeric(row.get('WT'), errors='coerce'),
            '40yd': pd.to_numeric(row.get('40'), errors='coerce'), 'Vertical': pd.to_numeric(row.get('VJ'), errors='coerce'),
            'Bench': pd.to_numeric(row.get('BP'), errors='coerce'), 'Broad Jump': _broad_2024_to_inches(row.get('BJ')),
            '3Cone': pd.to_numeric(row.get('3Cone'), errors='coerce'), 'Shuttle': pd.to_numeric(row.get('SS'), errors='coerce'),
            'Drafted': True, 'Round': _pick_to_round(pick_taken), 'Pick': pick_int,
            'RAS': pd.to_numeric(row.get('RAS'), errors='coerce'), 'arm_length_inches': _arm_2024_to_inches(row.get('Arm')),
        })
    ot_2024 = pd.DataFrame(ot_2024_list)
    print(f'Loaded 2024 OTs: {len(ot_2024)} from 2024 Draft - Public - OT.csv')
else:
    ot_2024 = pd.DataFrame(columns=['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS', 'arm_length_inches'])
    print('2024 Draft - Public - OT.csv not found.')

# --- 2025 from combine ---
combine_2025_path = os.path.join(DATA_RAW, 'GabrielGTB 2025 NFL Combine - Master List.csv')
def _ht_2025_to_inches(ht):
    if pd.isna(ht) or str(ht).strip() == '':
        return np.nan
    s = str(int(float(ht))).zfill(4)
    if len(s) < 4:
        return np.nan
    return int(s[0]) * 12 + int(s[1:3]) + (int(s[3]) if len(s) > 3 else 0) / 8.0
def _broad_2025_to_inches(broad):
    if pd.isna(broad) or str(broad).strip() == '':
        return np.nan
    s = str(int(float(broad)))
    if len(s) < 3:
        return np.nan
    return int(s[:-2]) * 12 + int(s[-2:])

ot_2025_list = []
if os.path.exists(combine_2025_path):
    c25 = pd.read_csv(combine_2025_path)
    c25_ot = c25[c25['Position'].astype(str).str.strip().str.upper().isin(['OT', 'T'])].copy()
    for _, row in c25_ot.iterrows():
        ot_2025_list.append({
            'Year': 2025, 'Player': row['Name'], 'Pos': 'OT', 'School': str(row.get('School', '')).replace('Syracruse', 'Syracuse').strip(),
            'Height': _ht_2025_to_inches(row.get('Height (FIIE)')), 'Weight': pd.to_numeric(row.get('Weight (lbs.)'), errors='coerce'),
            '40yd': pd.to_numeric(row.get('40-yard Dash (seconds)'), errors='coerce'),
            'Vertical': pd.to_numeric(row.get('Vertical Jump (inches)'), errors='coerce'),
            'Bench': pd.to_numeric(row.get('Bench Press (reps)'), errors='coerce'),
            'Broad Jump': _broad_2025_to_inches(row.get('Broad Jump (FFII)')),
            '3Cone': pd.to_numeric(row.get('Three-cone Drill (seconds)'), errors='coerce'),
            'Shuttle': pd.to_numeric(row.get('20-yard Shuttle (seconds)'), errors='coerce'),
            'Drafted': True, 'Round': np.nan, 'Pick': np.nan,
            'RAS': pd.to_numeric(row.get('RAS'), errors='coerce'),
            'arm_length_inches': pd.to_numeric(row.get('Arm Length (inches)'), errors='coerce'),
        })
    ot_2025_from_combine = pd.DataFrame(ot_2025_list)
    draft_picks_2025_path = os.path.join(DATA_RAW, '2025_draft_picks.csv')
    ot_drafted_2025_path = os.path.join(SCRIPT_DIR, 'ot_drafted_2025.csv')
    def _norm_name(n):
        return re.sub(r'\s+Jr\.?$|\s+III$|\s+II$|\s+IV$', '', str(n).strip(), flags=re.IGNORECASE).strip() if pd.notna(n) else ''
    def _norm_school(s):
        x = str(s).strip() if pd.notna(s) else ''
        aliases = {'Penn St.': 'Penn State', 'Ohio St.': 'Ohio State', 'Florida St.': 'Florida State', 'Ole Miss': 'Mississippi', 'Syracruse': 'Syracuse'}
        return aliases.get(x, x)
    round_pick_2025 = None
    if os.path.exists(draft_picks_2025_path):
        draft_all = pd.read_csv(draft_picks_2025_path)
        draft_ot = draft_all[draft_all['Pos'].astype(str).str.upper().isin(['OT', 'T', 'OL'])].copy()
        if not draft_ot.empty:
            draft_ot = draft_ot.rename(columns={'Rnd': 'Round'})
            draft_ot['Player_norm'] = draft_ot['Player'].map(_norm_name)
            draft_ot['School_norm'] = draft_ot['School'].map(_norm_school)
            round_pick_2025 = draft_ot[['Player_norm', 'School_norm', 'Round', 'Pick']].drop_duplicates()
    ot_2025_from_combine = ot_2025_from_combine.drop(columns=['Round', 'Pick'], errors='ignore')
    ot_2025_from_combine['Player_norm'] = ot_2025_from_combine['Player'].map(_norm_name)
    ot_2025_from_combine['School_norm'] = ot_2025_from_combine['School'].map(_norm_school)
    if round_pick_2025 is not None and not round_pick_2025.empty:
        ot_2025_from_combine = ot_2025_from_combine.merge(round_pick_2025, on=['Player_norm', 'School_norm'], how='left')
        ot_2025_from_combine = ot_2025_from_combine.drop(columns=['Player_norm', 'School_norm'], errors='ignore')
    else:
        ot_2025_from_combine = ot_2025_from_combine.drop(columns=['Player_norm', 'School_norm'], errors='ignore')
    ot_2025 = ot_2025_from_combine
    print(f'Loaded 2025 OTs: {len(ot_2025)} from combine')
else:
    ot_2025 = pd.DataFrame(columns=['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS', 'arm_length_inches'])
    print('GabrielGTB 2025 NFL Combine - Master List.csv not found.')

# Training / testing / 2026
ot_training_data = nfl_combine_data_ot[nfl_combine_data_ot['Year'].between(2015, 2023)].copy()
ot_testing_data = pd.concat([ot_2024, ot_2025], ignore_index=True)
if not ot_testing_data.empty:
    ot_testing_data['Year'] = ot_testing_data['Year'].astype(int)
    ot_testing_data['Drafted'] = True
    ot_testing_data = ot_testing_data[ot_testing_data['Round'].notna() & (ot_testing_data['Round'] != 8)].copy()
if 'arm_length_inches' not in ot_testing_data.columns and not ot_testing_data.empty:
    ot_testing_data['arm_length_inches'] = np.nan
cols_drop = ['speed_score', 'explosive_score', 'agility_score']
ot_testing_data = ot_testing_data.drop(columns=cols_drop, errors='ignore')

# 2026
ot_drafted_2026_path = os.path.join(SCRIPT_DIR, 'ot_drafted_2026.csv')
if os.path.exists(ot_drafted_2026_path):
    ot_2026_raw = pd.read_csv(ot_drafted_2026_path)
    if 'Player' in ot_2026_raw.columns and 'School' in ot_2026_raw.columns:
        ot_2026 = ot_2026_raw.copy()
        ot_2026['Year'] = 2026
        ot_2026['Pos'] = 'OT'
        for col in ['Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'arm_length_inches', 'RAS']:
            if col not in ot_2026.columns:
                ot_2026[col] = np.nan
        ot_2026['40yd'] = pd.to_numeric(ot_2026.get('40yd', ot_2026.get('40', np.nan)), errors='coerce')
        ot_2026_processed = ot_2026.drop(columns=cols_drop, errors='ignore')
        print(f'Loaded 2026 OTs: {len(ot_2026)} from ot_drafted_2026.csv')
    else:
        ot_2026_processed = pd.DataFrame()
        print('ot_drafted_2026.csv missing Player/School')
else:
    ot_2026_processed = pd.DataFrame()
    print('No ot_drafted_2026.csv; 2026 will be empty.')

# --- Normalize and add PFF ---
def normalize_player_name(name):
    s = str(name).strip().upper()
    s = re.sub(r'\s+(III|II|JR|SR|JR\.|SR\.)$', '', s)
    s = re.sub(r'[.\',\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_pff_school(name):
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    mapping = {
        'OHIO STATE': 'Ohio State', 'FLORIDA ST': 'Florida State', 'PENN STATE': 'Penn State',
        'NOTRE DAME': 'Notre Dame', 'MICHIGAN': 'Michigan', 'GEORGIA': 'Georgia', 'ALABAMA': 'Alabama',
        'OREGON ST': 'Oregon State', 'OREGON': 'Oregon', 'WASHINGTON': 'Washington', 'DUKE': 'Duke',
        'NORTH CAROLINA': 'North Carolina', 'N CAROLINA': 'North Carolina', 'NC STATE': 'North Carolina State',
        'VIRGINIA TECH': 'Virginia Tech', 'VA TECH': 'Virginia Tech',
        'UCLA': 'UCLA', 'USC': 'USC', 'CAL': 'California', 'STANFORD': 'Stanford',
        'OKLAHOMA': 'Oklahoma', 'TEXAS': 'Texas', 'LSU': 'LSU', 'AUBURN': 'Auburn', 'TENNESSEE': 'Tennessee',
        'BYU': 'BYU', 'HOUSTON': 'Houston', 'YALE': 'Yale', 'ARIZONA': 'Arizona', 'ARIZONA ST': 'Arizona State',
        'KENTUCKY': 'Kentucky', 'CONNECTICUT': 'Connecticut', 'UCONN': 'Connecticut',
        'E KENTUCKY': 'Eastern Kentucky', 'GA STATE': 'Georgia State',
        'WASH STATE': 'Washington State', 'S CAROLINA': 'South Carolina', 'COLO STATE': 'Colorado State',
        'MICH STATE': 'Michigan State', 'MISS STATE': 'Mississippi State', 'OLE MISS': 'Mississippi',
        'BOISE ST': 'Boise State', 'BOSTON COL': 'Boston College', 'KANSAS ST': 'Kansas State',
        'S DIEGO ST': 'San Diego State', 'W MICHIGAN': 'Western Michigan', 'W VIRGINIA': 'West Virginia',
        'TEXAS A&M': 'Texas A&M', 'NWESTERN': 'Northwestern', 'MIAMI FL': 'Miami', 'MIAMI OH': 'Miami (OH)',
    }
    return mapping.get(name, name.title())

def normalize_combine_school(name):
    if pd.isna(name):
        return name
    x = str(name).strip()
    # Match PFF: uppercase school names from draft/combine CSVs -> same canonical as normalize_pff_school
    upper_mapping = {
        'NOTRE DAME': 'Notre Dame', 'ALABAMA': 'Alabama', 'PENN STATE': 'Penn State', 'WASHINGTON': 'Washington',
        'OREGON ST': 'Oregon State', 'OREGON': 'Oregon', 'GEORGIA': 'Georgia', 'HOUSTON': 'Houston', 'YALE': 'Yale',
        'OKLAHOMA': 'Oklahoma', 'MARYLAND': 'Maryland', 'KANSAS': 'Kansas', 'TCU': 'TCU', 'MISSOURI': 'Missouri',
        'TEXAS': 'Texas', 'KANSAS ST': 'Kansas State', 'LA LAFAYET': 'Louisiana', 'EASTERN KENTUCKY': 'Eastern Kentucky',
        'UTAH': 'Utah', 'PITTSBURGH': 'Pittsburgh', 'GA STATE': 'Georgia State', 'MICHIGAN': 'Michigan',
        'CONNECTICUT': 'Connecticut', 'UCONN': 'Connecticut', 'UCF': 'UCF', 'OHIO STATE': 'Ohio State', 'FLORIDA STATE': 'Florida State',
        'FINDLAY': 'Findlay',
        'WEST. MICHIGAN': 'Western Michigan', 'MISSISSIPPI ST.': 'Mississippi State', 'ARIZONA ST.': 'Arizona State',
        'BOISE ST.': 'Boise State', 'NORTH DAKOTA ST.': 'North Dakota State', 'BOSTON COL.': 'Boston College',
        'WASH STATE': 'Washington State', 'S CAROLINA': 'South Carolina', 'COLO STATE': 'Colorado State',
        'MICH STATE': 'Michigan State', 'MISS STATE': 'Mississippi State', 'OLE MISS': 'Mississippi',
        'BOISE ST': 'Boise State', 'ARIZONA ST': 'Arizona State', 'OHIO ST.': 'Ohio State',
        'VIRGINIA TECH': 'Virginia Tech', 'VA TECH': 'Virginia Tech', 'S DIEGO ST': 'San Diego State',
        'W MICHIGAN': 'Western Michigan', 'W VIRGINIA': 'West Virginia', 'NC STATE': 'North Carolina State',
        'N CAROLINA': 'North Carolina', 'NORTH CAROLINA': 'North Carolina', 'TEXAS A&M': 'Texas A&M',
        'NWESTERN': 'Northwestern', 'NORTHWESTERN': 'Northwestern',
    }
    if x.upper() in upper_mapping:
        return upper_mapping[x.upper()]
    alias = {
        'Ole Miss': 'Mississippi', 'Miami (FL)': 'Miami', 'Southern California': 'USC', 'Ohio St.': 'Ohio State',
        'Florida St.': 'Florida State', 'Penn St.': 'Penn State', 'North Carolina St.': 'North Carolina State',
        'NC State': 'North Carolina State', 'Oregon St.': 'Oregon State', 'Washington St.': 'Washington State',
        'Cal': 'California', 'Brigham Young': 'BYU', 'Central Florida': 'UCF', 'LA-Lafayette': 'Louisiana',
        'West. Michigan': 'Western Michigan', 'Mississippi St.': 'Mississippi State', 'Arizona St.': 'Arizona State',
        'Boise St.': 'Boise State', 'North Dakota St.': 'North Dakota State', 'Boston Col.': 'Boston College',
        'Southern Utah St.': 'Southern Utah',
    }
    return alias.get(x, x)

def add_pff_data(combine_df, pff_df):
    combine_df = combine_df.copy()
    pff_n = pff_df.copy()
    pff_n['School_normalized'] = pff_n['School'].apply(normalize_pff_school)
    pff_n['Player_normalized'] = pff_n['Player'].apply(normalize_player_name)
    combine_df['School_normalized'] = combine_df['School'].apply(normalize_combine_school)
    pff_value_cols = [c for c in pff_df.columns if c not in ('Player', 'School', 'Year')]
    def lookup(row):
        draft_year = int(row['Year'])
        final_season = draft_year - 1
        player = normalize_player_name(row['Player'])
        school = row['School_normalized']
        mask = (pff_n['Player_normalized'] == player) & (pff_n['School_normalized'] == school) & (pff_n['Year'] == final_season)
        match = pff_n.loc[mask]
        if match.empty:
            return pd.Series({c: None for c in pff_value_cols})
        return pd.Series({c: match.iloc[0][c] for c in pff_value_cols})
    res = combine_df.apply(lookup, axis=1)
    for c in res.columns:
        combine_df[c] = res[c]
    return combine_df.drop(columns=['School_normalized'], errors='ignore')

def add_ras_data(combine_df, ras_subset):
    combine_df = combine_df.copy()
    ras_n = ras_subset.copy()
    ras_n['Year'] = ras_n['Year'].astype(int)
    ras_n['Name_n'] = ras_n['Name'].apply(normalize_player_name)
    # RAS College -> same canonical as normalize_combine_school; then normalize so combine match works
    ras_school = {
        'Miami (FL)': 'Miami', 'Southern California': 'USC', 'Ohio St.': 'Ohio State', 'Florida St.': 'Florida State',
        'Ole Miss': 'Mississippi', 'Penn St.': 'Penn State', 'NC State': 'North Carolina State', 'Oregon St.': 'Oregon State',
        'Louisiana State': 'LSU', 'Miami OH': 'Miami (OH)', 'Central Florida': 'UCF', 'North Dakota': 'North Dakota State',
        'Washington St.': 'Washington State', 'West. Michigan': 'Western Michigan',
    }
    def _normalize_ras_college(x):
        if pd.isna(x):
            return x
        raw = str(x).strip()
        canonical = ras_school.get(raw, raw)
        return normalize_combine_school(canonical)
    ras_n['College_n'] = ras_n['College'].apply(_normalize_ras_college)
    # Combine normalized name -> RAS may use alternate spelling (so we also match these)
    ras_name_variants = {
        'YOSH NIJMAN': ['YOSUAH NIJMAN'],
        'GARRETT BOLLES': ['GARETT BOLLES'],
    }
    def lookup_ras(row):
        player = normalize_player_name(row['Player'])
        school = normalize_combine_school(row['School'])
        year = int(row['Year'])
        names_to_try = [player] + ras_name_variants.get(player, [])
        m = (ras_n['Name_n'].isin(names_to_try)) & (ras_n['College_n'] == school) & (ras_n['Year'] == year)
        hit = ras_n.loc[m]
        return pd.Series({'RAS': hit.iloc[0]['RAS'] if not hit.empty else None})
    ras_cols = combine_df.apply(lookup_ras, axis=1)
    combine_df['RAS'] = ras_cols['RAS']
    return combine_df

def add_arm_length(combine_df, arm_df):
    combine_df = combine_df.copy()
    combine_df = combine_df.drop(columns=['arm_length_inches'], errors='ignore')
    if arm_df.empty or 'arm_length_inches' not in arm_df.columns:
        combine_df['arm_length_inches'] = np.nan
        return combine_df
    combine_df = combine_df.merge(arm_df[['Player', 'Year', 'arm_length_inches']], on=['Player', 'Year'], how='left')
    return combine_df

# Apply PFF, RAS, arm
ot_training_data = add_pff_data(ot_training_data, pff_data)
ot_testing_data = add_pff_data(ot_testing_data, pff_data)
if not ot_2026_processed.empty:
    ot_2026_processed = add_pff_data(ot_2026_processed, pff_data)

ot_training_data = add_ras_data(ot_training_data, ras_ot)
ot_testing_data = add_ras_data(ot_testing_data, ras_ot)
if not ot_2026_processed.empty:
    ot_2026_processed = add_ras_data(ot_2026_processed, ras_ot)

ot_training_data = add_arm_length(ot_training_data, arm_length_df)
ot_testing_data = add_arm_length(ot_testing_data, arm_length_df)
if not ot_2026_processed.empty:
    ot_2026_processed = add_arm_length(ot_2026_processed, arm_length_df)

ot_training_data = ot_training_data.drop(columns=cols_drop, errors='ignore')
ot_testing_data = ot_testing_data.drop(columns=cols_drop, errors='ignore')

training_cols_order = ['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick', 'RAS', 'arm_length_inches',
    'true_pass_set_pressure_rate', 'true_pass_set_sack_rate', 'snap_counts_pass_block', 'snap_counts_run_block', 'grades_run_block', 'gap_rate', 'zone_rate', 'penalty_rate']
training_cols_order = [c for c in training_cols_order if c in ot_training_data.columns or c in pff_data.columns]
ot_training_data = ot_training_data[[c for c in training_cols_order if c in ot_training_data.columns]]
ot_testing_data = ot_testing_data[[c for c in training_cols_order if c in ot_testing_data.columns]]

ot_training_data.to_csv(os.path.join(DATA_PROCESSED, 'ot_training.csv'), index=False)
ot_testing_data.to_csv(os.path.join(DATA_PROCESSED, 'ot_testing.csv'), index=False)

if not ot_2026_processed.empty:
    ot_2026_cols = [c for c in ['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'RAS', 'arm_length_inches',
        'true_pass_set_pressure_rate', 'true_pass_set_sack_rate', 'snap_counts_pass_block', 'snap_counts_run_block', 'grades_run_block', 'gap_rate', 'zone_rate', 'penalty_rate'] if c in ot_2026_processed.columns]
    ot_2026_final = ot_2026_processed[ot_2026_cols].copy()
    ot_2026_final.to_csv(ot_drafted_2026_path, index=False)
    print(f'Saved ot_drafted_2026.csv: {len(ot_2026_final)}')
else:
    # Write minimal header-only so round regression can load
    pd.DataFrame(columns=['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'RAS', 'arm_length_inches',
        'true_pass_set_pressure_rate', 'true_pass_set_sack_rate', 'snap_counts_pass_block', 'snap_counts_run_block', 'grades_run_block', 'gap_rate', 'zone_rate', 'penalty_rate']).to_csv(ot_drafted_2026_path, index=False)
    print('Saved ot_drafted_2026.csv (empty)')

print(f'Saved ot_training.csv: {len(ot_training_data)} (2015-2023)')
print(f'Saved ot_testing.csv: {len(ot_testing_data)} (2024-2025)')
print(f'Columns: {list(ot_training_data.columns)}')
