"""
LB data cleaning: combine LBs (ILB/LB/OLB) + PFF Pass_Rush, Run_Defense, Pass_Coverage.
- PFF: match by Player + School + Year; prefer LB/ILB/OLB when dedup.
- RAS for LB. Optional arm length (mockdraftable_lb_arm_length.csv).
- Output: lb_training.csv (2015-2023), lb_testing.csv (2024-2026), lb_drafted_2026.csv.
Run from LB/ directory.
"""
import os
import re
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Load combine, LB only
nfl_combine_data = pd.read_csv(os.path.join(DATA_RAW, 'nfl_combine_2010_to_2023.csv'))
LB_POSITIONS = ['ILB', 'LB', 'OLB']
nfl_combine_data_lb = nfl_combine_data[nfl_combine_data['Pos'].isin(LB_POSITIONS)].copy()
print(f"LB combine rows: {len(nfl_combine_data_lb)} (Pos in {LB_POSITIONS})")

# PFF: prefer LB/ILB/OLB when same player/school/year
POSITION_PRIORITY = {'LB': 0, 'ILB': 1, 'OLB': 2}  # others 99

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
    pff_data = pd.DataFrame(columns=['Player', 'School', 'Year', 'true_pass_set_pass_rush_win_rate',
                                     'pass_rush_win_rate', 'snap_counts_pass_rush'])
    print('No PFF pass rush files found.')
else:
    pff_data = pd.concat(pff_files, ignore_index=True)
    pff_data['_pos_order'] = pff_data['position'].map(POSITION_PRIORITY).fillna(99)
    pff_data = pff_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    pff_data = pff_data.drop(columns=['_pos_order', 'position'], errors='ignore')
    print(f'PFF pass rush records (after dedup): {len(pff_data)}')

# --- Run Defense: stop_percent, missed_tackle_rate, avg_depth_of_tackle, snap_counts_run, forced_fumbles ---
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
    merge_cols = ['Player', 'School', 'Year'] + [c for c in ['stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles'] if c in run_defense_data.columns]
    pff_data = pff_data.merge(run_defense_data[merge_cols], on=['Player', 'School', 'Year'], how='left')
    print(f'Merged run defense; columns: {list(pff_data.columns)}')
else:
    for c in ['stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles']:
        pff_data[c] = None
    print('No run defense files found.')

# --- Pass Coverage ---
coverage_cols = ['player', 'team_name', 'position', 'yards_per_coverage_snap', 'forced_incompletion_rate',
                 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target']
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
        for c in ['yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target']:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
        coverage_files.append(sub)
        print(f'Loaded PFF pass coverage {year}: {len(sub)} players')

if coverage_files:
    coverage_data = pd.concat(coverage_files, ignore_index=True)
    coverage_data['_pos_order'] = coverage_data['position'].map(POSITION_PRIORITY).fillna(99)
    coverage_data = coverage_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    coverage_data = coverage_data.drop(columns=['_pos_order', 'position'], errors='ignore')
    merge_cov = ['Player', 'School', 'Year'] + [c for c in ['yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target'] if c in coverage_data.columns]
    pff_data = pff_data.merge(coverage_data[merge_cov], on=['Player', 'School', 'Year'], how='left')
    print(f'Merged pass coverage; columns: {list(pff_data.columns)}')
else:
    for c in ['yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent', 'interceptions', 'pass_break_ups', 'coverage_snaps_per_target']:
        pff_data[c] = None
    print('No pass coverage files found.')

# RAS for LB
ras_df = pd.read_csv(os.path.join(DATA_RAW, 'ras.csv'))
ras_df = ras_df[ras_df['Pos'].isin(LB_POSITIONS)].copy()
ras_df['RAS'] = pd.to_numeric(ras_df['RAS'], errors='coerce')
ras_df['Year'] = ras_df['Year'].astype(int)
ras_lb = ras_df[['Name', 'Year', 'RAS', 'College']].drop_duplicates(subset=['Name', 'Year'])
print(f'RAS LB records: {len(ras_lb)}')

# Arm length (optional; from scrape_mockdraftable_arm_length_for_our_lbs.py)
arm_path = os.path.join(DATA_RAW, 'mockdraftable_lb_arm_length.csv')
if os.path.exists(arm_path):
    arm_length_df = pd.read_csv(arm_path)
    arm_length_df['Year'] = arm_length_df['Year'].astype(int)
    arm_length_df = arm_length_df.drop_duplicates(subset=['Player', 'Year'], keep='first')
    arm_length_df = arm_length_df[['Player', 'Year', 'arm_length_inches']].copy()
    arm_length_df['arm_length_inches'] = pd.to_numeric(arm_length_df['arm_length_inches'], errors='coerce')
    print(f'Arm length LB: {len(arm_length_df)} records ({arm_length_df["arm_length_inches"].notna().sum()} with values)')
else:
    arm_length_df = pd.DataFrame(columns=['Player', 'Year', 'arm_length_inches'])
    print('No mockdraftable_lb_arm_length.csv; arm_length_inches will be empty.')

# Splits: training 2015-2023, testing 2024-2026 from drafted CSVs
lb_training_data = nfl_combine_data_lb[nfl_combine_data_lb['Year'].between(2015, 2023)].copy()

lb_2024 = pd.read_csv(os.path.join(SCRIPT_DIR, 'lb_drafted_2024.csv'))
lb_2025 = pd.read_csv(os.path.join(SCRIPT_DIR, 'lb_drafted_2025.csv'))
lb_2026 = pd.read_csv(os.path.join(SCRIPT_DIR, 'lb_drafted_2026.csv'))
lb_testing_data = pd.concat([lb_2024, lb_2025, lb_2026], ignore_index=True)
lb_testing_data['Year'] = lb_testing_data['Year'].astype(int)
lb_testing_data['Drafted'] = True

cols_drop = ['Sacks_cumulative', 'TFL_cumulative', 'QB_Hurry_cumulative', 'PD_cumulative', 'SOLO_cumulative', 'TOT_cumulative',
             'Sacks_final_season', 'TFL_final_season', 'QB_Hurry_final_season', 'PD_final_season', 'SOLO_final_season', 'TOT_final_season',
             'speed_score', 'explosive_score', 'agility_score']
lb_testing_data = lb_testing_data.drop(columns=cols_drop, errors='ignore')

lb_2026_processed = lb_2026.copy()
lb_2026_processed['Year'] = lb_2026_processed['Year'].astype(int)
lb_2026_processed['Drafted'] = True
lb_2026_processed = lb_2026_processed.drop(columns=cols_drop, errors='ignore')


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
        'OHIO STATE': 'Ohio State', 'OHIO ST': 'Ohio State',
        'FLORIDA ST': 'Florida State', 'FLORIDA STATE': 'Florida State',
        'KANSAS ST': 'Kansas State', 'KANSAS STATE': 'Kansas State',
        'IOWA ST': 'Iowa State', 'IOWA STATE': 'Iowa State',
        'OKLAHOMA ST': 'Oklahoma State', 'OKLAHOMA STATE': 'Oklahoma State', 'OKLAHOMA': 'Oklahoma',
        'PENN ST': 'Penn State', 'PENN STATE': 'Penn State',
        'SAN DIEGO ST': 'San Diego State', 'SAN DIEGO STATE': 'San Diego State',
        'SAN JOSE ST': 'San Jose State', 'SAN JOSE STATE': 'San Jose State',
        'MISSISSIPPI ST': 'Mississippi State', 'MISS STATE': 'Mississippi State', 'MISSISSIPPI STATE': 'Mississippi State',
        'MICHIGAN ST': 'Michigan State', 'MICHIGAN STATE': 'Michigan State',
        'NORTH CAROLINA ST': 'North Carolina State', 'NC STATE': 'North Carolina State', 'NORTH CAROLINA': 'North Carolina',
        'SOUTH CAROLINA': 'South Carolina', 'NWESTERN': 'Northwestern', 'NORTHWESTERN': 'Northwestern',
        'SOUTHERN CAL': 'USC', 'SOUTHERN CALIFORNIA': 'USC', 'CENTRAL FLORIDA': 'UCF', 'UCF': 'UCF',
        'BRIGHAM YOUNG': 'BYU', 'MIAMI (FL)': 'Miami', 'MIAMI FL': 'Miami', 'MIAMI': 'Miami', 'MIAMI OH': 'Miami (OH)',
        'OLE MISS': 'Mississippi', 'ALABAMA-BIRMINGHAM': 'UAB', 'UAB': 'UAB', 'TENN-CHATTANOOGA': 'Chattanooga',
        'C MICHIGAN': 'Central Michigan', 'CENTRAL MICHIGAN': 'Central Michigan',
        'W MICHIGAN': 'Western Michigan', 'WESTERN MICHIGAN': 'Western Michigan',
        'E MICHIGAN': 'Eastern Michigan', 'EASTERN MICHIGAN': 'Eastern Michigan',
        'FRESNO ST': 'Fresno State', 'FRESNO STATE': 'Fresno State',
        'BOISE ST': 'Boise State', 'BOISE STATE': 'Boise State',
        'ARIZONA ST': 'Arizona State', 'ARIZONA STATE': 'Arizona State',
        'OREGON ST': 'Oregon State', 'OREGON STATE': 'Oregon State',
        'COLORADO ST': 'Colorado State', 'COLORADO STATE': 'Colorado State',
        'UTAH ST': 'Utah State', 'UTAH STATE': 'Utah State',
        'ALABAMA': 'Alabama', 'ARKANSAS': 'Arkansas', 'COLORADO': 'Colorado', 'KENTUCKY': 'Kentucky',
        'UCLA': 'UCLA', 'LSU': 'LSU', 'TCU': 'TCU', 'USC': 'USC',
        'WASHINGTON STATE': 'Washington State', 'WSU': 'Washington State', 'WASH STATE': 'Washington State',
        'COLO STATE': 'Colorado State', 'BOSTON COL': 'Boston College', 'BOSTON COLLEGE': 'Boston College',
        'VA TECH': 'Virginia Tech', 'VIRGINIA TECH': 'Virginia Tech',
        'TEXAS ST': 'Texas State', 'TEXAS STATE': 'Texas State',
        'LA TECH': 'Louisiana Tech', 'LOUISIANA TECH': 'Louisiana Tech',
        'OKLA STATE': 'Oklahoma State', 'MICH STATE': 'Michigan State',
        'APPALACHIAN ST': 'Appalachian State', 'APPALACHIAN STATE': 'Appalachian State', 'APP ST': 'Appalachian State',
        'APP STATE': 'Appalachian State', 'N CAROLINA': 'North Carolina', 'S CAROLINA': 'South Carolina',
        'FLORIDA ATLANTIC': 'Florida Atlantic', 'FAU': 'Florida Atlantic',
        'TEXAS SAN ANTONIO': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio', 'TEXAS TECH': 'Texas Tech',
        'TOLEDO': 'Toledo', 'GA SOUTHRN': 'Georgia Southern', 'GEORGIA SOUTHERN': 'Georgia Southern',
        'WYOMING': 'Wyoming', 'UNLV': 'UNLV', 'S DIEGO ST': 'San Diego State', 'S JOSE ST': 'San Jose State',
        'GA TECH': 'Georgia Tech', 'GA STATE': 'Georgia State',
        'W VIRGINIA': 'West Virginia', 'WEST VIRGINIA': 'West Virginia',
        'TEXAS A&M': 'Texas A&M', 'TEXAS AM': 'Texas A&M',
        'WAKE': 'Wake Forest', 'CAL': 'California',
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
        'San Diego St.': 'San Diego State', 'San Diego State': 'San Diego State',
        'San Jose St.': 'San Jose State', 'San Jose State': 'San Jose State',
        'Boston Col.': 'Boston College', 'Boston College': 'Boston College',
        'Alabama-Birmingham': 'UAB', 'Tenn-Chattanooga': 'Chattanooga', 'Miami (Ohio)': 'Miami (OH)',
        'Washington State': 'Washington State', 'Washington St.': 'Washington State', 'Colorado State': 'Colorado State',
        'Northwestern': 'Northwestern', 'LSU': 'LSU', 'Virginia Tech': 'Virginia Tech',
        'Texas State': 'Texas State', 'Louisiana Tech': 'Louisiana Tech',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State', 'North Carolina St.': 'North Carolina State',
        'Appalachian State': 'Appalachian State', 'Appalachian St.': 'Appalachian State', 'App St.': 'Appalachian State',
        'Florida Atlantic': 'Florida Atlantic', 'Texas-San Antonio': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio',
        'Toledo': 'Toledo', 'Georgia Southern': 'Georgia Southern', 'Ga. Southern': 'Georgia Southern',
        'Kentucky': 'Kentucky', 'TCU': 'TCU', 'Texas A&M': 'Texas A&M',
        'Arizona St.': 'Arizona State', 'Arizona State': 'Arizona State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
        'West Virginia': 'West Virginia', 'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
        'Montana St.': 'Montana State', 'Montana State': 'Montana State',
    }
    return alias.get(name, name)


def add_pff_data(combine_df, pff_df):
    combine_df = combine_df.copy()
    pff_n = pff_df.copy()
    pff_n['School_normalized'] = pff_n['School'].apply(normalize_pff_school)
    pff_n['Player_normalized'] = pff_n['Player'].apply(normalize_player_name)
    combine_df['School_normalized'] = combine_df['School'].apply(normalize_combine_school)

    pff_value_cols = [c for c in pff_df.columns if c not in ('Player', 'School', 'Year')]

    # Combine normalized name -> PFF normalized name (no punctuation; spelling/nickname variants)
    player_nickname_map = {
        'YANNICK CUDJOEVIRGIL': 'YANNIK CUDJOEVIRGIL',
        'LORENZO MAULDIN': 'LORENZO MAULDIN IV',
        'CAMERON BROWN': 'CAM BROWN',
        'JOSH UCHE': 'JOSHUA UCHE',
        'WILLIAM BRADLEYKING': 'WILL BRADLEYKING',
        'TAKKARIST MCKINLEY': 'TAKK MCKINLEY',
        'JOE TRYON': 'JOE TRYONSHOYINKA',
        'SCOOTA HARRIS': 'DEJON HARRIS',
    }
    # (player_norm, school_norm) -> PFF school to use
    player_school_pff_override = {}
    # (player_norm, school_norm, draft_year) -> PFF year to use (e.g. opt-out or injury)
    player_school_year_override = {
        ('JACK CICHY', 'Wisconsin', 2018): 2016,   # injured 2017; use 2016
        ('MICAH PARSONS', 'Penn State', 2021): 2019,  # opted out 2020; use 2019
        ('JOE TRYONSHOYINKA', 'Washington', 2021): 2019,  # 2020 season limited; use 2019
        ('SHAQ SMITH', 'Maryland', 2021): 2019,  # use 2019 PFF data
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
        'Miami (FL)': 'Miami', 'Miami': 'Miami', 'Miami (Ohio)': 'Miami (OH)',
        'Southern California': 'USC', 'USC': 'USC', 'UCLA': 'UCLA',
        'Central Florida': 'UCF', 'UCF': 'UCF', 'Brigham Young': 'BYU', 'BYU': 'BYU',
        'Ole Miss': 'Mississippi', 'Mississippi': 'Mississippi', 'Ohio St.': 'Ohio State', 'Ohio State': 'Ohio State',
        'Florida St.': 'Florida State', 'Florida State': 'Florida State',
        'Oklahoma St.': 'Oklahoma State', 'Oklahoma State': 'Oklahoma State', 'Oklahoma': 'Oklahoma',
        'Penn St.': 'Penn State', 'Penn State': 'Penn State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
        'Virginia Tech': 'Virginia Tech', 'Texas State': 'Texas State', 'Louisiana Tech': 'Louisiana Tech',
        'Appalachian State': 'Appalachian State', 'Florida Atlantic': 'Florida Atlantic',
        'Texas-San Antonio': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio',
        'Toledo': 'Toledo', 'Georgia Southern': 'Georgia Southern', 'Kentucky': 'Kentucky',
        'TCU': 'TCU', 'Texas Christian': 'TCU', 'Louisiana State': 'LSU', 'LSU': 'LSU',
        'Boston Col.': 'Boston College', 'Boston College': 'Boston College',
        'San Diego St.': 'San Diego State', 'San Diego State': 'San Diego State',
        'San Jose St.': 'San Jose State', 'San Jose State': 'San Jose State',
        'Kansas St.': 'Kansas State', 'Kansas State': 'Kansas State',
        'Iowa St.': 'Iowa State', 'Iowa State': 'Iowa State',
        'Alabama-Birmingham': 'UAB', 'Tenn-Chattanooga': 'Chattanooga',
        'Washington State': 'Washington State', 'Colorado State': 'Colorado State', 'Northwestern': 'Northwestern',
        'Arizona St.': 'Arizona State', 'Arizona State': 'Arizona State',
        'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
        'West Virginia': 'West Virginia', 'Texas A&M': 'Texas A&M',
        'Georgia Tech': 'Georgia Tech', 'North Carolina': 'North Carolina', 'South Carolina': 'South Carolina',
        'Montana St.': 'Montana State', 'Montana State': 'Montana State',
        'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
        'Washington St.': 'Washington State',
        'North Carolina St.': 'North Carolina State',
        'Ala-Birmingham': 'UAB',
        'Texas AM': 'Texas A&M',  # RAS sometimes has "Texas AM" (no &)
    }
    ras_n['College_n'] = ras_n['College'].apply(lambda x: ras_school.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    # Combine normalized name -> RAS normalized name (spelling variants / nicknames)
    ras_name_alias = {
        'JOSHUA MCMILLON': 'JOSHUA MCMILLAN',
        'HENRY TOOTOO': 'HENRY TOO TOO',  # To'oTo'o (combine) -> To'o To'o (RAS) normalized
        'SCOOTA HARRIS': 'DEJON HARRIS',  # Scoota is De'Jon Harris's nickname
    }

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
    Add arm_length_inches by left merge on Player + Year.
    """
    combine_df = combine_df.copy()
    combine_df = combine_df.drop(columns=['arm_length_inches'], errors='ignore')
    if arm_df.empty or 'arm_length_inches' not in arm_df.columns:
        combine_df['arm_length_inches'] = None
        return combine_df
    combine_df = combine_df.merge(
        arm_df[['Player', 'Year', 'arm_length_inches']],
        on=['Player', 'Year'],
        how='left'
    )
    return combine_df


# Apply PFF, then RAS
lb_training_data = add_pff_data(lb_training_data, pff_data)
lb_testing_data = add_pff_data(lb_testing_data, pff_data)
lb_2026_processed = add_pff_data(lb_2026_processed, pff_data)

# INT_rate, PBU_rate (after PFF merge)
for _df in (lb_training_data, lb_testing_data, lb_2026_processed):
    snap = _df.get('snap_counts_coverage')
    interceptions = _df.get('interceptions')
    pbu = _df.get('pass_break_ups')
    _df['INT_rate'] = np.nan
    _df['PBU_rate'] = np.nan
    if snap is not None and interceptions is not None:
        _df['INT_rate'] = np.where(pd.notna(snap) & (snap > 0), pd.to_numeric(interceptions, errors='coerce') / snap, np.nan)
    if snap is not None and pbu is not None:
        _df['PBU_rate'] = np.where(pd.notna(snap) & (snap > 0), pd.to_numeric(pbu, errors='coerce') / snap, np.nan)

lb_training_data = add_ras_data(lb_training_data, ras_lb)
lb_testing_data = add_ras_data(lb_testing_data, ras_lb)
lb_2026_processed = add_ras_data(lb_2026_processed, ras_lb)

lb_training_data = add_arm_length(lb_training_data, arm_length_df)
lb_testing_data = add_arm_length(lb_testing_data, arm_length_df)
lb_2026_processed = add_arm_length(lb_2026_processed, arm_length_df)

lb_training_data = lb_training_data.drop(columns=cols_drop, errors='ignore')
lb_testing_data = lb_testing_data.drop(columns=cols_drop, errors='ignore')

# Column order
training_cols_order = ['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical',
                      'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick',
                      'RAS', 'arm_length_inches',
                      'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate', 'snap_counts_pass_rush',
                      'stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles',
                      'yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent',
                      'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'INT_rate', 'PBU_rate']
lb_training_data = lb_training_data[[c for c in training_cols_order if c in lb_training_data.columns]]
lb_testing_data = lb_testing_data[[c for c in training_cols_order if c in lb_testing_data.columns]]

lb_training_data.to_csv(os.path.join(DATA_PROCESSED, 'lb_training.csv'), index=False)
lb_testing_data.to_csv(os.path.join(DATA_PROCESSED, 'lb_testing.csv'), index=False)

lb_2026_cols = ['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight',
                '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
                'RAS', 'arm_length_inches',
                'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate', 'snap_counts_pass_rush',
                'stop_percent', 'missed_tackle_rate', 'avg_depth_of_tackle', 'snap_counts_run', 'forced_fumbles',
                'yards_per_coverage_snap', 'forced_incompletion_rate', 'snap_counts_coverage', 'coverage_percent',
                'interceptions', 'pass_break_ups', 'coverage_snaps_per_target', 'INT_rate', 'PBU_rate']
lb_2026_final = lb_2026_processed[[c for c in lb_2026_cols if c in lb_2026_processed.columns]]
lb_2026_final.to_csv(os.path.join(SCRIPT_DIR, 'lb_drafted_2026.csv'), index=False)

arm_train = lb_training_data['arm_length_inches'].notna().sum()
arm_test = lb_testing_data['arm_length_inches'].notna().sum()
arm_2026 = lb_2026_final['arm_length_inches'].notna().sum()
print(f'\nSaved lb_training.csv: {len(lb_training_data)} (2015-2023)')
print(f'Saved lb_testing.csv: {len(lb_testing_data)} (2024-2026)')
print(f'Saved lb_drafted_2026.csv: {len(lb_2026_final)}')
print(f'Arm length coverage: Training {arm_train}/{len(lb_training_data)}, Testing {arm_test}/{len(lb_testing_data)}, 2026 {arm_2026}/{len(lb_2026_final)}')
print(f'Columns: {list(lb_training_data.columns)}')
