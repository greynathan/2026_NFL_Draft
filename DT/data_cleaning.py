"""
DT data cleaning: same metrics as Edges.
- PFF Pass_Rush + Run_Defense (all positions; match by name+school+year; prefer DI when dedup).
- RAS (Pos == 'DT').
- Optional arm length (mockdraftable_dt_arm_length.csv).
- No college stats (Sacks/TFL/QB Hurry).
Output: dt_training.csv, dt_testing.csv, updated dt_drafted_2026.csv.
Run from DT/ directory.
"""
import pandas as pd
import os
import re

# Load combine, DT only
nfl_combine_data = pd.read_csv('../data/raw/nfl_combine_2010_to_2023.csv')
nfl_combine_data_dt = nfl_combine_data[nfl_combine_data['Pos'] == 'DT'].copy()
print(nfl_combine_data_dt.head())

# PFF: same as Edges (all positions; dedup by Player/School/Year, prefer DI for DT)
PFF_PASS_RUSH_DIR = '../data/raw/pff/Pass_Rush'
PFF_RUN_DEFENSE_DIR = '../data/raw/pff/Run_Defense'
POSITION_PRIORITY = {'DI': 0}  # interior first; others 99

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

run_defense_files = []
for year in range(2014, 2026):
    path = os.path.join(PFF_RUN_DEFENSE_DIR, f'{year}_run_defense_summary.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        cols = ['player', 'team_name', 'position', 'stop_percent']
        sub = df[[c for c in cols if c in df.columns]].copy()
        if 'stop_percent' not in sub.columns:
            continue
        if 'position' not in sub.columns:
            sub['position'] = None
        sub['Year'] = year
        sub = sub.rename(columns={'player': 'Player', 'team_name': 'School'})
        run_defense_files.append(sub)
        print(f'Loaded PFF run defense {year}: {len(sub)} players')

if run_defense_files:
    run_defense_data = pd.concat(run_defense_files, ignore_index=True)
    run_defense_data['stop_percent'] = pd.to_numeric(run_defense_data['stop_percent'], errors='coerce')
    run_defense_data['_pos_order'] = run_defense_data['position'].map(POSITION_PRIORITY).fillna(99)
    run_defense_data = run_defense_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    run_defense_data = run_defense_data.drop(columns=['_pos_order', 'position'], errors='ignore')
    pff_data = pff_data.merge(run_defense_data[['Player', 'School', 'Year', 'stop_percent']],
                              on=['Player', 'School', 'Year'], how='left')
    print(f'Merged run defense into PFF; columns: {list(pff_data.columns)}')
else:
    pff_data['stop_percent'] = None
    print('No run defense files found.')

# RAS for DT
ras_df = pd.read_csv('../data/raw/ras.csv')
ras_df = ras_df[ras_df['Pos'] == 'DT'].copy()
ras_df['RAS'] = pd.to_numeric(ras_df['RAS'], errors='coerce')
ras_df['Year'] = ras_df['Year'].astype(int)
ras_dt = ras_df[['Name', 'Year', 'RAS', 'College']].drop_duplicates(subset=['Name', 'Year'])
print(f'RAS DT records: {len(ras_dt)}')

# Arm length (optional)
arm_path = '../data/raw/mockdraftable_dt_arm_length.csv'
if os.path.exists(arm_path):
    arm_length_df = pd.read_csv(arm_path)
    arm_length_df['Year'] = arm_length_df['Year'].astype(int)
    arm_length_df = arm_length_df.drop_duplicates(subset=['Player', 'Year'], keep='first')
    arm_length_df = arm_length_df[['Player', 'Year', 'arm_length_inches']].copy()
    arm_length_df['arm_length_inches'] = pd.to_numeric(arm_length_df['arm_length_inches'], errors='coerce')
    print(f'Arm length DT: {len(arm_length_df)} records')
else:
    arm_length_df = pd.DataFrame(columns=['Player', 'Year', 'arm_length_inches'])
    print('No mockdraftable_dt_arm_length.csv; arm_length_inches will be empty.')

# Splits: training 2015-2023, testing 2024-2026 from drafted CSVs
dt_training_data = nfl_combine_data_dt[nfl_combine_data_dt['Year'].between(2015, 2023)].copy()

dt_2024 = pd.read_csv('dt_drafted_2024.csv')
dt_2025 = pd.read_csv('dt_drafted_2025.csv')
dt_2026 = pd.read_csv('dt_drafted_2026.csv')
dt_testing_data = pd.concat([dt_2024, dt_2025, dt_2026], ignore_index=True)
dt_testing_data['Year'] = dt_testing_data['Year'].astype(int)
dt_testing_data['Drafted'] = True

cols_drop = ['Sacks_cumulative', 'TFL_cumulative', 'QB_Hurry_cumulative',
             'Sacks_final_season', 'TFL_final_season', 'QB_Hurry_final_season',
             'speed_score', 'explosive_score', 'agility_score']
dt_testing_data = dt_testing_data.drop(columns=cols_drop, errors='ignore')

dt_2026_processed = dt_2026.copy()
dt_2026_processed['Year'] = dt_2026_processed['Year'].astype(int)
dt_2026_processed['Drafted'] = True
dt_2026_processed = dt_2026_processed.drop(columns=cols_drop, errors='ignore')


def normalize_player_name(name):
    s = str(name).strip().upper()
    s = re.sub(r'\s+(III|II|JR|SR|JR\.|SR\.)$', '', s)
    s = re.sub(r'[.\',\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# PFF school mapping (same as Edges)
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
        'Washington State': 'Washington State', 'Colorado State': 'Colorado State',
        'Northwestern': 'Northwestern', 'LSU': 'LSU', 'Virginia Tech': 'Virginia Tech',
        'Texas State': 'Texas State', 'Louisiana Tech': 'Louisiana Tech',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
        'Appalachian State': 'Appalachian State', 'Appalachian St.': 'Appalachian State', 'App St.': 'Appalachian State',
        'Florida Atlantic': 'Florida Atlantic', 'Texas-San Antonio': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio',
        'Toledo': 'Toledo', 'Georgia Southern': 'Georgia Southern', 'Ga. Southern': 'Georgia Southern',
        'Kentucky': 'Kentucky', 'TCU': 'TCU',
        'Arizona St.': 'Arizona State', 'Arizona State': 'Arizona State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
        'West Virginia': 'West Virginia',
    }
    return alias.get(name, name)


def add_pff_data(combine_df, pff_df):
    combine_df = combine_df.copy()
    pff_n = pff_df.copy()
    pff_n['School_normalized'] = pff_n['School'].apply(normalize_pff_school)
    pff_n['Player_normalized'] = pff_n['Player'].apply(normalize_player_name)
    combine_df['School_normalized'] = combine_df['School'].apply(normalize_combine_school)

    # Alternate spellings / nicknames: combine name -> PFF normalized name
    player_nickname_map = {
        'JOHNNY NEWTON': 'JERZHAN NEWTON',   # Jer'Zhan "Johnny" Newton (Illinois)
        'DAVON GAUDCHAUX': 'DAVON GODCHAUX', # common misspelling: Gaudchaux -> Godchaux (LSU)
        'JOSHUA FRAZIER': 'JOSH FRAZIER',    # PFF uses "Josh Frazier" (Alabama)
        'THOMAS BOOKER': 'THOMAS BOOKER IV',  # PFF uses "Thomas Booker IV" (Stanford)
    }
    # (player_normalized, combine_school_normalized) -> PFF school to use (final season elsewhere)
    player_school_pff_override = {
        ('DJ JONES', 'Akron'): 'Mississippi',       # Ole Miss for final season (2016)
        ('TAYLOR UPSHAW', 'Michigan'): 'Arizona',  # transferred to Arizona for 2023
    }
    # (player_normalized, school_normalized, draft_year) -> PFF year to use (e.g. opt-out: use prior year)
    player_school_year_override = {
        ('TYLER SHELVIN', 'LSU', 2021): 2019,  # opted out 2020; use 2019
        ('CALEB BANKS', 'Florida', 2026): 2024,  # injury in 2025; use 2024 (healthy season)
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
            out = {'true_pass_set_pass_rush_win_rate': None, 'pass_rush_win_rate': None,
                   'snap_counts_pass_rush': None}
            if 'stop_percent' in pff_n.columns:
                out['stop_percent'] = None
            return pd.Series(out)
        r = match.iloc[0]
        out = {'true_pass_set_pass_rush_win_rate': r['true_pass_set_pass_rush_win_rate'],
               'pass_rush_win_rate': r['pass_rush_win_rate'], 'snap_counts_pass_rush': r['snap_counts_pass_rush']}
        if 'stop_percent' in pff_n.columns:
            out['stop_percent'] = r['stop_percent']
        return pd.Series(out)

    res = combine_df.apply(lookup, axis=1)
    for c in res.columns:
        combine_df[c] = res[c]
    return combine_df.drop(columns=['School_normalized'], errors='ignore')


def add_ras_data(combine_df, ras_subset):
    """RAS_subset has Name, Year, RAS, College."""
    combine_df = combine_df.copy()
    ras_n = ras_subset.copy()
    ras_n['Year'] = ras_n['Year'].astype(int)
    ras_n['Name_n'] = ras_n['Name'].apply(normalize_player_name)
    # Map RAS College to same canonical names as normalize_combine_school (for matching)
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
        'West Virginia': 'West Virginia',
        'Georgia Tech': 'Georgia Tech', 'North Carolina': 'North Carolina', 'South Carolina': 'South Carolina',
    }
    ras_n['College_n'] = ras_n['College'].apply(lambda x: ras_school.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)

    # Combine may have different spelling than RAS (e.g. Gaudchaux vs Godchaux)
    ras_name_alias = {'DAVON GAUDCHAUX': 'DAVON GODCHAUX'}

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
    combine_df = combine_df.drop(columns=['arm_length_inches'], errors='ignore')
    if arm_df.empty or 'arm_length_inches' not in arm_df.columns:
        combine_df['arm_length_inches'] = None
        return combine_df
    return combine_df.merge(arm_df[['Player', 'Year', 'arm_length_inches']], on=['Player', 'Year'], how='left')


# Apply PFF, RAS, arm length
dt_training_data = add_pff_data(dt_training_data, pff_data)
dt_testing_data = add_pff_data(dt_testing_data, pff_data)
dt_2026_processed = add_pff_data(dt_2026_processed, pff_data)

dt_training_data = add_ras_data(dt_training_data, ras_dt)
dt_testing_data = add_ras_data(dt_testing_data, ras_dt)
dt_2026_processed = add_ras_data(dt_2026_processed, ras_dt)

dt_training_data = add_arm_length(dt_training_data, arm_length_df)
dt_testing_data = add_arm_length(dt_testing_data, arm_length_df)
dt_2026_processed = add_arm_length(dt_2026_processed, arm_length_df)

# Drop any old college stats columns
dt_training_data = dt_training_data.drop(columns=cols_drop, errors='ignore')
dt_testing_data = dt_testing_data.drop(columns=cols_drop, errors='ignore')

# Column order: same as Edges
training_cols_order = ['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical',
                       'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick',
                       'RAS', 'arm_length_inches', 'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate',
                       'snap_counts_pass_rush', 'stop_percent']
dt_training_data = dt_training_data[[c for c in training_cols_order if c in dt_training_data.columns]]
dt_testing_data = dt_testing_data[[c for c in training_cols_order if c in dt_testing_data.columns]]

dt_training_data.to_csv('../data/processed/dt_training.csv', index=False)
dt_testing_data.to_csv('../data/processed/dt_testing.csv', index=False)

dt_2026_cols = ['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight',
                '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
                'RAS', 'arm_length_inches', 'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate',
                'snap_counts_pass_rush', 'stop_percent']
dt_2026_final = dt_2026_processed[[c for c in dt_2026_cols if c in dt_2026_processed.columns]]
dt_2026_final.to_csv('dt_drafted_2026.csv', index=False)

print(f'\nSaved dt_training.csv: {len(dt_training_data)} (2015-2023)')
print(f'Saved dt_testing.csv: {len(dt_testing_data)} (2024-2026)')
print(f'Saved dt_drafted_2026.csv: {len(dt_2026_final)}')
print(f'Columns: {list(dt_training_data.columns)}')
train_pr = dt_training_data['pass_rush_win_rate'].notna().sum()
train_stop = dt_training_data['stop_percent'].notna().sum()
train_ras = dt_training_data['RAS'].notna().sum()
print(f'RAS coverage training: {train_ras}/{len(dt_training_data)}')
print(f'PFF coverage training: pass_rush_win_rate {train_pr}/{len(dt_training_data)}, stop_percent {train_stop}/{len(dt_training_data)}')
