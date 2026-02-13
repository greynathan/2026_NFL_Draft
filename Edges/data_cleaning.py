import pandas as pd
import os

# Load the data
nfl_combine_data = pd.read_csv('../data/raw/nfl_combine_2010_to_2023.csv')

# Load defensive stats data (use processed file if available, otherwise try individual files)
if os.path.exists('../data/processed/defensive_stats_2016_to_2025.csv'):
    defensive_stats_data = pd.read_csv('../data/processed/defensive_stats_2016_to_2025.csv')
    # Filter to 2016-2022 to match original behavior
    defensive_stats_data = defensive_stats_data[defensive_stats_data['Season'].between(2016, 2022)]
else:
    # Try loading individual files if processed file doesn't exist
    defensive_stats_files = []
    for year in range(2016, 2023):
        file_path = f'../data/raw/defensive_stats_2016_to_2022/download-{year-2003}.csv'
        if os.path.exists(file_path):
            defensive_stats_files.append(pd.read_csv(file_path))
    if defensive_stats_files:
        defensive_stats_data = pd.concat(defensive_stats_files, ignore_index=True)
    else:
        print("Warning: No defensive stats files found. Continuing without defensive stats.")
        defensive_stats_data = pd.DataFrame()

# PFF data: pass rush and run defense (2014-2025).
# - Load ALL positions (no filter). We match to our edge list by Player + School + Year in add_pff_data,
#   so only edge-list players get PFF stats attached; position is irrelevant for who we match.
# - When the same player/school/year appears under multiple PFF positions (e.g. ED and LB), we keep one row:
#   prefer ED > DE > LB > other (so edge designation wins when present).
PFF_PASS_RUSH_DIR = '../data/raw/pff/Pass_Rush'
PFF_RUN_DEFENSE_DIR = '../data/raw/pff/Run_Defense'
POSITION_PRIORITY = {'ED': 0, 'DE': 1, 'LB': 2}  # others get 99 and sort last

# Load all PFF pass rush data (no position filter)
pff_files = []
for year in range(2014, 2026):
    file_path = os.path.join(PFF_PASS_RUSH_DIR, f'{year}_pass_rush_summary.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        pff_cols = ['player', 'team_name', 'position', 'true_pass_set_pass_rush_win_rate',
                    'pass_rush_win_rate', 'snap_counts_pass_rush']
        df_subset = df[[c for c in pff_cols if c in df.columns]].copy()
        if 'player' not in df_subset.columns or 'true_pass_set_pass_rush_win_rate' not in df_subset.columns:
            continue
        if 'position' not in df_subset.columns:
            df_subset['position'] = None
        df_subset['Year'] = year
        df_subset = df_subset.rename(columns={
            'player': 'Player',
            'team_name': 'School'
        })
        pff_files.append(df_subset)
        print(f'Loaded PFF pass rush for {year}: {len(df_subset)} players')

# Combine and dedupe by Player/School/Year (when multiple positions, prefer ED > DE > LB)
if not pff_files:
    pff_data = pd.DataFrame(columns=['Player', 'School', 'Year', 'true_pass_set_pass_rush_win_rate',
                                     'pass_rush_win_rate', 'snap_counts_pass_rush'])
    print('No PFF pass rush files found; PFF columns will be empty.')
else:
    pff_data = pd.concat(pff_files, ignore_index=True)
    pff_data['_pos_order'] = pff_data['position'].map(POSITION_PRIORITY).fillna(99)
    pff_data = pff_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    pff_data = pff_data.drop(columns=['_pos_order', 'position'], errors='ignore')
    print(f'Total PFF pass rush records (after dedup): {len(pff_data)}')

# Load all PFF run defense data (no position filter)
run_defense_files = []
for year in range(2014, 2026):
    file_path = os.path.join(PFF_RUN_DEFENSE_DIR, f'{year}_run_defense_summary.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        run_cols = ['player', 'team_name', 'position', 'stop_percent']
        df_subset = df[[c for c in run_cols if c in df.columns]].copy()
        if 'stop_percent' not in df_subset.columns:
            continue
        if 'position' not in df_subset.columns:
            df_subset['position'] = None
        df_subset['Year'] = year
        df_subset = df_subset.rename(columns={
            'player': 'Player',
            'team_name': 'School'
        })
        run_defense_files.append(df_subset)
        print(f'Loaded PFF run defense for {year}: {len(df_subset)} players')

if run_defense_files:
    run_defense_data = pd.concat(run_defense_files, ignore_index=True)
    run_defense_data['stop_percent'] = pd.to_numeric(run_defense_data['stop_percent'], errors='coerce')
    run_defense_data['_pos_order'] = run_defense_data['position'].map(POSITION_PRIORITY).fillna(99)
    run_defense_data = run_defense_data.sort_values('_pos_order').drop_duplicates(subset=['Player', 'School', 'Year'], keep='first')
    run_defense_data = run_defense_data.drop(columns=['_pos_order', 'position'], errors='ignore')
    # Merge stop_percent into pff_data on Player, School, Year (left merge to keep all pass rush rows)
    pff_data = pff_data.merge(
        run_defense_data[['Player', 'School', 'Year', 'stop_percent']],
        on=['Player', 'School', 'Year'],
        how='left'
    )
    print(f'Merged run defense stop_percent into PFF data; columns: {list(pff_data.columns)}')
else:
    pff_data['stop_percent'] = None
    print('No run defense files found; stop_percent will be empty.')

# Load RAS (Raw Athletic Score) data for edges
ras_df = pd.read_csv('../data/raw/ras.csv')
ras_df = ras_df[ras_df['Pos'].isin(['DE', 'EDGE'])].copy()
ras_df['RAS'] = pd.to_numeric(ras_df['RAS'], errors='coerce')
ras_df['Year'] = ras_df['Year'].astype(int)
ras_edges = ras_df[['Name', 'Year', 'RAS', 'College']].drop_duplicates(subset=['Name', 'Year'])
print(f'Loaded RAS data: {len(ras_edges)} records')

# Load MockDraftable arm length (scraped for our training/testing/2026 players)
arm_length_path = '../data/raw/mockdraftable_edge_arm_length.csv'
if os.path.exists(arm_length_path):
    arm_length_df = pd.read_csv(arm_length_path)
    arm_length_df['Year'] = arm_length_df['Year'].astype(int)
    # Keep one row per (Player, Year) - take first if duplicates
    arm_length_df = arm_length_df.drop_duplicates(subset=['Player', 'Year'], keep='first')
    arm_length_df = arm_length_df[['Player', 'Year', 'arm_length_inches']].copy()
    arm_length_df['arm_length_inches'] = pd.to_numeric(arm_length_df['arm_length_inches'], errors='coerce')
    print(f'Loaded arm length data: {len(arm_length_df)} records ({arm_length_df["arm_length_inches"].notna().sum()} with values)')
else:
    arm_length_df = pd.DataFrame(columns=['Player', 'Year', 'arm_length_inches'])
    print('No mockdraftable_edge_arm_length.csv found; arm_length_inches will be empty.')

# Clean the data
# Only defensive ends 
nfl_combine_data_de_only = nfl_combine_data[(nfl_combine_data['Pos'] == 'DE') | (nfl_combine_data['Pos'] == 'EDGE')]
print(nfl_combine_data_de_only.head())

# Update split: training (2015-2023)
edge_training_data = nfl_combine_data_de_only[nfl_combine_data_de_only['Year'].between(2015, 2023)].copy()

# Load testing data from drafted edges CSVs (2024-2026)
edges_2024 = pd.read_csv('edges_drafted_2024.csv')
edges_2025 = pd.read_csv('edges_drafted_2025.csv')
edges_2026 = pd.read_csv('edges_drafted_2026.csv')
edge_testing_data = pd.concat([edges_2024, edges_2025, edges_2026], ignore_index=True)

# Ensure Year column is int
edge_testing_data['Year'] = edge_testing_data['Year'].astype(int)

# Add Drafted column (all are drafted)
edge_testing_data['Drafted'] = True

# Drop old stats columns and calculated fields that will be recalculated
cols_to_drop_testing = ['Sacks_cumulative', 'TFL_cumulative', 'QB_Hurry_cumulative',
                        'Sacks_final_season', 'TFL_final_season', 'QB_Hurry_final_season',
                        'speed_score', 'explosive_score', 'agility_score']
edge_testing_data = edge_testing_data.drop(columns=cols_to_drop_testing, errors='ignore')

# Process edges_2026 separately to update edges_drafted_2026.csv
edges_2026_processed = edges_2026.copy()
edges_2026_processed['Year'] = edges_2026_processed['Year'].astype(int)
edges_2026_processed['Drafted'] = True
edges_2026_processed = edges_2026_processed.drop(columns=cols_to_drop_testing, errors='ignore')

# Ensure column order matches training data structure
# Training columns: Year, Player, Pos, School, Height, Weight, 40yd, Vertical, Bench, Broad Jump, 3Cone, Shuttle, Drafted, Round, Pick, true_pass_set_pass_rush_win_rate, pass_rush_win_rate, snap_counts_pass_rush


def get_college_stats(combine_df, defensive_stats_df):
    """
    Add Sacks, TFL, and QB Hurry columns (cumulative and final season) by matching
    combine players to defensive_stats. Matches on Player name + School/Team.
    """
    # Pivot defensive stats: filter to SACKS, TFL, QB HUR and reshape to wide
    stats_to_keep = ['SACKS', 'TFL', 'QB HUR']
    stats_filtered = defensive_stats_df[
        defensive_stats_df['StatType'].isin(stats_to_keep)
    ].copy()
    stats_filtered['Stat'] = pd.to_numeric(stats_filtered['Stat'], errors='coerce').fillna(0)

    # Pivot so each row is (Season, Player, Team) with SACKS, TFL, QB HUR columns
    stats_pivot = stats_filtered.pivot_table(
        index=['Season', 'Player', 'Team'],
        columns='StatType',
        values='Stat',
        aggfunc='sum'
    ).reset_index()

    # Rename QB HUR for consistency
    if 'QB HUR' in stats_pivot.columns:
        stats_pivot = stats_pivot.rename(columns={'QB HUR': 'QB_HUR'})

    # School name mapping: map variants to canonical form for matching
    # (combine uses abbreviations; defensive stats uses full names)
    school_alias = {
        'Ole Miss': 'Mississippi',
        'Miami (FL)': 'Miami',
        'Southern California': 'USC',
        'Central Florida': 'UCF',
        'Brigham Young': 'BYU',
        'Ohio St.': 'Ohio State',
        'Florida St.': 'Florida State',
        'Kansas St.': 'Kansas State',
        'Iowa St.': 'Iowa State',
        'Oklahoma St.': 'Oklahoma State',
        'Penn St.': 'Penn State',
        'San Diego St.': 'San Diego State',
        'San Jose St.': 'San José State',
        'Boston Col.': 'Boston College',
        'Alabama-Birmingham': 'UAB',
        'Tenn-Chattanooga': 'Chattanooga',
        'Arizona St.': 'Arizona State',
        'Michigan St.': 'Michigan State',
        'Mississippi St.': 'Mississippi State',
        'West Virginia': 'West Virginia',
    }

    def normalize_school(name):
        return school_alias.get(name, name) if pd.notna(name) else name

    stats_pivot['School_normalized'] = stats_pivot['Team'].apply(normalize_school)
    combine_df = combine_df.copy()
    combine_df['School_normalized'] = combine_df['School'].apply(normalize_school)

    def lookup_stats(row):
        draft_year = int(row['Year'])
        final_season = draft_year - 1  # last college season before draft
        player = row['Player']
        school = row['School_normalized']

        # Match player + school (use normalized names for both datasets)
        mask = (
            (stats_pivot['Player'] == player) &
            (stats_pivot['School_normalized'] == school)
        )

        player_stats = stats_pivot.loc[mask]
        if player_stats.empty:
            return pd.Series({
                'Sacks_cumulative': None, 'TFL_cumulative': None, 'QB_Hurry_cumulative': None,
                'Sacks_final_season': None, 'TFL_final_season': None, 'QB_Hurry_final_season': None,
            })

        # Cumulative: all seasons through final_season
        cumulative = player_stats[player_stats['Season'] <= final_season]
        sacks_cum = cumulative['SACKS'].sum() if 'SACKS' in cumulative.columns else 0
        tfl_cum = cumulative['TFL'].sum() if 'TFL' in cumulative.columns else 0
        qbh_cum = cumulative['QB_HUR'].sum() if 'QB_HUR' in cumulative.columns else 0

        # Final season only
        final = player_stats[player_stats['Season'] == final_season]
        sacks_final = final['SACKS'].sum() if 'SACKS' in final.columns else 0
        tfl_final = final['TFL'].sum() if 'TFL' in final.columns else 0
        qbh_final = final['QB_HUR'].sum() if 'QB_HUR' in final.columns else 0

        return pd.Series({
            'Sacks_cumulative': sacks_cum, 'TFL_cumulative': tfl_cum, 'QB_Hurry_cumulative': qbh_cum,
            'Sacks_final_season': sacks_final, 'TFL_final_season': tfl_final, 'QB_Hurry_final_season': qbh_final,
        })

    stats_cols = combine_df.apply(lookup_stats, axis=1)
    for col in stats_cols.columns:
        combine_df[col] = stats_cols[col]

    return combine_df.drop(columns=['School_normalized'], errors='ignore')


def add_pff_data(combine_df, pff_df):
    """
    Add PFF pass rush data by matching on Player name + School + Year.
    PFF Year represents the college season, so for a player drafted in Year Y,
    we match to PFF data from Year Y-1 (their final college season).
    """
    combine_df = combine_df.copy()
    
    # School name normalization for PFF data (PFF uses uppercase abbreviations)
    def normalize_pff_school(name):
        if pd.isna(name):
            return name
        name = str(name).strip().upper()
        # Handle common PFF naming conventions (uppercase abbreviations)
        school_mapping = {
            'OHIO STATE': 'Ohio State',
            'OHIO ST': 'Ohio State',
            'FLORIDA ST': 'Florida State',
            'FLORIDA STATE': 'Florida State',
            'KANSAS ST': 'Kansas State',
            'KANSAS STATE': 'Kansas State',
            'IOWA ST': 'Iowa State',
            'IOWA STATE': 'Iowa State',
            'OKLAHOMA ST': 'Oklahoma State',
            'OKLAHOMA STATE': 'Oklahoma State',
            'PENN ST': 'Penn State',
            'PENN STATE': 'Penn State',
            'SAN DIEGO ST': 'San Diego State',
            'S DIEGO ST': 'San Diego State',
            'SAN DIEGO STATE': 'San Diego State',
            'SAN JOSE ST': 'San Jose State',
            'S JOSE ST': 'San Jose State',
            'SAN JOSE STATE': 'San Jose State',
            'MISSISSIPPI ST': 'Mississippi State',
            'MISS STATE': 'Mississippi State',
            'MISSISSIPPI STATE': 'Mississippi State',
            'MICHIGAN ST': 'Michigan State',
            'MICHIGAN STATE': 'Michigan State',
            'NORTH CAROLINA ST': 'North Carolina State',
            'NC STATE': 'North Carolina State',
            'NORTH CAROLINA': 'North Carolina',
            'SOUTH CAROLINA': 'South Carolina',
            'NWESTERN': 'Northwestern',
            'NORTHWESTERN': 'Northwestern',
            'SOUTHERN CAL': 'USC',
            'SOUTHERN CALIFORNIA': 'USC',
            'CENTRAL FLORIDA': 'UCF',
            'UCF': 'UCF',  # Handle UCF directly
            'BRIGHAM YOUNG': 'BYU',
            'MIAMI (FL)': 'Miami',
            'MIAMI FL': 'Miami',
            'MIAMI': 'Miami',
            'MIAMI OH': 'Miami (OH)',  # Different school - Miami University (Ohio)
            'OLE MISS': 'Mississippi',
            'ALABAMA-BIRMINGHAM': 'UAB',
            'UAB': 'UAB',
            'TENN-CHATTANOOGA': 'Chattanooga',
            'C MICHIGAN': 'Central Michigan',
            'CENTRAL MICHIGAN': 'Central Michigan',
            'W MICHIGAN': 'Western Michigan',
            'WESTERN MICHIGAN': 'Western Michigan',
            'E MICHIGAN': 'Eastern Michigan',
            'EASTERN MICHIGAN': 'Eastern Michigan',
            'FRESNO ST': 'Fresno State',
            'FRESNO STATE': 'Fresno State',
            'BOISE ST': 'Boise State',
            'BOISE STATE': 'Boise State',
            'ARIZONA ST': 'Arizona State',
            'ARIZONA STATE': 'Arizona State',
            'OREGON ST': 'Oregon State',
            'OREGON STATE': 'Oregon State',
            'COLORADO ST': 'Colorado State',
            'COLORADO STATE': 'Colorado State',
            'UTAH ST': 'Utah State',
            'UTAH STATE': 'Utah State',
            'WYOMING': 'Wyoming',
            'UNLV': 'UNLV',
            'ALABAMA': 'Alabama',
            'ARKANSAS': 'Arkansas',
            'COLORADO': 'Colorado',
            'KENTUCKY': 'Kentucky',
            'UCLA': 'UCLA',
            'LSU': 'LSU',
            'TCU': 'TCU',
            'USC': 'USC',
            'SOUTHERN CAL': 'USC',
            'SOUTHERN CALIFORNIA': 'USC',
            'WASHINGTON STATE': 'Washington State',
            'WSU': 'Washington State',
            'WASH STATE': 'Washington State',
            'COLORADO STATE': 'Colorado State',
            'COLO STATE': 'Colorado State',
            'BOSTON COL': 'Boston College',
            'BOSTON COLLEGE': 'Boston College',
            'VA TECH': 'Virginia Tech',
            'VIRGINIA TECH': 'Virginia Tech',
            'TEXAS ST': 'Texas State',
            'TEXAS STATE': 'Texas State',
            'LA TECH': 'Louisiana Tech',
            'LOUISIANA TECH': 'Louisiana Tech',
            'OKLA STATE': 'Oklahoma State',
            'OKLAHOMA STATE': 'Oklahoma State',
            'OKLAHOMA': 'Oklahoma',  # University of Oklahoma (OU). Oklahoma State is OKLAHOMA ST / OKLAHOMA STATE
            'NC STATE': 'North Carolina State',
            'NORTH CAROLINA STATE': 'North Carolina State',
            'PENN STATE': 'Penn State',
            'PENN ST': 'Penn State',
            'MICHIGAN ST': 'Michigan State',
            'MICH STATE': 'Michigan State',
            'MICHIGAN STATE': 'Michigan State',
            'APPALACHIAN ST': 'Appalachian State',
            'APPALACHIAN STATE': 'Appalachian State',
            'APP ST': 'Appalachian State',
            'APP STATE': 'Appalachian State',  # PFF abbreviation
            'N CAROLINA': 'North Carolina',   # PFF abbreviation
            'S CAROLINA': 'South Carolina',   # PFF abbreviation
            'FLORIDA ATLANTIC': 'Florida Atlantic',
            'FAU': 'Florida Atlantic',
            'TEXAS SAN ANTONIO': 'Texas-San Antonio',
            'UTSA': 'Texas-San Antonio',
            'TEXAS TECH': 'Texas Tech',  # Different from UTSA
            'TOLEDO': 'Toledo',
            'GA SOUTHRN': 'Georgia Southern',
            'GEORGIA SOUTHERN': 'Georgia Southern',
            'GA TECH': 'Georgia Tech',
            'GA STATE': 'Georgia State',
            'W VIRGINIA': 'West Virginia',
            'WEST VIRGINIA': 'West Virginia',
            'WAKE': 'Wake Forest',  # Wake Forest
            'CAL': 'California',  # California (Berkeley)
            'CALIFORNIA': 'California',
            'FLORIDA': 'Florida',
        }
        # Check if exact match exists
        if name in school_mapping:
            return school_mapping[name]
        # Otherwise try title case (for names not in mapping)
        return name.title()
    
    # Normalize school names in both dataframes
    pff_df_normalized = pff_df.copy()
    pff_df_normalized['School_normalized'] = pff_df_normalized['School'].apply(normalize_pff_school)
    
    # Normalize combine school names (handle common variations)
    def normalize_combine_school(name):
        if pd.isna(name):
            return name
        name = str(name).strip()
        # Use the same school alias mapping from get_college_stats for consistency
        school_alias = {
            'Ole Miss': 'Mississippi',
            'Miami (FL)': 'Miami',
            'Miami': 'Miami',  # Handle both formats
            'Southern California': 'USC',
            'USC': 'USC',
            'UCLA': 'UCLA',
            'Central Florida': 'UCF',
            'UCF': 'UCF',  # Handle both formats
            'Brigham Young': 'BYU',
            'Ohio St.': 'Ohio State',
            'Ohio State': 'Ohio State',  # Handle both formats
            'Florida St.': 'Florida State',
            'Florida State': 'Florida State',  # Handle both formats
            'Kansas St.': 'Kansas State',
            'Kansas State': 'Kansas State',  # Handle both formats
            'Iowa St.': 'Iowa State',
            'Iowa State': 'Iowa State',  # Handle both formats
            'Oklahoma St.': 'Oklahoma State',
            'Oklahoma State': 'Oklahoma State',  # Handle both formats
            'Penn St.': 'Penn State',
            'Penn State': 'Penn State',  # Handle both formats
            'San Diego St.': 'San Diego State',
            'San Diego State': 'San Diego State',  # Handle both formats
            'San Jose St.': 'San Jose State',
            'San Jose State': 'San Jose State',  # Handle both formats
            'Boston Col.': 'Boston College',
            'Boston College': 'Boston College',  # Handle both formats
            'Alabama-Birmingham': 'UAB',
            'Tenn-Chattanooga': 'Chattanooga',
            'Miami (Ohio)': 'Miami (OH)',  # match PFF MIAMI OH
            'Washington State': 'Washington State',
            'Colorado State': 'Colorado State',
            'Northwestern': 'Northwestern',
            'LSU': 'LSU',
            'Virginia Tech': 'Virginia Tech',
            'Texas State': 'Texas State',
            'Louisiana Tech': 'Louisiana Tech',
            'Oklahoma State': 'Oklahoma State',
            'Oklahoma St.': 'Oklahoma State',
            'North Carolina State': 'North Carolina State',
            'NC State': 'North Carolina State',
            'Appalachian State': 'Appalachian State',
            'Appalachian St.': 'Appalachian State',
            'App St.': 'Appalachian State',
            'Florida Atlantic': 'Florida Atlantic',
            'Texas-San Antonio': 'Texas-San Antonio',
            'UTSA': 'Texas-San Antonio',
            'Toledo': 'Toledo',
            'Georgia Southern': 'Georgia Southern',
            'Ga. Southern': 'Georgia Southern',
            'Kentucky': 'Kentucky',
            'TCU': 'TCU',
            'Arizona St.': 'Arizona State',
            'Arizona State': 'Arizona State',
            'Michigan St.': 'Michigan State',
            'Michigan State': 'Michigan State',
            'Mississippi St.': 'Mississippi State',
            'Mississippi State': 'Mississippi State',
            'West Virginia': 'West Virginia',
        }
        return school_alias.get(name, name)
    
    combine_df['School_normalized'] = combine_df['School'].apply(normalize_combine_school)
    
    def normalize_player_name(name):
        """Normalize player name by removing punctuation and extra spaces for matching."""
        import re
        # Convert to uppercase, strip whitespace, remove periods and other punctuation
        normalized = str(name).strip().upper()
        # Remove suffixes like "III", "II", "JR", "SR" etc.
        normalized = re.sub(r'\s+(III|II|JR|SR|JR\.|SR\.)$', '', normalized)
        # Remove periods, commas, hyphens, apostrophes, etc. but keep spaces
        normalized = re.sub(r'[.\',\-]', '', normalized)
        # Normalize multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    # Pre-compute normalized PFF player names for efficient matching
    pff_df_normalized['Player_normalized'] = pff_df_normalized['Player'].apply(normalize_player_name)
    
    def lookup_pff_stats(row):
        draft_year = int(row['Year'])
        final_season = draft_year - 1  # PFF data is for the college season
        player = normalize_player_name(row['Player'])
        school = row['School_normalized']
        
        # Handle nickname/alternate name mappings (use normalized names)
        player_nickname_map = {
            'DEMEIOUN ROBINSON': 'CHOP ROBINSON',  # Also known as Chop Robinson
            'ALVIN DUPREE': 'BUD DUPREE',  # Alvin "Bud" Dupree
            'JAYSON OWEH': 'ODAFE OWEH',  # Odafe "Jayson" Oweh
            'OWAMAGBE ODIGHIZUWA': 'OWA ODIGHIZUWA',  # Owamagbe "Owa" Odighizuwa
            'OGBONNIA OKORONKWO': 'OGBO OKORONKWO',  # Ogbonnia "Ogbo" Okoronkwo
            'JOHNNY NEWTON': 'JERZHAN NEWTON',  # Jer'Zhan "Johnny" Newton (DT, Illinois)
            'DAVON GAUDCHAUX': 'DAVON GODCHAUX',  # common misspelling: Gaudchaux -> Godchaux (LSU DT)
            'EARNEST BROWN': 'EARNEST BROWN IV',  # PFF uses "Earnest Brown IV" (Northwestern)
            'ZACHARY CARTER': 'ZACH CARTER',  # PFF uses "Zach Carter" (Florida)
            'JOSHUA PASCHAL': 'JOSH PASCHAL',  # PFF uses "Josh Paschal" (Kentucky)
            'AMARÉ BARNO': 'AMARE BARNO',  # PFF uses "Amare Barno" (Virginia Tech)
        }
        player_to_search = player_nickname_map.get(player, player)
        
        # Match on Player + School + Year (final college season)
        # Use normalized player names (punctuation removed)
        # Try exact match first
        mask = (
            (pff_df_normalized['Player_normalized'] == player_to_search) &
            (pff_df_normalized['School_normalized'] == school) &
            (pff_df_normalized['Year'] == final_season)
        )
        
        # If no match and we have a nickname, try the original name too
        if not mask.any() and player_to_search != player:
            mask = (
                (pff_df_normalized['Player_normalized'] == player) &
                (pff_df_normalized['School_normalized'] == school) &
                (pff_df_normalized['Year'] == final_season)
            )
        
        pff_stats = pff_df_normalized.loc[mask]
        if pff_stats.empty:
            out = {
                'true_pass_set_pass_rush_win_rate': None,
                'pass_rush_win_rate': None,
                'snap_counts_pass_rush': None,
            }
            if 'stop_percent' in pff_df_normalized.columns:
                out['stop_percent'] = None
            return pd.Series(out)
        
        # Take the first match if multiple (shouldn't happen, but just in case)
        pff_row = pff_stats.iloc[0]
        out = {
            'true_pass_set_pass_rush_win_rate': pff_row['true_pass_set_pass_rush_win_rate'],
            'pass_rush_win_rate': pff_row['pass_rush_win_rate'],
            'snap_counts_pass_rush': pff_row['snap_counts_pass_rush'],
        }
        if 'stop_percent' in pff_df_normalized.columns:
            out['stop_percent'] = pff_row['stop_percent']
        return pd.Series(out)
    
    pff_cols = combine_df.apply(lookup_pff_stats, axis=1)
    for col in pff_cols.columns:
        combine_df[col] = pff_cols[col]
    
    return combine_df.drop(columns=['School_normalized'], errors='ignore')


def add_ras_data(combine_df, ras_df):
    """
    Add RAS (Raw Athletic Score) data by matching on Player name + Year.
    Uses normalized names and school mappings similar to PFF matching.
    """
    combine_df = combine_df.copy()
    
    def normalize_player_name(name):
        """Normalize player name by removing punctuation and suffixes."""
        import re
        normalized = str(name).strip().upper()
        # Remove suffixes like "III", "II", "JR", "SR" etc.
        normalized = re.sub(r'\s+(III|II|JR|SR|JR\.|SR\.)$', '', normalized)
        # Remove periods, commas, hyphens, apostrophes, etc. but keep spaces
        normalized = re.sub(r'[.\',\-]', '', normalized)
        # Normalize multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def normalize_ras_school(name):
        """Normalize RAS school name to match combine data format."""
        if pd.isna(name):
            return name
        name = str(name).strip()
        # RAS school mapping (RAS uses full school names, but may have variations)
        school_mapping = {
            'Miami (FL)': 'Miami',
            'Miami': 'Miami',
            'Miami (Ohio)': 'Miami (OH)',
            'Boston Col.': 'Boston College',
            'Boston College': 'Boston College',
            'Southern California': 'USC',
            'USC': 'USC',
            'UCLA': 'UCLA',
            'Central Florida': 'UCF',
            'UCF': 'UCF',
            'Brigham Young': 'BYU',
            'BYU': 'BYU',
            'Ole Miss': 'Mississippi',
            'Mississippi': 'Mississippi',
            'Ohio St.': 'Ohio State',
            'Ohio State': 'Ohio State',
            'Florida St.': 'Florida State',
            'Florida State': 'Florida State',
            'Oklahoma St.': 'Oklahoma State',
            'Oklahoma State': 'Oklahoma State',
            'Penn St.': 'Penn State',
            'Penn State': 'Penn State',
            'Michigan St.': 'Michigan State',
            'Michigan State': 'Michigan State',
            'North Carolina State': 'North Carolina State',
            'NC State': 'North Carolina State',
            'Virginia Tech': 'Virginia Tech',
            'Texas State': 'Texas State',
            'Louisiana Tech': 'Louisiana Tech',
            'Appalachian State': 'Appalachian State',
            'Florida Atlantic': 'Florida Atlantic',
            'Texas-San Antonio': 'Texas-San Antonio',
            'UTSA': 'Texas-San Antonio',
            'Toledo': 'Toledo',
            'Georgia Southern': 'Georgia Southern',
            'Kentucky': 'Kentucky',
            'TCU': 'TCU',
            'Texas Christian': 'TCU',  # RAS uses full name
            'Louisiana State': 'LSU',  # RAS uses full name
            'LSU': 'LSU',
            'San Diego St.': 'San Diego State',
            'San Diego State': 'San Diego State',
            'San Jose St.': 'San Jose State',
            'San Jose State': 'San Jose State',
            'Kansas St.': 'Kansas State',
            'Kansas State': 'Kansas State',
            'Iowa St.': 'Iowa State',
            'Iowa State': 'Iowa State',
            'Arizona St.': 'Arizona State',
            'Arizona State': 'Arizona State',
            'Mississippi St.': 'Mississippi State',
            'Mississippi State': 'Mississippi State',
            'West Virginia': 'West Virginia',
        }
        return school_mapping.get(name, name)
    
    def normalize_combine_school_for_ras(name):
        """Normalize combine school name for RAS matching."""
        if pd.isna(name):
            return name
        name = str(name).strip()
        # Use same mapping as normalize_combine_school
        school_alias = {
            'Ole Miss': 'Mississippi',
            'Miami (FL)': 'Miami',
            'Miami': 'Miami',
            'Southern California': 'USC',
            'USC': 'USC',
            'UCLA': 'UCLA',
            'Central Florida': 'UCF',
            'UCF': 'UCF',
            'Brigham Young': 'BYU',
            'Ohio St.': 'Ohio State',
            'Ohio State': 'Ohio State',
            'Florida St.': 'Florida State',
            'Florida State': 'Florida State',
            'Kansas St.': 'Kansas State',
            'Kansas State': 'Kansas State',
            'Iowa St.': 'Iowa State',
            'Iowa State': 'Iowa State',
            'Oklahoma St.': 'Oklahoma State',
            'Oklahoma State': 'Oklahoma State',
            'Penn St.': 'Penn State',
            'Penn State': 'Penn State',
            'San Diego St.': 'San Diego State',
            'San Diego State': 'San Diego State',
            'San Jose St.': 'San Jose State',
            'San Jose State': 'San Jose State',
            'Boston Col.': 'Boston College',
            'Boston College': 'Boston College',
            'Alabama-Birmingham': 'UAB',
            'Tenn-Chattanooga': 'Chattanooga',
            'Washington State': 'Washington State',
            'Colorado State': 'Colorado State',
            'Northwestern': 'Northwestern',
            'LSU': 'LSU',
            'Virginia Tech': 'Virginia Tech',
            'Texas State': 'Texas State',
            'Louisiana Tech': 'Louisiana Tech',
            'Oklahoma State': 'Oklahoma State',
            'North Carolina State': 'North Carolina State',
            'NC State': 'North Carolina State',
            'Appalachian State': 'Appalachian State',
            'Appalachian St.': 'Appalachian State',
            'App St.': 'Appalachian State',
            'Florida Atlantic': 'Florida Atlantic',
            'Texas-San Antonio': 'Texas-San Antonio',
            'UTSA': 'Texas-San Antonio',
            'Toledo': 'Toledo',
            'Georgia Southern': 'Georgia Southern',
            'Ga. Southern': 'Georgia Southern',
            'Kentucky': 'Kentucky',
            'TCU': 'TCU',
            'Texas Christian': 'TCU',
            'Louisiana State': 'LSU',
            'LSU': 'LSU',
            'Arizona St.': 'Arizona State',
            'Arizona State': 'Arizona State',
            'Michigan St.': 'Michigan State',
            'Michigan State': 'Michigan State',
            'Mississippi St.': 'Mississippi State',
            'Mississippi State': 'Mississippi State',
            'West Virginia': 'West Virginia',
        }
        return school_alias.get(name, name)
    
    # Normalize RAS data
    ras_df_normalized = ras_df.copy()
    ras_df_normalized['Name_normalized'] = ras_df_normalized['Name'].apply(normalize_player_name)
    ras_df_normalized['College_normalized'] = ras_df_normalized['College'].apply(normalize_ras_school)
    
    # Handle nickname/alternate name mappings (use normalized names)
    player_nickname_map = {
        'DEMEIOUN ROBINSON': 'CHOP ROBINSON',
        'ALVIN DUPREE': 'BUD DUPREE',
        'JAYSON OWEH': 'ODAFE OWEH',
        'OWAMAGBE ODIGHIZUWA': 'OWA ODIGHIZUWA',
        'OGBONNIA OKORONKWO': 'OGBO OKORONKWO',
    }
    
    def lookup_ras(row):
        player = normalize_player_name(row['Player'])
        school = normalize_combine_school_for_ras(row['School'])
        year = int(row['Year'])
        
        player_to_search = player_nickname_map.get(player, player)
        
        # Match on normalized Player + normalized School + Year
        mask = (
            (ras_df_normalized['Name_normalized'] == player_to_search) &
            (ras_df_normalized['College_normalized'] == school) &
            (ras_df_normalized['Year'] == year)
        )
        
        # If no match and we have a nickname, try the original name too
        if not mask.any() and player_to_search != player:
            mask = (
                (ras_df_normalized['Name_normalized'] == player) &
                (ras_df_normalized['College_normalized'] == school) &
                (ras_df_normalized['Year'] == year)
            )
        
        ras_match = ras_df_normalized.loc[mask]
        if ras_match.empty:
            return pd.Series({'RAS': None})
        
        # Take the first match if multiple
        return pd.Series({'RAS': ras_match.iloc[0]['RAS']})
    
    ras_cols = combine_df.apply(lookup_ras, axis=1)
    combine_df['RAS'] = ras_cols['RAS']

    return combine_df


def add_arm_length(combine_df, arm_df):
    """
    Add arm_length_inches by left merge on Player + Year.
    """
    combine_df = combine_df.drop(columns=['arm_length_inches'], errors='ignore')
    if arm_df.empty or 'arm_length_inches' not in arm_df.columns:
        combine_df['arm_length_inches'] = None
        return combine_df
    out = combine_df.merge(
        arm_df[['Player', 'Year', 'arm_length_inches']],
        on=['Player', 'Year'],
        how='left'
    )
    return out


# Skip college stats columns (QB_Hurry, TFL, Sacks) - no longer needed
# if not defensive_stats_data.empty:
#     edge_training_data = get_college_stats(edge_training_data, defensive_stats_data)
#     edge_testing_data = get_college_stats(edge_testing_data, defensive_stats_data)
# else:
#     print("Skipping college stats addition - no defensive stats data available")

# Add PFF data
edge_training_data = add_pff_data(edge_training_data, pff_data)
edge_testing_data = add_pff_data(edge_testing_data, pff_data)
edges_2026_processed = add_pff_data(edges_2026_processed, pff_data)

# Add RAS data
edge_training_data = add_ras_data(edge_training_data, ras_edges)
edge_testing_data = add_ras_data(edge_testing_data, ras_edges)
edges_2026_processed = add_ras_data(edges_2026_processed, ras_edges)

# Add arm length (MockDraftable); 2026 will be empty until we scrape
edge_training_data = add_arm_length(edge_training_data, arm_length_df)
edge_testing_data = add_arm_length(edge_testing_data, arm_length_df)
edges_2026_processed = add_arm_length(edges_2026_processed, arm_length_df)

# Drop college stats columns if they exist (no longer needed)
cols_to_drop = ['Sacks_cumulative', 'TFL_cumulative', 'QB_Hurry_cumulative',
                'Sacks_final_season', 'TFL_final_season', 'QB_Hurry_final_season']
edge_training_data = edge_training_data.drop(columns=cols_to_drop, errors='ignore')
edge_testing_data = edge_testing_data.drop(columns=cols_to_drop, errors='ignore')

# Reorder columns to match training data structure
training_cols_order = ['Year', 'Player', 'Pos', 'School', 'Height', 'Weight', '40yd', 'Vertical',
                       'Bench', 'Broad Jump', '3Cone', 'Shuttle', 'Drafted', 'Round', 'Pick',
                       'RAS', 'arm_length_inches', 'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate', 'snap_counts_pass_rush', 'stop_percent']
# Only include columns that exist in the dataframe
edge_testing_data = edge_testing_data[[col for col in training_cols_order if col in edge_testing_data.columns]]
edge_training_data = edge_training_data[[col for col in training_cols_order if col in edge_training_data.columns]]

# Save the data
edge_training_data.to_csv('../data/processed/edge_training.csv', index=False)
edge_testing_data.to_csv('../data/processed/edge_testing.csv', index=False)

# Save updated edges_drafted_2026.csv with PFF data and without old stats
# Reorder to match original CSV structure (Round, Pick, Player, Pos, School, Year, then combine metrics, then RAS, then PFF metrics)
edges_2026_cols_order = ['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight',
                         '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
                         'RAS', 'arm_length_inches', 'true_pass_set_pass_rush_win_rate', 'pass_rush_win_rate', 'snap_counts_pass_rush', 'stop_percent']
edges_2026_final = edges_2026_processed[[col for col in edges_2026_cols_order if col in edges_2026_processed.columns]]
edges_2026_final.to_csv('edges_drafted_2026.csv', index=False)

print(f'\nSaved edge_training.csv: {len(edge_training_data)} players (2015-2023)')
print(f'Saved edge_testing.csv: {len(edge_testing_data)} players (2024-2026)')
print(f'Saved edges_drafted_2026.csv: {len(edges_2026_final)} players (updated with PFF data, old stats removed)')
print(f'\nColumns in training data: {list(edge_training_data.columns)}')

# Validation: PFF and run defense mapping consistency
train_pr = edge_training_data['pass_rush_win_rate'].notna().sum()
train_stop = edge_training_data['stop_percent'].notna().sum()
test_pr = edge_testing_data['pass_rush_win_rate'].notna().sum()
test_stop = edge_testing_data['stop_percent'].notna().sum()
arm_train = edge_training_data['arm_length_inches'].notna().sum()
arm_test = edge_testing_data['arm_length_inches'].notna().sum()
arm_2026 = edges_2026_final['arm_length_inches'].notna().sum()
print(f'\nArm length coverage: Training {arm_train}/{len(edge_training_data)}, Testing {arm_test}/{len(edge_testing_data)}, 2026 {arm_2026}/{len(edges_2026_final)}')
print(f'\nPFF/Run defense coverage:')
print(f'  Training: pass_rush_win_rate {train_pr}/{len(edge_training_data)}, stop_percent {train_stop}/{len(edge_training_data)}')
print(f'  Testing:  pass_rush_win_rate {test_pr}/{len(edge_testing_data)}, stop_percent {test_stop}/{len(edge_testing_data)}')
# Rows with pass rush but no stop_percent are expected (run defense is ED-only; pass rush includes more)
inconsistent = ((edge_training_data['pass_rush_win_rate'].notna()) & (edge_training_data['stop_percent'].isna())).sum()
print(f'  Training: has pass_rush but no stop_percent (expected for some): {inconsistent}')
