import pandas as pd

# Load the data
nfl_combine_data = pd.read_csv('data/raw/nfl_combine_2010_to_2023.csv')

# Combine all the defensive stats data into a single dataframe
defensive_stats_data = pd.read_csv('data/processed/defensive_stats_2016_to_2022.csv')

# Clean the data
# Only defensive ends 
nfl_combine_data_dt_only = nfl_combine_data[(nfl_combine_data['Pos'] == 'DT')]
print(nfl_combine_data_dt_only.head())

dt_training_data = nfl_combine_data_dt_only[nfl_combine_data_dt_only['Year'] <= 2020]
dt_testing_data = nfl_combine_data_dt_only[nfl_combine_data_dt_only['Year'] > 2020]


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


# Add college stats columns
dt_training_data = get_college_stats(dt_training_data, defensive_stats_data)
dt_testing_data = get_college_stats(dt_testing_data, defensive_stats_data)

# Save the data
dt_training_data.to_csv('data/processed/dt_training_data.csv', index=False)
dt_testing_data.to_csv('data/processed/dt_testing_data.csv', index=False)
