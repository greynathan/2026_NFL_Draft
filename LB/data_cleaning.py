"""
LB data cleaning: filter combine to ILB/LB/OLB and merge college stats
(Sacks, TFL, QB HUR, PD, SOLO, TOT) from defensive_stats.
"""
import os
import pandas as pd

# Paths: run from project root or from LB/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# Load the data
nfl_combine_data = pd.read_csv(os.path.join(DATA_RAW, 'nfl_combine_2010_to_2023.csv'))
defensive_stats_data = pd.read_csv(os.path.join(DATA_PROCESSED, 'defensive_stats_2016_to_2025.csv'))

# Keep ILB, LB, OLB only
LB_POSITIONS = ['ILB', 'LB', 'OLB']
nfl_combine_data_lb = nfl_combine_data[nfl_combine_data['Pos'].isin(LB_POSITIONS)].copy()
print(f"LB combine rows: {len(nfl_combine_data_lb)} (Pos in {LB_POSITIONS})")
print(nfl_combine_data_lb.head())

lb_training_data = nfl_combine_data_lb[nfl_combine_data_lb['Year'] <= 2020].copy()
lb_testing_data = nfl_combine_data_lb[nfl_combine_data_lb['Year'] > 2020].copy()
print(f"Training: {len(lb_training_data)}, Testing: {len(lb_testing_data)}")


def get_college_stats(combine_df, defensive_stats_df):
    """
    Add college stat columns (cumulative and final season) by matching
    combine players to defensive_stats on Player name + School/Team.
    Stats: Sacks, TFL, QB Hurry, PD, SOLO, TOT.
    """
    stats_to_keep = ['SACKS', 'TFL', 'QB HUR', 'PD', 'SOLO', 'TOT']
    stats_filtered = defensive_stats_df[
        defensive_stats_df['StatType'].isin(stats_to_keep)
    ].copy()
    stats_filtered['Stat'] = pd.to_numeric(stats_filtered['Stat'], errors='coerce').fillna(0)

    stats_pivot = stats_filtered.pivot_table(
        index=['Season', 'Player', 'Team'],
        columns='StatType',
        values='Stat',
        aggfunc='sum'
    ).reset_index()

    if 'QB HUR' in stats_pivot.columns:
        stats_pivot = stats_pivot.rename(columns={'QB HUR': 'QB_HUR'})

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

    stat_cols = ['SACKS', 'TFL', 'QB_HUR', 'PD', 'SOLO', 'TOT']
    out_names = ['Sacks', 'TFL', 'QB_Hurry', 'PD', 'SOLO', 'TOT']

    def safe_sum(df, col):
        if col in df.columns:
            return df[col].sum()
        return 0

    def lookup_stats(row):
        draft_year = int(row['Year'])
        final_season = draft_year - 1
        player = row['Player']
        school = row['School_normalized']

        mask = (
            (stats_pivot['Player'] == player) &
            (stats_pivot['School_normalized'] == school)
        )
        player_stats = stats_pivot.loc[mask]
        if player_stats.empty:
            out = {}
            for name in out_names:
                out[f'{name}_cumulative'] = None
                out[f'{name}_final_season'] = None
            return pd.Series(out)

        cumulative = player_stats[player_stats['Season'] <= final_season]
        final = player_stats[player_stats['Season'] == final_season]

        out = {}
        for i, st in enumerate(stat_cols):
            key = out_names[i]
            out[f'{key}_cumulative'] = safe_sum(cumulative, st)
            out[f'{key}_final_season'] = safe_sum(final, st)
        return pd.Series(out)

    stats_cols = combine_df.apply(lookup_stats, axis=1)
    for col in stats_cols.columns:
        combine_df[col] = stats_cols[col]

    return combine_df.drop(columns=['School_normalized'], errors='ignore')


lb_training_data = get_college_stats(lb_training_data, defensive_stats_data)
lb_testing_data = get_college_stats(lb_testing_data, defensive_stats_data)

# Save
lb_training_data.to_csv(os.path.join(DATA_PROCESSED, 'lb_training_data.csv'), index=False)
lb_testing_data.to_csv(os.path.join(DATA_PROCESSED, 'lb_testing_data.csv'), index=False)
print(f"Saved lb_training_data.csv ({len(lb_training_data)} rows), lb_testing_data.csv ({len(lb_testing_data)} rows)")
