"""
Populate DT drafted CSVs (2024, 2025, 2026) from defensive_stats source.
Fetches sacks, TFL, QB Hurry stats from defensive_stats_2016_to_2025.
Standardizes format to match edges drafted files: float notation, 3 trailing commas.
Run from project root: python DT/populate_drafted_dts.py
"""
import pandas as pd
import os

# Paths (script may run from project root or DT/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_DIR = os.path.join(PROJECT_ROOT, 'DT')
DEFENSIVE_STATS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'defensive_stats_2016_to_2025.csv')

# Player name aliases: draft name -> defensive_stats name (for matching)
PLAYER_ALIAS = {
    'Johnny Newton': "Jer'Zhan Newton",
    'Jowon Briggs': 'Jowon Briggs',  # exact match
}

# School name mapping for defensive_stats Team column
SCHOOL_ALIAS = {
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
    'Michigan St.': 'Michigan State',
}


def load_defensive_stats():
    """Load and pivot defensive stats for SACKS, TFL, QB HUR."""
    df = pd.read_csv(DEFENSIVE_STATS_PATH)
    stats_to_keep = ['SACKS', 'TFL', 'QB HUR']
    filtered = df[df['StatType'].isin(stats_to_keep)].copy()
    filtered['Stat'] = pd.to_numeric(filtered['Stat'], errors='coerce').fillna(0)

    pivot = filtered.pivot_table(
        index=['Season', 'Player', 'Team'],
        columns='StatType',
        values='Stat',
        aggfunc='sum'
    ).reset_index()
    if 'QB HUR' in pivot.columns:
        pivot = pivot.rename(columns={'QB HUR': 'QB_HUR'})
    pivot['Team_norm'] = pivot['Team'].map(lambda x: SCHOOL_ALIAS.get(x, x))
    return pivot


def lookup_stats(stats_pivot, player: str, school: str, draft_year: int):
    """Look up cumulative and final-season stats for a player."""
    final_season = draft_year - 1
    school_norm = SCHOOL_ALIAS.get(school, school)
    player_match = PLAYER_ALIAS.get(player, player)

    mask = (stats_pivot['Player'] == player_match) & (stats_pivot['Team_norm'] == school_norm)
    rows = stats_pivot.loc[mask]
    if rows.empty:
        return None, None, None, None, None, None

    cum = rows[rows['Season'] <= final_season]
    fin = rows[rows['Season'] == final_season]
    sacks_cum = cum['SACKS'].sum() if 'SACKS' in cum.columns else 0
    tfl_cum = cum['TFL'].sum() if 'TFL' in cum.columns else 0
    qbh_cum = cum['QB_HUR'].sum() if 'QB_HUR' in cum.columns else 0
    sacks_fin = fin['SACKS'].sum() if 'SACKS' in fin.columns else 0
    tfl_fin = fin['TFL'].sum() if 'TFL' in fin.columns else 0
    qbh_fin = fin['QB_HUR'].sum() if 'QB_HUR' in fin.columns else 0
    return sacks_cum, tfl_cum, qbh_cum, sacks_fin, tfl_fin, qbh_fin


def _fmt(val):
    """Format value for CSV: float notation (e.g. 72.0), empty if NaN."""
    if pd.isna(val) or val == '' or (isinstance(val, float) and val != val):
        return ''
    try:
        f = float(val)
        return f'{f:.1f}' if f == int(f) else f'{f}'
    except (TypeError, ValueError):
        return str(val)


def format_row(row_dict):
    """Format a row to match edges CSV format: floats, 3 trailing commas."""
    cols = [
        'Round', 'Pick', 'Player', 'Pos', 'School', 'Year',
        'Height', 'Weight', '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
        'Sacks_cumulative', 'TFL_cumulative', 'QB_Hurry_cumulative',
        'Sacks_final_season', 'TFL_final_season', 'QB_Hurry_final_season',
        'speed_score', 'explosive_score', 'agility_score'
    ]
    parts = []
    for c in cols:
        v = row_dict.get(c, '')
        parts.append(_fmt(v))
    return ','.join(parts)


def process_drafted_csv(path: str, stats_pivot: pd.DataFrame) -> str:
    """Load drafted CSV, enrich with stats, return formatted CSV content."""
    cols = ['Round', 'Pick', 'Player', 'Pos', 'School', 'Year', 'Height', 'Weight',
            '40yd', 'Vertical', 'Bench', 'Broad Jump', '3Cone', 'Shuttle',
            'Sacks_cumulative', 'TFL_cumulative', 'QB_Hurry_cumulative',
            'Sacks_final_season', 'TFL_final_season', 'QB_Hurry_final_season']
    try:
        df = pd.read_csv(path, usecols=lambda c: c in cols or c in ['speed_score', 'explosive_score', 'agility_score'])
    except Exception:
        df = pd.read_csv(path, header=0, names=cols + ['speed_score', 'explosive_score', 'agility_score'])
    rows = []

    for _, r in df.iterrows():
        player = r['Player']
        school = r['School']
        year_val = r['Year']
        if pd.isna(year_val):
            continue
        year = int(year_val)

        sacks_cum, tfl_cum, qbh_cum, sacks_fin, tfl_fin, qbh_fin = lookup_stats(
            stats_pivot, player, school, year
        )

        row = {
            'Round': r['Round'],
            'Pick': r['Pick'],
            'Player': player,
            'Pos': r['Pos'],
            'School': school,
            'Year': year,
            'Height': r.get('Height', ''),
            'Weight': r.get('Weight', ''),
            '40yd': r.get('40yd', ''),
            'Vertical': r.get('Vertical', ''),
            'Bench': r.get('Bench', ''),
            'Broad Jump': r.get('Broad Jump', ''),
            '3Cone': r.get('3Cone', ''),
            'Shuttle': r.get('Shuttle', ''),
            'Sacks_cumulative': sacks_cum if sacks_cum is not None else r.get('Sacks_cumulative', ''),
            'TFL_cumulative': tfl_cum if tfl_cum is not None else r.get('TFL_cumulative', ''),
            'QB_Hurry_cumulative': qbh_cum if qbh_cum is not None else r.get('QB_Hurry_cumulative', ''),
            'Sacks_final_season': sacks_fin if sacks_fin is not None else r.get('Sacks_final_season', ''),
            'TFL_final_season': tfl_fin if tfl_fin is not None else r.get('TFL_final_season', ''),
            'QB_Hurry_final_season': qbh_fin if qbh_fin is not None else r.get('QB_Hurry_final_season', ''),
        }
        rows.append(format_row(row))

    header = 'Round,Pick,Player,Pos,School,Year,Height,Weight,40yd,Vertical,Bench,Broad Jump,3Cone,Shuttle,Sacks_cumulative,TFL_cumulative,QB_Hurry_cumulative,Sacks_final_season,TFL_final_season,QB_Hurry_final_season,speed_score,explosive_score,agility_score'
    return header + '\n' + '\n'.join(rows)


def main():
    stats_pivot = load_defensive_stats()
    for year in [2024, 2025, 2026]:
        path = os.path.join(DT_DIR, f'dt_drafted_{year}.csv')
        if not os.path.exists(path):
            print(f'Skipping {path} (not found)')
            continue
        content = process_drafted_csv(path, stats_pivot)
        with open(path, 'w') as f:
            f.write(content)
        print(f'Updated {path}')


if __name__ == '__main__':
    main()
