"""
One-off RAS triple-check for DT and Edges.
- Load processed CSVs + 2026 drafted; find missing RAS.
- For each missing: look in ras.csv (DT or DE/EDGE) for same Year + same normalized school.
- If exactly one RAS row at that school+year, or one with matching last name -> safe alias.
Run from project root: python verify_ras_dt_edges.py
"""
import os
import re
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')


def normalize_player_name(name):
    s = str(name).strip().upper()
    s = re.sub(r'\s+(III|II|JR|SR|JR\.|SR\.)$', '', s)
    s = re.sub(r'[.\',\-]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ---- DT: same ras_school and normalize_combine_school as DT/data_cleaning.py ----
DT_RAS_SCHOOL = {
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
    'Montana St.': 'Montana State', 'Montana State': 'Montana State',
    'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
    'Washington St.': 'Washington State', 'North Carolina St.': 'North Carolina State',
    'Ala-Birmingham': 'UAB', 'Texas AM': 'Texas A&M',
}


def dt_normalize_combine_school(name):
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
        'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
        'Florida Atlantic': 'Florida Atlantic', 'Texas-San Antonio': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio',
        'Toledo': 'Toledo', 'Georgia Southern': 'Georgia Southern', 'Ga. Southern': 'Georgia Southern',
        'Kentucky': 'Kentucky', 'TCU': 'TCU',
        'Arizona St.': 'Arizona State', 'Arizona State': 'Arizona State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
        'West Virginia': 'West Virginia', 'Montana St.': 'Montana State', 'Montana State': 'Montana State',
    }
    return alias.get(name, name)


# ---- Edges: same normalize_ras_school and normalize_combine_school_for_ras as Edges/data_cleaning.py ----
EDGE_RAS_SCHOOL = {
    'Miami (FL)': 'Miami', 'Miami': 'Miami', 'Miami (Ohio)': 'Miami (OH)',
    'Boston Col.': 'Boston College', 'Boston College': 'Boston College',
    'Southern California': 'USC', 'USC': 'USC', 'UCLA': 'UCLA',
    'Central Florida': 'UCF', 'UCF': 'UCF', 'Brigham Young': 'BYU', 'BYU': 'BYU',
    'Ole Miss': 'Mississippi', 'Mississippi': 'Mississippi', 'Ohio St.': 'Ohio State', 'Ohio State': 'Ohio State',
    'Florida St.': 'Florida State', 'Florida State': 'Florida State',
    'Oklahoma St.': 'Oklahoma State', 'Oklahoma State': 'Oklahoma State',
    'Penn St.': 'Penn State', 'Penn State': 'Penn State',
    'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
    'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
    'Virginia Tech': 'Virginia Tech', 'Texas State': 'Texas State', 'Louisiana Tech': 'Louisiana Tech',
    'Appalachian State': 'Appalachian State', 'Florida Atlantic': 'Florida Atlantic',
    'Texas-San Antonio': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio',
    'Toledo': 'Toledo', 'Georgia Southern': 'Georgia Southern', 'Kentucky': 'Kentucky',
    'TCU': 'TCU', 'Texas Christian': 'TCU', 'Louisiana State': 'LSU', 'LSU': 'LSU',
    'San Diego St.': 'San Diego State', 'San Diego State': 'San Diego State',
    'San Jose St.': 'San Jose State', 'San Jose State': 'San Jose State',
    'Kansas St.': 'Kansas State', 'Kansas State': 'Kansas State',
    'Iowa St.': 'Iowa State', 'Iowa State': 'Iowa State',
    'Arizona St.': 'Arizona State', 'Arizona State': 'Arizona State',
    'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
    'West Virginia': 'West Virginia',
    'Washington State': 'Washington State', 'Washington St.': 'Washington State',
    'North Carolina St.': 'North Carolina State', 'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
    'Texas AM': 'Texas A&M',
}


def edge_normalize_combine_school(name):
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
        'Oklahoma St.': 'Oklahoma State', 'Oklahoma State': 'Oklahoma State',
        'Penn St.': 'Penn State', 'Penn State': 'Penn State',
        'San Diego St.': 'San Diego State', 'San Diego State': 'San Diego State',
        'San Jose St.': 'San Jose State', 'San Jose State': 'San Jose State',
        'Boston Col.': 'Boston College', 'Boston College': 'Boston College',
        'Alabama-Birmingham': 'UAB', 'Tenn-Chattanooga': 'Chattanooga',
        'Washington State': 'Washington State', 'Colorado State': 'Colorado State',
        'Northwestern': 'Northwestern', 'LSU': 'LSU', 'Virginia Tech': 'Virginia Tech',
        'Texas State': 'Texas State', 'Louisiana Tech': 'Louisiana Tech',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
        'Appalachian State': 'Appalachian State', 'Appalachian St.': 'Appalachian State', 'App St.': 'Appalachian State',
        'Florida Atlantic': 'Florida Atlantic', 'Texas-San Antonio': 'Texas-San Antonio', 'UTSA': 'Texas-San Antonio',
        'Toledo': 'Toledo', 'Georgia Southern': 'Georgia Southern', 'Ga. Southern': 'Georgia Southern',
        'Kentucky': 'Kentucky', 'TCU': 'TCU', 'Louisiana State': 'LSU',
        'Arizona St.': 'Arizona State', 'Arizona State': 'Arizona State',
        'Michigan St.': 'Michigan State', 'Michigan State': 'Michigan State',
        'Mississippi St.': 'Mississippi State', 'Mississippi State': 'Mississippi State',
        'West Virginia': 'West Virginia', 'Oregon St.': 'Oregon State', 'Oregon State': 'Oregon State',
    }
    return alias.get(name, name)


def find_confirmed_aliases(missing_df, ras_n, normalize_combine_school_fn, ras_school_dict):
    """For each missing row, if RAS has same school+year with exactly one candidate or last-name match, return (combine_name_n, ras_name_n) or school fix."""
    ras_n = ras_n.copy()
    ras_n['College_n'] = ras_n['College'].apply(lambda x: ras_school_dict.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    name_aliases = []
    school_additions = []
    for _, row in missing_df.iterrows():
        player_n = normalize_player_name(row['Player'])
        school_n = normalize_combine_school_fn(row['School'])
        year = int(row['Year'])
        m = (ras_n['College_n'] == school_n) & (ras_n['Year'] == year)
        cand = ras_n.loc[m]
        if cand.empty:
            continue
        combine_last = player_n.split()[-1] if player_n else ''
        match = None
        if len(cand) == 1:
            match = cand.iloc[0]
        else:
            for _, c in cand.iterrows():
                if combine_last and combine_last in c['Name_n']:
                    match = c
                    break
        if match is not None:
            ras_name_n = match['Name_n']
            if ras_name_n != player_n:
                name_aliases.append((row['Player'], row['School'], year, match['Name'], player_n, ras_name_n))
            # Check if combine school needed (e.g. Montana St. -> Montana State)
            ras_college_raw = match.get('College', '')
            if str(ras_college_raw).strip() and normalize_combine_school_fn(row['School']) != ras_n.loc[m].iloc[0]['College_n']:
                pass  # school already matches after our normalize
    return name_aliases


def main():
    ras_path = os.path.join(DATA_RAW, 'ras.csv')
    ras_all = pd.read_csv(ras_path)
    ras_all['Year'] = ras_all['Year'].astype(int)
    ras_all['Name_n'] = ras_all['Name'].apply(normalize_player_name)

    # ----- DT -----
    print("=== DT ===")
    dt_train = pd.read_csv(os.path.join(DATA_PROCESSED, 'dt_training.csv'))
    dt_test = pd.read_csv(os.path.join(DATA_PROCESSED, 'dt_testing.csv'))
    dt_2026 = pd.read_csv(os.path.join(PROJECT_ROOT, 'DT', 'dt_drafted_2026.csv'))
    all_dt = pd.concat([dt_train, dt_test, dt_2026], ignore_index=True)
    missing_dt = all_dt[all_dt['RAS'].isna()][['Player', 'School', 'Year']].drop_duplicates()
    ras_dt = ras_all[ras_all['Pos'] == 'DT'].copy()
    dt_confirmed = find_confirmed_aliases(missing_dt, ras_dt, dt_normalize_combine_school, DT_RAS_SCHOOL)
    print(f"DT missing RAS: {len(missing_dt)}")
    print(f"DT confirmed name aliases (add to ras_name_alias): {len(dt_confirmed)}")
    for t in dt_confirmed:
        print(f"  # {t[0]} ({t[2]}): RAS '{t[3]}' -> alias '{t[4]}': '{t[5]}'")

    # ----- Edges -----
    print("\n=== Edges ===")
    edge_train = pd.read_csv(os.path.join(DATA_PROCESSED, 'edge_training.csv'))
    edge_test = pd.read_csv(os.path.join(DATA_PROCESSED, 'edge_testing.csv'))
    edge_2026 = pd.read_csv(os.path.join(PROJECT_ROOT, 'Edges', 'edges_drafted_2026.csv'))
    all_edge = pd.concat([edge_train, edge_test, edge_2026], ignore_index=True)
    missing_edge = all_edge[all_edge['RAS'].isna()][['Player', 'School', 'Year']].drop_duplicates()
    ras_edge = ras_all[ras_all['Pos'].isin(['DE', 'EDGE'])].copy()
    edge_confirmed = find_confirmed_aliases(missing_edge, ras_edge, edge_normalize_combine_school, EDGE_RAS_SCHOOL)
    print(f"Edges missing RAS: {len(missing_edge)}")
    print(f"Edges confirmed name aliases (add to player_nickname_map or ras alias): {len(edge_confirmed)}")
    for t in edge_confirmed[:40]:
        print(f"  # {t[0]} ({t[2]}): RAS '{t[3]}' -> alias '{t[4]}': '{t[5]}'")
    if len(edge_confirmed) > 40:
        print(f"  ... and {len(edge_confirmed) - 40} more")


if __name__ == '__main__':
    main()
