"""
Verify PFF and RAS coverage for LB pipeline.
- Loads lb_training.csv, lb_testing.csv, lb_drafted_2026.csv.
- Reports counts: total, with RAS, with PFF pass rush, run defense, coverage.
- Lists every player missing RAS or PFF and suggests overrides by searching raw PFF/RAS.
Run from LB/ directory. Run data_cleaning.py first.
"""
import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')


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
        'MISSISSIPPI ST': 'Mississippi State', 'MISS STATE': 'Mississippi State', 'MISSISSIPPI STATE': 'Mississippi State',
        'MICHIGAN ST': 'Michigan State', 'MICHIGAN STATE': 'Michigan State',
        'NORTH CAROLINA ST': 'North Carolina State', 'NC STATE': 'North Carolina State',
        'SOUTHERN CAL': 'USC', 'SOUTHERN CALIFORNIA': 'USC', 'CENTRAL FLORIDA': 'UCF', 'UCF': 'UCF',
        'BRIGHAM YOUNG': 'BYU', 'MIAMI (FL)': 'Miami', 'MIAMI FL': 'Miami', 'MIAMI': 'Miami',
        'OLE MISS': 'Mississippi', 'TEXAS A&M': 'Texas A&M', 'TEXAS AM': 'Texas A&M',
        'W VIRGINIA': 'West Virginia', 'WEST VIRGINIA': 'West Virginia',
        'LA TECH': 'Louisiana Tech', 'LOUISIANA TECH': 'Louisiana Tech',
        'VA TECH': 'Virginia Tech', 'VIRGINIA TECH': 'Virginia Tech',
    }
    return mapping.get(name, name.title())


def normalize_combine_school(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    alias = {
        'Ole Miss': 'Mississippi', 'Miami (FL)': 'Miami', 'Southern California': 'USC', 'Central Florida': 'UCF',
        'Brigham Young': 'BYU', 'Ohio St.': 'Ohio State', 'Florida St.': 'Florida State',
        'Kansas St.': 'Kansas State', 'Iowa St.': 'Iowa State', 'Oklahoma St.': 'Oklahoma State',
        'Penn St.': 'Penn State', 'San Diego St.': 'San Diego State', 'San Jose St.': 'San Jose State',
        'Boston Col.': 'Boston College', 'Alabama-Birmingham': 'UAB', 'Tenn-Chattanooga': 'Chattanooga',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
        'Michigan St.': 'Michigan State', 'Mississippi St.': 'Mississippi State',
        'Arizona St.': 'Arizona State', 'West Virginia': 'West Virginia', 'Texas A&M': 'Texas A&M',
    }
    return alias.get(name, name)


def load_all_lb():
    train = pd.read_csv(os.path.join(DATA_PROCESSED, 'lb_training.csv'))
    test = pd.read_csv(os.path.join(DATA_PROCESSED, 'lb_testing.csv'))
    draft26 = pd.read_csv(os.path.join(SCRIPT_DIR, 'lb_drafted_2026.csv'))
    train['source'] = 'training'
    test['source'] = 'testing'
    draft26['source'] = '2026'
    return pd.concat([train, test, draft26], ignore_index=True)


def report_coverage(df, label):
    n = len(df)
    ras = df['RAS'].notna().sum() if 'RAS' in df.columns else 0
    pr = df['pass_rush_win_rate'].notna().sum() if 'pass_rush_win_rate' in df.columns else 0
    stop = df['stop_percent'].notna().sum() if 'stop_percent' in df.columns else 0
    cov = df['snap_counts_coverage'].notna().sum() if 'snap_counts_coverage' in df.columns else 0
    print(f"\n{label}: n={n}")
    print(f"  RAS:                {ras}/{n} ({100*ras/n:.1f}%)" if n else "  RAS: 0")
    print(f"  PFF pass rush:      {pr}/{n} ({100*pr/n:.1f}%)" if n else "  PFF pass rush: 0")
    print(f"  PFF run defense:    {stop}/{n} ({100*stop/n:.1f}%)" if n else "  PFF run defense: 0")
    print(f"  PFF coverage:       {cov}/{n} ({100*cov/n:.1f}%)" if n else "  PFF coverage: 0")


def find_missing_and_suggest_ras(all_lb, ras_lb):
    """List LBs missing RAS and suggest RAS Name/College from ras.csv."""
    missing = all_lb[all_lb['RAS'].isna()][['Player', 'School', 'Year', 'source']].drop_duplicates()
    if missing.empty:
        print("\n--- RAS: no missing players ---")
        return
    print("\n--- Missing RAS ---")
    ras_n = ras_lb.copy()
    ras_n['Name_n'] = ras_n['Name'].apply(normalize_player_name)
    ras_school = {
        'Miami (FL)': 'Miami', 'Miami': 'Miami', 'Southern California': 'USC', 'USC': 'USC',
        'Central Florida': 'UCF', 'UCF': 'UCF', 'Brigham Young': 'BYU', 'BYU': 'BYU',
        'Ole Miss': 'Mississippi', 'Ohio St.': 'Ohio State', 'Florida St.': 'Florida State',
        'Oklahoma St.': 'Oklahoma State', 'Penn St.': 'Penn State', 'Michigan St.': 'Michigan State',
        'North Carolina State': 'North Carolina State', 'NC State': 'North Carolina State',
        'Louisiana State': 'LSU', 'LSU': 'LSU', 'Texas A&M': 'Texas A&M', 'Boston Col.': 'Boston College',
        'San Diego St.': 'San Diego State', 'San Jose St.': 'San Jose State',
        'Kansas St.': 'Kansas State', 'Iowa St.': 'Iowa State', 'Arizona St.': 'Arizona State',
        'Mississippi St.': 'Mississippi State', 'West Virginia': 'West Virginia',
    }
    ras_n['College_n'] = ras_n['College'].apply(lambda x: ras_school.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    suggestions = []
    for _, row in missing.iterrows():
        player_n = normalize_player_name(row['Player'])
        school_n = normalize_combine_school(row['School'])
        year = int(row['Year'])
        # Exact match
        m = (ras_n['Name_n'] == player_n) & (ras_n['College_n'] == school_n) & (ras_n['Year'] == year)
        if m.any():
            continue
        # Same year + same college (different name spelling)
        m2 = (ras_n['College_n'] == school_n) & (ras_n['Year'] == year)
        cand = ras_n.loc[m2]
        if not cand.empty:
            # Prefer last name match
            combine_last = player_n.split()[-1] if player_n else ''
            for _, c in cand.iterrows():
                if combine_last and combine_last in c['Name_n']:
                    suggestions.append((row['Player'], row['School'], year, c['Name'], c['College'], 'RAS name/college'))
                    break
            else:
                suggestions.append((row['Player'], row['School'], year, cand.iloc[0]['Name'], cand.iloc[0]['College'], 'RAS same school+year'))
        else:
            # Same year only
            m3 = ras_n['Year'] == year
            cand2 = ras_n.loc[m3]
            combine_last = player_n.split()[-1] if player_n else ''
            for _, c in cand2.iterrows():
                if combine_last and combine_last in c['Name_n']:
                    suggestions.append((row['Player'], row['School'], year, c['Name'], c['College'], 'RAS same year+last name'))
                    break
    for r in missing.itertuples(index=False):
        print(f"  {r.Player} | {r.School} | {r.Year} | {r.source}")
    if suggestions:
        print("\n  Suggested RAS overrides (check and add to data_cleaning ras_name_alias / ras_school):")
        for s in suggestions[:30]:
            print(f"    # {s[0]} ({s[2]}): RAS has '{s[3]}' @ {s[4]}")


def find_missing_and_suggest_pff(all_lb, pff_pr, pff_rd, pff_cov):
    """List LBs missing PFF (pass rush, run defense, or coverage) and suggest PFF player/school."""
    has_pr = all_lb['pass_rush_win_rate'].notna()
    has_rd = all_lb['stop_percent'].notna()
    has_cov = all_lb['snap_counts_coverage'].notna()
    missing_any = all_lb[~(has_pr & has_rd & has_cov)][['Player', 'School', 'Year', 'source']].drop_duplicates()
    if missing_any.empty:
        print("\n--- PFF: no missing players ---")
        return
    print("\n--- Missing PFF (pass rush, run defense, or coverage) ---")
    # Build one PFF lookup: pass rush has Player, School, Year (normalized)
    pff_pr['School_n'] = pff_pr['School'].apply(normalize_pff_school)
    pff_pr['Player_n'] = pff_pr['Player'].apply(normalize_player_name)
    suggestions = []
    for _, row in missing_any.iterrows():
        player_n = normalize_player_name(row['Player'])
        school_n = normalize_combine_school(row['School'])
        final_season = int(row['Year']) - 1
        mask = (pff_pr['Player_n'] == player_n) & (pff_pr['School_n'] == school_n) & (pff_pr['Year'] == final_season)
        if mask.any():
            continue
        # Same school + year in PFF?
        m2 = (pff_pr['School_n'] == school_n) & (pff_pr['Year'] == final_season)
        cand = pff_pr.loc[m2]
        combine_last = player_n.split()[-1] if player_n else ''
        for _, c in cand.iterrows():
            if combine_last and combine_last in c['Player_n']:
                suggestions.append((row['Player'], row['School'], int(row['Year']), c['Player'], c['School'], 'PFF same school+year+last'))
                break
        else:
            if not cand.empty:
                suggestions.append((row['Player'], row['School'], int(row['Year']), cand.iloc[0]['Player'], cand.iloc[0]['School'], 'PFF same school+year'))
    for r in missing_any.itertuples(index=False):
        print(f"  {r.Player} | {r.School} | {r.Year} | {r.source}")
    if suggestions:
        print("\n  Suggested PFF overrides (add to data_cleaning player_nickname_map / player_school_pff_override):")
        seen = set()
        for s in suggestions[:40]:
            key = (s[0], s[2])
            if key in seen:
                continue
            seen.add(key)
            print(f"    # {s[0]} ({s[2]}): PFF has '{s[3]}' @ {s[4]} -> {s[5]}")


def main():
    print("Loading LB processed data...")
    all_lb = load_all_lb()
    report_coverage(all_lb, "All LB (training + testing + 2026)")
    report_coverage(all_lb[all_lb['source'] == 'training'], "Training only")
    report_coverage(all_lb[all_lb['source'] == 'testing'], "Testing only")
    report_coverage(all_lb[all_lb['source'] == '2026'], "2026 only")

    print("\nLoading raw RAS (ILB/LB/OLB)...")
    ras = pd.read_csv(os.path.join(DATA_RAW, 'ras.csv'))
    ras = ras[ras['Pos'].isin(['ILB', 'LB', 'OLB'])].copy()
    ras['Year'] = ras['Year'].astype(int)
    find_missing_and_suggest_ras(all_lb, ras)

    print("\nLoading raw PFF pass rush (for suggestion lookup)...")
    pff_pr_list = []
    for year in range(2014, 2026):
        path = os.path.join(DATA_RAW, 'pff', 'Pass_Rush', f'{year}_pass_rush_summary.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'player' in df.columns and 'team_name' in df.columns:
                df = df.rename(columns={'player': 'Player', 'team_name': 'School'})
                df['Year'] = year
                pff_pr_list.append(df[['Player', 'School', 'Year']])
    pff_pr = pd.concat(pff_pr_list, ignore_index=True) if pff_pr_list else pd.DataFrame()
    pff_rd = pd.DataFrame()
    pff_cov = pd.DataFrame()
    find_missing_and_suggest_pff(all_lb, pff_pr, pff_rd, pff_cov)
    print("\n--- Summary ---")
    print("Remaining missing RAS: often not on ras.football for that year/school (do not add wrong-player overrides).")
    print("Remaining missing PFF: many are small schools or PFF lists under different position/name (verify before adding).")
    print("Done. Add only verified overrides to LB/data_cleaning.py then re-run data_cleaning.py and this script.")


if __name__ == '__main__':
    main()
