#!/usr/bin/env python3
"""
Scrape MockDraftable for arm length only for players in our edge_training.csv
and edge_testing.csv. Saves to data/raw/mockdraftable_edge_arm_length.csv for
use in data_cleaning (merge by name + year).
"""
import re
import time
import csv
import os
import urllib.request

BASE = "https://www.mockdraftable.com"
DELAY = 0.6  # seconds between requests

# Paths relative to script (Edges/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
TRAINING_PATH = os.path.join(DATA_DIR, "processed", "edge_training.csv")
TESTING_PATH = os.path.join(DATA_DIR, "processed", "edge_testing.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "raw", "mockdraftable_edge_arm_length.csv")


def name_to_slug(name):
    """Convert 'J.T. Tuimoloau' -> 'j-t-tuimoloau'; 'Al-Quadin Muhammad' -> 'al-quadin-muhammad'."""
    if not name or not isinstance(name, str):
        return ""
    # Lowercase, keep letters digits spaces and hyphens; collapse punctuation to nothing
    s = re.sub(r"[^a-z0-9\s-]", "", name.lower().strip())
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


def fetch(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse_inches(raw):
    if not raw:
        return None
    raw = (raw.replace("\u215b", ".125").replace("\u00bc", ".25").replace("\u2153", ".333")
           .replace("\u215c", ".375").replace("\u00bd", ".5").replace("\u215d", ".625")
           .replace("\u2154", ".667").replace("\u00be", ".75").replace("\u215e", ".875"))
    raw = re.sub(r"\s+", " ", raw).strip()
    m = re.search(r"(\d+)\s*(?:\.(\d+))?\s*(?:\s*(\d+)\s*/\s*(\d+))?\s*[\"']?", raw)
    if not m:
        return None
    whole = int(m.group(1))
    frac = 0.0
    if m.group(2) is not None:
        frac = int(m.group(2)) / (10 ** len(m.group(2)))
    elif m.lastindex >= 4 and m.group(3) and m.group(4):
        num, den = int(m.group(3)), int(m.group(4))
        if den:
            frac = num / den
    return round(whole + frac, 2)


def extract_arm_length_and_year(html):
    """Return (arm_length, draft_year) or (None, None)."""
    year = None
    m = re.search(r"Draft\s*Class\s*:\s*(\d{4})", html, re.I)
    if m:
        year = m.group(1)
    # Arm length: pipe table or HTML table
    arm_length = None
    m = re.search(r"Arm\s+Length\s*\|\s*([^|]+?)\s*\|", html, re.I)
    if m:
        arm_length = _parse_inches(m.group(1).strip())
    if arm_length is None:
        m = re.search(r"<td[^>]*>\s*Arm\s+Length\s*</td>\s*<td[^>]*>\s*([^<]+?)\s*</td>", html, re.I | re.DOTALL)
        if m:
            arm_length = _parse_inches(m.group(1).strip())
    if arm_length is None:
        m = re.search(r"Arm\s+Length\s*</td>\s*<td[^>]*>([^<]+)</td>", html, re.I)
        if m:
            arm_length = _parse_inches(m.group(1).strip())
    return arm_length, year


def main():
    import pandas as pd
    train = pd.read_csv(TRAINING_PATH)
    test = pd.read_csv(TESTING_PATH)
    # Unique (Player, Year, School)
    combined = pd.concat([
        train[["Player", "Year", "School"]],
        test[["Player", "Year", "School"]],
    ], ignore_index=True).drop_duplicates()
    players = list(combined.itertuples(index=False, name=None))  # (Player, Year, School)[]
    print(f"Loaded {len(players)} unique players from edge_training + edge_testing.")

    rows = []
    for i, (player_name, year, school) in enumerate(players):
        slug = name_to_slug(player_name)
        if not slug:
            rows.append({"Player": player_name, "Year": year, "School": school or "", "arm_length_inches": ""})
            continue
        url = f"{BASE}/player/{slug}?position=EDGE"
        try:
            html = fetch(url)
            arm_length, page_year = extract_arm_length_and_year(html)
            # Confirm it's the right player (draft year should match)
            if page_year and str(int(year)) != str(page_year):
                # Wrong player (same slug, different year) - try with year suffix
                url2 = f"{BASE}/player/{slug}-{int(year)}?position=EDGE"
                try:
                    html2 = fetch(url2)
                    arm_length, page_year = extract_arm_length_and_year(html2)
                except Exception:
                    pass
            rows.append({
                "Player": player_name,
                "Year": int(year),
                "School": school if pd.notna(school) else "",
                "arm_length_inches": arm_length if arm_length is not None else "",
            })
        except Exception as e:
            rows.append({"Player": player_name, "Year": int(year), "School": school or "", "arm_length_inches": ""})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(players)} ...")
        time.sleep(DELAY)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Player", "Year", "School", "arm_length_inches"])
        w.writeheader()
        w.writerows(rows)
    with_arm = sum(1 for r in rows if r.get("arm_length_inches") not in ("", None))
    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")
    print(f"Arm length found for {with_arm} players.")


if __name__ == "__main__":
    main()
