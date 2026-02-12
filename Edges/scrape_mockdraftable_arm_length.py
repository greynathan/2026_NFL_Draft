#!/usr/bin/env python3
"""
Scrape MockDraftable for arm length of every EDGE player (1999-2025).
Saves to data/raw/mockdraftable_edge_arm_length.csv
"""
import re
import time
import csv
import urllib.request
from urllib.parse import urljoin, quote

BASE = "https://www.mockdraftable.com"
SEARCH_URL = BASE + "/search?position=EDGE&beginYear=1999&endYear=2025&sort=DESC&page={page}"
OUTPUT_PATH = "data/raw/mockdraftable_edge_arm_length.csv"

# Be polite: delay between requests (seconds)
DELAY_SEARCH = 0.8
DELAY_PLAYER = 0.5
MAX_SEARCH_PAGES = 20
# Set to a positive int to only scrape that many players (for testing); 0 = all
MAX_PLAYERS_LIMIT = 0  # 0 = scrape all players


def fetch(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8", errors="replace")


def extract_player_links(html):
    # Links like href="/player/aaron-lynch?position=EDGE" or full URL
    pattern = r'href="(?:https://www\.mockdraftable\.com)?(/player/[^"]+\?position=EDGE)"'
    links = re.findall(pattern, html, re.I)
    # Also get name/school/year from the link text if present: [DE South Florida, 2014](url)
    # For now just collect URLs; we'll get name/school/year from player page
    seen = set()
    out = []
    for link in links:
        full = link if link.startswith("http") else (BASE + link.split("?")[0] + "?position=EDGE")
        if full not in seen:
            seen.add(full)
            out.append(full)
    return out


def _parse_inches(raw):
    """Parse measurement string to inches (float). E.g. 34\", 33 1/2\", 33½\", 32⅛\"."""
    if not raw:
        return None
    # Unicode fractions: ⅛ ¼ ⅓ ⅜ ½ ⅝ ⅔ ¾ ⅞
    raw = (raw.replace("\u215b", ".125").replace("\u00bc", ".25").replace("\u2153", ".333")
           .replace("\u215c", ".375").replace("\u00bd", ".5").replace("\u215d", ".625")
           .replace("\u2154", ".667").replace("\u00be", ".75").replace("\u215e", ".875"))
    raw = re.sub(r"\s+", " ", raw).strip()
    # Match: 34, 34", 33 1/2, 33.5, 32.125
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


def extract_arm_length_and_info(html, url):
    """Extract name, school, year, arm length from player page HTML."""
    name = school = year = arm_length = None
    # Name: <title>Name - MockDraftable</title>
    m = re.search(r"<title>([^<]+)\s*-\s*MockDraftable", html, re.I)
    if m:
        name = m.group(1).strip()
    # Draft class: Draft Class: 2014 or draftClass: 2014
    m = re.search(r"Draft\s*Class\s*:\s*(\d{4})", html, re.I)
    if m:
        year = m.group(1)
    # School: School: South Florida
    m = re.search(r"School\s*:\s*([^\n<]+)", html, re.I)
    if m:
        school = m.group(1).strip()
    # Arm length: try multiple patterns (site may serve HTML or pipe table in different contexts)
    # 1) Pipe-style table (e.g. in pre or as text): | Arm Length | 34" | 70 |
    m = re.search(r"Arm\s+Length\s*\|\s*([^|]+?)\s*\|", html, re.I)
    if m:
        arm_length = _parse_inches(m.group(1).strip())
    if arm_length is None:
        # 2) HTML table: <td>Arm Length</td><td>34"</td>
        m = re.search(r"<td[^>]*>\s*Arm\s+Length\s*</td>\s*<td[^>]*>\s*([^<]+?)\s*</td>", html, re.I | re.DOTALL)
        if m:
            arm_length = _parse_inches(m.group(1).strip())
    if arm_length is None:
        # 3) Tight HTML: Arm Length</td><td>34"
        m = re.search(r"Arm\s+Length\s*</td>\s*<td[^>]*>([^<]+)</td>", html, re.I)
        if m:
            arm_length = _parse_inches(m.group(1).strip())
    return name, school, year, arm_length


def main():
    all_links = []
    for page in range(1, MAX_SEARCH_PAGES + 1):
        url = SEARCH_URL.format(page=page)
        print(f"Fetching search page {page}...", end=" ")
        try:
            html = fetch(url)
            links = extract_player_links(html)
            if not links:
                print("no links, stopping.")
                break
            all_links.extend(links)
            print(f"found {len(links)} players (total {len(all_links)})")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(DELAY_SEARCH)

    # Deduplicate preserving order
    seen = set()
    unique_links = []
    for L in all_links:
        if L not in seen:
            seen.add(L)
            unique_links.append(L)

    if MAX_PLAYERS_LIMIT:
        unique_links = unique_links[:MAX_PLAYERS_LIMIT]
        print(f"Limiting to first {len(unique_links)} players (test run).")
    print(f"\nScraping {len(unique_links)} player pages for arm length...")
    rows = []
    for i, url in enumerate(unique_links):
        try:
            html = fetch(url)
            name, school, year, arm_length = extract_arm_length_and_info(html, url)
            rows.append({
                "name": name or "",
                "school": school or "",
                "year": year or "",
                "arm_length_inches": arm_length if arm_length is not None else "",
                "url": url,
            })
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(unique_links)} ...")
        except Exception as e:
            print(f"  Error {url}: {e}")
            rows.append({"name": "", "school": "", "year": "", "arm_length_inches": "", "url": url})
        time.sleep(DELAY_PLAYER)

    # Save to project data/raw (script lives in Edges/)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mockdraftable_edge_arm_length.csv")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "school", "year", "arm_length_inches", "url"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_file}")
    with_arm = sum(1 for r in rows if r["arm_length_inches"] != "")
    print(f"Arm length present for {with_arm} players.")


if __name__ == "__main__":
    main()
