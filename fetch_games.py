"""Fetch historical MLB regular-season games + starting pitchers to data/games.csv."""

import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import pandas as pd
import statsapi

SEASONS = list(range(2015, 2025))
DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "games.csv"

SEASON_RANGE = ("03-15", "10-05")


def fetch_season(year: int) -> list:
    last_err = None
    for attempt in range(4):
        try:
            return statsapi.schedule(
                start_date=f"{year}-{SEASON_RANGE[0]}",
                end_date=f"{year}-{SEASON_RANGE[1]}",
            )
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"\n  attempt {attempt+1} failed ({e.__class__.__name__}), retrying in {wait}s...", end="", flush=True)
            time.sleep(wait)
    raise last_err


def to_clean_row(g: dict, year: int) -> dict:
    return {
        "game_id": g["game_id"],
        "date": g["game_date"],
        "season": year,
        "home_team": g["home_name"],
        "away_team": g["away_name"],
        "home_id": g["home_id"],
        "away_id": g["away_id"],
        "home_score": g["home_score"],
        "away_score": g["away_score"],
        "home_pitcher": g.get("home_probable_pitcher") or "",
        "away_pitcher": g.get("away_probable_pitcher") or "",
        "venue": g.get("venue_name") or "",
        "venue_id": g.get("venue_id") or 0,
    }


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    all_games = []
    for year in SEASONS:
        print(f"Fetching {year}...", end=" ", flush=True)
        t = time.time()
        raw = fetch_season(year)
        regular = [g for g in raw if g["game_type"] == "R" and g["status"] == "Final"]
        rows = [to_clean_row(g, year) for g in regular]
        print(f"{len(rows)} games ({time.time()-t:.1f}s)")
        all_games.extend(rows)
        time.sleep(0.3)

    df = pd.DataFrame(all_games)
    df = df.drop_duplicates(subset=["game_id"]).sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} games to {OUTPUT}")
    print(f"\nGames with named home starter:  {(df['home_pitcher'] != '').sum()}")
    print(f"Games with named away starter:  {(df['away_pitcher'] != '').sum()}")


if __name__ == "__main__":
    main()
