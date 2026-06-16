"""Fetch per-pitcher season stats for all relevant seasons via MLB-StatsAPI."""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import statsapi

DATA = Path(__file__).parent / "data"
SEASONS = range(2013, 2026)
OUTPUT = DATA / "pitcher_season_stats.csv"


def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def fetch_season(year: int) -> list:
    result = statsapi.get("stats", {
        "group": "pitching",
        "stats": "season",
        "sportId": 1,
        "season": year,
        "gameType": "R",
        "limit": 2000,
        "playerPool": "All",
    })
    splits = result.get("stats", [{}])[0].get("splits", [])
    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})
        rows.append({
            "season": year,
            "player_id": player.get("id"),
            "player_name": player.get("fullName"),
            "team_name": team.get("name"),
            "games_started": stat.get("gamesStarted") or 0,
            "innings_pitched": safe_float(stat.get("inningsPitched")),
            "era": safe_float(stat.get("era")),
            "whip": safe_float(stat.get("whip")),
            "k_per_9": safe_float(stat.get("strikeoutsPer9Inn")),
            "bb_per_9": safe_float(stat.get("walksPer9Inn")),
            "hr_per_9": safe_float(stat.get("homeRunsPer9Inn")),
            "strikeouts": stat.get("strikeOuts") or 0,
            "walks": stat.get("baseOnBalls") or 0,
            "hits_allowed": stat.get("hits") or 0,
        })
    return rows


def main() -> None:
    DATA.mkdir(exist_ok=True)
    all_rows = []
    for year in SEASONS:
        print(f"  {year}...", end=" ", flush=True)
        rows = fetch_season(year)
        print(f"{len(rows)} pitchers")
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} pitcher-season rows to {OUTPUT}")
    print(f"\nStarters with 10+ IP: {len(df[df['innings_pitched'] >= 10])}")
    print(f"\nSample top by IP (2024):")
    sample = df[df["season"] == 2024].nlargest(5, "innings_pitched")[
        ["player_name", "games_started", "innings_pitched", "era", "whip", "k_per_9"]
    ]
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
