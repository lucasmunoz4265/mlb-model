"""Engineer per-game features for MLB with no look-ahead bias."""

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_elo import run_pitcher_elo

DATA = Path(__file__).parent / "data"
WIN_WINDOW = 20
RUN_WINDOW = 20
NEUTRAL_RUNS = 4.5

# League-average pitcher stats for unknown/rookie pitchers
LEAGUE_AVG_PITCHER = {
    "era": 4.20, "whip": 1.30, "k_per_9": 8.5, "bb_per_9": 3.2,
}


def load_pitcher_lookup() -> dict:
    """Build (player_name, season) -> stats dict from pitcher_season_stats.csv."""
    path = DATA / "pitcher_season_stats.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df = df.sort_values("innings_pitched", ascending=False)
    df = df.drop_duplicates(subset=["player_name", "season"], keep="first")
    lookup = {}
    for r in df.itertuples(index=False):
        lookup[(r.player_name, r.season)] = {
            "era": r.era if r.era > 0 else LEAGUE_AVG_PITCHER["era"],
            "whip": r.whip if r.whip > 0 else LEAGUE_AVG_PITCHER["whip"],
            "k_per_9": r.k_per_9 if r.k_per_9 > 0 else LEAGUE_AVG_PITCHER["k_per_9"],
            "bb_per_9": r.bb_per_9 if r.bb_per_9 > 0 else LEAGUE_AVG_PITCHER["bb_per_9"],
        }
    return lookup


def get_prior_stats(pitcher_name: str, current_season: int, lookup: dict) -> dict:
    return lookup.get((pitcher_name, current_season - 1), LEAGUE_AVG_PITCHER)

# Park run factors (approx. from public 3-year averages, hitter-friendly > 1.0).
PARK_FACTOR = {
    "Coors Field": 1.18,
    "Great American Ball Park": 1.06,
    "Globe Life Field": 1.05,
    "Globe Life Park in Arlington": 1.06,
    "Citizens Bank Park": 1.05,
    "Yankee Stadium": 1.04,
    "Fenway Park": 1.04,
    "Camden Yards": 1.03,
    "Oriole Park at Camden Yards": 1.03,
    "Wrigley Field": 1.02,
    "Truist Park": 1.02,
    "Rogers Centre": 1.02,
    "Minute Maid Park": 1.01,
    "Daikin Park": 1.01,
    "Chase Field": 1.01,
    "Nationals Park": 1.00,
    "PNC Park": 1.00,
    "Target Field": 1.00,
    "Progressive Field": 1.00,
    "Comerica Park": 0.99,
    "Angel Stadium": 0.99,
    "American Family Field": 0.99,
    "Miller Park": 0.99,
    "Busch Stadium": 0.98,
    "Dodger Stadium": 0.98,
    "Citi Field": 0.97,
    "Guaranteed Rate Field": 0.97,
    "loanDepot park": 0.96,
    "Marlins Park": 0.96,
    "T-Mobile Park": 0.96,
    "Safeco Field": 0.96,
    "Kauffman Stadium": 0.96,
    "Oakland Coliseum": 0.95,
    "O.co Coliseum": 0.95,
    "RingCentral Coliseum": 0.95,
    "Tropicana Field": 0.95,
    "Oracle Park": 0.92,
    "AT&T Park": 0.92,
    "Petco Park": 0.93,
    "Angel Stadium of Anaheim": 0.99,
    "SunTrust Park": 1.02,
    "U.S. Cellular Field": 0.97,
    "Turner Field": 1.02,
    "Sahlen Field": 1.00,
}


def park_factor(venue: str) -> float:
    return PARK_FACTOR.get(venue, 1.00)


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"])
    games = games.sort_values("date").reset_index(drop=True)
    games = games.fillna({"home_pitcher": "", "away_pitcher": ""})

    history, _, _ = run_pitcher_elo(games)
    games["team_elo_home"] = history["team_h"].values
    games["team_elo_away"] = history["team_a"].values
    games["pit_elo_home"] = history["pit_h"].values
    games["pit_elo_away"] = history["pit_a"].values
    games["elo_p_home"] = history["p_home"].values

    pitcher_lookup = load_pitcher_lookup()

    runs_for: dict = defaultdict(lambda: deque(maxlen=RUN_WINDOW))
    runs_against: dict = defaultdict(lambda: deque(maxlen=RUN_WINDOW))
    wins: dict = defaultdict(lambda: deque(maxlen=WIN_WINDOW))
    last_played: dict = {}

    rows = []
    for g in games.itertuples(index=False):
        h, a = g.home_team, g.away_team
        def_prev = g.date - pd.Timedelta(days=5)
        rh = min(5, (g.date - last_played.get(h, def_prev)).days)
        ra = min(5, (g.date - last_played.get(a, def_prev)).days)

        home_pit_stats = get_prior_stats(g.home_pitcher, g.season, pitcher_lookup)
        away_pit_stats = get_prior_stats(g.away_pitcher, g.season, pitcher_lookup)

        rows.append({
            "game_id": g.game_id,
            "date": g.date,
            "season": g.season,
            "home_team": h,
            "away_team": a,
            "team_elo_home": g.team_elo_home,
            "team_elo_away": g.team_elo_away,
            "pit_elo_home": g.pit_elo_home,
            "pit_elo_away": g.pit_elo_away,
            "elo_p_home": g.elo_p_home,
            "home_runs_for_avg": np.mean(runs_for[h]) if runs_for[h] else NEUTRAL_RUNS,
            "home_runs_against_avg": np.mean(runs_against[h]) if runs_against[h] else NEUTRAL_RUNS,
            "home_winrate": np.mean(wins[h]) if wins[h] else 0.5,
            "away_runs_for_avg": np.mean(runs_for[a]) if runs_for[a] else NEUTRAL_RUNS,
            "away_runs_against_avg": np.mean(runs_against[a]) if runs_against[a] else NEUTRAL_RUNS,
            "away_winrate": np.mean(wins[a]) if wins[a] else 0.5,
            "rest_home": rh,
            "rest_away": ra,
            "park_factor": park_factor(g.venue),
            "home_pit_era": home_pit_stats["era"],
            "home_pit_whip": home_pit_stats["whip"],
            "home_pit_k9": home_pit_stats["k_per_9"],
            "home_pit_bb9": home_pit_stats["bb_per_9"],
            "away_pit_era": away_pit_stats["era"],
            "away_pit_whip": away_pit_stats["whip"],
            "away_pit_k9": away_pit_stats["k_per_9"],
            "away_pit_bb9": away_pit_stats["bb_per_9"],
            "home_won": int(g.home_score > g.away_score),
        })

        runs_for[h].append(g.home_score)
        runs_against[h].append(g.away_score)
        wins[h].append(1 if g.home_score > g.away_score else 0)
        runs_for[a].append(g.away_score)
        runs_against[a].append(g.home_score)
        wins[a].append(1 if g.away_score > g.home_score else 0)
        last_played[h] = g.date
        last_played[a] = g.date

    return pd.DataFrame(rows)


if __name__ == "__main__":
    games = pd.read_csv(DATA / "games.csv")
    features = build_features(games)
    features.to_csv(DATA / "features.csv", index=False)
    print(f"Built features for {len(features)} games -> data/features.csv")
    print(f"\nMissing park factor (defaulted to 1.0): {(features['park_factor'] == 1.0).sum()} games")
    unmapped = games[~games["venue"].isin(PARK_FACTOR.keys())]["venue"].value_counts().head(5)
    if len(unmapped) > 0:
        print("Top unmapped venues (sanity check):")
        print(unmapped)
