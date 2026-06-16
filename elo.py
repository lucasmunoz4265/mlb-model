"""Team Elo rating model for MLB games."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL_RATING = 1500
K = 4               # low because baseball has high per-game variance
HCA = 24            # ~54% home win rate
SEASON_REGRESSION = 0.35

DATA = Path(__file__).parent / "data"


def expected_home_win(rating_home: float, rating_away: float) -> float:
    diff = rating_home + HCA - rating_away
    return 1 / (1 + 10 ** (-diff / 400))


def run_elo(games: pd.DataFrame) -> tuple:
    ratings: dict = defaultdict(lambda: INITIAL_RATING)
    rows = []
    prev_season = None

    for g in games.itertuples(index=False):
        if prev_season is not None and g.season != prev_season:
            for team in list(ratings):
                ratings[team] = INITIAL_RATING + (ratings[team] - INITIAL_RATING) * (1 - SEASON_REGRESSION)

        r_home = ratings[g.home_team]
        r_away = ratings[g.away_team]
        p_home = expected_home_win(r_home, r_away)
        home_won = int(g.home_score > g.away_score)

        rows.append({
            "game_id": g.game_id,
            "date": g.date,
            "season": g.season,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_rating_pre": r_home,
            "away_rating_pre": r_away,
            "p_home": p_home,
            "home_won": home_won,
        })

        update = K * (home_won - p_home)
        ratings[g.home_team] = r_home + update
        ratings[g.away_team] = r_away - update
        prev_season = g.season

    return pd.DataFrame(rows), dict(ratings)


def peak_ratings(history: pd.DataFrame) -> pd.DataFrame:
    home = history[["date", "season", "home_team", "home_rating_pre"]].rename(
        columns={"home_team": "team", "home_rating_pre": "rating"}
    )
    away = history[["date", "season", "away_team", "away_rating_pre"]].rename(
        columns={"away_team": "team", "away_rating_pre": "rating"}
    )
    stacked = pd.concat([home, away], ignore_index=True)
    idx = stacked.groupby("team")["rating"].idxmax()
    return stacked.loc[idx].sort_values("rating", ascending=False).reset_index(drop=True)


def main() -> None:
    games = pd.read_csv(DATA / "games.csv")
    history, final = run_elo(games)
    history.to_csv(DATA / "elo_history.csv", index=False)

    print(f"Processed {len(history)} games\n")

    correct = (history["p_home"] > 0.5) == history["home_won"].astype(bool)
    print(f"Pick-the-favorite accuracy: {correct.mean():.1%}")
    print(f"  (baseball baseline: ~54% home wins, ~60% with team strength)")

    eps = 1e-15
    p = history["p_home"].clip(eps, 1 - eps)
    y = history["home_won"]
    log_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    print(f"Log loss: {log_loss:.4f} (0.693 = coin flip)\n")

    print("Final ratings — end of 2024 (top 10):")
    for team, rating in sorted(final.items(), key=lambda x: -x[1])[:10]:
        print(f"  {team}: {rating:.0f}")

    print("\nPeak ratings across all 10 seasons (top 12):")
    peaks = peak_ratings(history)
    for _, row in peaks.head(12).iterrows():
        print(f"  {row['team']}: {row['rating']:.0f}  ({row['date'][:10]})")


if __name__ == "__main__":
    main()
