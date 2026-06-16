"""Team Elo + per-pitcher Elo. Pitcher ratings are added to team rating for predictions."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from elo import INITIAL_RATING, HCA, SEASON_REGRESSION, peak_ratings

K_TEAM = 4
K_PITCHER = 8
PITCHER_WEIGHT = 0.5

DATA = Path(__file__).parent / "data"


def expected_home_win(rating_diff: float) -> float:
    return 1 / (1 + 10 ** (-rating_diff / 400))


def run_pitcher_elo(games: pd.DataFrame) -> tuple:
    team_ratings: dict = defaultdict(lambda: INITIAL_RATING)
    pitcher_ratings: dict = defaultdict(lambda: INITIAL_RATING)
    rows = []
    prev_season = None

    for g in games.itertuples(index=False):
        if prev_season is not None and g.season != prev_season:
            for t in list(team_ratings):
                team_ratings[t] = INITIAL_RATING + (team_ratings[t] - INITIAL_RATING) * (1 - SEASON_REGRESSION)
            for p in list(pitcher_ratings):
                pitcher_ratings[p] = INITIAL_RATING + (pitcher_ratings[p] - INITIAL_RATING) * (1 - SEASON_REGRESSION)

        team_h = team_ratings[g.home_team]
        team_a = team_ratings[g.away_team]
        pit_h = pitcher_ratings[g.home_pitcher] if g.home_pitcher else INITIAL_RATING
        pit_a = pitcher_ratings[g.away_pitcher] if g.away_pitcher else INITIAL_RATING

        diff = (team_h - team_a) + PITCHER_WEIGHT * (pit_h - pit_a) + HCA
        p_home = expected_home_win(diff)
        home_won = int(g.home_score > g.away_score)

        rows.append({
            "game_id": g.game_id,
            "date": g.date,
            "season": g.season,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_pitcher": g.home_pitcher,
            "away_pitcher": g.away_pitcher,
            "team_h": team_h, "team_a": team_a,
            "pit_h": pit_h, "pit_a": pit_a,
            "p_home": p_home,
            "home_won": home_won,
        })

        team_update = K_TEAM * (home_won - p_home)
        pitcher_update = K_PITCHER * (home_won - p_home)
        team_ratings[g.home_team] = team_h + team_update
        team_ratings[g.away_team] = team_a - team_update
        if g.home_pitcher:
            pitcher_ratings[g.home_pitcher] = pit_h + pitcher_update
        if g.away_pitcher:
            pitcher_ratings[g.away_pitcher] = pit_a - pitcher_update

        prev_season = g.season

    return pd.DataFrame(rows), dict(team_ratings), dict(pitcher_ratings)


def main() -> None:
    games = pd.read_csv(DATA / "games.csv")
    games = games.dropna(subset=["home_pitcher", "away_pitcher"])
    games["home_pitcher"] = games["home_pitcher"].fillna("")
    games["away_pitcher"] = games["away_pitcher"].fillna("")

    print("Running team-only Elo for comparison...")
    from elo import run_elo
    history_team, _ = run_elo(games)
    team_acc = ((history_team["p_home"] > 0.5) == history_team["home_won"].astype(bool)).mean()

    print("Running team + pitcher Elo...")
    history, final_team, final_pit = run_pitcher_elo(games)
    history.to_csv(DATA / "pitcher_elo_history.csv", index=False)

    pit_acc = ((history["p_home"] > 0.5) == history["home_won"].astype(bool)).mean()
    eps = 1e-15
    p = history["p_home"].clip(eps, 1 - eps)
    y = history["home_won"]
    log_loss_pit = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()

    p2 = history_team["p_home"].clip(eps, 1 - eps)
    y2 = history_team["home_won"]
    log_loss_team = -(y2 * np.log(p2) + (1 - y2) * np.log(1 - p2)).mean()

    print()
    print("=" * 60)
    print("TEAM-ONLY  vs  TEAM + PITCHER")
    print("=" * 60)
    print(f"  Accuracy:  {team_acc:.2%}    {pit_acc:.2%}    (Δ {(pit_acc-team_acc)*100:+.2f}pp)")
    print(f"  Log loss:  {log_loss_team:.4f}    {log_loss_pit:.4f}    (Δ {log_loss_pit-log_loss_team:+.4f})")

    print("\nFinal team ratings — end of 2024 (top 10):")
    for t, r in sorted(final_team.items(), key=lambda x: -x[1])[:10]:
        print(f"  {t}: {r:.0f}")

    print("\nTop pitchers by peak Elo across all 10 seasons:")
    home_pit = history[["date", "home_pitcher", "pit_h"]].rename(
        columns={"home_pitcher": "pitcher", "pit_h": "rating"}
    )
    away_pit = history[["date", "away_pitcher", "pit_a"]].rename(
        columns={"away_pitcher": "pitcher", "pit_a": "rating"}
    )
    all_pit = pd.concat([home_pit, away_pit], ignore_index=True)
    all_pit = all_pit[all_pit["pitcher"] != ""]
    starts = all_pit.groupby("pitcher").size()
    qualified = all_pit[all_pit["pitcher"].isin(starts[starts >= 50].index)]
    idx = qualified.groupby("pitcher")["rating"].idxmax()
    peaks = qualified.loc[idx].sort_values("rating", ascending=False).head(15)
    for _, row in peaks.iterrows():
        print(f"  {row['pitcher']}: {row['rating']:.0f}  ({row['date'][:10]})")


if __name__ == "__main__":
    main()
