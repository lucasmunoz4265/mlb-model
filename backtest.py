"""Backtest MLB model predictions vs historical sportsbook moneylines."""

from pathlib import Path

import numpy as np
import pandas as pd

from pitcher_elo import run_pitcher_elo
from odds import american_to_prob, american_to_decimal

DATA = Path(__file__).parent / "data"

TEAM_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Indians",
    "COL": "Colorado Rockies",
    "CWS": "Chicago White Sox",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF": "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}


def load_market_odds() -> pd.DataFrame:
    odds = pd.read_csv(DATA / "odds_raw/oddsDataMLB.csv")
    odds = odds.dropna(subset=["moneyLine", "oppMoneyLine"])
    odds["team_full"] = odds["team"].map(TEAM_MAP)
    odds["opp_full"] = odds["opponent"].map(TEAM_MAP)
    odds = odds.dropna(subset=["team_full", "opp_full"])
    odds["date"] = pd.to_datetime(odds["date"])

    home_view = odds.rename(columns={
        "team_full": "home_team",
        "opp_full": "away_team",
        "moneyLine": "home_ml",
        "oppMoneyLine": "away_ml",
    })[["date", "home_team", "away_team", "home_ml", "away_ml"]]
    return home_view


def join_predictions_with_odds(history: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"])
    return history.merge(odds, on=["date", "home_team", "away_team"], how="inner")


def add_market_probs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_imp = df["home_ml"].apply(american_to_prob)
    away_imp = df["away_ml"].apply(american_to_prob)
    total = home_imp + away_imp
    df["market_p_home"] = home_imp / total
    df["market_p_away"] = away_imp / total
    df["home_decimal"] = df["home_ml"].apply(american_to_decimal)
    df["away_decimal"] = df["away_ml"].apply(american_to_decimal)
    df["edge_home"] = df["p_home"] * df["home_decimal"] - 1
    df["edge_away"] = (1 - df["p_home"]) * df["away_decimal"] - 1
    return df


def simulate_bets(df: pd.DataFrame, edge_threshold: float, stake: float = 100.0) -> dict:
    bets_home = df[df["edge_home"] >= edge_threshold].copy()
    bets_home["won"] = bets_home["home_won"] == 1
    bets_home["payout"] = np.where(bets_home["won"], stake * (bets_home["home_decimal"] - 1), -stake)

    bets_away = df[df["edge_away"] >= edge_threshold].copy()
    bets_away["won"] = bets_away["home_won"] == 0
    bets_away["payout"] = np.where(bets_away["won"], stake * (bets_away["away_decimal"] - 1), -stake)

    all_bets = pd.concat([bets_home, bets_away], ignore_index=True)
    if len(all_bets) == 0:
        return {"threshold": edge_threshold, "bets": 0, "wins": 0, "win_rate": 0, "roi": 0, "profit": 0}

    return {
        "threshold": edge_threshold,
        "bets": len(all_bets),
        "wins": int(all_bets["won"].sum()),
        "win_rate": all_bets["won"].mean(),
        "profit": all_bets["payout"].sum(),
        "roi": all_bets["payout"].sum() / (len(all_bets) * stake),
    }


def main() -> None:
    print("Loading data...")
    games = pd.read_csv(DATA / "games.csv")
    games = games.fillna({"home_pitcher": "", "away_pitcher": ""})
    history, _, _ = run_pitcher_elo(games)
    odds = load_market_odds()
    print(f"  {len(games)} games, {len(odds)} games with odds")

    merged = join_predictions_with_odds(history, odds)
    print(f"  {len(merged)} games successfully joined (model x market)\n")

    df = add_market_probs(merged)

    print("=" * 70)
    print("MODEL vs MARKET")
    print("=" * 70)
    corr = df[["p_home", "market_p_home"]].corr().iloc[0, 1]
    mae = (df["p_home"] - df["market_p_home"]).abs().mean()
    print(f"Correlation (model prob vs market no-vig prob):  {corr:.3f}")
    print(f"Mean absolute disagreement:                       {mae:.3f} ({mae*100:.1f}pp)")
    acc = ((df["p_home"] > 0.5) == df["home_won"].astype(bool)).mean()
    market_acc = ((df["market_p_home"] > 0.5) == df["home_won"].astype(bool)).mean()
    print(f"Model picks favorite right:   {acc:.1%}")
    print(f"Market picks favorite right:  {market_acc:.1%}")

    print("\n" + "=" * 70)
    print("BACKTEST — flat $100 per bet on positive-edge games")
    print("=" * 70)
    print(f"\n{'Threshold':>10}  {'Bets':>6}  {'Wins':>5}  {'WinRate':>8}  {'Profit':>12}  {'ROI':>8}")
    print("-" * 70)
    for threshold in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]:
        r = simulate_bets(df, threshold)
        if r["bets"] > 0:
            print(f"  {threshold:>+8.2%}  {r['bets']:>6}  {r['wins']:>5}  {r['win_rate']:>7.1%}  ${r['profit']:>10,.0f}  {r['roi']:>+7.2%}")


if __name__ == "__main__":
    main()
