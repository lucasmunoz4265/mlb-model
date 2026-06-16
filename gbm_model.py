"""Gradient boosting model trained walk-forward, compared to Elo baselines."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import build_features
from backtest import load_market_odds, join_predictions_with_odds, add_market_probs, simulate_bets

DATA = Path(__file__).parent / "data"

FEATURE_COLS = [
    "team_elo_home", "team_elo_away",
    "pit_elo_home", "pit_elo_away",
    "elo_p_home",
    "home_runs_for_avg", "home_runs_against_avg", "home_winrate",
    "away_runs_for_avg", "away_runs_against_avg", "away_winrate",
    "rest_home", "rest_away",
    "park_factor",
    "home_pit_era", "home_pit_whip", "home_pit_k9", "home_pit_bb9",
    "away_pit_era", "away_pit_whip", "away_pit_k9", "away_pit_bb9",
]

FIRST_TEST_SEASON = 2018


def walk_forward_predict(features: pd.DataFrame) -> pd.DataFrame:
    features = features.sort_values("date").reset_index(drop=True)
    test_seasons = sorted(features.loc[features["season"] >= FIRST_TEST_SEASON, "season"].unique())

    parts = []
    for season in test_seasons:
        train = features[features["season"] < season]
        test = features[features["season"] == season]
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
        )
        model.fit(train[FEATURE_COLS], train["home_won"])
        probs = model.predict_proba(test[FEATURE_COLS])[:, 1]
        test = test.copy()
        test["gbm_p_home"] = probs
        parts.append(test)
        print(f"  {season}: trained on {len(train)} games, predicted {len(test)}")

    return pd.concat(parts, ignore_index=True)


def log_loss(p: pd.Series, y: pd.Series) -> float:
    eps = 1e-15
    p = p.clip(eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def backtest_with(preds: pd.DataFrame, p_col: str, odds: pd.DataFrame) -> pd.DataFrame:
    df = preds[["date", "home_team", "away_team", "home_won", p_col]].copy()
    df = df.rename(columns={p_col: "p_home"})
    df["date"] = pd.to_datetime(df["date"])
    merged = df.merge(odds, on=["date", "home_team", "away_team"], how="inner")
    return add_market_probs(merged)


def main() -> None:
    print("Building features...")
    games = pd.read_csv(DATA / "games.csv")
    features = build_features(games)
    print(f"  {len(features)} games\n")

    print("Walk-forward training:")
    preds = walk_forward_predict(features)
    preds.to_csv(DATA / "gbm_predictions.csv", index=False)
    print(f"\n  {len(preds)} predictions saved\n")

    y = preds["home_won"]
    elo_acc = ((preds["elo_p_home"] > 0.5) == y.astype(bool)).mean()
    gbm_acc = ((preds["gbm_p_home"] > 0.5) == y.astype(bool)).mean()
    elo_ll = log_loss(preds["elo_p_home"], y)
    gbm_ll = log_loss(preds["gbm_p_home"], y)

    print("=" * 70)
    print("MODEL ACCURACY (test seasons 2018-2024)")
    print("=" * 70)
    print(f"{'':30s}  {'Elo':>10s}  {'GBM':>10s}  {'Δ':>10s}")
    print(f"{'Pick favorite right':30s}  {elo_acc:>10.2%}  {gbm_acc:>10.2%}  {(gbm_acc-elo_acc)*100:>+8.2f}pp")
    print(f"{'Log loss (lower=better)':30s}  {elo_ll:>10.4f}  {gbm_ll:>10.4f}  {gbm_ll-elo_ll:>+10.4f}")

    print("\n" + "=" * 70)
    print("BACKTEST ROI BY EDGE THRESHOLD")
    print("=" * 70)
    odds = load_market_odds()
    elo_bt = backtest_with(preds, "elo_p_home", odds)
    gbm_bt = backtest_with(preds, "gbm_p_home", odds)

    print(f"{'Threshold':>10}  {'Elo bets':>10}  {'Elo ROI':>10}  {'GBM bets':>10}  {'GBM ROI':>10}")
    print("-" * 60)
    for thr in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]:
        re = simulate_bets(elo_bt, thr)
        rg = simulate_bets(gbm_bt, thr)
        print(f"  {thr:>+8.2%}  {re['bets']:>10}  {re['roi']:>+9.2%}  {rg['bets']:>10}  {rg['roi']:>+9.2%}")


if __name__ == "__main__":
    main()
