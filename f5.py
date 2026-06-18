"""First-5-innings (F5) model + backtest.

F5 outcomes are driven by the starting pitchers + offense, with the bullpen
removed — exactly what the team+pitcher Elo captures, minus its noisiest part.
We reconstruct historical F5 results from the inning-by-inning linescore, join
the PRE-GAME Elo ratings (from pitcher_elo's walk-forward history, no look-ahead),
fit the F5 win model, and validate out-of-sample vs the full-game baseline.

Build the F5 results cache (once, ~1 call/season, free statsapi):
    python f5.py --build --start 2022 --end 2025
Backtest the model:
    python f5.py --backtest
"""

from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import statsapi

from pitcher_elo import run_pitcher_elo, expected_home_win

DATA = Path(__file__).parent / "data"
F5_FILE = DATA / "f5_results.csv"


# --------------------------------------------------------------------------- #
# Build the historical F5 results cache                                        #
# --------------------------------------------------------------------------- #
def fetch_f5_results(start_year: int, end_year: int) -> pd.DataFrame:
    """Per game: first-5-innings runs for home/away, via one schedule call per
    season with the linescore hydrated."""
    rows = []
    for yr in range(start_year, end_year + 1):
        print(f"  fetching {yr} linescores...")
        sched = statsapi.get("schedule", {"sportId": 1,
            "startDate": f"{yr}-03-01", "endDate": f"{yr}-11-15", "hydrate": "linescore"})
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("detailedState") != "Final":
                    continue
                innings = (g.get("linescore") or {}).get("innings", [])
                if len(innings) < 5:
                    continue
                h5 = sum(int((i.get("home") or {}).get("runs", 0) or 0) for i in innings[:5])
                a5 = sum(int((i.get("away") or {}).get("runs", 0) or 0) for i in innings[:5])
                rows.append({"game_id": g["gamePk"], "h5": h5, "a5": a5})
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    print(f"  {len(df)} games with F5 results")
    return df


def build(start_year: int, end_year: int) -> None:
    df = fetch_f5_results(start_year, end_year)
    df.to_csv(F5_FILE, index=False)
    print(f"Saved {len(df)} F5 results to {F5_FILE}")


# --------------------------------------------------------------------------- #
# Assemble the modeling dataset: pre-game Elo features + F5 outcome            #
# --------------------------------------------------------------------------- #
def build_dataset() -> pd.DataFrame:
    games = pd.read_csv(DATA / "games.csv")
    games = games.dropna(subset=["home_pitcher", "away_pitcher"])
    games["home_pitcher"] = games["home_pitcher"].fillna("")
    games["away_pitcher"] = games["away_pitcher"].fillna("")
    history, _, _ = run_pitcher_elo(games)          # pre-game ratings per game

    f5 = pd.read_csv(F5_FILE)
    df = history.merge(f5, on="game_id", how="inner")
    df["team_diff"] = df["team_h"] - df["team_a"]
    df["pit_diff"] = df["pit_h"] - df["pit_a"]
    df["f5_margin"] = df["h5"] - df["a5"]
    df["f5_home_win"] = (df["f5_margin"] > 0).astype(int)
    df["f5_tie"] = (df["f5_margin"] == 0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Backtest                                                                     #
# --------------------------------------------------------------------------- #
def _acc_logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    acc = ((p > 0.5) == y.astype(bool)).mean()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return acc, ll


def backtest() -> None:
    from sklearn.linear_model import LogisticRegression
    full = build_dataset()
    tie_rate = full["f5_tie"].mean()
    # The F5 ML is 2-way with ties pushing, so the bet is decided ONLY among
    # non-tied games — measure the model there.
    df = full[full["f5_tie"] == 0].copy()
    seasons = sorted(df["season"].unique())
    print(f"F5 dataset: {len(full):,} games ({len(df):,} decided), seasons {seasons[0]}–{seasons[-1]}")
    print(f"  Ties through 5: {tie_rate:.1%} (these PUSH) | Home wins among decided: {df['f5_home_win'].mean():.1%}")

    # Walk-forward: train on all but the last 2 seasons, test on the last 2.
    test_seasons = seasons[-2:]
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]
    print(f"  Train: {len(train):,} games ({seasons[0]}–{test_seasons[0]-1}) | "
          f"Test: {len(test):,} games ({test_seasons[0]}–{test_seasons[-1]})")

    y_test = test["f5_home_win"].values

    # Baseline A: full-game Elo prob used directly to predict F5.
    accA, llA = _acc_logloss(test["p_home"].values, y_test)

    # Model B: logistic regression fit to F5 outcomes on the two Elo differentials.
    X_tr = train[["team_diff", "pit_diff"]].values
    X_te = test[["team_diff", "pit_diff"]].values
    clf = LogisticRegression()
    clf.fit(X_tr, train["f5_home_win"].values)
    pB = clf.predict_proba(X_te)[:, 1]
    accB, llB = _acc_logloss(pB, y_test)

    # Baseline: always pick home.
    base_home = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n{'='*60}\nF5 HOME-LEAD PREDICTION (out-of-sample test)\n{'-'*60}")
    print(f"  {'Model':<28}{'Acc':>8}{'LogLoss':>10}")
    print(f"  {'always home':<28}{base_home:>7.1%}{'—':>10}")
    print(f"  {'full-game Elo (baseline)':<28}{accA:>7.1%}{llA:>10.4f}")
    print(f"  {'F5-tuned logistic':<28}{accB:>7.1%}{llB:>10.4f}")
    tw, pw = clf.coef_[0]
    print(f"\n  F5 weights: team_diff×{tw:.4f}, pit_diff×{pw:.4f}, intercept {clf.intercept_[0]:+.3f}")
    print(f"  Pitcher/team weight ratio: {pw/tw:.2f}x  "
          f"(full-game model uses 0.50x — higher here = pitchers matter more in F5, as expected)")
    print(f"\n  Read: if F5-tuned beats the full-game baseline on log loss, the F5-specific")
    print(f"  weighting is real signal. Ties ({df['f5_tie'].mean():.0%}) need a 3-way market treatment later.")


def backtest_totals() -> None:
    """F5 TOTAL runs model. Predict first-5 total from the starters' + offense
    Elo, then Poisson around it for over/under, validated out-of-sample.
    Totals are where the project notes say books are softest."""
    from sklearn.linear_model import LinearRegression
    from props import over_prob  # Poisson P(total > line)
    df = build_dataset()
    df["f5_total"] = df["h5"] + df["a5"]
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-2:]
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]

    feats = ["pit_h", "pit_a", "team_h", "team_a"]
    reg = LinearRegression().fit(train[feats].values, train["f5_total"].values)
    pred = reg.predict(test[feats].values)
    actual = test["f5_total"].values
    mae = np.abs(pred - actual).mean()
    base_pred = train["f5_total"].mean()
    base_mae = np.abs(base_pred - actual).mean()

    print(f"F5 TOTALS — {len(df):,} games, mean F5 total {df['f5_total'].mean():.2f} runs")
    print(f"  Train {len(train):,} / Test {len(test):,}")
    print(f"\n{'='*60}\nF5 TOTAL RUNS — out-of-sample\n{'-'*60}")
    print(f"  Predict league-avg ({base_pred:.2f}) every game:  MAE {base_mae:.3f}")
    print(f"  Elo-features model:                       MAE {mae:.3f}  ({(mae-base_mae)/base_mae*100:+.1f}%)")
    print(f"\n  Over/under calibration (Poisson around predicted total):")
    print(f"    {'Line':>5}{'ModelP(over)':>14}{'Actual over':>13}{'Diff':>7}")
    for ln in (3.5, 4.5, 5.5):
        mp = np.mean([over_prob(ln, max(0.5, p)) for p in pred])
        af = (actual > ln).mean()
        print(f"    {ln:>5.1f}{mp*100:>13.0f}%{af*100:>12.0f}%{(mp-af)*100:>+6.0f}%")
    print(f"\n  Read: if the Elo model barely beats the league-avg baseline, win-based")
    print(f"  Elo doesn't capture run environment — we'd need run stats (ERA/wOBA) for totals.")


def main():
    ap = argparse.ArgumentParser(description="F5 (first 5 innings) model + backtest.")
    ap.add_argument("--build", action="store_true", help="fetch + cache historical F5 results")
    ap.add_argument("--start", type=int, default=2022)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--backtest", action="store_true", help="run the F5 win backtest")
    ap.add_argument("--totals", action="store_true", help="run the F5 totals backtest")
    args = ap.parse_args()
    if args.build:
        build(args.start, args.end)
    if args.backtest:
        backtest()
    if args.totals:
        backtest_totals()
    if not (args.build or args.backtest or args.totals):
        ap.print_help()


if __name__ == "__main__":
    main()
