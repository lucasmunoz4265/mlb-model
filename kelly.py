"""Kelly + flat bet sizing simulator for MLB model."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from backtest import load_market_odds, add_market_probs

DATA = Path(__file__).parent / "data"
INITIAL_BANKROLL = 1000.0
MAX_BET_FRACTION = 0.01
MIN_BANKROLL = 10.0


def kelly_fraction(p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    f = (p * (b + 1) - 1) / b
    return max(0.0, f)


def select_bet(row) -> Optional[Tuple[float, float, bool]]:
    if row.edge_home > 0:
        return row.p_home, row.home_decimal, row.home_won == 1
    if row.edge_away > 0:
        return 1 - row.p_home, row.away_decimal, row.home_won == 0
    return None


def simulate_kelly(df: pd.DataFrame, fraction: float, cap: float = MAX_BET_FRACTION) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_dd = 0.0
    n_bets = n_wins = 0
    busted = False

    for row in df.itertuples(index=False):
        if bankroll < MIN_BANKROLL:
            busted = True
            break
        pick = select_bet(row)
        if pick is None:
            continue
        p, decimal, won = pick
        f_full = kelly_fraction(p, decimal)
        if f_full <= 0:
            continue
        f = min(f_full * fraction, cap)
        stake = bankroll * f
        n_bets += 1
        if won:
            bankroll += stake * (decimal - 1)
            n_wins += 1
        else:
            bankroll -= stake
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)

    return _result(f"{fraction:g}x Kelly", bankroll, n_bets, n_wins, max_dd, busted)


def simulate_flat(df: pd.DataFrame, stake: float) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_dd = 0.0
    n_bets = n_wins = 0
    busted = False

    for row in df.itertuples(index=False):
        if bankroll < stake:
            busted = True
            break
        pick = select_bet(row)
        if pick is None:
            continue
        _, decimal, won = pick
        n_bets += 1
        if won:
            bankroll += stake * (decimal - 1)
            n_wins += 1
        else:
            bankroll -= stake
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)

    return _result(f"Flat ${stake:.0f}", bankroll, n_bets, n_wins, max_dd, busted)


def _result(label, bankroll, n_bets, n_wins, max_dd, busted):
    return {
        "strategy": label,
        "busted": busted,
        "final": bankroll,
        "growth": bankroll / INITIAL_BANKROLL,
        "bets": n_bets,
        "wins": n_wins,
        "win_rate": n_wins / n_bets if n_bets else 0,
        "max_drawdown": max_dd,
    }


def main() -> None:
    preds = pd.read_csv(DATA / "gbm_predictions.csv")
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.rename(columns={"elo_p_home": "p_home"})

    odds = load_market_odds()
    df = preds.merge(odds, on=["date", "home_team", "away_team"], how="inner")
    df = add_market_probs(df)
    print(f"  {len(df)} games joined with odds\n")

    print("=" * 100)
    print("MLB BANKROLL SIMULATION — starting $1,000, using Elo predictions")
    print("=" * 100)
    print(f"{'EdgeMin':>8} {'Strategy':>20}  {'Bets':>5}  {'WinRate':>7}  {'Final $':>10}  {'Growth':>7}  {'MaxDD':>6}  {'Bust?':>5}")
    print("-" * 100)

    def show(r, edge):
        bust = "Yes" if r["busted"] else ""
        print(f"{edge:>7.0%}  {r['strategy']:>20}  {r['bets']:>5}  {r['win_rate']:>6.1%}  ${r['final']:>9,.0f}  {r['growth']:>6.2f}x  {r['max_drawdown']:>5.1%}  {bust:>5}")

    for edge_min in [0.05, 0.10, 0.15]:
        sub = df[(df["edge_home"] >= edge_min) | (df["edge_away"] >= edge_min)].copy()
        if len(sub) < 50:
            continue
        for s in [simulate_flat(sub, 10), simulate_flat(sub, 25),
                  simulate_kelly(sub, 0.05), simulate_kelly(sub, 0.1), simulate_kelly(sub, 0.25)]:
            show(s, edge_min)
        print()


if __name__ == "__main__":
    main()
