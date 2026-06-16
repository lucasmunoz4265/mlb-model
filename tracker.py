"""Track recommended bets and their outcomes.

Usage:
  python tracker.py update    Fetch results for any pending bets
  python tracker.py summary   Show running performance
  python tracker.py recent    Show the last 20 bets logged
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import statsapi

DATA = Path(__file__).parent / "data"
LOG_FILE = DATA / "bet_log.csv"

COLUMNS = [
    "logged_at", "game_date", "game_id", "home_team", "away_team",
    "bet_side", "bet_team", "pitcher", "opp_pitcher",
    "odds_american", "odds_decimal",
    "model_p", "market_p", "edge", "stake",
    "status", "actual_winner", "profit",
]


def ensure_log() -> pd.DataFrame:
    if not LOG_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(LOG_FILE, index=False)
    return pd.read_csv(LOG_FILE)


def log_bet(row: dict) -> None:
    """Append one recommended bet to the log as 'pending'.
    If a pending bet already exists for this game+side, update it instead."""
    df = ensure_log()
    row = {**row, "status": "pending", "actual_winner": "", "profit": ""}
    existing_mask = (
        (df["game_id"].astype(str) == str(row["game_id"]))
        & (df["bet_side"] == row["bet_side"])
        & (df["status"] == "pending")
    )
    if existing_mask.any():
        for col, val in row.items():
            df.loc[existing_mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


def update_pending() -> None:
    df = ensure_log()
    pending = df[df["status"] == "pending"]
    print(f"Checking {len(pending)} pending bets...")
    updated = 0
    for idx, row in pending.iterrows():
        try:
            games = statsapi.schedule(game_id=int(row["game_id"]))
            if not games:
                continue
            g = games[0]
            if g["status"] != "Final":
                continue
            home_won = g["home_score"] > g["away_score"]
            bet_won = (row["bet_side"] == "home" and home_won) or (row["bet_side"] == "away" and not home_won)
            stake = float(row["stake"])
            decimal = float(row["odds_decimal"])
            profit = stake * (decimal - 1) if bet_won else -stake
            df.at[idx, "status"] = "won" if bet_won else "lost"
            df.at[idx, "actual_winner"] = g["winning_team"]
            df.at[idx, "profit"] = round(profit, 2)
            updated += 1
        except Exception as e:
            print(f"  Error checking game {row['game_id']}: {e}")
    df.to_csv(LOG_FILE, index=False)
    print(f"Updated {updated} bets.")


def summary() -> None:
    df = ensure_log()
    if df.empty:
        print("No bets logged yet. Run tonight.py to generate recommendations.")
        return
    finished = df[df["status"].isin(["won", "lost"])]
    pending = df[df["status"] == "pending"]

    print("=" * 60)
    print("BET TRACKER — running performance")
    print("=" * 60)
    print(f"  Total logged:    {len(df)}")
    print(f"  Pending:         {len(pending)}")
    print(f"  Completed:       {len(finished)}")
    if finished.empty:
        print("\nNo completed bets yet — come back after games finish.")
        return

    finished["profit_num"] = finished["profit"].astype(float)
    finished["stake_num"] = finished["stake"].astype(float)

    wins = (finished["status"] == "won").sum()
    losses = (finished["status"] == "lost").sum()
    total_wagered = finished["stake_num"].sum()
    total_profit = finished["profit_num"].sum()
    roi = total_profit / total_wagered if total_wagered > 0 else 0
    win_rate = wins / len(finished)

    print(f"\n  Win rate:        {win_rate:.1%} ({wins}-{losses})")
    print(f"  Total wagered:   ${total_wagered:.2f}")
    print(f"  Total profit:    ${total_profit:+.2f}")
    print(f"  ROI:             {roi*100:+.2f}%")

    print("\nBy edge bucket:")
    finished["edge_num"] = finished["edge"].astype(float)
    for lo, hi in [(0.05, 0.08), (0.08, 0.12), (0.12, 0.20), (0.20, 1.0)]:
        sub = finished[(finished["edge_num"] >= lo) & (finished["edge_num"] < hi)]
        if len(sub) == 0:
            continue
        sub_wins = (sub["status"] == "won").sum()
        sub_wagered = sub["stake_num"].sum()
        sub_profit = sub["profit_num"].sum()
        sub_roi = sub_profit / sub_wagered if sub_wagered > 0 else 0
        print(f"  edge {lo:.0%}-{hi:.0%}: {len(sub):>3} bets, win {sub_wins/len(sub)*100:>4.1f}%, ROI {sub_roi*100:+5.2f}%")


def recent(n: int = 20) -> None:
    df = ensure_log()
    if df.empty:
        print("No bets logged yet.")
        return
    print(f"Last {min(n, len(df))} bets logged:\n")
    cols = ["game_date", "bet_team", "odds_american", "edge", "stake", "status", "profit"]
    print(df[cols].tail(n).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "update":
        update_pending()
    elif cmd == "summary":
        summary()
    elif cmd == "recent":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        recent(n)
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
