"""Track recommended bets and their outcomes.

Usage:
  python tracker.py update    Fetch results for any pending bets
  python tracker.py summary   Show running performance
  python tracker.py recent    Show the last 20 bets logged
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import statsapi

from db import read_all, insert_or_update, update_bet, is_supabase_active


def ensure_log() -> pd.DataFrame:
    return read_all()


def log_bet(row: dict) -> None:
    insert_or_update(row)


def manual_resolve(bet_id, won: bool) -> None:
    df = read_all()
    if bet_id not in df.index:
        return
    row = df.loc[bet_id]
    stake = float(row["stake"]) if pd.notna(row["stake"]) else 0
    decimal = float(row["odds_decimal"]) if pd.notna(row["odds_decimal"]) else 0
    profit = stake * (decimal - 1) if won else -stake
    update_bet(bet_id, {
        "status": "won" if won else "lost",
        "profit": round(profit, 2),
    })


def update_pending() -> None:
    df = read_all()
    pending = df[df["status"] == "pending"]
    print(f"Checking {len(pending)} pending bets...")
    updated = 0
    skipped_manual = 0
    for bet_id, row in pending.iterrows():
        gid = row.get("game_id")
        if not gid or pd.isna(gid) or str(gid).strip() == "":
            skipped_manual += 1
            continue
        bet_type = str(row.get("bet_type") or "moneyline").lower()
        if bet_type not in ("moneyline", ""):
            skipped_manual += 1
            continue
        try:
            games = statsapi.schedule(game_id=int(float(gid)))
            if not games:
                continue
            g = games[0]
            if g["status"] != "Final":
                continue
            home_won = g["home_score"] > g["away_score"]
            bet_won = (row["bet_side"] == "home" and home_won) or (row["bet_side"] == "away" and not home_won)
            stake = float(row["stake"]) if pd.notna(row["stake"]) else 0
            decimal = float(row["odds_decimal"]) if pd.notna(row["odds_decimal"]) else 0
            profit = stake * (decimal - 1) if bet_won else -stake
            update_bet(bet_id, {
                "status": "won" if bet_won else "lost",
                "actual_winner": g["winning_team"],
                "profit": round(profit, 2),
            })
            updated += 1
        except Exception as e:
            print(f"  Error checking game {gid}: {e}")
    print(f"Updated {updated} bets. ({skipped_manual} manual bets skipped — resolve them in dashboard.)")


def summary() -> None:
    df = read_all()
    if df.empty:
        print("No bets logged yet.")
        return
    finished = df[df["status"].isin(["won", "lost"])]
    pending = df[df["status"] == "pending"]

    backend = "Supabase" if is_supabase_active() else "CSV (local)"
    print("=" * 60)
    print(f"BET TRACKER — running performance (storage: {backend})")
    print("=" * 60)
    print(f"  Total logged:    {len(df)}")
    print(f"  Pending:         {len(pending)}")
    print(f"  Completed:       {len(finished)}")
    if finished.empty:
        print("\nNo completed bets yet.")
        return

    finished = finished.copy()
    finished["profit_num"] = pd.to_numeric(finished["profit"], errors="coerce").fillna(0)
    finished["stake_num"] = pd.to_numeric(finished["stake"], errors="coerce").fillna(0)
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


def recent(n: int = 20) -> None:
    df = read_all()
    if df.empty:
        print("No bets logged yet.")
        return
    print(f"Last {min(n, len(df))} bets logged:\n")
    cols = ["game_date", "bet_team", "odds_american", "edge", "stake", "status", "profit"]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].tail(n).to_string())


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
