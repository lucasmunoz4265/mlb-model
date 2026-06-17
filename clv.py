"""Closing Line Value (CLV) logger — moneyline bets.

CLV is the single best early signal of whether a model has real edge: if you
consistently get a better price than the market's CLOSING line, you're beating
the market regardless of any individual win/loss. This script snapshots the
current FanDuel moneyline (run it close to first pitch = the closing line),
matches it to each pending moneyline bet in the log, and records:

    close_odds     the closing American odds for the side you bet
    close_decimal  decimal form
    clv_pct        (your decimal odds / closing decimal) - 1
                   > 0  → you beat the close (got a better price) = good
                   < 0  → the market moved against you

Only moneylines for now (1 Odds API credit per run — h2h covers the whole
slate). Props CLV is per-game/credit-heavy and waits until the model is sharper.

CLI:
    python clv.py            # capture closing lines for pending ML bets (1 credit)
    python clv.py --dry-run  # compute + print, write nothing (still 1 credit to fetch)
    python clv.py --summary  # show CLV stats from what's already recorded (0 credits)
"""

from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from tonight import normalize, load_key, format_american
from odds import american_to_decimal, american_to_prob
from db import read_all, update_bet, is_supabase_active

H2H_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"


def fetch_closing_h2h(api_key: str) -> tuple:
    """FanDuel moneyline for the whole slate — 1 credit (h2h only)."""
    r = requests.get(H2H_URL, params={
        "apiKey": api_key, "regions": "us", "markets": "h2h",
        "bookmakers": "fanduel", "oddsFormat": "american",
    }, timeout=15)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def closing_price_for(bet_team: str, odds_for_game: dict):
    """The current FanDuel ML for the team the bet is on, or None."""
    if not odds_for_game:
        return None
    fd = next((b for b in odds_for_game.get("bookmakers", []) if b["key"] == "fanduel"), None)
    if not fd:
        return None
    for m in fd.get("markets", []):
        if m["key"] != "h2h":
            continue
        for o in m["outcomes"]:
            if normalize(o["name"]) == normalize(bet_team):
                return o["price"]
    return None


def capture(dry_run: bool = False) -> None:
    df = read_all()
    pending_ml = df[(df["status"] == "pending")
                    & (df["bet_type"].fillna("").astype(str).str.lower() == "moneyline")]
    if pending_ml.empty:
        print("No pending moneyline bets to capture CLV for.")
        return

    api_key = os.environ.get("ODDS_API_KEY") or load_key()  # env first (CI), then .env
    odds_data, remaining = fetch_closing_h2h(api_key)
    # Only pre-game events — never overwrite a closing line with in-game odds.
    now_utc = datetime.now(timezone.utc)
    def pregame(o):
        try:
            return datetime.fromisoformat(o["commence_time"].replace("Z", "+00:00")) > now_utc
        except Exception:
            return True
    pre = [o for o in odds_data if pregame(o)]
    print(f"Fetched FanDuel ML for {len(odds_data)} games (1 credit); "
          f"{len(pre)} still pre-game. Credits remaining: {remaining}")
    odds_by_pair = {(normalize(o["home_team"]), normalize(o["away_team"])): o for o in pre}

    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    for bet_id, row in pending_ml.iterrows():
        pair = (normalize(str(row["home_team"])), normalize(str(row["away_team"])))
        close_ml = closing_price_for(str(row["bet_team"]), odds_by_pair.get(pair))
        if close_ml is None:
            print(f"  no current ML found for {row['bet_team']} ({pair[1]} @ {pair[0]}) — skipped")
            continue
        bet_dec = float(row["odds_decimal"]) if pd.notna(row["odds_decimal"]) else \
            american_to_decimal(float(row["odds_american"]))
        close_dec = american_to_decimal(close_ml)
        clv_pct = bet_dec / close_dec - 1
        rows.append((bet_id, row, close_ml, close_dec, clv_pct))
        if not dry_run:
            update_bet(bet_id, {
                "close_odds": float(close_ml), "close_decimal": round(close_dec, 4),
                "clv_pct": round(clv_pct, 4), "clv_at": now,
            })

    if not rows:
        print("No bets matched the current slate.")
        return

    print(f"\n{'Bet':<26} {'Your odds':>10} {'Close':>8} {'CLV':>8}")
    print("-" * 56)
    for _bid, row, close_ml, _cd, clv in rows:
        print(f"{str(row['bet_team'])[:26]:<26} {format_american(int(row['odds_american'])):>10} "
              f"{format_american(int(close_ml)):>8} {clv*100:>+7.1f}%")
    avg = sum(r[4] for r in rows) / len(rows)
    beat = sum(1 for r in rows if r[4] > 0)
    print("-" * 56)
    print(f"Avg CLV: {avg*100:+.1f}%  |  Beat the close: {beat}/{len(rows)}")
    print(("DRY RUN — nothing written." if dry_run else
           f"Saved to {'Supabase' if is_supabase_active() else 'CSV'}.")
          + "  (Run near first pitch for true closing lines.)")


def summary() -> None:
    df = read_all()
    have = df[pd.to_numeric(df.get("clv_pct"), errors="coerce").notna()] \
        if "clv_pct" in df.columns else df.iloc[0:0]
    if have.empty:
        print("No CLV recorded yet. Run `python clv.py` near game time.")
        return
    clv = pd.to_numeric(have["clv_pct"], errors="coerce")
    print(f"CLV recorded on {len(have)} bets")
    print(f"  Average CLV:      {clv.mean()*100:+.2f}%")
    print(f"  Beat the close:   {(clv > 0).sum()}/{len(have)} ({(clv > 0).mean()*100:.0f}%)")
    print(f"  Best / worst:     {clv.max()*100:+.1f}% / {clv.min()*100:+.1f}%")
    print("\n  (+) average CLV over many bets = the model is finding real value.")


def main():
    ap = argparse.ArgumentParser(description="Log closing-line value for pending moneyline bets.")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, write nothing")
    ap.add_argument("--summary", action="store_true", help="show recorded CLV stats (0 credits)")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        capture(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
