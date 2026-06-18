"""Line-shopping + arbitrage tracker for the Puerto Rico books.

For each game it pulls every book's moneyline in ONE Odds API call (cost is per
market, NOT per book — so all your books = 1 credit), shows the BEST price for
each side and which book has it, and flags arbitrage (when the best prices on
opposite sides together imply < 100% → locked profit).

This is Path-A value: free EV from always taking the best number, plus the
occasional guaranteed arb — no need to out-model a sharp book.

CLI:
    python shop.py                 # moneyline line-shop + arb (1 credit)
    python shop.py --market totals # totals (2-way over/under per total line)
    python shop.py --all-books     # show every US book, not just the PR set
"""

from __future__ import annotations

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import requests

from odds import american_to_decimal

# The Odds API bookmaker keys for the books live in Puerto Rico.
# (Caesars appears as williamhill_us on the API; both kept just in case.)
PR_BOOKS = {"draftkings", "fanduel", "betmgm", "williamhill_us", "caesars"}
H2H_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"


def load_key() -> str:
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k
    from pathlib import Path
    for line in (Path(__file__).parent / ".env").read_text().splitlines():
        if line.startswith("ODDS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ODDS_API_KEY not found")


def american(dec: float) -> str:
    """Decimal -> American odds string, for display."""
    if dec >= 2:
        return f"+{round((dec - 1) * 100)}"
    return f"{round(-100 / (dec - 1))}"


def fetch(api_key: str, market: str) -> tuple:
    r = requests.get(H2H_URL, params={"apiKey": api_key, "regions": "us",
        "markets": market, "oddsFormat": "american"}, timeout=15)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def best_prices(event: dict, market: str, books: set) -> dict:
    """{outcome_name: (best_decimal, book_key)} across the allowed books.
    For totals, outcome key is e.g. 'Over 8.5'."""
    best = {}
    for b in event.get("bookmakers", []):
        if b["key"] not in books:
            continue
        for m in b.get("markets", []):
            if m["key"] != market:
                continue
            for o in m["outcomes"]:
                name = o["name"]
                if o.get("point") is not None:
                    name = f"{o['name']} {o['point']}"
                dec = american_to_decimal(o["price"])
                if name not in best or dec > best[name][0]:
                    best[name] = (dec, b["key"])
    return best


def arb(best: dict, sides: tuple):
    """If the two named sides exist, return (roi, payout_inv) for an arb, else None."""
    if not all(s in best for s in sides):
        return None
    inv = sum(1.0 / best[s][0] for s in sides)
    return (1.0 / inv - 1.0, inv) if inv < 1.0 else None


def main():
    ap = argparse.ArgumentParser(description="PR line-shopping + arbitrage tracker.")
    ap.add_argument("--market", default="h2h", choices=["h2h", "totals"])
    ap.add_argument("--all-books", action="store_true", help="include every US book")
    args = ap.parse_args()

    api_key = load_key()
    data, remaining = fetch(api_key, args.market)
    books = None if args.all_books else PR_BOOKS
    seen = sorted({b["key"] for e in data for b in e.get("bookmakers", [])})
    if not args.all_books:
        seen = [b for b in seen if b in PR_BOOKS]
    print(f"{len(data)} games | books used: {', '.join(seen) or '(none matched PR set)'} "
          f"| credits left: {remaining}\n")

    arbs = []
    for e in data:
        away, home = e["away_team"], e["home_team"]
        allowed = {b["key"] for b in e.get("bookmakers", [])} if args.all_books else PR_BOOKS
        best = best_prices(e, args.market, allowed)
        if not best:
            continue
        print(f"{away} @ {home}")
        for name, (dec, book) in sorted(best.items()):
            print(f"   best {name:<22} {american(dec):>6}  @ {book}")
        # Arb check: moneyline = the two team names; totals = Over/Under per line.
        if args.market == "h2h":
            res = arb(best, (away, home))
            if res:
                roi, _ = res
                arbs.append((f"{away} @ {home}", roi))
                print(f"   *** ARB: {roi*100:.2f}% guaranteed "
                      f"({away} @ {best[away][1]} + {home} @ {best[home][1]}) ***")
        else:
            points = {n.rsplit(" ", 1)[1] for n in best if n.startswith(("Over", "Under"))}
            for pt in points:
                res = arb(best, (f"Over {pt}", f"Under {pt}"))
                if res:
                    roi, _ = res
                    arbs.append((f"{away} @ {home} O/U {pt}", roi))
                    print(f"   *** ARB on {pt}: {roi*100:.2f}% guaranteed ***")
        print()

    print("=" * 50)
    if arbs:
        print(f"{len(arbs)} arbitrage opportunit(ies):")
        for label, roi in sorted(arbs, key=lambda x: -x[1]):
            print(f"  {roi*100:>5.2f}%  {label}")
    else:
        print("No arbs right now — but the 'best @ book' lines above are still "
              "free EV vs betting one book blindly.")


if __name__ == "__main__":
    main()
