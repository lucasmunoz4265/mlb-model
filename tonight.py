"""Tonight's MLB slate: model predictions vs FanDuel lines, with edge analysis."""

import warnings
warnings.filterwarnings("ignore")

from datetime import date
from pathlib import Path

import pandas as pd
import requests
import statsapi

from pitcher_elo import (
    run_pitcher_elo, expected_home_win, HCA, PITCHER_WEIGHT, INITIAL_RATING,
)
from odds import american_to_prob, american_to_decimal

DATA = Path(__file__).parent / "data"

TEAM_ALIAS = {
    "Athletics": "Oakland Athletics",
}


def load_key() -> str:
    env = (Path(__file__).parent / ".env").read_text()
    for line in env.splitlines():
        if line.startswith("ODDS_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ODDS_API_KEY not found in .env")


def normalize(name: str) -> str:
    return TEAM_ALIAS.get(name, name)


def get_current_ratings():
    print("Computing current Elo ratings from all historical games...")
    games = pd.read_csv(DATA / "games.csv")
    games = games.fillna({"home_pitcher": "", "away_pitcher": ""})
    _, team_ratings, pitcher_ratings = run_pitcher_elo(games)
    print(f"  {len(team_ratings)} teams, {len(pitcher_ratings)} pitchers tracked\n")
    return team_ratings, pitcher_ratings


def fetch_tonight():
    today = date.today().isoformat()
    return today, statsapi.schedule(start_date=today, end_date=today)


def fetch_fanduel_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "bookmakers": "fanduel",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def predict_game(g, team_ratings, pitcher_ratings):
    home = normalize(g["home_name"])
    away = normalize(g["away_name"])
    home_pit_name = g.get("home_probable_pitcher") or ""
    away_pit_name = g.get("away_probable_pitcher") or ""

    team_h = team_ratings.get(home, INITIAL_RATING)
    team_a = team_ratings.get(away, INITIAL_RATING)
    pit_h = pitcher_ratings.get(home_pit_name, INITIAL_RATING)
    pit_a = pitcher_ratings.get(away_pit_name, INITIAL_RATING)

    diff = (team_h - team_a) + PITCHER_WEIGHT * (pit_h - pit_a) + HCA
    p_home = expected_home_win(diff)
    return {
        "game_id": g["game_id"],
        "game_date": g["game_date"],
        "home": home, "away": away,
        "home_pitcher": home_pit_name, "away_pitcher": away_pit_name,
        "team_h": team_h, "team_a": team_a,
        "pit_h": pit_h, "pit_a": pit_a,
        "p_home": p_home,
    }


def join_odds(prediction, odds_for_game):
    """Find FanDuel's ML lines for this matchup."""
    if not odds_for_game or not odds_for_game.get("bookmakers"):
        return None
    fd = next((b for b in odds_for_game["bookmakers"] if b["key"] == "fanduel"), None)
    if not fd:
        return None
    out = {"ml_home": None, "ml_away": None, "total": None,
           "over_odds": None, "under_odds": None,
           "spread_home": None, "spread_home_odds": None,
           "spread_away": None, "spread_away_odds": None}
    for m in fd["markets"]:
        if m["key"] == "h2h":
            for o in m["outcomes"]:
                team = normalize(o["name"])
                if team == prediction["home"]:
                    out["ml_home"] = o["price"]
                elif team == prediction["away"]:
                    out["ml_away"] = o["price"]
        elif m["key"] == "totals":
            for o in m["outcomes"]:
                if o["name"] == "Over":
                    out["total"] = o.get("point")
                    out["over_odds"] = o["price"]
                else:
                    out["under_odds"] = o["price"]
        elif m["key"] == "spreads":
            for o in m["outcomes"]:
                team = normalize(o["name"])
                if team == prediction["home"]:
                    out["spread_home"] = o.get("point")
                    out["spread_home_odds"] = o["price"]
                elif team == prediction["away"]:
                    out["spread_away"] = o.get("point")
                    out["spread_away_odds"] = o["price"]
    return out


def format_american(odds):
    if odds is None:
        return "—"
    return f"+{odds}" if odds > 0 else str(odds)


def main(edge_threshold=0.05, bankroll=500.0, kelly_frac=0.05) -> None:
    api_key = load_key()
    team_ratings, pitcher_ratings = get_current_ratings()

    today, games = fetch_tonight()
    print(f"Tonight's slate ({today}): {len(games)} games")

    odds_data, remaining = fetch_fanduel_odds(api_key)
    print(f"Odds API: {len(odds_data)} games returned, {remaining} requests left this month\n")

    odds_by_pair = {}
    for o in odds_data:
        key = (normalize(o["home_team"]), normalize(o["away_team"]))
        odds_by_pair[key] = o

    rows = []
    for g in games:
        pred = predict_game(g, team_ratings, pitcher_ratings)
        odds = join_odds(pred, odds_by_pair.get((pred["home"], pred["away"])))
        if odds is None:
            continue

        if odds["ml_home"] is not None and odds["ml_away"] is not None:
            home_imp = american_to_prob(odds["ml_home"])
            away_imp = american_to_prob(odds["ml_away"])
            total_imp = home_imp + away_imp
            market_p_home = home_imp / total_imp
            home_decimal = american_to_decimal(odds["ml_home"])
            away_decimal = american_to_decimal(odds["ml_away"])
            edge_home = pred["p_home"] * home_decimal - 1
            edge_away = (1 - pred["p_home"]) * away_decimal - 1
        else:
            market_p_home = home_decimal = away_decimal = None
            edge_home = edge_away = None

        rows.append({**pred, **odds,
                     "market_p_home": market_p_home,
                     "home_decimal": home_decimal, "away_decimal": away_decimal,
                     "edge_home": edge_home, "edge_away": edge_away})

    if not rows:
        print("No games with FanDuel odds available.")
        return

    print("=" * 130)
    print(f"{'Matchup':<55} {'Model':>7} {'Market':>7} {'FD Home':>9} {'FD Away':>9} {'EdgeH':>7} {'EdgeA':>7}")
    print("-" * 130)
    sorted_rows = sorted(rows, key=lambda r: max(r["edge_home"] or -1, r["edge_away"] or -1), reverse=True)
    for r in sorted_rows:
        matchup = f"{r['away']} @ {r['home']}"
        model = f"{r['p_home']*100:.0f}%"
        market = f"{r['market_p_home']*100:.0f}%" if r['market_p_home'] is not None else "—"
        fdh = format_american(r['ml_home'])
        fda = format_american(r['ml_away'])
        eh = f"{r['edge_home']*100:+.1f}%" if r['edge_home'] is not None else "—"
        ea = f"{r['edge_away']*100:+.1f}%" if r['edge_away'] is not None else "—"
        print(f"{matchup:<55} {model:>7} {market:>7} {fdh:>9} {fda:>9} {eh:>7} {ea:>7}")

    print()
    print("=" * 130)
    print(f"RECOMMENDED BETS (edge >= {edge_threshold*100:.0f}%, sized at {kelly_frac:g}x Kelly on ${bankroll:.0f} bankroll)")
    print("=" * 130)
    recs = []
    for r in sorted_rows:
        for side, p, decimal, ml, edge_val in [
            ("home", r["p_home"], r["home_decimal"], r["ml_home"], r["edge_home"]),
            ("away", 1 - r["p_home"], r["away_decimal"], r["ml_away"], r["edge_away"]),
        ]:
            if edge_val is None or edge_val < edge_threshold:
                continue
            team = r["home"] if side == "home" else r["away"]
            b = decimal - 1
            f_full = max(0, (p * (b + 1) - 1) / b)
            f_capped = min(f_full * kelly_frac, 0.01)
            stake = bankroll * f_capped
            recs.append((team, side, ml, edge_val, stake, r))

    if not recs:
        print("\n  No bets with edge >= threshold.")
        return

    from datetime import datetime
    from tracker import log_bet
    logged_at = datetime.now().isoformat(timespec="seconds")

    for team, side, ml, edge_val, stake, r in recs:
        print(f"\n  Bet: {team} ML at {format_american(ml)}")
        print(f"    Matchup: {r['away']} @ {r['home']}")
        print(f"    Pitchers: {r['away_pitcher']} ({r['pit_a']:.0f}) vs {r['home_pitcher']} ({r['pit_h']:.0f})")
        print(f"    Model: {r['p_home']*100:.1f}% home   Market (no-vig): {r['market_p_home']*100:.1f}% home")
        print(f"    Edge: {edge_val*100:+.1f}%   Recommended stake: ${stake:.2f}")

        model_p_bet = r["p_home"] if side == "home" else 1 - r["p_home"]
        market_p_bet = r["market_p_home"] if side == "home" else 1 - r["market_p_home"]
        decimal = r["home_decimal"] if side == "home" else r["away_decimal"]
        pitcher = r["home_pitcher"] if side == "home" else r["away_pitcher"]
        opp_pitcher = r["away_pitcher"] if side == "home" else r["home_pitcher"]

        log_bet({
            "logged_at": logged_at,
            "game_date": r["game_date"],
            "game_id": r["game_id"],
            "home_team": r["home"],
            "away_team": r["away"],
            "bet_side": side,
            "bet_team": team,
            "pitcher": pitcher,
            "opp_pitcher": opp_pitcher,
            "odds_american": ml,
            "odds_decimal": round(decimal, 4),
            "model_p": round(model_p_bet, 4),
            "market_p": round(market_p_bet, 4),
            "edge": round(edge_val, 4),
            "stake": round(stake, 2),
        })

    print(f"\n  ✓ Logged {len(recs)} recommendations to data/bet_log.csv")
    print(f"    After games finish, run: python tracker.py update")
    print(f"    See performance:         python tracker.py summary")


if __name__ == "__main__":
    main()
