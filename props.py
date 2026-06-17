"""Player prop modeling — pitcher strikeout props (v1).

Models a starter's strikeouts as Poisson(lambda) where

    lambda = (K/9 ÷ 9) × expected_IP × opponent_K_adjustment

then compares model P(over)/P(under) to FanDuel's prop line to find +EV bets.
Strikeout counts are well-approximated by a Poisson process, so this is a
principled first model. Inputs:
  - K/9 and IP/start: pitcher's current-season stats (statsapi), falling back
    to the most recent season in data/pitcher_season_stats.csv.
  - opponent_K_adjustment: opposing team's K-rate ÷ league average (statsapi),
    falling back to 1.0 (neutral) if team hitting stats can't be fetched.

CREDIT NOTE: The Odds API charges (markets × games) credits. We request only
the `pitcher_strikeouts` market = ~1 credit per game. Results are cached per
day in data/props_cache_<date>.json so re-runs cost nothing.

CLI:
    python props.py             # model tonight's K props (fetches odds — uses credits!)
    python props.py --limit 1   # only the first game (~1 credit) — for testing
    python props.py --cached    # use today's cache only, no API call (0 credits)
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
warnings.filterwarnings("ignore")

from datetime import date
from pathlib import Path

import pandas as pd
import requests
import statsapi

from tonight import normalize, load_key, format_american
from odds import american_to_prob, american_to_decimal

DATA = Path(__file__).parent / "data"

PROP_MARKET = "pitcher_strikeouts"
DEFAULT_IP_PER_START = 5.4   # league-typical modern starter when we lack data
IP_BOUNDS = (3.5, 7.0)


# --------------------------------------------------------------------------- #
# Poisson helpers (no scipy dependency)                                        #
# --------------------------------------------------------------------------- #
def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)."""
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def over_prob(point: float, lam: float) -> float:
    """P(strikeouts beats the line). For a half-point line (e.g. 5.5),
    'Over' means X >= 6, i.e. 1 - P(X <= 5) = 1 - cdf(floor(point))."""
    return max(0.0, min(1.0, 1.0 - poisson_cdf(math.floor(point), lam)))


# --------------------------------------------------------------------------- #
# Pitcher profile: K/9 and innings-per-start                                   #
# --------------------------------------------------------------------------- #
def _parse_ip(ip) -> float:
    """MLB innings-pitched notation: '45.1' = 45 + 1/3, '45.2' = 45 + 2/3."""
    try:
        s = str(ip)
        whole, _, frac = s.partition(".")
        outs = {"": 0, "0": 0, "1": 1, "2": 2}.get(frac, 0)
        return int(whole) + outs / 3.0
    except Exception:
        return 0.0


def _profile_from_csv(name: str) -> dict | None:
    path = DATA / "pitcher_season_stats.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df[df["player_name"].str.lower() == name.lower()]
    if rows.empty:
        return None
    row = rows.sort_values("season").iloc[-1]  # most recent season on file
    gs = float(row.get("games_started") or 0)
    ip = float(row.get("innings_pitched") or 0)
    ip_per_start = ip / gs if gs > 0 else DEFAULT_IP_PER_START
    return {
        "k_per_9": float(row.get("k_per_9") or 0) or None,
        "ip_per_start": ip_per_start,
        "source": f"csv {int(row['season'])}",
    }


def get_pitcher_profile(name: str, season: int) -> dict | None:
    """Current-season K/9 and IP/start via statsapi, falling back to CSV history."""
    if not name:
        return None
    try:
        matches = statsapi.lookup_player(name)
        if matches:
            pid = matches[0]["id"]
            data = statsapi.player_stat_data(pid, group="pitching", type="season",
                                             season=season)
            splits = data.get("stats") or []
            if splits:
                stat = splits[0].get("stats", {})
                gs = float(stat.get("gamesStarted") or 0)
                ip = _parse_ip(stat.get("inningsPitched") or 0)
                k9 = stat.get("strikeoutsPer9Inn")
                k9 = float(k9) if k9 not in (None, "", "-.--") else None
                ip_per_start = ip / gs if gs > 0 else DEFAULT_IP_PER_START
                # Require a meaningful current-season sample, else fall back.
                if k9 and ip >= 10:
                    return {"k_per_9": k9, "ip_per_start": ip_per_start,
                            "source": f"statsapi {season}"}
    except Exception:
        pass
    return _profile_from_csv(name)


# --------------------------------------------------------------------------- #
# Opponent team strikeout adjustment                                           #
# --------------------------------------------------------------------------- #
_team_k_cache: dict | None = None


def get_team_k_adjustments(season: int) -> dict:
    """Map team_name -> (team K-rate ÷ league-avg K-rate). 1.0 = neutral.
    Degrades to an empty dict (callers treat missing as 1.0) on any failure."""
    global _team_k_cache
    if _team_k_cache is not None:
        return _team_k_cache
    adj = {}
    try:
        raw = statsapi.get("teams_stats", {
            "season": season, "group": "hitting", "stats": "season", "sportIds": 1,
        })
        splits = raw["stats"][0]["splits"]
        rates = {}
        for sp in splits:
            team = normalize(sp["team"]["name"])
            stat = sp["stat"]
            pa = float(stat.get("plateAppearances") or 0)
            so = float(stat.get("strikeOuts") or 0)
            if pa > 0:
                rates[team] = so / pa
        if rates:
            league_avg = sum(rates.values()) / len(rates)
            adj = {t: r / league_avg for t, r in rates.items()}
    except Exception:
        adj = {}
    _team_k_cache = adj
    return adj


# --------------------------------------------------------------------------- #
# The model                                                                    #
# --------------------------------------------------------------------------- #
def expected_strikeouts(profile: dict, opp_adj: float) -> float:
    k9 = profile["k_per_9"]
    ip = max(IP_BOUNDS[0], min(IP_BOUNDS[1], profile["ip_per_start"]))
    return (k9 / 9.0) * ip * opp_adj


# --------------------------------------------------------------------------- #
# The Odds API: events + per-event prop odds (the credit-costly part)          #
# --------------------------------------------------------------------------- #
BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"


def fetch_events(api_key: str) -> list:
    """List today's MLB events with their event ids. This endpoint is FREE."""
    r = requests.get(f"{BASE}/events", params={"apiKey": api_key}, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_event_props(api_key: str, event_id: str) -> tuple:
    """Fetch FanDuel pitcher-strikeout props for one event. Costs ~1 credit."""
    r = requests.get(
        f"{BASE}/events/{event_id}/odds",
        params={"apiKey": api_key, "regions": "us", "markets": PROP_MARKET,
                "bookmakers": "fanduel", "oddsFormat": "american"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def cache_path(day: str) -> Path:
    return DATA / f"props_cache_{day}.json"


def load_cached_props(day: str) -> dict | None:
    p = cache_path(day)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def save_cached_props(day: str, payload: dict) -> None:
    cache_path(day).write_text(json.dumps(payload, indent=2))


def fetch_props_cached(api_key: str, day: str, limit: int | None = None,
                       force: bool = False) -> dict:
    """Return {event_id: event_odds_json} for the day, fetching only what's
    missing from cache. Set force=True to refetch everything (spends credits)."""
    cached = (None if force else load_cached_props(day)) or {"events": {}, "remaining": None}
    events = fetch_events(api_key)
    if limit:
        events = events[:limit]
    remaining = cached.get("remaining")
    for ev in events:
        eid = ev["id"]
        if eid in cached["events"] and not force:
            continue
        try:
            data, remaining = fetch_event_props(api_key, eid)
            cached["events"][eid] = data
        except Exception as e:
            print(f"  prop fetch failed for {ev.get('home_team')} vs {ev.get('away_team')}: {e}")
    cached["remaining"] = remaining
    save_cached_props(day, cached)
    return cached


# --------------------------------------------------------------------------- #
# Parse prop odds -> per-pitcher lines                                         #
# --------------------------------------------------------------------------- #
def extract_pitcher_lines(event_odds: dict) -> dict:
    """From one event's odds JSON, return {pitcher_name: {point, over, under}}."""
    out = {}
    for b in event_odds.get("bookmakers", []):
        if b["key"] != "fanduel":
            continue
        for m in b.get("markets", []):
            if m["key"] != PROP_MARKET:
                continue
            for o in m["outcomes"]:
                pitcher = o.get("description") or ""
                entry = out.setdefault(pitcher, {"point": o.get("point"),
                                                 "over": None, "under": None})
                if o["name"].lower() == "over":
                    entry["over"] = o["price"]
                    entry["point"] = o.get("point")
                elif o["name"].lower() == "under":
                    entry["under"] = o["price"]
    return out


# --------------------------------------------------------------------------- #
# Tie it together: model every prop and compute edges                          #
# --------------------------------------------------------------------------- #
def build_pitcher_opponent_map(games) -> dict:
    """From a statsapi schedule, map each probable pitcher -> the team they FACE."""
    out = {}
    for g in games:
        home, away = normalize(g["home_name"]), normalize(g["away_name"])
        hp = (g.get("home_probable_pitcher") or "").strip()
        ap = (g.get("away_probable_pitcher") or "").strip()
        if hp:
            out[hp] = away   # home starter faces the away lineup
        if ap:
            out[ap] = home
    return out


def model_props(events_odds: dict, season: int, opponent_map: dict | None = None) -> list:
    """Given {event_id: odds_json}, return a list of modeled prop rows with edges.
    opponent_map (pitcher -> opposing team) sharpens the K adjustment; without it
    we average both clubs as a neutral-safe fallback."""
    team_adj = get_team_k_adjustments(season)
    opponent_map = opponent_map or {}
    rows = []
    for event_odds in events_odds.values():
        home = normalize(event_odds.get("home_team", ""))
        away = normalize(event_odds.get("away_team", ""))
        lines = extract_pitcher_lines(event_odds)
        for pitcher, line in lines.items():
            if line["point"] is None or line["over"] is None or line["under"] is None:
                continue
            profile = get_pitcher_profile(pitcher, season)
            if not profile or not profile.get("k_per_9"):
                continue
            # Adjust for the lineup the pitcher actually faces. If we know it
            # (from the schedule), use that team's K-rate; otherwise average both.
            opp_team = opponent_map.get(pitcher.strip())
            if opp_team and opp_team in team_adj:
                opp_adj = team_adj[opp_team]
            else:
                opp_adj = (team_adj.get(home, 1.0) + team_adj.get(away, 1.0)) / 2
            lam = expected_strikeouts(profile, opp_adj)
            p_over = over_prob(line["point"], lam)
            p_under = 1.0 - p_over
            over_dec = american_to_decimal(line["over"])
            under_dec = american_to_decimal(line["under"])
            edge_over = p_over * over_dec - 1
            edge_under = p_under * under_dec - 1
            # Market no-vig probability (for display/context)
            io, iu = american_to_prob(line["over"]), american_to_prob(line["under"])
            mkt_over = io / (io + iu) if (io + iu) else None
            rows.append({
                "pitcher": pitcher, "home": home, "away": away,
                "line": line["point"], "lambda": lam,
                "k_per_9": profile["k_per_9"], "ip_per_start": profile["ip_per_start"],
                "source": profile["source"], "opp_adj": opp_adj,
                "over_odds": line["over"], "under_odds": line["under"],
                "over_dec": over_dec, "under_dec": under_dec,
                "p_over": p_over, "p_under": p_under,
                "mkt_over": mkt_over,
                "edge_over": edge_over, "edge_under": edge_under,
            })
    return rows


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Model tonight's pitcher strikeout props.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N games (testing — saves credits)")
    ap.add_argument("--cached", action="store_true",
                    help="use today's cache only, make no API call (0 credits)")
    ap.add_argument("--force", action="store_true",
                    help="refetch all games even if cached (spends credits)")
    ap.add_argument("--edge", type=float, default=0.05, help="min edge to flag")
    args = ap.parse_args()

    today = date.today().isoformat()
    season = date.today().year

    if args.cached:
        payload = load_cached_props(today)
        if not payload:
            print(f"No cache for {today}. Run without --cached to fetch (uses credits).")
            return
        print(f"Using cached props for {today} ({len(payload['events'])} games, 0 credits).")
    else:
        api_key = load_key()
        n = f"first {args.limit} game(s)" if args.limit else "all games"
        print(f"Fetching pitcher_strikeouts props for {n} "
              f"(~1 credit/game, FanDuel)...")
        payload = fetch_props_cached(api_key, today, limit=args.limit, force=args.force)
        print(f"  Odds API credits remaining: {payload.get('remaining')}")

    opponent_map = build_pitcher_opponent_map(statsapi.schedule(start_date=today, end_date=today))
    rows = model_props(payload["events"], season, opponent_map)
    if not rows:
        print("No modelable pitcher props found (need probable pitchers with stats + posted lines).")
        return

    rows.sort(key=lambda r: max(r["edge_over"], r["edge_under"]), reverse=True)
    print("\n" + "=" * 100)
    print(f"{'Pitcher':<22} {'Matchup':<24} {'Line':>5} {'Proj':>6} "
          f"{'Over':>7} {'Under':>7} {'EdgeO':>7} {'EdgeU':>7}")
    print("-" * 100)
    for r in rows:
        matchup = f"{r['away'][:10]} @ {r['home'][:10]}"
        print(f"{r['pitcher'][:22]:<22} {matchup:<24} {r['line']:>5.1f} "
              f"{r['lambda']:>6.2f} {format_american(r['over_odds']):>7} "
              f"{format_american(r['under_odds']):>7} "
              f"{r['edge_over']*100:>+6.1f}% {r['edge_under']*100:>+6.1f}%")

    print("\n" + "=" * 100)
    print(f"+EV PROPS (edge >= {args.edge*100:.0f}%)")
    print("-" * 100)
    flagged = []
    for r in rows:
        for side, p, dec, odds, edge in [
            ("Over", r["p_over"], r["over_dec"], r["over_odds"], r["edge_over"]),
            ("Under", r["p_under"], r["under_dec"], r["under_odds"], r["edge_under"]),
        ]:
            if edge >= args.edge:
                flagged.append((r, side, p, odds, edge))
    if not flagged:
        print("  None tonight at this threshold.")
    for r, side, p, odds, edge in flagged:
        print(f"  {r['pitcher']} {side} {r['line']} Ks at {format_american(odds)}  "
              f"(proj {r['lambda']:.2f} | model {p*100:.0f}% | edge {edge*100:+.1f}% | {r['source']})")


if __name__ == "__main__":
    main()
