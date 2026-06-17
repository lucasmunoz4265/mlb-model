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

PITCHER_K_MARKET = "pitcher_strikeouts"
BATTER_HITS_MARKET = "batter_hits"
DEFAULT_MARKETS = [PITCHER_K_MARKET, BATTER_HITS_MARKET]

# Pitcher-K model
DEFAULT_IP_PER_START = 5.4   # league-typical modern starter when we lack data
IP_BOUNDS = (3.5, 7.0)

# Batter-hits model
LEAGUE_BAA = 0.245           # league avg batting-average-against, for opp-pitcher scaling
DEFAULT_AB_PER_GAME = 3.9
AB_BOUNDS = (3.0, 4.6)
OPP_ADJ_BOUNDS = (0.85, 1.15)
P_HIT_BOUNDS = (0.05, 0.60)

# Calibration corrections — empirically derived from backtest_props.py over two
# out-of-sample season pairs (2023→24 and 2024→25), which both showed the raw
# model leaning ~3-6% toward the OVER (it over-projects). These shrink the
# projection to re-center calibration. Re-run backtest_props.py if inputs change.
PITCHER_K_CALIBRATION = 0.95   # raw K projection ran ~0.25 K high
BATTER_HIT_CALIBRATION = 0.90  # raw hit rate over-projected P(1+ hit) by ~6%


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
# Binomial helpers — for batter hits (fixed small number of at-bats)           #
# --------------------------------------------------------------------------- #
def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_over_prob(point: float, n: int, p: float) -> float:
    """P(count beats the line) for Binomial(n, p). 'Over 0.5' = P(X >= 1)."""
    k = math.floor(point) + 1
    if k > n:
        return 0.0
    return max(0.0, min(1.0, sum(binom_pmf(i, n, p) for i in range(k, n + 1))))


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
    return (k9 / 9.0) * ip * opp_adj * PITCHER_K_CALIBRATION


# --------------------------------------------------------------------------- #
# The Odds API: events + per-event prop odds (the credit-costly part)          #
# --------------------------------------------------------------------------- #
BASE = "https://api.the-odds-api.com/v4/sports/baseball_mlb"


def fetch_events(api_key: str) -> list:
    """List today's MLB events with their event ids. This endpoint is FREE."""
    r = requests.get(f"{BASE}/events", params={"apiKey": api_key}, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_event_props(api_key: str, event_id: str, markets: list | None = None) -> tuple:
    """Fetch FanDuel props for one event. Costs (len(markets)) credits."""
    markets = markets or [PITCHER_K_MARKET]
    r = requests.get(
        f"{BASE}/events/{event_id}/odds",
        params={"apiKey": api_key, "regions": "us", "markets": ",".join(markets),
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


def fetch_props_cached(api_key: str, day: str, markets: list | None = None,
                       limit: int | None = None, force: bool = False) -> dict:
    """Return {event_id: event_odds_json} for the day, fetching only what's
    missing from cache. If new markets are requested that the cache doesn't have,
    events are refetched so the cache holds all requested markets.
    Set force=True to refetch everything (spends credits)."""
    markets = markets or [PITCHER_K_MARKET]
    cached = (None if force else load_cached_props(day)) or \
        {"events": {}, "remaining": None, "markets": []}
    # If the cache lacks any requested market, its event payloads are stale → refetch.
    need_new_markets = not set(markets).issubset(set(cached.get("markets") or []))
    do_force = force or need_new_markets
    events = fetch_events(api_key)
    if limit:
        events = events[:limit]
    remaining = cached.get("remaining")
    for ev in events:
        eid = ev["id"]
        if eid in cached["events"] and not do_force:
            continue
        try:
            data, remaining = fetch_event_props(api_key, eid, markets)
            cached["events"][eid] = data
        except Exception as e:
            print(f"  prop fetch failed for {ev.get('home_team')} vs {ev.get('away_team')}: {e}")
    cached["remaining"] = remaining
    cached["markets"] = list(markets) if do_force else \
        sorted(set(cached.get("markets") or []) | set(markets))
    save_cached_props(day, cached)
    return cached


# --------------------------------------------------------------------------- #
# Parse prop odds -> per-pitcher lines                                         #
# --------------------------------------------------------------------------- #
def extract_lines(event_odds: dict, market_key: str) -> dict:
    """From one event's odds JSON, return {player_name: {point, over, under}}
    for the given prop market (player name comes from the outcome 'description')."""
    out = {}
    for b in event_odds.get("bookmakers", []):
        if b["key"] != "fanduel":
            continue
        for m in b.get("markets", []):
            if m["key"] != market_key:
                continue
            for o in m["outcomes"]:
                player = o.get("description") or ""
                entry = out.setdefault(player, {"point": o.get("point"),
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
        lines = extract_lines(event_odds, PITCHER_K_MARKET)
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
# Batter hits: profile, opponent pitcher, model                                #
# --------------------------------------------------------------------------- #
def get_batter_profile(name: str, season: int) -> dict | None:
    """Current-season AVG, at-bats/game, SLG, and team for a hitter via statsapi.
    Returns None if the player has too small a sample (props need regulars)."""
    if not name:
        return None
    try:
        matches = statsapi.lookup_player(name)
        if not matches:
            return None
        pid = matches[0]["id"]
        data = statsapi.player_stat_data(pid, group="hitting", type="season", season=season)
        splits = data.get("stats") or []
        if not splits:
            return None
        s = splits[0].get("stats", {})
        ab = float(s.get("atBats") or 0)
        g = float(s.get("gamesPlayed") or 0)
        avg = s.get("avg")
        avg = float(avg) if avg not in (None, "", ".---") else None
        if not avg or ab < 20:        # need a real sample to trust the rate
            return None
        slg = s.get("slg")
        slg = float(slg) if slg not in (None, "", ".---") else avg
        ab_per_game = ab / g if g > 0 else DEFAULT_AB_PER_GAME
        return {"avg": avg, "ab_per_game": ab_per_game, "slg": slg,
                "team": normalize(data.get("current_team") or ""),
                "source": f"statsapi {season}"}
    except Exception:
        return None


_pitcher_baa_cache: dict = {}


def get_pitcher_baa(name: str, season: int) -> float | None:
    """Opposing starter's batting-average-against (statsapi 'avg'). Cached."""
    if not name:
        return None
    if name in _pitcher_baa_cache:
        return _pitcher_baa_cache[name]
    baa = None
    try:
        matches = statsapi.lookup_player(name)
        if matches:
            data = statsapi.player_stat_data(matches[0]["id"], group="pitching",
                                             type="season", season=season)
            splits = data.get("stats") or []
            if splits:
                s = splits[0].get("stats", {})
                ip = _parse_ip(s.get("inningsPitched") or 0)
                v = s.get("avg")
                if v not in (None, "", ".---") and ip >= 10:
                    baa = float(v)
    except Exception:
        baa = None
    _pitcher_baa_cache[name] = baa
    return baa


def build_team_opponent_pitcher_map(games) -> dict:
    """From a statsapi schedule, map each team -> the opposing probable pitcher
    they face (home lineup faces the away starter, and vice versa)."""
    out = {}
    for g in games:
        home, away = normalize(g["home_name"]), normalize(g["away_name"])
        hp = (g.get("home_probable_pitcher") or "").strip()
        ap = (g.get("away_probable_pitcher") or "").strip()
        if ap:
            out[home] = ap   # home batters face the away starter
        if hp:
            out[away] = hp
    return out


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def model_batter_props(events_odds: dict, season: int,
                       team_opp_pitcher: dict | None = None) -> list:
    """Model batter HITS as Binomial(at-bats, per-AB hit prob), where the hit
    prob is the batter's AVG scaled by the opposing starter's BAA vs league.
    team_opp_pitcher (team -> opposing starter) sharpens the adjustment."""
    team_opp_pitcher = team_opp_pitcher or {}
    rows = []
    for event_odds in events_odds.values():
        home = normalize(event_odds.get("home_team", ""))
        away = normalize(event_odds.get("away_team", ""))
        lines = extract_lines(event_odds, BATTER_HITS_MARKET)
        for batter, line in lines.items():
            if line["point"] is None or line["over"] is None or line["under"] is None:
                continue
            prof = get_batter_profile(batter, season)
            if not prof:
                continue
            opp_pitcher = team_opp_pitcher.get(prof["team"])
            opp_adj = 1.0
            if opp_pitcher:
                baa = get_pitcher_baa(opp_pitcher, season)
                if baa:
                    opp_adj = _clamp(baa / LEAGUE_BAA, *OPP_ADJ_BOUNDS)
            p_hit = _clamp(prof["avg"] * opp_adj * BATTER_HIT_CALIBRATION, *P_HIT_BOUNDS)
            n = max(1, round(_clamp(prof["ab_per_game"], *AB_BOUNDS)))
            p_over = binom_over_prob(line["point"], n, p_hit)
            p_under = 1.0 - p_over
            over_dec = american_to_decimal(line["over"])
            under_dec = american_to_decimal(line["under"])
            io, iu = american_to_prob(line["over"]), american_to_prob(line["under"])
            rows.append({
                "batter": batter, "home": home, "away": away,
                "line": line["point"], "team": prof["team"],
                "avg": prof["avg"], "ab_per_game": prof["ab_per_game"],
                "p_hit": p_hit, "n_ab": n, "opp_pitcher": opp_pitcher or "—",
                "opp_adj": opp_adj, "source": prof["source"],
                "over_odds": line["over"], "under_odds": line["under"],
                "over_dec": over_dec, "under_dec": under_dec,
                "p_over": p_over, "p_under": p_under,
                "mkt_over": io / (io + iu) if (io + iu) else None,
                "edge_over": p_over * over_dec - 1,
                "edge_under": p_under * under_dec - 1,
            })
    return rows


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _flag_ev(rows, edge):
    """Return (row, side, p, odds, edge) tuples that meet the edge threshold."""
    out = []
    for r in rows:
        for side, p, odds, e in [
            ("Over", r["p_over"], r["over_odds"], r["edge_over"]),
            ("Under", r["p_under"], r["under_odds"], r["edge_under"]),
        ]:
            if e >= edge:
                out.append((r, side, p, odds, e))
    return out


def main():
    ap = argparse.ArgumentParser(description="Model tonight's MLB player props (pitcher Ks + batter hits).")
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N games (testing — saves credits)")
    ap.add_argument("--cached", action="store_true",
                    help="use today's cache only, make no API call (0 credits)")
    ap.add_argument("--force", action="store_true",
                    help="refetch all games even if cached (spends credits)")
    ap.add_argument("--ks-only", action="store_true", help="pitcher strikeouts only (1 credit/game)")
    ap.add_argument("--hits-only", action="store_true", help="batter hits only (1 credit/game)")
    ap.add_argument("--edge", type=float, default=0.05, help="min edge to flag")
    args = ap.parse_args()

    today = date.today().isoformat()
    season = date.today().year
    markets = DEFAULT_MARKETS
    if args.ks_only:
        markets = [PITCHER_K_MARKET]
    elif args.hits_only:
        markets = [BATTER_HITS_MARKET]

    if args.cached:
        payload = load_cached_props(today)
        if not payload:
            print(f"No cache for {today}. Run without --cached to fetch (uses credits).")
            return
        print(f"Using cached props for {today} ({len(payload['events'])} games, 0 credits).")
    else:
        api_key = load_key()
        n = f"first {args.limit} game(s)" if args.limit else "all games"
        per_game = len(markets)
        print(f"Fetching {','.join(markets)} for {n} (~{per_game} credit(s)/game, FanDuel)...")
        payload = fetch_props_cached(api_key, today, markets=markets,
                                     limit=args.limit, force=args.force)
        print(f"  Odds API credits remaining: {payload.get('remaining')}")

    schedule = statsapi.schedule(start_date=today, end_date=today)

    if PITCHER_K_MARKET in markets:
        rows = model_props(payload["events"], season, build_pitcher_opponent_map(schedule))
        rows.sort(key=lambda r: max(r["edge_over"], r["edge_under"]), reverse=True)
        print("\n" + "=" * 92 + "\nPITCHER STRIKEOUTS\n" + "-" * 92)
        for r in rows:
            print(f"  {r['pitcher'][:22]:<22} {r['away'][:9]+' @ '+r['home'][:9]:<24} "
                  f"line {r['line']:<4.1f} proj {r['lambda']:>5.2f}  "
                  f"O {format_american(r['over_odds']):>6} {r['edge_over']*100:>+6.1f}%  "
                  f"U {format_american(r['under_odds']):>6} {r['edge_under']*100:>+6.1f}%")
        print(f"\n  +EV (edge >= {args.edge*100:.0f}%):")
        flagged = _flag_ev(rows, args.edge)
        for r, side, p, odds, e in flagged or []:
            print(f"    {r['pitcher']} {side} {r['line']:g} Ks at {format_american(odds)} "
                  f"(proj {r['lambda']:.2f} | {p*100:.0f}% | {e*100:+.1f}%)")
        if not flagged:
            print("    none")

    if BATTER_HITS_MARKET in markets:
        brows = model_batter_props(payload["events"], season,
                                   build_team_opponent_pitcher_map(schedule))
        brows.sort(key=lambda r: max(r["edge_over"], r["edge_under"]), reverse=True)
        print("\n" + "=" * 92 + "\nBATTER HITS\n" + "-" * 92)
        for r in brows:
            print(f"  {r['batter'][:22]:<22} {r['away'][:9]+' @ '+r['home'][:9]:<24} "
                  f"line {r['line']:<4.1f} p_hit {r['p_hit']:.3f} x{r['n_ab']}  "
                  f"O {format_american(r['over_odds']):>6} {r['edge_over']*100:>+6.1f}%  "
                  f"U {format_american(r['under_odds']):>6} {r['edge_under']*100:>+6.1f}%")
        print(f"\n  +EV (edge >= {args.edge*100:.0f}%):")
        flagged = _flag_ev([{**r, "pitcher": r["batter"]} for r in brows], args.edge)
        for r, side, p, odds, e in flagged or []:
            print(f"    {r['batter']} {side} {r['line']:g} Hits at {format_american(odds)} "
                  f"(p_hit {r['p_hit']:.3f} | {p*100:.0f}% | {e*100:+.1f}% | vs {r['opp_pitcher']})")
        if not flagged:
            print("    none")


if __name__ == "__main__":
    main()
