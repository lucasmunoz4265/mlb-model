"""Calibration backtest for the prop models — does the model's number hold up?

This does NOT need historical betting lines (those don't exist for props). Instead
it checks the only thing we can check for free: are the model's PROBABILITIES
accurate? It uses one season's stats to project every game of the NEXT season
(strictly out-of-sample, no look-ahead) and compares:

  - Projection bias  : mean(actual - projected)  — should be near 0
  - MAE              : average miss per game
  - Line calibration : for each line, model's avg P(over) vs the actual fraction
                       of games that went over. If the model says 55% and 55%
                       happen, it's calibrated.

Opponent adjustment is neutralized here (centered ~1.0) so we test the core
projection. The live model uses current-season-to-date stats, which is FRESHER
than this prior-season test — so real calibration should be at least this good.

CLI:
    python backtest_props.py                 # 2024 stats -> 2025 outcomes
    python backtest_props.py --prior 2023 --test 2024
    python backtest_props.py --pitchers-only | --batters-only
"""

from __future__ import annotations

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import statsapi

from props import (over_prob, binom_over_prob, _parse_ip, _clamp, normalize,
                   AB_BOUNDS, DEFAULT_AB_PER_GAME,
                   PITCHER_K_CALIBRATION, BATTER_HIT_CALIBRATION)

DATA = Path(__file__).parent / "data"
K_LINES = [3.5, 4.5, 5.5, 6.5, 7.5]
HIT_LINES = [0.5, 1.5, 2.5]

# Toggled by --raw to compare corrected vs uncorrected calibration.
K_CAL = PITCHER_K_CALIBRATION
HIT_CAL = BATTER_HIT_CALIBRATION


def game_log(pid, group: str, season: int) -> list:
    """Per-game stat splits for a player in a season (free statsapi)."""
    res = statsapi.get("person", {"personId": pid,
        "hydrate": f"stats(group=[{group}],type=[gameLog],season={season},gameType=R)"})
    people = res.get("people", [])
    if not people or not people[0].get("stats"):
        return []
    return people[0]["stats"][0].get("splits", [])


def _f(stat, key, default=0.0):
    v = stat.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Pitcher strikeouts                                                           #
# --------------------------------------------------------------------------- #
def calibrate_pitchers(prior: int, test: int, min_gs: int = 15) -> dict:
    csv = pd.read_csv(DATA / "pitcher_season_stats.csv")
    pool = csv[(csv["season"] == prior) & (csv["games_started"] >= min_gs)
               & (csv["k_per_9"] > 0) & (csv["innings_pitched"] > 0)]
    print(f"  pitcher pool ({prior}, GS>={min_gs}): {len(pool)} starters — fetching {test} game logs...")

    lambdas, actuals = [], []
    n_pitchers = 0
    for _, row in pool.iterrows():
        ip_per_start = float(row["innings_pitched"]) / float(row["games_started"])
        lam = (float(row["k_per_9"]) / 9.0) * _clamp(ip_per_start, 3.5, 7.0) * K_CAL
        try:
            gl = game_log(int(row["player_id"]), "pitching", test)
        except Exception:
            continue
        starts = [int(_f(s["stat"], "strikeOuts")) for s in gl
                  if int(_f(s["stat"], "gamesStarted")) == 1]
        if not starts:
            continue
        n_pitchers += 1
        for ks in starts:
            lambdas.append(lam)
            actuals.append(ks)
    return _report("PITCHER STRIKEOUTS", prior, test, n_pitchers, lambdas, actuals,
                   K_LINES, lambda lam, line: over_prob(line, lam), "K")


# --------------------------------------------------------------------------- #
# Batter hits                                                                  #
# --------------------------------------------------------------------------- #
def calibrate_batters(prior: int, test: int, min_ab: int = 300, sample: int = 90) -> dict:
    res = statsapi.get("stats", {"stats": "season", "group": "hitting", "season": prior,
                                 "gameType": "R", "sportId": 1, "limit": sample})
    splits = res["stats"][0]["splits"]
    pool = [(s["player"]["id"], s["player"]["fullName"], _f(s["stat"], "avg"),
             _f(s["stat"], "atBats"), _f(s["stat"], "gamesPlayed"))
            for s in splits if _f(s["stat"], "atBats") >= min_ab and _f(s["stat"], "avg") > 0]
    print(f"  batter pool ({prior}, AB>={min_ab}): {len(pool)} regulars — fetching {test} game logs...")

    projes, actuals, params = [], [], []
    n_batters = 0
    for pid, _name, avg, ab, g in pool:
        n_ab = max(1, round(_clamp(ab / g if g else DEFAULT_AB_PER_GAME, *AB_BOUNDS)))
        p = avg * HIT_CAL
        try:
            gl = game_log(pid, "hitting", test)
        except Exception:
            continue
        games = [int(_f(s["stat"], "hits")) for s in gl if _f(s["stat"], "atBats") > 0]
        if not games:
            continue
        n_batters += 1
        for h in games:
            projes.append(n_ab * p)
            actuals.append(h)
            params.append((n_ab, p))
    # batter line calibration needs (n, p) per game, so handle separately
    return _report_batters("BATTER HITS", prior, test, n_batters, projes, actuals, params)


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _calib_table(model_probs_by_line: dict, actual_frac_by_line: dict, unit_lines, label):
    print(f"  Line calibration (model P(over) vs actual over-rate):")
    print(f"    {'Line':>5} {'ModelP':>8} {'Actual':>8} {'Diff':>7}")
    errs = []
    for ln in unit_lines:
        mp = model_probs_by_line[ln]
        af = actual_frac_by_line[ln]
        errs.append(abs(mp - af))
        print(f"    {ln:>5.1f} {mp*100:>7.0f}% {af*100:>7.0f}% {(mp-af)*100:>+6.0f}%")
    print(f"    mean abs calibration error: {sum(errs)/len(errs)*100:.1f}%")
    return sum(errs) / len(errs)


def _report(title, prior, test, n_players, projes, actuals, lines, prob_fn, unit):
    n = len(actuals)
    if n == 0:
        print(f"\n{title}: no data."); return {}
    bias = sum(a - p for a, p in zip(actuals, projes)) / n
    mae = sum(abs(a - p) for a, p in zip(actuals, projes)) / n
    print(f"\n{'='*60}\n{title} — calibration ({prior} stats → {test} games)\n{'-'*60}")
    print(f"  Players: {n_players} | Games: {n:,}")
    print(f"  Projection bias: {bias:+.2f} {unit} (actual - projected; ~0 = unbiased)")
    print(f"  MAE: {mae:.2f} {unit} per game")
    model_probs = {ln: sum(prob_fn(p, ln) for p in projes) / n for ln in lines}
    actual_frac = {ln: sum(1 for a in actuals if a > ln) / n for ln in lines}
    err = _calib_table(model_probs, actual_frac, lines, unit)
    return {"bias": bias, "mae": mae, "calib_error": err, "n": n}


def _report_batters(title, prior, test, n_players, projes, actuals, params):
    n = len(actuals)
    if n == 0:
        print(f"\n{title}: no data."); return {}
    bias = sum(a - p for a, p in zip(actuals, projes)) / n
    mae = sum(abs(a - p) for a, p in zip(actuals, projes)) / n
    print(f"\n{'='*60}\n{title} — calibration ({prior} stats → {test} games)\n{'-'*60}")
    print(f"  Players: {n_players} | Games: {n:,}")
    print(f"  Projection bias: {bias:+.2f} H (actual - projected; ~0 = unbiased)")
    print(f"  MAE: {mae:.2f} H per game")
    model_probs = {ln: sum(binom_over_prob(ln, npar, ppar) for npar, ppar in params) / n
                   for ln in HIT_LINES}
    actual_frac = {ln: sum(1 for a in actuals if a > ln) / n for ln in HIT_LINES}
    err = _calib_table(model_probs, actual_frac, HIT_LINES, "H")
    return {"bias": bias, "mae": mae, "calib_error": err, "n": n}


# --------------------------------------------------------------------------- #
# Phase 0: game-aware backtest — project each start vs that game's opponent     #
# --------------------------------------------------------------------------- #
def team_k_tables(season: int, cache: bool = True) -> dict:
    """Per-team strikeout rate vs LHP/RHP (and overall) for a season, with the
    league averages. Cached to disk — opponent K-tendencies drive the handedness
    adjustment. Every PA is vs an L or R pitcher, so overall = (Lso+Rso)/(Lpa+Rpa)."""
    path = DATA / f"team_k_by_hand_{season}.json"
    if cache and path.exists():
        return json.loads(path.read_text())
    teams = [t for t in statsapi.get("teams", {"sportId": 1})["teams"]]
    by_team, tot = {}, {"L": [0.0, 0.0], "R": [0.0, 0.0]}
    print(f"  building team K-vs-hand table for {season} ({len(teams)} teams)...")
    for t in teams:
        rec, so_all, pa_all = {}, 0.0, 0.0
        for code, hand in (("vl", "L"), ("vr", "R")):
            try:
                s = statsapi.get("team_stats", {"teamId": t["id"], "stats": "statSplits",
                    "sitCodes": code, "group": "hitting", "season": season})["stats"][0]["splits"][0]["stat"]
                so, pa = float(s["strikeOuts"]), float(s["plateAppearances"])
            except Exception:
                so = pa = 0.0
            rec[hand] = so / pa if pa else None
            so_all += so; pa_all += pa
            tot[hand][0] += so; tot[hand][1] += pa
        rec["all"] = so_all / pa_all if pa_all else None
        by_team[normalize(t["name"])] = rec
    league = {"L": tot["L"][0] / tot["L"][1], "R": tot["R"][0] / tot["R"][1],
              "all": (tot["L"][0] + tot["R"][0]) / (tot["L"][1] + tot["R"][1])}
    out = {"teams": by_team, "league": league}
    if cache:
        path.write_text(json.dumps(out, indent=2))
    return out


def _metrics(projes, actuals, lines, prob_fn) -> tuple:
    """(bias, mae, mean-abs line-calibration-error) for a set of projections."""
    n = len(actuals)
    bias = sum(a - p for a, p in zip(actuals, projes)) / n
    mae = sum(abs(a - p) for a, p in zip(actuals, projes)) / n
    errs = []
    for ln in lines:
        mp = sum(prob_fn(p, ln) for p in projes) / n
        af = sum(1 for a in actuals if a > ln) / n
        errs.append(abs(mp - af))
    return bias, mae, sum(errs) / len(errs)


def calibrate_pitchers_gameaware(prior: int, test: int, min_gs: int = 15) -> dict:
    """Project each start against that game's ACTUAL opponent, in 3 opponent
    modes, so we can measure whether opponent/handedness awareness helps:
      neutral  — no opponent adjustment (today's live baseline behavior)
      overall  — opponent's overall K-rate vs league
      hand     — opponent's K-rate vs the pitcher's handedness (the Phase-1 idea)
    Opponent K-rates use the TEST season (same info the live model uses)."""
    tbl = team_k_tables(test)
    teams_k, league = tbl["teams"], tbl["league"]
    csv = pd.read_csv(DATA / "pitcher_season_stats.csv")
    pool = csv[(csv["season"] == prior) & (csv["games_started"] >= min_gs)
               & (csv["k_per_9"] > 0) & (csv["innings_pitched"] > 0)]
    print(f"  pitcher pool ({prior}, GS>={min_gs}): {len(pool)} starters — fetching {test} logs...")

    actuals = []
    proj = {"neutral": [], "overall": [], "hand": [], "ip_ceiling": []}
    n_pitchers = 0
    for _, row in pool.iterrows():
        k9_term = float(row["k_per_9"]) / 9.0
        base = k9_term * _clamp(
            float(row["innings_pitched"]) / float(row["games_started"]), 3.5, 7.0) * K_CAL
        try:
            res = statsapi.get("person", {"personId": int(row["player_id"]),
                "hydrate": f"stats(group=[pitching],type=[gameLog],season={test},gameType=R)"})
            person = res["people"][0]
            hand = person.get("pitchHand", {}).get("code", "R")
            splits = person["stats"][0]["splits"]
        except Exception:
            continue
        starts = [s for s in splits if int(_f(s["stat"], "gamesStarted")) == 1]
        if not starts:
            continue
        n_pitchers += 1
        for s in starts:
            opp = normalize(s.get("opponent", {}).get("name", ""))
            rec = teams_k.get(opp, {})
            adj_overall = (rec.get("all") or league["all"]) / league["all"]
            hk = rec.get(hand)
            adj_hand = (hk / league[hand]) if hk else 1.0
            actual_ip = _parse_ip(s["stat"].get("inningsPitched", 0))
            actuals.append(int(_f(s["stat"], "strikeOuts")))
            proj["neutral"].append(base)
            proj["overall"].append(base * adj_overall)
            proj["hand"].append(base * adj_hand)
            # Diagnostic: perfect knowledge of THIS start's innings (look-ahead) —
            # shows the ceiling of how much a better expected-innings model could help.
            proj["ip_ceiling"].append(k9_term * actual_ip * K_CAL)

    n = len(actuals)
    print(f"\n{'='*64}\nPITCHER STRIKEOUTS — game-aware opponent test ({prior}→{test})\n{'-'*64}")
    print(f"  Pitchers: {n_pitchers} | Starts: {n:,}")
    print(f"  {'Mode':<22}{'Bias':>8}{'MAE':>8}{'CalibErr':>10}")
    results = {}
    labels = {"neutral": "neutral", "overall": "overall K%", "hand": "by handedness",
              "ip_ceiling": "PERFECT-IP (ceiling)"}
    for mode in ("neutral", "overall", "hand", "ip_ceiling"):
        bias, mae, ce = _metrics(proj[mode], actuals, K_LINES, lambda lam, ln: over_prob(ln, lam))
        results[mode] = {"bias": bias, "mae": mae, "calib_error": ce}
        print(f"  {labels[mode]:<22}{bias:>+8.2f}{mae:>8.3f}{ce*100:>9.1f}%")
    base_mae = results["neutral"]["mae"]
    realistic = ("neutral", "overall", "hand")
    best = min(realistic, key=lambda m: results[m]["mae"])
    print(f"\n  Best realistic feature: {best}  (MAE {results[best]['mae']:.3f} vs "
          f"neutral {base_mae:.3f}, {(results[best]['mae']-base_mae)/base_mae*100:+.1f}%)")
    ip_mae = results["ip_ceiling"]["mae"]
    print(f"  Perfect-IP ceiling: MAE {ip_mae:.3f} ({(ip_mae-base_mae)/base_mae*100:+.1f}% vs neutral) "
          f"— the MOST a better innings model could buy. If small, K variance is mostly irreducible.")
    return results


def main():
    ap = argparse.ArgumentParser(description="Calibration backtest for prop models.")
    ap.add_argument("--prior", type=int, default=2024, help="season to build stats from")
    ap.add_argument("--test", type=int, default=2025, help="season to test outcomes on")
    ap.add_argument("--pitchers-only", action="store_true")
    ap.add_argument("--batters-only", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="disable the calibration corrections (see uncorrected model)")
    ap.add_argument("--gameaware", action="store_true",
                    help="game-aware pitcher test: compare neutral vs overall vs handedness opponent")
    args = ap.parse_args()

    if args.raw:
        global K_CAL, HIT_CAL
        K_CAL, HIT_CAL = 1.0, 1.0

    if args.gameaware:
        print(f"Game-aware backtest — {args.prior} stats project {args.test} starts "
              f"vs each game's real opponent.")
        calibrate_pitchers_gameaware(args.prior, args.test)
        print("\nReading the result: lower MAE + lower CalibErr = the opponent feature helps. "
              "If 'by handedness' wins, it's worth adding to the live model.")
        return

    mode = "RAW (no correction)" if args.raw else \
        f"corrected (K×{K_CAL}, hit×{HIT_CAL})"
    print(f"Calibration backtest — {args.prior} stats project {args.test} outcomes "
          f"[{mode}].")
    if not args.batters_only:
        calibrate_pitchers(args.prior, args.test)
    if not args.pitchers_only:
        calibrate_batters(args.prior, args.test)
    print("\nReading the result: bias near 0 and small line-calibration error (<~4-5%) "
          "means the model's probabilities are trustworthy to bet from.")


if __name__ == "__main__":
    main()
