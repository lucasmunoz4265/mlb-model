"""Streamlit dashboard for the MLB model. Run with: streamlit run dashboard.py"""

import warnings
warnings.filterwarnings("ignore")

from datetime import date
from pathlib import Path

import pandas as pd
import requests
import statsapi
import streamlit as st

from pitcher_elo import (
    run_pitcher_elo, expected_home_win, HCA, PITCHER_WEIGHT, INITIAL_RATING,
)
from odds import american_to_prob, american_to_decimal
from tonight import normalize, format_american
from tracker import log_bet, ensure_log

DATA = Path(__file__).parent / "data"


def load_api_key():
    # Streamlit Cloud secrets first (production)
    try:
        if "ODDS_API_KEY" in st.secrets:
            return st.secrets["ODDS_API_KEY"]
    except Exception:
        pass
    # Local .env fallback (development)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


@st.cache_data(ttl=3600, show_spinner="Computing Elo ratings from history...")
def get_current_ratings():
    games = pd.read_csv(DATA / "games.csv")
    games = games.fillna({"home_pitcher": "", "away_pitcher": ""})
    _, team_ratings, pitcher_ratings = run_pitcher_elo(games)
    return team_ratings, pitcher_ratings


def get_tonight_games():
    """Not cached — game statuses change throughout the night and we always want fresh."""
    today = date.today().isoformat()
    return today, statsapi.schedule(start_date=today, end_date=today)


@st.cache_data(ttl=300, show_spinner="Checking pending bet results...")
def auto_update_pending():
    """Run tracker.update_pending and return (updated_count, timestamp)."""
    from tracker import update_pending as _update
    from io import StringIO
    import sys
    from datetime import datetime
    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        _update()
    finally:
        sys.stdout = old
    return buf.getvalue(), datetime.now().strftime("%Y-%m-%d %H:%M")


# 30-min cache (shared across all visitors) so loading the public app makes at
# most ~2 paid fetches/hour. Only h2h + totals — spreads were fetched but never
# used. Pre-game lines barely move, so 30 min is plenty fresh.
@st.cache_data(ttl=1800, show_spinner="Fetching FanDuel odds...")
def get_fanduel_odds(api_key):
    if not api_key:
        return [], None
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": api_key, "regions": "us",
        "markets": "h2h,totals",
        "bookmakers": "fanduel", "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def predict_game(g, team_ratings, pitcher_ratings):
    home = normalize(g["home_name"])
    away = normalize(g["away_name"])
    home_pit = g.get("home_probable_pitcher") or ""
    away_pit = g.get("away_probable_pitcher") or ""
    team_h = team_ratings.get(home, INITIAL_RATING)
    team_a = team_ratings.get(away, INITIAL_RATING)
    pit_h = pitcher_ratings.get(home_pit, INITIAL_RATING)
    pit_a = pitcher_ratings.get(away_pit, INITIAL_RATING)
    diff = (team_h - team_a) + PITCHER_WEIGHT * (pit_h - pit_a) + HCA
    return {
        "game_id": g["game_id"], "game_date": g["game_date"],
        "home": home, "away": away,
        "home_pitcher": home_pit, "away_pitcher": away_pit,
        "team_h": team_h, "team_a": team_a, "pit_h": pit_h, "pit_a": pit_a,
        "p_home": expected_home_win(diff),
    }


def fanduel_lines_for_game(prediction, odds_for_game):
    if not odds_for_game or not odds_for_game.get("bookmakers"):
        return None
    fd = next((b for b in odds_for_game["bookmakers"] if b["key"] == "fanduel"), None)
    if not fd:
        return None
    out = {"ml_home": None, "ml_away": None, "total": None,
           "over_odds": None, "under_odds": None}
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
    return out


def compute_edges(rows):
    """Add no-vig probs, decimal odds, edges to each row."""
    out = []
    for r in rows:
        if r["ml_home"] is None or r["ml_away"] is None:
            continue
        home_imp = american_to_prob(r["ml_home"])
        away_imp = american_to_prob(r["ml_away"])
        total_imp = home_imp + away_imp
        r["market_p_home"] = home_imp / total_imp
        r["home_decimal"] = american_to_decimal(r["ml_home"])
        r["away_decimal"] = american_to_decimal(r["ml_away"])
        r["edge_home"] = r["p_home"] * r["home_decimal"] - 1
        r["edge_away"] = (1 - r["p_home"]) * r["away_decimal"] - 1
        out.append(r)
    return out


def kelly_stake(p, decimal, bankroll, fraction, cap=0.01):
    b = decimal - 1
    if b <= 0:
        return 0
    f_full = max(0, (p * (b + 1) - 1) / b)
    f = min(f_full * fraction, cap)
    return bankroll * f


def main():
    st.set_page_config(page_title="MLB Model", page_icon="⚾", layout="wide")
    st.title("⚾ MLB Model — Live Dashboard")

    st.sidebar.header("Settings")
    edge_threshold = st.sidebar.slider("Min edge to flag bet", 0.0, 0.25, 0.05, 0.01,
                                       format="%.2f")
    bankroll = st.sidebar.number_input("Bankroll ($)", 100, 100000, 500, step=50)
    kelly_frac = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.05, 0.01,
                                   help="0.05 = 5% Kelly. Higher = bigger bets, more variance.")
    log_enabled = st.sidebar.toggle("Log new recs to tracker", value=False,
                                    help="When ON, hitting 'Log All' saves bets to bet_log.csv")
    st.sidebar.divider()
    if st.sidebar.button("Refresh data (force)"):
        st.cache_data.clear()
        st.rerun()

    api_key = load_api_key()
    if not api_key:
        st.error("No API key in .env. Add ODDS_API_KEY=... to the .env file.")
        st.stop()

    try:
        team_ratings, pitcher_ratings = get_current_ratings()
        today, games = get_tonight_games()
        odds_data, remaining = get_fanduel_odds(api_key)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

    st.caption(f"Date: **{today}** • Games tonight: **{len(games)}** • "
               f"Odds API credits remaining: **{remaining}**")

    odds_by_pair = {(normalize(o["home_team"]), normalize(o["away_team"])): o for o in odds_data}

    NON_PREGAME_KEYWORDS = ("final", "game over", "in progress", "suspended",
                            "postponed", "cancelled", "completed")
    from datetime import datetime, timezone

    def is_pregame(g):
        # Check 1: explicit non-pregame status (handles status="In Progress" etc.)
        status_lower = str(g.get("status") or "").lower()
        for kw in NON_PREGAME_KEYWORDS:
            if kw in status_lower:
                return False
        # Check 2: scheduled start time in the past
        dt_str = g.get("game_datetime") or ""
        if dt_str:
            try:
                game_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if game_dt < datetime.now(timezone.utc):
                    return False
            except Exception:
                pass
        # Check 3: must affirmatively be in a pregame state if status is set
        if status_lower and status_lower not in ("scheduled", "pre-game", "warmup", "delayed", ""):
            return False
        return True

    pregame_games = [g for g in games if is_pregame(g)]
    n_started = len(games) - len(pregame_games)
    if n_started > 0:
        st.info(f"⏱️ {n_started} of tonight's {len(games)} games have already started — they're hidden because comparing pre-game model predictions to live in-game odds produces meaningless edges.")

    rows = []
    for g in pregame_games:
        pred = predict_game(g, team_ratings, pitcher_ratings)
        lines = fanduel_lines_for_game(pred, odds_by_pair.get((pred["home"], pred["away"])))
        if lines is None:
            continue
        rows.append({**pred, **lines})
    rows = compute_edges(rows)
    # Safety net: edges over 50% are almost certainly stale/live odds, not real pre-game edges.
    # Real pre-game MLB MLs rarely produce edges over 25%.
    rows = [r for r in rows if abs(r.get("edge_home", 0) or 0) < 0.50 and abs(r.get("edge_away", 0) or 0) < 0.50]
    rows.sort(key=lambda r: max(r["edge_home"], r["edge_away"]), reverse=True)

    tab1, tab2, tab_props, tab_bet, tab3 = st.tabs(
        ["🎯 Recommended Bets", "📋 Full Slate", "⚾ Player Props", "📝 Place a Bet", "📈 Performance"])

    with tab1:
        recs = []
        for r in rows:
            for side, p, decimal, ml, edge_val in [
                ("home", r["p_home"], r["home_decimal"], r["ml_home"], r["edge_home"]),
                ("away", 1 - r["p_home"], r["away_decimal"], r["ml_away"], r["edge_away"]),
            ]:
                if edge_val < edge_threshold:
                    continue
                team = r["home"] if side == "home" else r["away"]
                stake = kelly_stake(p, decimal, bankroll, kelly_frac)
                recs.append((team, side, ml, edge_val, stake, p, r))

        st.subheader(f"{len(recs)} bets at edge ≥ {edge_threshold:.0%}")
        if not recs:
            st.info("No bets meet the edge threshold tonight. Lower the threshold in the sidebar to see more options.")
        else:
            for team, side, ml, edge_val, stake, p, r in recs:
                with st.container(border=True):
                    cols = st.columns([3, 1, 1, 1, 1])
                    cols[0].markdown(f"**{team} ML at {format_american(ml)}**  \n"
                                     f"_{r['away']} @ {r['home']}_  \n"
                                     f"<small>{r['away_pitcher']} ({r['pit_a']:.0f}) vs "
                                     f"{r['home_pitcher']} ({r['pit_h']:.0f})</small>",
                                     unsafe_allow_html=True)
                    cols[1].metric("Model", f"{p*100:.1f}%")
                    market_p_bet = r["market_p_home"] if side == "home" else 1 - r["market_p_home"]
                    cols[2].metric("Market", f"{market_p_bet*100:.1f}%")
                    cols[3].metric("Edge", f"{edge_val*100:+.1f}%")
                    cols[4].metric("Stake", f"${stake:.2f}")

            if log_enabled and st.button("📝 Log all recommendations to tracker"):
                from datetime import datetime
                logged_at = datetime.now().isoformat(timespec="seconds")
                for team, side, ml, edge_val, stake, p, r in recs:
                    market_p_bet = r["market_p_home"] if side == "home" else 1 - r["market_p_home"]
                    decimal = r["home_decimal"] if side == "home" else r["away_decimal"]
                    pitcher = r["home_pitcher"] if side == "home" else r["away_pitcher"]
                    opp_pitcher = r["away_pitcher"] if side == "home" else r["home_pitcher"]
                    log_bet({
                        "logged_at": logged_at, "game_date": r["game_date"],
                        "game_id": r["game_id"], "home_team": r["home"], "away_team": r["away"],
                        "bet_side": side, "bet_team": team,
                        "pitcher": pitcher, "opp_pitcher": opp_pitcher,
                        "odds_american": ml, "odds_decimal": round(decimal, 4),
                        "model_p": round(p, 4), "market_p": round(market_p_bet, 4),
                        "edge": round(edge_val, 4), "stake": round(stake, 2),
                    })
                st.success(f"Logged {len(recs)} bets to bet_log.csv")

    with tab2:
        st.subheader("Full slate — all games with FanDuel odds")
        if not rows:
            st.info("No games loaded yet.")
        else:
            display_df = pd.DataFrame([{
                "Matchup": f"{r['away']} @ {r['home']}",
                "Model home %": f"{r['p_home']*100:.0f}%",
                "Market home %": f"{r['market_p_home']*100:.0f}%",
                "FD Home": format_american(r['ml_home']),
                "FD Away": format_american(r['ml_away']),
                "Total": r['total'] or "—",
                "Edge Home": f"{r['edge_home']*100:+.1f}%",
                "Edge Away": f"{r['edge_away']*100:+.1f}%",
            } for r in rows])
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab_props:
        st.subheader("⚾ Player Props")
        st.caption("Pitcher strikeouts modeled as Poisson(expected K); batter hits as "
                   "Binomial(at-bats, hit prob) adjusted for the opposing starter. "
                   "Each prop market costs **~1 credit per game** to fetch.")
        from props import (load_cached_props, fetch_props_cached, model_props,
                           model_batter_props, build_pitcher_opponent_map,
                           build_team_opponent_pitcher_map,
                           PITCHER_K_MARKET, BATTER_HITS_MARKET)
        season = date.today().year
        gid_by_pair = {(normalize(g["home_name"]), normalize(g["away_name"])): g
                       for g in games}
        cached = load_cached_props(today)

        PROP_TYPES = {"Pitcher Ks": PITCHER_K_MARKET, "Batter Hits": BATTER_HITS_MARKET}
        pc1, pc2 = st.columns([3, 2])
        chosen = pc1.multiselect("Prop types", list(PROP_TYPES), default=list(PROP_TYPES),
                                 help="Each type adds ~1 credit per game.")
        markets = [PROP_TYPES[c] for c in chosen]
        max_g = max(len(games), 1)
        limit = pc2.number_input("Games to fetch", 1, max_g, max_g, step=1)
        cost = int(limit) * max(len(markets), 1)
        fetch_clicked = st.button(f"💸 Fetch props (~{cost} credits)", disabled=not markets)
        if cached:
            st.caption(f"📦 Cache: {len(cached['events'])} games for {today} "
                       f"({', '.join(cached.get('markets', []) or ['—'])}) — viewing is free")

        def model_all(events):
            pitchers = (model_props(events, season, build_pitcher_opponent_map(games))
                        if PITCHER_K_MARKET in markets else [])
            batters = (model_batter_props(events, season, build_team_opponent_pitcher_map(games))
                       if BATTER_HITS_MARKET in markets else [])
            return pitchers, batters

        if fetch_clicked and markets:
            with st.spinner("Fetching props + modeling..."):
                payload = fetch_props_cached(api_key, today, markets=markets, limit=int(limit))
                st.session_state["prop_rows"], st.session_state["batter_rows"] = model_all(payload["events"])
                st.session_state["prop_remaining"] = payload.get("remaining")
        elif cached and "prop_rows" not in st.session_state:
            with st.spinner("Modeling cached props..."):
                st.session_state["prop_rows"], st.session_state["batter_rows"] = model_all(cached["events"])

        if st.session_state.get("prop_remaining"):
            st.caption(f"Odds API credits remaining: **{st.session_state['prop_remaining']}**")

        def render_props(rows, name_field, stat_label, subtitle_fn, table_fn, key_prefix):
            """Shared renderer for a prop type: +EV cards (with optional logging) + full table."""
            if not rows:
                return
            rows = sorted(rows, key=lambda r: max(r["edge_over"], r["edge_under"]), reverse=True)
            flagged = [(r, side, p, odds, dec, edge)
                       for r in rows
                       for side, p, odds, dec, edge in [
                           ("Over", r["p_over"], r["over_odds"], r["over_dec"], r["edge_over"]),
                           ("Under", r["p_under"], r["under_odds"], r["under_dec"], r["edge_under"])]
                       if edge >= edge_threshold]
            st.markdown(f"**{len(flagged)} {stat_label} props at edge ≥ {edge_threshold:.0%}**")
            for r, side, p, odds, dec, edge in flagged:
                with st.container(border=True):
                    cols = st.columns([3, 1, 1, 1, 1])
                    cols[0].markdown(
                        f"**{r[name_field]} {side} {r['line']:g} {stat_label} at {format_american(odds)}**  \n"
                        f"_{r['away']} @ {r['home']}_  \n<small>{subtitle_fn(r)}</small>",
                        unsafe_allow_html=True)
                    cols[1].metric("Model", f"{p*100:.0f}%")
                    cols[2].metric("Line", f"{r['line']:g}")
                    cols[3].metric("Edge", f"{edge*100:+.1f}%")
                    stake = kelly_stake(p, dec, bankroll, kelly_frac)
                    cols[4].metric("Stake", f"${stake:.2f}")
                    if log_enabled and cols[4].button("Log bet", key=f"{key_prefix}_{r[name_field]}_{side}"):
                        from datetime import datetime
                        g = gid_by_pair.get((r["home"], r["away"]), {})
                        log_bet({
                            "logged_at": datetime.now().isoformat(timespec="seconds"),
                            "game_date": g.get("game_date", today), "game_id": g.get("game_id", ""),
                            "home_team": r["home"], "away_team": r["away"],
                            "bet_side": side.lower(), "bet_team": r[name_field],
                            "pitcher": r[name_field] if name_field == "pitcher" else "", "opp_pitcher": "",
                            "odds_american": odds, "odds_decimal": round(dec, 4),
                            "model_p": round(p, 4),
                            "market_p": round(r["mkt_over"] if side == "Over" else 1 - r["mkt_over"], 4)
                                        if r.get("mkt_over") else "",
                            "edge": round(edge, 4), "stake": round(stake, 2),
                            "source": "model", "bet_type": "prop",
                            "description": f"{r[name_field]} {side} {r['line']:g} {stat_label}",
                        })
                        st.success(f"Logged: {r[name_field]} {side} {r['line']:g} {stat_label}")
            if not flagged:
                st.info("None meet the edge threshold. Lower it in the sidebar to see more.")
            with st.expander(f"📊 Full {stat_label} table"):
                st.dataframe(table_fn(rows), use_container_width=True, hide_index=True)

        prop_rows = st.session_state.get("prop_rows")
        batter_rows = st.session_state.get("batter_rows")
        if not prop_rows and not batter_rows:
            st.info("No props loaded yet. Pick prop types and click **Fetch props** "
                    "(uses credits), or no cache exists for today.")

        if prop_rows:
            st.markdown("### ⚾ Pitcher Strikeouts")
            render_props(
                prop_rows, "pitcher", "Ks",
                lambda r: f"proj {r['lambda']:.2f} K • K/9 {r['k_per_9']:.1f} • {r['source']}",
                lambda rows: pd.DataFrame([{
                    "Pitcher": r["pitcher"], "Matchup": f"{r['away']} @ {r['home']}",
                    "Line": f"{r['line']:g}", "Proj K": f"{r['lambda']:.2f}", "K/9": f"{r['k_per_9']:.1f}",
                    "Over": format_american(r["over_odds"]), "Under": format_american(r["under_odds"]),
                    "Edge Over": f"{r['edge_over']*100:+.1f}%", "Edge Under": f"{r['edge_under']*100:+.1f}%",
                } for r in rows]),
                "kprop")

        if batter_rows:
            st.markdown("### 🏏 Batter Hits")
            render_props(
                batter_rows, "batter", "Hits",
                lambda r: f"p_hit {r['p_hit']:.3f} × {r['n_ab']} AB • AVG {r['avg']:.3f} • vs {r['opp_pitcher']}",
                lambda rows: pd.DataFrame([{
                    "Batter": r["batter"], "Matchup": f"{r['away']} @ {r['home']}",
                    "Line": f"{r['line']:g}", "P(hit)": f"{r['p_hit']:.3f}", "AVG": f"{r['avg']:.3f}",
                    "Over": format_american(r["over_odds"]), "Under": format_american(r["under_odds"]),
                    "Edge Over": f"{r['edge_over']*100:+.1f}%", "Edge Under": f"{r['edge_under']*100:+.1f}%",
                    "Opp SP": r["opp_pitcher"],
                } for r in rows]),
                "hprop")

    with tab_bet:
        st.subheader("📝 Log a bet you placed")
        st.caption("Use this to track YOUR actual bets — model picks, gut picks, props, anything.")

        bet_kind = st.radio("What kind of bet?", ["Tonight's moneyline", "Custom / prop / other"],
                            horizontal=True)

        if bet_kind == "Tonight's moneyline":
            if not rows:
                st.info("No games loaded yet.")
            else:
                game_labels = [f"{r['away']} @ {r['home']}" for r in rows]
                pick = st.selectbox("Game", game_labels)
                game = rows[game_labels.index(pick)]
                side_label = st.radio(
                    "Side",
                    [f"{game['away']} ({format_american(game['ml_away'])})",
                     f"{game['home']} ({format_american(game['ml_home'])})"],
                    horizontal=True,
                )
                if side_label.startswith(game["home"]):
                    side, team, default_odds = "home", game["home"], int(game["ml_home"])
                    pitcher, opp_pitcher = game["home_pitcher"], game["away_pitcher"]
                    model_p_val = game["p_home"]
                    market_p_val = game["market_p_home"]
                    edge_val = game["edge_home"]
                else:
                    side, team, default_odds = "away", game["away"], int(game["ml_away"])
                    pitcher, opp_pitcher = game["away_pitcher"], game["home_pitcher"]
                    model_p_val = 1 - game["p_home"]
                    market_p_val = 1 - game["market_p_home"]
                    edge_val = game["edge_away"]

                c1, c2 = st.columns(2)
                odds = c1.number_input("Odds (American)", value=default_odds, step=5)
                stake = c2.number_input("Stake ($)", min_value=0.01, value=5.0, step=0.5)

                st.caption(f"Model: {model_p_val*100:.1f}%  •  Market (no-vig): {market_p_val*100:.1f}%  •  Edge: {edge_val*100:+.1f}%")

                if st.button("✅ Log this bet"):
                    from datetime import datetime
                    decimal = american_to_decimal(odds)
                    log_bet({
                        "logged_at": datetime.now().isoformat(timespec="seconds"),
                        "game_date": game["game_date"],
                        "game_id": game["game_id"],
                        "home_team": game["home"], "away_team": game["away"],
                        "bet_side": side, "bet_team": team,
                        "pitcher": pitcher, "opp_pitcher": opp_pitcher,
                        "odds_american": odds, "odds_decimal": round(decimal, 4),
                        "model_p": round(model_p_val, 4),
                        "market_p": round(market_p_val, 4),
                        "edge": round(edge_val, 4),
                        "stake": round(stake, 2),
                        "source": "manual", "bet_type": "moneyline", "description": "",
                    })
                    st.success(f"Logged: {team} ML at {format_american(odds)} for ${stake:.2f}")
                    st.rerun()

        else:
            st.markdown("For player props, parlays, futures, or any bet without a clean ML/total match.")
            description = st.text_input("Description", placeholder="e.g., Valdez 5+ Ks, or 3-leg parlay")
            c1, c2 = st.columns(2)
            odds_manual = c1.number_input("Odds (American)", value=-110, step=5, key="manual_odds")
            stake_manual = c2.number_input("Stake ($)", min_value=0.01, value=5.0, step=0.5, key="manual_stake")
            if st.button("✅ Log this bet", key="log_manual"):
                from datetime import datetime
                if not description.strip():
                    st.error("Add a description so you remember what this bet was.")
                else:
                    decimal = american_to_decimal(odds_manual)
                    log_bet({
                        "logged_at": datetime.now().isoformat(timespec="seconds"),
                        "game_date": today, "game_id": "",
                        "home_team": "", "away_team": "",
                        "bet_side": "", "bet_team": "",
                        "pitcher": "", "opp_pitcher": "",
                        "odds_american": odds_manual, "odds_decimal": round(decimal, 4),
                        "model_p": "", "market_p": "", "edge": "",
                        "stake": round(stake_manual, 2),
                        "source": "manual", "bet_type": "other",
                        "description": description.strip(),
                    })
                    st.success(f"Logged: {description} at {format_american(odds_manual)} for ${stake_manual:.2f}")
                    st.rerun()

    with tab3:
        st.subheader("📈 Performance")

        # Auto-resolve pending ML bets on page load (cached 5 min)
        update_log, last_check = auto_update_pending()
        c_status, c_btn = st.columns([4, 1])
        c_status.caption(f"Auto-check ran at **{last_check}** — {update_log.strip()}")
        if c_btn.button("🔄 Check now"):
            auto_update_pending.clear()
            st.rerun()

        df = ensure_log()
        if df.empty:
            st.info("No bets logged yet. Place bets in 'Place a Bet' or log model recs.")
        else:
            df["source"] = df["source"].fillna("").replace("", "model")
            view = st.radio("View", ["My bets only", "Model recs only", "All"], horizontal=True, index=0)
            if view == "Model recs only":
                df = df[df["source"] == "model"]
            elif view == "My bets only":
                df = df[df["source"] == "manual"]

            if df.empty:
                st.info(f"No bets to show in '{view}' yet. Go to 'Place a Bet' to log one.")
                return

            finished = df[df["status"].isin(["won", "lost"])].copy()
            pending = df[df["status"] == "pending"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Total logged", len(df))
            c2.metric("Pending", len(pending))
            c3.metric("Completed", len(finished))

            if not finished.empty:
                finished["profit_num"] = pd.to_numeric(finished["profit"], errors="coerce").fillna(0)
                finished["stake_num"] = pd.to_numeric(finished["stake"], errors="coerce").fillna(0)
                wins = (finished["status"] == "won").sum()
                wagered = finished["stake_num"].sum()
                profit = finished["profit_num"].sum()
                roi = profit / wagered if wagered > 0 else 0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Win rate", f"{wins/len(finished)*100:.1f}%")
                c2.metric("Wagered", f"${wagered:.0f}")
                c3.metric("Profit", f"${profit:+.2f}", delta_color="normal")
                c4.metric("ROI", f"{roi*100:+.2f}%")

            clv_series = pd.to_numeric(df.get("clv_pct"), errors="coerce").dropna() \
                if "clv_pct" in df.columns else pd.Series(dtype=float)
            if not clv_series.empty:
                st.markdown("**📉 Closing Line Value (CLV)**")
                v1, v2, v3 = st.columns(3)
                v1.metric("Avg CLV", f"{clv_series.mean()*100:+.2f}%")
                v2.metric("Beat the close", f"{(clv_series > 0).mean()*100:.0f}%",
                          help=f"{(clv_series > 0).sum()} of {len(clv_series)} bets got a better price than the close")
                v3.metric("Bets tracked", len(clv_series))
                st.caption("Positive average CLV over many bets = you're beating the market — the "
                           "best early sign the model has real edge. Updated automatically 3×/day by the cloud job.")

            def bet_label(row):
                bt = str(row.get("bet_type") or "").lower()
                desc = str(row.get("description") or "").strip()
                if desc:
                    return desc
                team = row.get("bet_team") or "—"
                return f"{team} ML" if bt == "moneyline" else team

            def fmt_bets(d, extra_cols):
                out = pd.DataFrame(index=d.index)
                out["Date"] = d["game_date"]
                out["Bet"] = d.apply(bet_label, axis=1)
                out["Type"] = d["bet_type"].fillna("").str.capitalize()
                out["Odds"] = pd.to_numeric(d["odds_american"], errors="coerce").apply(
                    lambda x: format_american(int(x)) if pd.notna(x) else "—")
                out["Stake"] = pd.to_numeric(d["stake"], errors="coerce").apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "—")
                out["Edge"] = pd.to_numeric(d["edge"], errors="coerce").apply(
                    lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
                if "close_odds" in d.columns:
                    out["Close"] = pd.to_numeric(d["close_odds"], errors="coerce").apply(
                        lambda x: format_american(int(x)) if pd.notna(x) else "—")
                if "clv_pct" in d.columns:
                    out["CLV"] = pd.to_numeric(d["clv_pct"], errors="coerce").apply(
                        lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "—")
                for c in extra_cols:
                    out[c] = d[c.lower()] if c.lower() in d.columns else "—"
                return out

            if not pending.empty:
                st.subheader("⏳ Pending bets")
                pend_view = pending.copy()
                pend_view["Status"] = "pending"
                st.dataframe(fmt_bets(pend_view, ["Status"]),
                             use_container_width=True, hide_index=True)

            manual_pending = pending[
                (pending["bet_type"].fillna("").astype(str).str.lower().isin(["other", "parlay"]))
                | (pending["game_id"].fillna("").astype(str).str.strip() == "")
            ]
            if not manual_pending.empty:
                st.subheader("⏳ Pending — manual resolve")
                st.caption("Auto-resolution works for moneylines via game outcomes. Custom bets (props/parlays) need to be marked here.")
                from tracker import manual_resolve
                for idx, row in manual_pending.iterrows():
                    desc = row.get("description") or f"{row.get('bet_team')}"
                    cols = st.columns([4, 1, 1, 1])
                    cols[0].markdown(f"**{desc}** @ {format_american(int(row['odds_american']))} • ${float(row['stake']):.2f}")
                    if cols[1].button("✅ Won", key=f"w{idx}"):
                        manual_resolve(idx, True); st.rerun()
                    if cols[2].button("❌ Lost", key=f"l{idx}"):
                        manual_resolve(idx, False); st.rerun()

            if not finished.empty:
                st.subheader("Recent results")
                res = fmt_bets(finished, ["Status"])
                res["Profit"] = pd.to_numeric(finished["profit"], errors="coerce").apply(
                    lambda x: f"${x:+.2f}" if pd.notna(x) else "—")
                st.dataframe(res.tail(30), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
