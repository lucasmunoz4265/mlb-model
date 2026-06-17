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


@st.cache_data(ttl=600, show_spinner="Loading tonight's games...")
def get_tonight_games():
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


@st.cache_data(ttl=300, show_spinner="Fetching FanDuel odds...")
def get_fanduel_odds(api_key):
    if not api_key:
        return [], None
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
    params = {
        "apiKey": api_key, "regions": "us",
        "markets": "h2h,spreads,totals",
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

    PREGAME_STATUSES = {"Scheduled", "Pre-Game", "Warmup", "Delayed"}
    from datetime import datetime, timezone

    def is_pregame(g):
        if g.get("status") not in PREGAME_STATUSES:
            return False
        # Belt-and-suspenders: also check start time in case cached status is stale
        dt_str = g.get("game_datetime") or ""
        if dt_str:
            try:
                game_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if game_dt < datetime.now(timezone.utc):
                    return False
            except Exception:
                pass
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
    rows.sort(key=lambda r: max(r["edge_home"], r["edge_away"]), reverse=True)

    tab1, tab2, tab_bet, tab3 = st.tabs(["🎯 Recommended Bets", "📋 Full Slate", "📝 Place a Bet", "📈 Performance"])

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

            manual_pending = pending[
                (pending["bet_type"].fillna("").astype(str).str.lower().isin(["other", "prop", "parlay"]))
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
                display_cols = ["game_date", "bet_team", "description", "odds_american", "stake", "source", "status", "profit"]
                available_cols = [c for c in display_cols if c in finished.columns]
                st.dataframe(finished[available_cols].tail(30), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
