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
from tracker import log_bet, LOG_FILE, ensure_log

DATA = Path(__file__).parent / "data"


def load_api_key():
    env = (Path(__file__).parent / ".env").read_text()
    for line in env.splitlines():
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

    rows = []
    for g in games:
        pred = predict_game(g, team_ratings, pitcher_ratings)
        lines = fanduel_lines_for_game(pred, odds_by_pair.get((pred["home"], pred["away"])))
        if lines is None:
            continue
        rows.append({**pred, **lines})
    rows = compute_edges(rows)
    rows.sort(key=lambda r: max(r["edge_home"], r["edge_away"]), reverse=True)

    tab1, tab2, tab3 = st.tabs(["🎯 Recommended Bets", "📋 Full Slate", "📈 Performance"])

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

    with tab3:
        st.subheader("Bet tracker performance")
        df = ensure_log()
        if df.empty:
            st.info("No bets logged yet. Go to Recommended Bets and click 'Log all'.")
        else:
            finished = df[df["status"].isin(["won", "lost"])].copy()
            pending = df[df["status"] == "pending"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Bets logged", len(df))
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
                c3.metric("Profit", f"${profit:+.2f}")
                c4.metric("ROI", f"{roi*100:+.2f}%")

                st.subheader("Recent results")
                display = finished[["game_date", "bet_team", "odds_american", "edge",
                                    "stake", "status", "profit"]].tail(20)
                st.dataframe(display, use_container_width=True, hide_index=True)
            else:
                st.warning("No bets resolved yet. Run `python tracker.py update` after games conclude.")


if __name__ == "__main__":
    main()
