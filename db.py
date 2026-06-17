"""Supabase storage backend for bet_log. Falls back to local CSV if not configured."""

import os
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
LOG_FILE = DATA / "bet_log.csv"

NUMERIC_FIELDS = ("odds_american", "odds_decimal", "model_p", "market_p", "edge", "stake", "profit")
COLUMNS = [
    "logged_at", "game_date", "game_id", "home_team", "away_team",
    "bet_side", "bet_team", "pitcher", "opp_pitcher",
    "odds_american", "odds_decimal",
    "model_p", "market_p", "edge", "stake",
    "status", "actual_winner", "profit",
    "source", "bet_type", "description",
]


def _read_env_file() -> dict:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return {}
    out = {}
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _get_credentials() -> tuple:
    """Look for SUPABASE_URL/SUPABASE_KEY in env vars, Streamlit secrets, or .env."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if url and key:
        return url, key
    try:
        import streamlit as st
        if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
            return st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"]
    except Exception:
        pass
    env = _read_env_file()
    return env.get("SUPABASE_URL"), env.get("SUPABASE_KEY")


_client = None


def supabase_client():
    global _client
    if _client is not None:
        return _client
    url, key = _get_credentials()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        print(f"Failed to init Supabase client: {e}")
        return None


def _coerce_numeric(record: dict) -> dict:
    out = dict(record)
    for k in NUMERIC_FIELDS:
        if k in out:
            v = out[k]
            if v in (None, "", "nan") or (isinstance(v, float) and pd.isna(v)):
                out[k] = None
            else:
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    out[k] = None
    return out


def read_all() -> pd.DataFrame:
    """Return the full bet_log as a DataFrame, indexed by id (Supabase) or row position (CSV)."""
    client = supabase_client()
    if client:
        try:
            res = client.table("bet_log").select("*").order("id").execute()
            df = pd.DataFrame(res.data) if res.data else pd.DataFrame(columns=COLUMNS + ["id"])
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = None
            if "id" in df.columns:
                df = df.set_index("id")
            return df
        except Exception as e:
            print(f"Supabase read failed: {e}, falling back to CSV")
    if not LOG_FILE.exists():
        LOG_FILE.parent.mkdir(exist_ok=True)
        pd.DataFrame(columns=COLUMNS).to_csv(LOG_FILE, index=False)
    df = pd.read_csv(LOG_FILE)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = "" if col in ("source", "bet_type", "description") else None
    return df


def insert_or_update(row: dict) -> None:
    """Insert a pending bet, or update an existing pending bet for the same game+side."""
    record = _coerce_numeric({**row, "status": "pending", "actual_winner": "", "profit": None})
    client = supabase_client()
    if client:
        try:
            gid = str(record.get("game_id") or "")
            side = record.get("bet_side") or ""
            existing = (client.table("bet_log").select("id")
                        .eq("game_id", gid).eq("bet_side", side).eq("status", "pending")
                        .execute())
            if existing.data and gid:
                client.table("bet_log").update(record).eq("id", existing.data[0]["id"]).execute()
            else:
                client.table("bet_log").insert(record).execute()
            return
        except Exception as e:
            print(f"Supabase write failed: {e}, falling back to CSV")
    # CSV fallback
    df = read_all()
    mask = (df["game_id"].astype(str) == str(record.get("game_id") or "")) & \
           (df["bet_side"] == record.get("bet_side")) & (df["status"] == "pending")
    if mask.any() and record.get("game_id"):
        for col, val in record.items():
            df.loc[mask, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)


def update_bet(bet_id, fields: dict) -> None:
    """Update a single bet by id (Supabase) or by index (CSV)."""
    client = supabase_client()
    fields = _coerce_numeric(fields)
    if client:
        try:
            client.table("bet_log").update(fields).eq("id", int(bet_id)).execute()
            return
        except Exception as e:
            print(f"Supabase update failed: {e}, falling back to CSV")
    df = read_all()
    for col, val in fields.items():
        df.at[bet_id, col] = val
    df.to_csv(LOG_FILE, index=False)


def is_supabase_active() -> bool:
    return supabase_client() is not None
