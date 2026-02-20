import io
import re
from typing import Optional, Tuple, Dict

import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------
HDA_MARKET = "Home Draw Away, Ordinary Time"
SUPER_ODDS_MARKET = "3-way (0% margin), Ordinary Time"   # promo: "Szuper Odds"

# Canonical column names we will work with
CANON = {
    "operator": "Operator",
    "user name": "User Name",
    "user id": "User ID",
    "bet id": "Bet ID",
    "bet type": "Bet Type",
    "event": "Event",
    "sport": "Sport",
    "league": "League",
    "market": "Market",
    "pick": "Pick",
    "stake (eur)": "Stake (EUR)",
    "stake (user currency)": "Stake (User Currency)",
    "user currency": "User Currency",
    "odds": "Odds",
    "possible profit (eur)": "Possible profit (EUR)",
    "possible profit (user currency)": "Possible profit (User Currency)",
    "placed date": "Placed Date",
    "settlement status": "Settlement status",
}

REQUIRED_CANON = list(CANON.values())


# ----------------------------
# Helpers
# ----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the app tolerant to small export differences like:
      - 'User name' vs 'User Name'
      - 'Placed date' vs 'Placed Date'
    """
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = _norm(col)
        if key in CANON:
            rename_map[col] = CANON[key]
    return df.rename(columns=rename_map).copy()


def validate_columns(df: pd.DataFrame) -> list:
    return [c for c in REQUIRED_CANON if c not in df.columns]


def parse_teams_from_event(event: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort parse of home/away team names from the Event column.
    Supports common separators like ' v ', ' vs ', '-', '–', '—', ':'.
    Returns (home, away) or (None, None) if not parseable.
    """
    if event is None:
        return None, None
    e = str(event).strip()
    seps = [r"\s+v\s+", r"\s+vs\.?\s+", r"\s+-\s+", r"\s+–\s+", r"\s+—\s+", r"\s+:\s+"]
    for sep in seps:
        parts = re.split(sep, e, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return parts[0].strip(), parts[1].strip()
    return None, None


def classify_pick(pick: str, home: Optional[str], away: Optional[str]) -> Optional[str]:
    """
    Classify pick into one of: 'home', 'draw', 'away' when possible.
    Works with:
      - textual: Home/Away/Draw, 1/X/2, H/D/A
      - team name picks: matches home/away tokens against team names
    """
    p = _norm(pick)

    # explicit codes
    if p in {"home", "h", "1", "1.", "1)", "team1", "t1"}:
        return "home"
    if p in {"away", "a", "2", "2.", "2)", "team2", "t2"}:
        return "away"
    if p in {"draw", "d", "x", "tie"}:
        return "draw"

    # team-name match
    if home:
        hn = _norm(home)
        if hn and (hn == p or hn in p or p in hn):
            return "home"
    if away:
        an = _norm(away)
        if an and (an == p or an in p or p in an):
            return "away"

    return None


def realized_profit(row: pd.Series) -> Optional[float]:
    """
    Use Possible profit (User Currency) when Settlement status is Won.
    Lost -> 0.
    Anything else -> None (ignored).
    """
    status = _norm(row.get("Settlement status", ""))
    if status == "won":
        return float(row.get("Possible profit (User Currency)", 0) or 0)
    if status == "lost":
        return 0.0
    return None


# ----------------------------
# Core detection logic
# ----------------------------
def detect_suspicious(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      suspicious_bets_df: bets belonging to users/events that match the pattern
      summary_df: per-user market profit summary

    Flag rules (event-level, per user):
      - Flag if on the HDA market the user placed BOTH 'home' and 'away' picks
        for the same event. (Super Odds is optional.)
    """
    df = df.copy()

    # Normalize fields that commonly contain trailing spaces in exports
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Pick"] = df["Pick"].astype(str).str.strip()

    # Work with normalized market for robust matching (handles double spaces etc.)
    df["_market_norm"] = df["Market"].apply(_norm)

    hda_norm = _norm(HDA_MARKET)
    super_norm = _norm(SUPER_ODDS_MARKET)

    # compute realized profit
    df["Realized Profit (User Currency)"] = df.apply(realized_profit, axis=1)

    # parse teams
    teams = df["Event"].apply(parse_teams_from_event)
    df["_home_team"] = teams.apply(lambda t: t[0])
    df["_away_team"] = teams.apply(lambda t: t[1])

    df["_pick_side"] = [
        classify_pick(p, h, a) for p, h, a in zip(df["Pick"], df["_home_team"], df["_away_team"])
    ]

    # Event-level suspicious pattern: HDA contains BOTH home and away
    group_cols = ["User ID", "Event"]

    def _event_flag(g: pd.DataFrame) -> bool:
        hda = g[g["_market_norm"] == hda_norm]
        if hda.empty:
            return False
        sides = set(hda["_pick_side"].dropna().tolist())
        return ("home" in sides) and ("away" in sides)

    flags = (
        df.groupby(group_cols, dropna=False)
          .apply(_event_flag)
          .rename("Suspicious")
          .reset_index()
    )
    suspicious_events = flags[flags["Suspicious"]].copy()

    if suspicious_events.empty:
        empty_summary = pd.DataFrame(columns=[
            "User ID", "User Name",
            f"Profit: {HDA_MARKET}", f"Profit: {SUPER_ODDS_MARKET}",
            "Total Profit (User Currency)", "Flagged Events", "Flagged Bets"
        ])
        return df.iloc[0:0].copy(), empty_summary

    # Join back to get relevant bets
    df2 = df.merge(suspicious_events[group_cols], on=group_cols, how="inner")

    # Keep only bets on the two relevant markets for reporting.
    suspicious_bets = df2[df2["_market_norm"].isin([hda_norm, super_norm])].copy()

    # Build summary (only Won/Lost contribute; others ignored)
    profit_rows = suspicious_bets.dropna(subset=["Realized Profit (User Currency)"]).copy()

    # For pivot readability, use the original configured market names for columns,
    # but group by normalized market (then map back).
    profit_rows["_market_label"] = profit_rows["_market_norm"].map({
        hda_norm: HDA_MARKET,
        super_norm: SUPER_ODDS_MARKET
    })

    prof = (
        profit_rows.groupby(["User ID", "User Name", "_market_label"])["Realized Profit (User Currency)"]
                  .sum()
                  .reset_index()
    )

    pivot = prof.pivot_table(
        index=["User ID", "User Name"],
        columns="_market_label",
        values="Realized Profit (User Currency)",
        fill_value=0.0
    )
    pivot = pivot.rename(columns={
        HDA_MARKET: f"Profit: {HDA_MARKET}",
        SUPER_ODDS_MARKET: f"Profit: {SUPER_ODDS_MARKET}"
    }).reset_index()

    # ensure profit columns exist even if one market is absent
    if f"Profit: {HDA_MARKET}" not in pivot.columns:
        pivot[f"Profit: {HDA_MARKET}"] = 0.0
    if f"Profit: {SUPER_ODDS_MARKET}" not in pivot.columns:
        pivot[f"Profit: {SUPER_ODDS_MARKET}"] = 0.0

    # add counts
    ev_count = suspicious_events.groupby("User ID")["Event"].nunique().rename("Flagged Events")
    bet_count = suspicious_bets.groupby("User ID")["Bet ID"].nunique().rename("Flagged Bets")

    summary = (
        pivot.merge(ev_count.reset_index(), on="User ID", how="left")
             .merge(bet_count.reset_index(), on="User ID", how="left")
    )
    summary["Flagged Events"] = summary["Flagged Events"].fillna(0).astype(int)
    summary["Flagged Bets"] = summary["Flagged Bets"].fillna(0).astype(int)

    summary["Total Profit (User Currency)"] = summary[f"Profit: {HDA_MARKET}"] + summary[f"Profit: {SUPER_ODDS_MARKET}"]
    summary = summary.sort_values(["Total Profit (User Currency)"], ascending=False, kind="mergesort")

    # Bets sheet columns
    keep_cols = [
        "Operator", "User Name", "User ID", "Bet ID", "Bet Type", "Event", "Sport", "League",
        "Market", "Pick", "Stake (User Currency)", "User Currency", "Odds",
        "Placed Date", "Settlement status", "Realized Profit (User Currency)"
    ]
    keep_cols = [c for c in keep_cols if c in suspicious_bets.columns]
    bets_out = suspicious_bets[keep_cols].copy()
    bets_out = bets_out.sort_values(["User ID", "Event", "Market", "Placed Date"], kind="mergesort")

    return bets_out, summary


def to_excel_bytes(bets_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        bets_df.to_excel(writer, index=False, sheet_name="Bets")
        for sheet_name in ["Summary", "Bets"]:
            ws = writer.book[sheet_name]
            ws.freeze_panes = "A2"
    return output.getvalue()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Early Settlement ügyeskedők szűrése", layout="wide")

st.title("Early Settlement ügyeskedők szűrése")
st.caption(
    "CSV feltöltés -> gyanús felhasználók és fogadások listázása. "
    "A találat feltétele: ugyanazon eseményen a HDA piacon fogadás hazaira és vendégre is."
)

with st.expander("Mit keresünk pontosan?", expanded=False):
    st.markdown(
        f"""
- Csak a **{HDA_MARKET}** piacon van Early Settlement.
- A **{SUPER_ODDS_MARKET}** piacon (Szuper Odds) nincs Early Settlement.
- **Gyanús**, ha ugyanazon felhasználó ugyanazon eseményen a **{HDA_MARKET}** piacon fogad **hazaira és vendégre is**.
- A Szuper Odds fogadás **nem kötelező** a találathoz, de ha jelen van, a riportban látszani fog.
- Profit számítás:
  - `Settlement status = Won` -> realizált profit = `Possible profit (User Currency)`
  - `Settlement status = Lost` -> realizált profit = 0
  - minden más státusz -> kihagyjuk az elszámolásból
        """
    )

uploaded = st.file_uploader("Tölts fel egy CSV-t (fogadások export)", type=["csv"])

if uploaded:
    # robust CSV read: try comma, then semicolon
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, sep=";")

    df = canonicalize_columns(df_raw)

    missing = validate_columns(df)
    if missing:
        st.error("Hiányzó oszlop(ok): " + ", ".join(missing))
        st.stop()

    st.subheader("Beolvasott adatok (minta)")
    st.dataframe(df.head(50), use_container_width=True)

    bets_df, summary_df = detect_suspicious(df)

    st.subheader("Eredmény")
    if summary_df.empty:
        st.info("Nem találtam olyan felhasználót/eseményt, ami megfelel a gyanús mintának ebben a fájlban.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Gyanús felhasználók", int(summary_df["User ID"].nunique()))
            st.metric("Gyanús események (összesen)", int(summary_df["Flagged Events"].sum()))
            st.metric("Gyanús fogadások (összesen)", int(summary_df["Flagged Bets"].sum()))
        with c2:
            st.dataframe(summary_df, use_container_width=True)

        st.markdown("**Gyanús fogadások listája (részletek):**")
        st.dataframe(bets_df, use_container_width=True, height=420)

        xls_bytes = to_excel_bytes(bets_df, summary_df)
        st.download_button(
            label="Eredmény letöltése (XLSX)",
            data=xls_bytes,
            file_name="early_settlement_suspicious_users.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Tölts fel egy CSV fájlt a feldolgozáshoz.")
