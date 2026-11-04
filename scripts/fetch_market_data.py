#!/usr/bin/env python3
"""Fetch end-of-day US market data and write it to data/latest.json.

Sources: Yahoo Finance (through yfinance, unofficial), the US Treasury daily
par yield curve, and Wikipedia's S&P 500 and Nasdaq 100 constituent tables.
A failed section is recorded under "errors"; the recap falls back to the web
for it. If the session itself cannot be established, nothing is overwritten.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
USER_AGENT = "market-recap-data/1.0 (personal end-of-day market recap)"
EASTERN = ZoneInfo("America/New_York")

EQUITIES = {"SPY": "SPY", "DJI": "^DJI", "QQQ": "QQQ", "RSP": "RSP", "VIX": "^VIX", "VVIX": "^VVIX"}
SECTORS = {
    "XLK": "Technology (software, semiconductors, hardware)",
    "XLC": "Communication services (internet platforms, media, telecom)",
    "XLY": "Consumer discretionary (retail, autos, travel, restaurants)",
    "XLF": "Financials (banks, insurers, payments, asset managers)",
    "XLI": "Industrials (aerospace, machinery, transport, construction)",
    "XLE": "Energy (oil and gas producers, refiners, services)",
    "XLB": "Materials (chemicals, mining, packaging)",
    "XLV": "Health care (pharma, biotech, medical devices, insurers)",
    "XLP": "Consumer staples (food, beverages, household goods, grocers)",
    "XLU": "Utilities (electric, gas and water companies)",
    "XLRE": "Real estate (REITs)",
}
# group, name, Yahoo symbol, unit
COMMODITIES = [
    ("Precious metals", "COMEX gold", "GC=F", "USD/oz"),
    ("Precious metals", "COMEX silver", "SI=F", "USD/oz"),
    ("Precious metals", "NYMEX platinum", "PL=F", "USD/oz"),
    ("Precious metals", "NYMEX palladium", "PA=F", "USD/oz"),
    ("Base metals", "COMEX copper", "HG=F", "USD/lb"),
    ("Base metals", "COMEX aluminium", "ALI=F", "USD/t"),
    ("Ferrous", "SGX/CME iron ore 62% Fe", "TIO=F", "USD/t"),
    ("Ferrous", "CME US Midwest HRC", "HRC=F", "USD/short ton"),
    ("Agriculture", "CBOT wheat", "ZW=F", "US cents/bu"),
    ("Agriculture", "CBOT corn", "ZC=F", "US cents/bu"),
    ("Agriculture", "CBOT soybeans", "ZS=F", "US cents/bu"),
    ("Softs", "ICE coffee", "KC=F", "US cents/lb"),
    ("Softs", "ICE cocoa", "CC=F", "USD/t"),
    ("Softs", "ICE sugar", "SB=F", "US cents/lb"),
    ("Softs", "ICE cotton", "CT=F", "US cents/lb"),
    ("Softs", "ICE orange juice", "OJ=F", "US cents/lb"),
    ("Energy", "NYMEX WTI", "CL=F", "USD/bbl"),
    ("Energy", "ICE Brent", "BZ=F", "USD/bbl"),
    ("Energy", "NYMEX natural gas", "NG=F", "USD/MMBtu"),
]
TENORS = {"3-month": "3 Mo", "2-year": "2 Yr", "5-year": "5 Yr", "7-year": "7 Yr", "10-year": "10 Yr", "30-year": "30 Yr"}
MOVER_COUNT = 5
SUSPECT_MOVE_PCT = 40.0


def http_get(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def download_closes(symbols: list[str]) -> pd.DataFrame:
    """Return unadjusted daily closes, one column per symbol, dates ascending."""
    raw = yf.download(
        symbols, period="10d", interval="1d", auto_adjust=False,
        group_by="column", threads=True, progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    closes = raw["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    closes = closes.copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes.sort_index()


def session_date(spy: pd.Series, now_et: datetime) -> pd.Timestamp:
    """Latest SPY bar that is complete: today's bar only counts after the close."""
    bars = spy.dropna()
    if bars.empty:
        raise ValueError("no SPY data")
    today = pd.Timestamp(now_et.date())
    if now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 5):
        bars = bars[bars.index < today]
    if bars.empty:
        raise ValueError("no completed SPY session")
    return bars.index[-1]


def quote(closes: pd.DataFrame, symbol: str, session: pd.Timestamp) -> dict | None:
    """Close and % change, only if the symbol has a bar for this exact session."""
    if symbol not in closes.columns:
        return None
    series = closes[symbol].dropna()
    if session not in series.index:
        return None
    prior = series[series.index < session]
    if prior.empty or prior.iloc[-1] == 0:
        return None
    close, previous = float(series.loc[session]), float(prior.iloc[-1])
    return {
        "close": round(close, 4),
        "previous_close": round(previous, 4),
        "previous_date": prior.index[-1].date().isoformat(),
        "change_pct": round((close / previous - 1) * 100, 2),
    }


def parse_constituents(html: str, index_name: str) -> dict[str, dict]:
    """Find the constituents table in a Wikipedia page by its column names."""
    for table in pd.read_html(io.StringIO(html)):
        columns = {str(c).strip().lower(): c for c in table.columns}
        symbol_col = columns.get("symbol") or columns.get("ticker")
        sub_col = columns.get("gics sub-industry") or columns.get("icb subsector")
        if symbol_col is None or sub_col is None or len(table) < 90:
            continue
        name_col = columns.get("security") or columns.get("company")
        sector_col = columns.get("gics sector") or columns.get("icb industry")
        out = {}
        for _, row in table.iterrows():
            ticker = str(row[symbol_col]).strip().replace(".", "-")
            out[ticker] = {
                "company": str(row[name_col]).strip() if name_col is not None else ticker,
                "sector": str(row[sector_col]).strip() if sector_col is not None else "",
                "industry": str(row[sub_col]).strip(),
                "indices": [index_name],
            }
        return out
    raise ValueError(f"constituents table not found for {index_name}")


def load_universe() -> dict[str, dict]:
    """Both indices, merged. Falls back to the last good copy if a fetch fails."""
    cache = DATA_DIR / "constituents.json"
    try:
        universe = parse_constituents(
            http_get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"), "S&P 500")
        for ticker, info in parse_constituents(
                http_get("https://en.wikipedia.org/wiki/Nasdaq-100"), "Nasdaq 100").items():
            if ticker in universe:
                universe[ticker]["indices"].append("Nasdaq 100")
            else:
                universe[ticker] = info
    except Exception:
        if cache.exists():
            return json.loads(cache.read_text())
        raise
    DATA_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(universe, indent=1) + "\n")
    return universe


def build_movers(closes: pd.DataFrame, universe: dict[str, dict], session: pd.Timestamp) -> dict:
    rows, missing = [], []
    for ticker, info in universe.items():
        q = quote(closes, ticker, session)
        if q is None:
            missing.append(ticker)
            continue
        rows.append({"ticker": ticker, **info, "change_pct": q["change_pct"], "close": q["close"],
                     "needs_check": abs(q["change_pct"]) >= SUSPECT_MOVE_PCT})
    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return {
        "universe_size": len(universe),
        "priced": len(rows),
        "missing_tickers": sorted(missing),
        "advancers": sum(r["change_pct"] > 0 for r in rows),
        "decliners": sum(r["change_pct"] < 0 for r in rows),
        "gainers": rows[:MOVER_COUNT],
        "losers": rows[::-1][:MOVER_COUNT],
    }


def parse_treasury(text: str, session: pd.Timestamp) -> dict:
    rows = list(csv.DictReader(io.StringIO(text)))
    dated = sorted(
        ((datetime.strptime(r["Date"], "%m/%d/%Y").date(), r) for r in rows), key=lambda x: x[0])
    if len(dated) < 2:
        raise ValueError("fewer than two Treasury rows")
    (prior_date, prior), (latest_date, latest) = dated[-2], dated[-1]
    if latest_date > session.date():
        earlier = [d for d in dated if d[0] <= session.date()]
        if len(earlier) < 2:
            raise ValueError("no Treasury rows for session")
        (prior_date, prior), (latest_date, latest) = earlier[-2], earlier[-1]
    yields, missing = {}, []
    for tenor, column in TENORS.items():
        try:
            level, previous = float(latest[column]), float(prior[column])
        except (KeyError, TypeError, ValueError):
            missing.append(tenor)
            continue
        yields[tenor] = {"yield_pct": level, "change_bp": round((level - previous) * 100)}
    out = {
        "source": "US Treasury daily par yield curve",
        "as_of": latest_date.isoformat(),
        "previous_date": prior_date.isoformat(),
        "matches_session": latest_date == session.date(),
        "yields": yields,
        "missing_tenors": missing,
    }
    for name, short, long in (("2s10s", "2-year", "10-year"), ("5s30s", "5-year", "30-year")):
        if short in yields and long in yields:
            level = (yields[long]["yield_pct"] - yields[short]["yield_pct"]) * 100
            out[name] = {"spread_bp": round(level),
                         "change_bp": yields[long]["change_bp"] - yields[short]["change_bp"]}
    return out


def fetch_treasury(session: pd.Timestamp) -> dict:
    year = session.year
    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
           f"&field_tdr_date_value={year}&page&_format=csv")
    return parse_treasury(http_get(url), session)


def build(now_utc: datetime | None = None) -> dict:
    now_utc = now_utc or datetime.now(timezone.utc)
    errors: dict[str, str] = {}

    universe: dict[str, dict] = {}
    try:
        universe = load_universe()
    except Exception as exc:  # recorded; the recap uses the web for movers
        errors["movers"] = f"constituents: {exc}"

    fixed = list(EQUITIES.values()) + list(SECTORS) + [c[2] for c in COMMODITIES]
    closes = download_closes(sorted(set(fixed) | set(universe)))
    if "SPY" not in closes.columns:
        raise RuntimeError("SPY missing from download")
    session = session_date(closes["SPY"], now_utc.astimezone(EASTERN))

    missing: list[str] = []
    equities = {}
    for label, symbol in EQUITIES.items():
        q = quote(closes, symbol, session)
        if q:
            equities[label] = q
        else:
            missing.append(label)
    sectors = []
    for ticker, description in SECTORS.items():
        q = quote(closes, ticker, session)
        if q:
            sectors.append({"ticker": ticker, "description": description, **q})
        else:
            missing.append(ticker)
    sectors.sort(key=lambda s: s["change_pct"], reverse=True)
    commodities = []
    for group, name, symbol, unit in COMMODITIES:
        q = quote(closes, symbol, session)
        if q:
            commodities.append({"group": group, "name": name, "symbol": symbol, "unit": unit, **q})
        else:
            missing.append(name)

    result = {
        "session_date": session.date().isoformat(),
        "generated_at_utc": now_utc.isoformat(timespec="seconds"),
        "price_source": "Yahoo Finance via yfinance (unofficial; unadjusted closes)",
        "equities": equities,
        "sectors": sectors,
        "commodities": commodities,
        "commodity_note": ("Close is Yahoo's latest price when this file was generated, which can "
                           "differ from the official settlement. Treat as indicative: use reported "
                           "settlements where available. Thin contracts can be stale."),
        "missing_symbols": missing,
    }
    if universe:
        result["movers"] = build_movers(closes, universe, session)
    try:
        result["treasury"] = fetch_treasury(session)
    except Exception as exc:
        errors["treasury"] = str(exc)
    result["errors"] = errors
    return result


def main() -> int:
    try:
        result = build()
    except Exception as exc:
        print(f"Fetch failed, keeping the previous file: {exc}", file=sys.stderr)
        return 1
    DATA_DIR.mkdir(exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=False) + "\n"
    (DATA_DIR / "latest.json").write_text(text)
    history = DATA_DIR / "history"
    history.mkdir(exist_ok=True)
    (history / f"{result['session_date']}.json").write_text(text)
    print(f"Wrote session {result['session_date']}; missing: {result['missing_symbols']}; "
          f"errors: {result['errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
