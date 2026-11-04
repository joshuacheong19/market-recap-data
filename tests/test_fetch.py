"""Offline tests: every network call is replaced with sample data."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import fetch_market_data as f  # noqa: E402

DAYS = pd.to_datetime(["2026-11-02", "2026-11-03", "2026-11-04"])
SESSION = pd.Timestamp("2026-11-03")


def test_session_ignores_todays_unfinished_bar():
    spy = pd.Series([752.1, 754.35, 760.0], index=DAYS)
    during = datetime(2026, 11, 4, 11, 0, tzinfo=f.EASTERN)
    after = datetime(2026, 11, 4, 16, 20, tzinfo=f.EASTERN)
    assert f.session_date(spy, during) == SESSION
    assert f.session_date(spy, after) == DAYS[2]


def test_quote_change_and_stale_symbol_rejected():
    closes = pd.DataFrame({"A": [100.0, 110.0, None], "STALE": [5.0, None, None]}, index=DAYS)
    q = f.quote(closes, "A", SESSION)
    assert q["change_pct"] == 10.0 and q["previous_date"] == "2026-11-02"
    assert f.quote(closes, "STALE", SESSION) is None
    assert f.quote(closes, "NOPE", SESSION) is None


def test_parse_constituents_and_dot_tickers():
    rows = "".join(
        f"<tr><td>T{i}</td><td>Co {i}</td><td>Tech</td><td>Software</td></tr>" for i in range(99))
    html = ("<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>"
            "<th>GICS Sub-Industry</th></tr>"
            f"<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>Insurance</td></tr>{rows}</table>")
    parsed = f.parse_constituents(html, "S&P 500")
    assert parsed["BRK-B"]["industry"] == "Insurance"
    assert len(parsed) == 100


def test_parse_constituents_missing_table_raises():
    with pytest.raises(ValueError):
        f.parse_constituents("<table><tr><th>x</th></tr><tr><td>1</td></tr></table>", "S&P 500")


def test_movers_rank_breadth_and_missing():
    closes = pd.DataFrame(
        {f"T{i}": [100.0, 100.0 + i, None] for i in range(-6, 7)}, index=DAYS)
    universe = {f"T{i}": {"company": "c", "sector": "s", "industry": "i", "indices": ["S&P 500"]}
                for i in range(-6, 7)}
    universe["GONE"] = {"company": "c", "sector": "s", "industry": "i", "indices": ["S&P 500"]}
    m = f.build_movers(closes, universe, SESSION)
    assert [r["ticker"] for r in m["gainers"]][:2] == ["T6", "T5"]
    assert m["losers"][0]["ticker"] == "T-6"
    assert m["advancers"] == 6 and m["decliners"] == 6
    assert m["missing_tickers"] == ["GONE"] and m["priced"] == 13


CSV = ('Date,"3 Mo","2 Yr","5 Yr","7 Yr","10 Yr","30 Yr"\n'
       "11/04/2026,4.30,4.70,4.80,4.90,5.00,5.40\n"
       "11/03/2026,4.22,4.63,4.76,4.84,4.97,5.33\n"
       "11/02/2026,4.21,4.59,4.70,4.79,4.91,5.28\n")


def test_treasury_uses_session_row_and_spreads():
    t = f.parse_treasury(CSV, SESSION)
    assert t["as_of"] == "2026-11-03" and t["matches_session"] is True
    assert t["yields"]["10-year"] == {"yield_pct": 4.97, "change_bp": 6}
    assert t["2s10s"] == {"spread_bp": 34, "change_bp": 2}
    assert t["5s30s"] == {"spread_bp": 57, "change_bp": -1}


def test_treasury_flags_row_not_yet_published():
    t = f.parse_treasury(CSV, pd.Timestamp("2026-11-05"))
    assert t["as_of"] == "2026-11-04" and t["matches_session"] is False


def test_build_end_to_end(monkeypatch):
    symbols = list(f.EQUITIES.values()) + list(f.SECTORS) + [c[2] for c in f.COMMODITIES]
    universe = {f"T{i}": {"company": "c", "sector": "s", "industry": "i", "indices": ["S&P 500"]}
                for i in range(12)}
    all_symbols = symbols + list(universe)
    table = pd.DataFrame({s: [100.0, 101.0, None] for s in all_symbols}, index=DAYS)
    table = table.drop(columns=["^VVIX"])
    monkeypatch.setattr(f, "load_universe", lambda: universe)
    monkeypatch.setattr(f, "download_closes", lambda _: table)
    monkeypatch.setattr(f, "http_get", lambda _: CSV)
    now = datetime(2026, 11, 3, 22, 15, tzinfo=timezone.utc)
    out = f.build(now)
    assert out["session_date"] == "2026-11-03"
    assert out["equities"]["QQQ"]["change_pct"] == 1.0
    assert out["missing_symbols"] == ["VVIX"]
    assert out["treasury"]["matches_session"] is True
    assert len(out["sectors"]) == 11 and out["errors"] == {}


def test_failed_treasury_is_recorded_not_fatal(monkeypatch):
    table = pd.DataFrame({"SPY": [100.0, 101.0, None]}, index=DAYS)
    monkeypatch.setattr(f, "load_universe", lambda: (_ for _ in ()).throw(ValueError("wiki down")))
    monkeypatch.setattr(f, "download_closes", lambda _: table)
    monkeypatch.setattr(f, "http_get", lambda _: (_ for _ in ()).throw(OSError("blocked")))
    out = f.build(datetime(2026, 11, 3, 22, 15, tzinfo=timezone.utc))
    assert "treasury" in out["errors"] and "movers" in out["errors"]
    assert out["equities"]["SPY"]["change_pct"] == 1.0


def test_download_closes_handles_yfinance_shape(monkeypatch):
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "QQQ"]])
    raw = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                       index=pd.to_datetime(["2026-11-03", "2026-11-02"]), columns=columns)
    monkeypatch.setattr(f.yf, "download", lambda *a, **k: raw)
    closes = f.download_closes(["SPY", "QQQ"])
    assert list(closes.index) == list(pd.to_datetime(["2026-11-02", "2026-11-03"]))
    assert closes.loc["2026-11-03", "QQQ"] == 2.0
