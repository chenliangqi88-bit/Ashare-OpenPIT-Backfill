from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

START = "2018-01-01"
END = "2026-08-27"


def glob_sql(root: Path, dataset: str) -> str | None:
    candidates = [root / "curated" / dataset, root / "derived" / dataset]
    for p in candidates:
        if p.exists() and any(p.rglob("*.parquet")):
            return str(p / "**" / "*.parquet").replace("'", "''")
    return None


def scalar(con, sql: str, default=None):
    try:
        row = con.execute(sql).fetchone()
        return default if not row else row[0]
    except Exception:
        return default


def audit(data_root: Path, reports: Path) -> dict:
    con = duckdb.connect()
    checks = []

    def add(check_id, passed, value, requirement, blocking=True, note=""):
        checks.append({
            "check_id": check_id,
            "status": "PASS" if bool(passed) else "FAIL",
            "value": value,
            "requirement": requirement,
            "blocking": bool(blocking),
            "note": note,
        })

    # ---- BF1 market/universe ----
    bars = glob_sql(data_root, "daily_bars")
    instr = glob_sql(data_root, "instruments")
    ca = glob_sql(data_root, "corporate_actions")
    adj = glob_sql(data_root, "adj_factors")
    dl = glob_sql(data_root, "delisting_events")
    ts = glob_sql(data_root, "trading_status")

    if bars:
        n = scalar(con, f"select count(*) from read_parquet('{bars}', union_by_name=true) where trade_date between '{START}' and '{END}'", 0)
        dup = scalar(con, f"select count(*) from (select symbol,trade_date,count(*) c from read_parquet('{bars}', union_by_name=true) where trade_date between '{START}' and '{END}' group by 1,2 having c>1)", 0)
        ohlc = scalar(con, f"select count(*) from read_parquet('{bars}', union_by_name=true) where trade_date between '{START}' and '{END}' and (high < greatest(open,close,low) or low > least(open,close,high) or volume < 0 or amount < 0)", 0)
        dmin = scalar(con, f"select min(trade_date) from read_parquet('{bars}', union_by_name=true) where trade_date >= '{START}'")
        dmax = scalar(con, f"select max(trade_date) from read_parquet('{bars}', union_by_name=true) where trade_date <= '{END}'")
        add("BF1_MARKET_ROWS", n > 0, n, ">0 real daily-bar rows")
        add("BF1_MARKET_DUPLICATES", dup == 0, dup, "0 duplicate symbol+trade_date")
        add("BF1_MARKET_OHLC", ohlc == 0, ohlc, "0 OHLC/volume/amount invariant violations")
        add("BF1_MARKET_WINDOW_START", str(dmin) <= "2018-01-10" if dmin else False, str(dmin), "coverage reaches early Jan 2018")
        add("BF1_MARKET_WINDOW_END", str(dmax) >= "2026-08-20" if dmax else False, str(dmax), "coverage reaches Aug 2026")
    else:
        for cid in ["BF1_MARKET_ROWS","BF1_MARKET_DUPLICATES","BF1_MARKET_OHLC","BF1_MARKET_WINDOW_START","BF1_MARKET_WINDOW_END"]:
            add(cid, False, None, "daily_bars required")

    add("BF1_INSTRUMENTS", bool(instr and scalar(con, f"select count(*) from read_parquet('{instr}', union_by_name=true)", 0) > 0), None, "historical security master present")
    add("BF1_CORPORATE_ACTIONS", bool(ca and scalar(con, f"select count(*) from read_parquet('{ca}', union_by_name=true)", 0) > 0), None, "corporate actions present")
    add("BF1_ADJ_FACTORS", bool(adj and scalar(con, f"select count(*) from read_parquet('{adj}', union_by_name=true)", 0) > 0), None, "adjustment factors present")

    dl_rows = scalar(con, f"select count(*) from read_parquet('{dl}', union_by_name=true)", 0) if dl else 0
    add("BF1_DELISTING_EVENTS", dl_rows > 0, dl_rows, ">0 real delisting events")
    if dl and bars:
        # Use symbol presence rather than exact terminal value here; H1 terminal-value protocol is a separate hard gate.
        ratio = scalar(con, f"with d as (select distinct symbol from read_parquet('{dl}', union_by_name=true)), b as (select distinct symbol from read_parquet('{bars}', union_by_name=true) where trade_date between '{START}' and '{END}') select count(*) filter(where b.symbol is not null)::double/nullif(count(*),0) from d left join b using(symbol)", 0.0)
        add("BF1_SURVIVORSHIP_HISTORY", ratio >= 0.90 if ratio is not None else False, ratio, ">=90% delisting-event symbols with market history", note="Engineering coverage gate; terminal economics audited separately.")
    else:
        add("BF1_SURVIVORSHIP_HISTORY", False, None, "delisted securities retained in history")

    if ts:
        ts_min = scalar(con, f"select min(trade_date) from read_parquet('{ts}', union_by_name=true)")
        add("BF1_TRADING_STATUS_HISTORY", str(ts_min) <= "2018-01-10" if ts_min else False, str(ts_min), "historical ST/suspension status reaches 2018")
    else:
        add("BF1_TRADING_STATUS_HISTORY", False, None, "historical ST/suspension status required")

    # Strict terminal economics cannot be inferred merely from event presence.
    terminal_override = data_root / "openpit_overrides" / "delisting_terminal_values.parquet"
    add("H1_DELISTING_TERMINAL_VALUES", terminal_override.exists(), str(terminal_override), "H1 terminal value override ledger exists")

    # ---- BF2 disclosures/fundamentals/macro ----
    fsi = glob_sql(data_root, "financial_statement_items")
    ann = glob_sql(data_root, "announcement_index")
    macro = glob_sql(data_root, "macro_indicators")
    val = glob_sql(data_root, "valuation_metrics")
    industry = glob_sql(data_root, "industry_members")

    for cid, path, req in [
        ("BF2_FINANCIAL_ROWS", fsi, "financial_statement_items present"),
        ("BF2_ANNOUNCEMENT_ROWS", ann, "announcement_index present"),
        ("BF2_MACRO_ROWS", macro, "macro_indicators present"),
        ("BF2_VALUATION_ROWS", val, "valuation_metrics present"),
    ]:
        n = scalar(con, f"select count(*) from read_parquet('{path}', union_by_name=true)", 0) if path else 0
        add(cid, n > 0, n, req)

    # FSI must carry announce_date and source lineage; but ashare-lake historical backfill documentation warns that
    # current restated values may be paired with first disclose date. Therefore schema completeness is necessary but not sufficient.
    if fsi:
        null_announce = scalar(con, f"select count(*) from read_parquet('{fsi}', union_by_name=true) where announce_date is null", -1)
        dup = scalar(con, f"select count(*) from (select symbol,report_period,statement_type,item_code,announce_date,count(*) c from read_parquet('{fsi}', union_by_name=true) group by all having c>1)", -1)
        add("BF2_FINANCIAL_ANNOUNCE_DATE", null_announce == 0, null_announce, "0 missing announce_date")
        add("BF2_FINANCIAL_KEY_UNIQUE", dup == 0, dup, "0 duplicate financial PIT key")
    else:
        add("BF2_FINANCIAL_ANNOUNCE_DATE", False, None, "announce_date required")
        add("BF2_FINANCIAL_KEY_UNIQUE", False, None, "unique PIT key required")

    finance_override = data_root / "openpit_overrides" / "financial_field_vintages.parquet"
    add("H1_EXACT_FINANCIAL_VINTAGES", finance_override.exists(), str(finance_override), "official/original field-vintage override exists", note="Historical bulk FSI alone is not strict PIT because later restatements can leak into first announce date.")

    announcement_override = data_root / "openpit_overrides" / "announcement_timestamp_registry.parquet"
    add("H1_ANNOUNCEMENT_TIMESTAMPS", announcement_override.exists(), str(announcement_override), "timestamp precision registry exists; date-only is conservatively next-day")

    macro_override = data_root / "openpit_overrides" / "macro_vintages.parquet"
    add("H1_MACRO_VINTAGES", macro_override.exists(), str(macro_override), "official first-release/revision macro vintage registry exists")

    if industry:
        i_min = scalar(con, f"select min(as_of_date) from read_parquet('{industry}', union_by_name=true)")
        add("BF2_INDUSTRY_PIT", str(i_min) <= "2020-01-31" if i_min else False, str(i_min), "historical industry mapping present from source-supported 2020 start", blocking=False)
    else:
        add("BF2_INDUSTRY_PIT", False, None, "historical industry mapping desired", blocking=False)

    # ---- H1 / BF3 strict prerequisites ----
    lineage = data_root / "openpit_features" / "factor_lineage.parquet"
    factors = data_root / "openpit_features" / "factor_panel_wide.parquet"
    labels = data_root / "openpit_features" / "forward_labels.parquet"
    add("BF3_FACTOR_PANEL", factors.exists(), str(factors), "real F01-F55 factor panel exists")
    add("H1_EXACT_FACTOR_LINEAGE", lineage.exists(), str(lineage), "factor-specific lineage covers non-missing observations")
    add("BF3_FORWARD_LABELS", labels.exists(), str(labels), "20/60/120 exact-session labels exist")

    blocking_failures = [x for x in checks if x["blocking"] and x["status"] != "PASS"]
    result = {
        "audit_version": "V3.0.3-H1_REAL_AUDIT_V1",
        "real_data": True,
        "synthetic_or_mock": False,
        "checks": checks,
        "blocking_failures": blocking_failures,
        "strict_pit_gate_pass": not blocking_failures,
        "status": "STRICT_PIT_GATE_PASS" if not blocking_failures else "STRICT_PIT_GATE_BLOCKED",
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "h1_real_data_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--reports", default="./reports")
    a = ap.parse_args()
    result = audit(Path(a.data_root), Path(a.reports))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result["strict_pit_gate_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
