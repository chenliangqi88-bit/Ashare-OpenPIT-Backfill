from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

CORRECTION_RE = re.compile(r"(更正|修订|修正|补充|勘误|更新)")
REPORT_RE = re.compile(r"(年度报告|半年度报告|季度报告|年报|半年报|一季报|三季报|业绩预告|业绩快报)")


def glob_path(root: Path, dataset: str) -> str | None:
    for base in [root / "curated" / dataset, root / "derived" / dataset]:
        if base.exists() and any(base.rglob("*.parquet")):
            return str(base / "**" / "*.parquet")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    root = Path(a.data_root)
    out = Path(a.out_root) if a.out_root else root / "openpit_overrides"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    summary = {}

    # 1) CNINFO announcement date -> next calendar day 00:00 Asia/Shanghai.
    # This is intentionally conservative: date-only disclosure is never usable during the disclosure date.
    ann = glob_path(root, "announcement_index")
    if ann:
        q = f"""
        select announcement_id, symbol, title, category, url, announce_date, source, data_version, fetched_at
        from read_parquet('{ann}', union_by_name=true)
        where announce_date between date '2018-01-01' and date '2026-08-27'
        """
        d = con.execute(q).df()
        if len(d):
            d["announce_date"] = pd.to_datetime(d["announce_date"])
            d["publication_time_precision"] = "date_only"
            # store tz-aware UTC timestamp corresponding to next day 00:00 Shanghai
            local = d["announce_date"].dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1)
            d["available_at"] = local.dt.tz_convert("UTC")
            d["timing_policy"] = "DATE_ONLY_NEXT_CALENDAR_DAY_00_SHANGHAI"
            d["strict_timing_eligible"] = True
            d.to_parquet(out / "announcement_timestamp_registry.parquet", index=False)
            summary["announcement_timestamp_registry_rows"] = len(d)

            corr = d[d["title"].fillna("").map(lambda x: bool(CORRECTION_RE.search(str(x)) and REPORT_RE.search(str(x))))].copy()
            corr["requires_original_field_reconstruction"] = True
            corr["strict_financial_use_before_reconstruction"] = False
            corr.to_parquet(out / "financial_correction_announcement_queue.parquet", index=False)
            summary["financial_correction_queue_rows"] = len(corr)

    # 2) Financial bulk backfill quarantine map. Historical bulk values are useful bootstrap values but
    # cannot be called strict PIT until an original-vintage source or equivalent proof exists.
    fsi = glob_path(root, "financial_statement_items")
    if fsi:
        q = f"""
        select symbol, report_period, statement_type, item_code, item_value, announce_date,
               source, data_version, fetched_at
        from read_parquet('{fsi}', union_by_name=true)
        """
        f = con.execute(q).df()
        if len(f):
            f["strict_pit_eligible"] = False
            f["quarantine_reason"] = "BULK_BACKFILL_CURRENT_RESTATEMENT_VALUE_MAY_BE_PAIRED_TO_FIRST_DISCLOSURE_DATE"
            f["required_override"] = "CNINFO_OR_EXCHANGE_ORIGINAL_FIELD_VINTAGE"
            f.to_parquet(out / "financial_bulk_quarantine.parquet", index=False)
            summary["financial_bulk_quarantine_rows"] = len(f)

    # 3) Macro candidate availability. The data-lake contract describes obs_date as observation/publication date,
    # but this is not uniformly strong enough for H1 strict release-vintage semantics. Preserve it as a candidate,
    # conservatively next-day, and require official source confirmation for strict admission.
    macro = glob_path(root, "macro_indicators")
    if macro:
        m = con.execute(f"select * from read_parquet('{macro}', union_by_name=true)").df()
        if len(m) and "obs_date" in m:
            m["obs_date"] = pd.to_datetime(m["obs_date"])
            local = m["obs_date"].dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1)
            m["candidate_available_at"] = local.dt.tz_convert("UTC")
            m["strict_pit_eligible"] = False
            m["required_override"] = "OFFICIAL_FIRST_RELEASE_OR_DAILY_NONREVISING_SOURCE_PROOF"
            m.to_parquet(out / "macro_vintage_candidates.parquet", index=False)
            summary["macro_candidate_rows"] = len(m)

    # 4) Delisting candidate terminal prices. Never silently accept these as final economic value.
    dl = glob_path(root, "delisting_events")
    bars = glob_path(root, "daily_bars")
    if dl and bars:
        d = con.execute(f"select * from read_parquet('{dl}', union_by_name=true)").df()
        if len(d) and "symbol" in d:
            syms = pd.DataFrame({"symbol": d["symbol"].astype(str).unique()})
            con.register("_delist_symbols", syms)
            last = con.execute(f"""
                select b.symbol, arg_max(b.close, b.trade_date) as last_close,
                       max(b.trade_date) as last_trade_date
                from read_parquet('{bars}', union_by_name=true) b
                join _delist_symbols d using(symbol)
                group by b.symbol
            """).df()
            x = d.merge(last, on="symbol", how="left")
            x["terminal_value_candidate"] = x["last_close"]
            x["terminal_resolution"] = "LAST_OBSERVED_TRADE_CANDIDATE"
            x["strict_pit_eligible"] = False
            x["requires_official_resolution"] = True
            x.to_parquet(out / "delisting_terminal_value_queue.parquet", index=False)
            summary["delisting_terminal_queue_rows"] = len(x)

    (out / "override_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
