from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

START = "2018-01-01"
END = "2026-08-27"


def run(cmd: list[str], name: str, log_dir: Path, required: bool = True, timeout: int | None = None) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    out = p.stdout or ""
    (log_dir / f"{name}.log").write_text(out, encoding="utf-8", errors="replace")
    rec = {
        "name": name,
        "cmd": cmd,
        "returncode": p.returncode,
        "required": required,
        "elapsed_seconds": round(time.time() - started, 2),
        "status": "PASS" if p.returncode == 0 else ("FAIL" if required else "WARN"),
    }
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_stats(data_root: Path, dataset: str) -> dict:
    roots = [data_root / "curated" / dataset, data_root / "derived" / dataset]
    files = []
    for r in roots:
        if r.exists():
            files.extend(sorted(r.rglob("*.parquet")))
    return {
        "dataset": dataset,
        "parquet_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "sample_sha256": {p.relative_to(data_root).as_posix(): sha256_file(p) for p in files[:5]},
        "present": bool(files),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="./real_pipeline")
    ap.add_argument("--profile", default="full")
    a = ap.parse_args()

    ws = Path(a.workspace).resolve()
    data_root = ws / "ashare-lake"
    cfg = ws / "configs" / "ashare-lake.toml"
    logs = ws / "logs"
    reports = ws / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)

    commands: list[dict] = []

    # Version-lock runtime metadata before any collection.
    versions = {}
    for cmd_name in ["asl", "python"]:
        try:
            cp = subprocess.run([cmd_name, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            versions[cmd_name] = (cp.stdout or "").strip()
        except Exception as exc:
            versions[cmd_name] = f"ERROR:{exc}"
    (reports / "runtime_versions.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")

    commands.append(run(["asl", "config", "init", "--data-root", str(data_root), "--force"], "00_config_init", logs, True, 300))
    # ashare-lake writes config under cwd/configs; copy/discover it into workspace if needed.
    default_cfg = Path("configs/ashare-lake.toml")
    if default_cfg.exists() and default_cfg.resolve() != cfg:
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_bytes(default_cfg.read_bytes())
    if not cfg.exists() and default_cfg.exists():
        cfg = default_cfg.resolve()

    commands.append(run(["asl", "config", "validate", "--config", str(cfg)], "01_config_validate", logs, True, 300))
    commands.append(run(["asl", "servers", "test", "--config", str(cfg)], "02_sources_test", logs, False, 600))

    # Full market/reference init. This creates instruments/calendar/corporate actions/raw daily/index/status + derive/audit.
    commands.append(run(["asl", "init", "--profile", a.profile, "--config", str(cfg)], "10_init_full", logs, True, 18000))

    # Derived/repair jobs that are essential to survivorship and return reconstruction.
    jobs = [
        ("11_delisted", ["asl", "delisted", "backfill", "--config", str(cfg)], True, 7200),
        ("12_adj_factors", ["asl", "derive", "adj_factors", "--config", str(cfg)], True, 7200),
        ("20_financials", ["asl", "backfill", "financial_statement_items", "--start", START, "--end", END, "--config", str(cfg)], True, 14400),
        ("21_announcements", ["asl", "backfill", "announcement_index", "--start", START, "--end", END, "--config", str(cfg)], True, 14400),
        ("22_earnings_schedule", ["asl", "backfill", "earnings_disclosure_schedule", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
        ("23_valuation", ["asl", "backfill", "valuation_metrics", "--start", START, "--end", END, "--config", str(cfg)], True, 14400),
        ("24_share_structure", ["asl", "backfill", "share_structure", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
        ("25_margin", ["asl", "backfill", "margin_trading", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
        ("26_northbound", ["asl", "backfill", "northbound_flows", "--start", START, "--end", END, "--config", str(cfg)], False, 3600),
        ("27_industry", ["asl", "backfill", "industry_members", "--start", "2020-01-01", "--end", END, "--config", str(cfg)], False, 7200),
        ("28_macro", ["asl", "backfill", "macro_indicators", "--start", START, "--end", END, "--config", str(cfg)], True, 7200),
        ("29_unlocks", ["asl", "backfill", "share_unlock_schedule", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
        ("30_regulatory", ["asl", "backfill", "regulatory_events", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
        ("31_block_trades", ["asl", "backfill", "block_trades", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
        ("32_institutional", ["asl", "backfill", "institutional_holdings", "--start", START, "--end", END, "--config", str(cfg)], False, 7200),
    ]
    for name, cmd, required, timeout in jobs:
        try:
            commands.append(run(cmd, name, logs, required, timeout))
        except subprocess.TimeoutExpired:
            commands.append({"name": name, "cmd": cmd, "required": required, "returncode": None, "status": "FAIL" if required else "WARN", "error": "TIMEOUT"})

    # Finalize and audit after all available backfills.
    commands.append(run(["asl", "derive", "adj_factors", "--config", str(cfg)], "80_rederive_adj", logs, True, 7200))
    commands.append(run(["asl", "audit", "--full", "--config", str(cfg)], "90_audit_full", logs, False, 3600))
    commands.append(run(["asl", "status", "--datasets", "--config", str(cfg)], "91_status_datasets", logs, False, 1200))
    commands.append(run(["asl", "catalog", "--config", str(cfg)], "92_catalog", logs, False, 1200))

    critical = [
        "instruments", "trading_calendar", "daily_bars", "corporate_actions", "adj_factors", "delisting_events",
        "financial_statement_items", "announcement_index", "valuation_metrics", "macro_indicators"
    ]
    optional = [
        "trading_status", "earnings_disclosure_schedule", "share_structure", "margin_trading", "northbound_flows",
        "industry_members", "share_unlock_schedule", "regulatory_events", "block_trades", "institutional_holdings"
    ]
    stats = {x: dataset_stats(data_root, x) for x in critical + optional}
    required_cmd_fail = [x for x in commands if x.get("required") and x.get("status") != "PASS"]
    missing_critical = [k for k in critical if not stats[k]["present"]]

    result = {
        "pipeline_version": "V3.0.3-H1_REAL_BOOTSTRAP_V1",
        "window": [START, END],
        "ashare_lake_version_pin": "0.6.0",
        "real_data": True,
        "synthetic_or_mock": False,
        "commands": commands,
        "dataset_stats": stats,
        "required_command_failures": required_cmd_fail,
        "missing_critical_datasets": missing_critical,
        "real_bootstrap_pass": not required_cmd_fail and not missing_critical,
        "strict_pit_gate_pass": False,
        "strict_pit_blockers": [
            "Historical financial backfill may contain current restated values paired to first announce_date; correction-vintage override is still required.",
            "Historical ST/suspension coverage must pass H1 official/effective-date audit.",
            "Delisting terminal economic values must pass H1 protocol, not only delisting-event presence.",
            "Exact factor-specific lineage and source_max_timestamp audit must pass before calibration.",
            "Official disclosure timestamp precision / restatement chain must pass BF2/H1 gates.",
        ],
        "status": "REAL_BOOTSTRAP_PASS__STRICT_PIT_GATE_PENDING" if (not required_cmd_fail and not missing_critical) else "REAL_BOOTSTRAP_INCOMPLETE",
    }
    (reports / "real_bootstrap_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if not result["real_bootstrap_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
