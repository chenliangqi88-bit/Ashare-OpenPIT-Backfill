from __future__ import annotations

import argparse
import calendar
import subprocess
import time
from datetime import date


def run_chunk(start: date, end: date, cfg: str, attempts: int, sleep_seconds: int) -> None:
    cmd = [
        "asl", "backfill", "announcement_index",
        "--start", start.isoformat(),
        "--end", end.isoformat(),
        "--config", cfg,
    ]
    for attempt in range(1, attempts + 1):
        print(f"[announcement] {start}..{end} attempt {attempt}/{attempts}", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return
        if attempt < attempts:
            delay = sleep_seconds * attempt
            print(f"[announcement] retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise SystemExit(f"announcement chunk failed after {attempts} attempts: {start}..{end}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--sleep-seconds", type=int, default=20)
    args = p.parse_args()

    today = date(2026, 8, 27)
    for month in range(1, 13):
        start = date(args.year, month, 1)
        if start > today:
            break
        last = calendar.monthrange(args.year, month)[1]
        end = min(date(args.year, month, last), today)
        run_chunk(start, end, args.config, args.attempts, args.sleep_seconds)


if __name__ == "__main__":
    main()
