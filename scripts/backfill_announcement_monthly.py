from __future__ import annotations

import argparse
import calendar
import subprocess
import time
from datetime import date, timedelta


def run_chunk(
    start: date,
    end: date,
    cfg: str,
    attempts: int,
    sleep_seconds: int,
    timeout_seconds: int,
) -> None:
    cmd = [
        "asl", "backfill", "announcement_index",
        "--start", start.isoformat(),
        "--end", end.isoformat(),
        "--config", cfg,
    ]
    for attempt in range(1, attempts + 1):
        print(f"[announcement] {start}..{end} attempt {attempt}/{attempts}", flush=True)
        try:
            proc = subprocess.run(cmd, timeout=timeout_seconds)
            if proc.returncode == 0:
                return
            print(
                f"[announcement] chunk exited rc={proc.returncode}: {start}..{end}",
                flush=True,
            )
        except subprocess.TimeoutExpired:
            print(
                f"[announcement] chunk timed out after {timeout_seconds}s: {start}..{end}",
                flush=True,
            )
        if attempt < attempts:
            delay = sleep_seconds * attempt
            print(f"[announcement] retrying in {delay}s", flush=True)
            time.sleep(delay)
    raise SystemExit(f"announcement chunk failed after {attempts} attempts: {start}..{end}")


def iter_chunks(year: int, start_month: int, end_month: int, chunk_days: int, today: date):
    for month in range(start_month, end_month + 1):
        month_start = date(year, month, 1)
        if month_start > today:
            break
        month_last = calendar.monthrange(year, month)[1]
        month_end = min(date(year, month, month_last), today)
        if chunk_days <= 0:
            yield month_start, month_end
            continue
        start = month_start
        while start <= month_end:
            end = min(start + timedelta(days=chunk_days - 1), month_end)
            yield start, end
            start = end + timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--sleep-seconds", type=int, default=20)
    p.add_argument("--start-month", type=int, default=1)
    p.add_argument("--end-month", type=int, default=12)
    p.add_argument(
        "--chunk-days",
        type=int,
        default=0,
        help="0 keeps calendar-month chunks; positive values split each month into bounded day chunks",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=2700,
        help="hard timeout per asl announcement_index subprocess attempt",
    )
    args = p.parse_args()

    if not 1 <= args.start_month <= 12 or not 1 <= args.end_month <= 12:
        p.error("start/end month must be in 1..12")
    if args.start_month > args.end_month:
        p.error("start-month must not exceed end-month")
    if args.chunk_days < 0:
        p.error("chunk-days must be >= 0")
    if args.timeout_seconds <= 0:
        p.error("timeout-seconds must be > 0")

    today = date(2026, 8, 27)
    for start, end in iter_chunks(
        args.year,
        args.start_month,
        args.end_month,
        args.chunk_days,
        today,
    ):
        run_chunk(
            start,
            end,
            args.config,
            args.attempts,
            args.sleep_seconds,
            args.timeout_seconds,
        )


if __name__ == "__main__":
    main()
