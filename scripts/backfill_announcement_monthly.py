from __future__ import annotations

import argparse
import calendar
import subprocess
import time
from datetime import date, timedelta


def _run_attempts(
    start: date,
    end: date,
    cfg: str,
    attempts: int,
    sleep_seconds: int,
    timeout_seconds: int,
) -> bool:
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
                return True
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
    return False


def run_chunk(
    start: date,
    end: date,
    cfg: str,
    attempts: int,
    sleep_seconds: int,
    timeout_seconds: int,
    adaptive_split: bool,
) -> None:
    if _run_attempts(start, end, cfg, attempts, sleep_seconds, timeout_seconds):
        return

    # Engineering-only fallback: preserve the exact requested PIT date range but
    # isolate a stalled provider response to progressively smaller subranges.
    # A single-day failure is never suppressed or converted to PASS.
    if adaptive_split and start < end:
        span_days = (end - start).days + 1
        left_days = span_days // 2
        left_end = start + timedelta(days=left_days - 1)
        right_start = left_end + timedelta(days=1)
        print(
            f"[announcement] splitting failed chunk {start}..{end} into "
            f"{start}..{left_end} and {right_start}..{end}",
            flush=True,
        )
        run_chunk(
            start,
            left_end,
            cfg,
            attempts,
            sleep_seconds,
            timeout_seconds,
            adaptive_split,
        )
        run_chunk(
            right_start,
            end,
            cfg,
            attempts,
            sleep_seconds,
            timeout_seconds,
            adaptive_split,
        )
        return

    raise SystemExit(f"announcement chunk failed after {attempts} attempts: {start}..{end}")


def iter_month_chunks(year: int, start_month: int, end_month: int, chunk_days: int, today: date):
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


def iter_range_chunks(start: date, end: date, chunk_days: int):
    if chunk_days <= 0:
        yield start, end
        return
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int)
    p.add_argument("--config", required=True)
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--sleep-seconds", type=int, default=20)
    p.add_argument("--start-month", type=int, default=1)
    p.add_argument("--end-month", type=int, default=12)
    p.add_argument("--start-date", type=date.fromisoformat)
    p.add_argument("--end-date", type=date.fromisoformat)
    p.add_argument(
        "--chunk-days",
        type=int,
        default=0,
        help="0 keeps the whole selected range; positive values split it into bounded day chunks",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=2700,
        help="hard timeout per asl announcement_index subprocess attempt",
    )
    p.add_argument(
        "--no-adaptive-split",
        action="store_true",
        help="disable recursive date-range bisection after a bounded chunk exhausts retries",
    )
    args = p.parse_args()

    if args.attempts <= 0:
        p.error("attempts must be > 0")
    if args.chunk_days < 0:
        p.error("chunk-days must be >= 0")
    if args.timeout_seconds <= 0:
        p.error("timeout-seconds must be > 0")

    today = date(2026, 8, 27)
    use_exact_range = args.start_date is not None or args.end_date is not None
    if use_exact_range:
        if args.start_date is None or args.end_date is None:
            p.error("start-date and end-date must be supplied together")
        if args.year is not None:
            p.error("year cannot be combined with start-date/end-date")
        if args.start_date > args.end_date:
            p.error("start-date must not exceed end-date")
        if args.start_date > today:
            return
        range_end = min(args.end_date, today)
        chunks = iter_range_chunks(args.start_date, range_end, args.chunk_days)
    else:
        if args.year is None:
            p.error("year is required unless start-date/end-date are supplied")
        if not 1 <= args.start_month <= 12 or not 1 <= args.end_month <= 12:
            p.error("start/end month must be in 1..12")
        if args.start_month > args.end_month:
            p.error("start-month must not exceed end-month")
        chunks = iter_month_chunks(
            args.year,
            args.start_month,
            args.end_month,
            args.chunk_days,
            today,
        )

    for start, end in chunks:
        run_chunk(
            start,
            end,
            args.config,
            args.attempts,
            args.sleep_seconds,
            args.timeout_seconds,
            adaptive_split=not args.no_adaptive_split,
        )


if __name__ == "__main__":
    main()
