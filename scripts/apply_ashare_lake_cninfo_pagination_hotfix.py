from __future__ import annotations

"""Backport the upstream CNINFO pagination termination fix to ashare-lake 0.6.0.

Upstream commit d765ad52fd5ae6403dfd7f0ec3034174a51546ce documents the
exact failure mode: CNINFO can keep ``hasMore`` true after ``totalpages`` is
exhausted and even repeat page 1 for overshot page numbers, producing an
unbounded announcement backfill.  This script applies only that termination
guard to the installed 0.6.0 adapter.  It does not change dataset schema,
PIT dates, filtering, gates, or model semantics.
"""

from importlib import metadata, util
from pathlib import Path

EXPECTED_VERSION = "0.6.0"
UPSTREAM_COMMIT = "d765ad52fd5ae6403dfd7f0ec3034174a51546ce"


def main() -> None:
    version = metadata.version("ashare-lake")
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"refusing CNINFO hotfix: expected ashare-lake {EXPECTED_VERSION}, got {version}"
        )

    spec = util.find_spec("ashare_lake.adapters.cninfo.announcements")
    if spec is None or spec.origin is None:
        raise SystemExit("cannot locate ashare_lake.adapters.cninfo.announcements")
    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")

    # Idempotent success if the exact upstream guard is already present.
    if "page >= total_pages" in text and 'total_pages = data.get("totalpages")' in text:
        print(f"CNINFO pagination guard already present in {path}")
        return

    old_batch = '''            batch = data.get("announcements") or []\n            if not batch:\n                break\n            for item in batch:\n'''
    new_batch = '''            batch = data.get("announcements") or []\n            if not batch:\n                break\n            total_pages = data.get("totalpages")\n            for item in batch:\n'''
    if old_batch not in text:
        raise SystemExit(
            "refusing CNINFO hotfix: ashare-lake adapter does not match the expected 0.6.0 source"
        )
    text = text.replace(old_batch, new_batch, 1)

    old_tail = '''            if not data.get("hasMore"):\n                break\n            page += 1\n'''
    new_tail = '''            if isinstance(total_pages, int) and page >= total_pages:\n                # Exact upstream d765ad5 termination fix: CNINFO hasMore can remain\n                # true after its own totalpages and otherwise loop indefinitely.\n                break\n            if not data.get("hasMore"):\n                break\n            page += 1\n'''
    if old_tail not in text:
        raise SystemExit(
            "refusing CNINFO hotfix: expected pagination tail was not found; no patch written"
        )
    text = text.replace(old_tail, new_tail, 1)
    path.write_text(text, encoding="utf-8")

    verify = path.read_text(encoding="utf-8")
    if "page >= total_pages" not in verify or 'total_pages = data.get("totalpages")' not in verify:
        raise SystemExit("CNINFO hotfix verification failed")
    print(
        f"Applied upstream CNINFO pagination termination fix {UPSTREAM_COMMIT} to {path}"
    )


if __name__ == "__main__":
    main()
