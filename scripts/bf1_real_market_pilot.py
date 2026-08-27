from __future__ import annotations
import argparse, json, time, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def get_bs():
    import baostock as bs
    return bs

def login(bs):
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f'BaoStock login failed: {lg.error_code} {lg.error_msg}')

def rs_to_df(rs):
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != '0':
        raise RuntimeError(f'BaoStock query error: {rs.error_code} {rs.error_msg}')
    return pd.DataFrame(rows, columns=rs.fields)

def stock_basic_all(bs):
    rs = bs.query_stock_basic(code='', code_name='')
    d = rs_to_df(rs)
    if len(d):
        for c in ['ipoDate', 'outDate']:
            if c in d:
                d[c] = pd.to_datetime(d[c], errors='coerce')
    return d

def calendar(bs, year):
    rs = bs.query_trade_dates(start_date=f'{year}-01-01', end_date=f'{year}-12-31')
    d = rs_to_df(rs)
    if len(d):
        d['calendar_date'] = pd.to_datetime(d['calendar_date'])
        d['is_trading_day'] = pd.to_numeric(d['is_trading_day'], errors='coerce').fillna(0).astype(int)
    return d

def snapshot_codes(bs, days):
    codes = set()
    for day in days:
        rs = bs.query_all_stock(day=pd.Timestamp(day).strftime('%Y-%m-%d'))
        try:
            d = rs_to_df(rs)
            if 'code' in d:
                codes.update(d['code'].dropna().astype(str))
        except Exception:
            pass
        time.sleep(0.03)
    return sorted(codes)

def codes_for_year(bs, year, cal):
    d = stock_basic_all(bs)
    y0 = pd.Timestamp(f'{year}-01-01')
    y1 = pd.Timestamp(f'{year}-12-31')
    codes = []
    if len(d) and {'code', 'ipoDate', 'outDate'}.issubset(d.columns):
        ipo = pd.to_datetime(d['ipoDate'], errors='coerce')
        out = pd.to_datetime(d['outDate'], errors='coerce')
        mask = (ipo.isna() | (ipo <= y1)) & (out.isna() | (out >= y0))
        codes = sorted(d.loc[mask, 'code'].dropna().astype(str).unique())
    trade = cal.loc[cal['is_trading_day'].eq(1), 'calendar_date'].sort_values()
    if len(trade):
        month_ends = trade.groupby(trade.dt.to_period('M')).max().tolist()
        codes = sorted(set(codes) | set(snapshot_codes(bs, month_ends)))
    return codes

def pull_one(bs, code, year, retries=3):
    last = None
    for attempt in range(retries):
        try:
            rs = bs.query_history_k_data_plus(
                code,
                FIELDS,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
                frequency='d',
                adjustflag='3'
            )
            d = rs_to_df(rs)
            if d.empty:
                return d
            d['trade_date'] = pd.to_datetime(d['date'])
            d = d.rename(columns={'tradestatus': 'trade_status', 'isST': 'is_st', 'turn': 'turnover'})
            for c in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turnover', 'pctChg', 'trade_status', 'is_st', 'adjustflag']:
                if c in d:
                    d[c] = pd.to_numeric(d[c], errors='coerce')
            if (d['adjustflag'].dropna() != 3).any():
                raise RuntimeError('non-unadjusted rows detected')
            d['security_id'] = d['code'].str.replace('sh.', '', regex=False).str.replace('sz.', '', regex=False).str.replace('bj.', '', regex=False)
            ex = np.where(d['code'].str.startswith('sh.'), 'SSE', np.where(d['code'].str.startswith('sz.'), 'SZSE', 'BSE'))
            suffix = np.where(ex == 'SSE', '.SH', np.where(ex == 'SZSE', '.SZ', '.BJ'))
            d['security_id'] = d['security_id'] + suffix
            d['exchange'] = ex
            d['raw_adjustment_flag'] = d['adjustflag']
            return d[['trade_date', 'security_id', 'exchange', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turnover', 'trade_status', 'is_st', 'raw_adjustment_flag']]
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'{code}: {last}')

def audit_chunk(d):
    errors = []
    if d.duplicated(['trade_date', 'security_id']).any():
        errors.append('duplicate keys')
    if (d['raw_adjustment_flag'].dropna() != 3).any():
        errors.append('non-unadjusted rows')
    trad = pd.to_numeric(d['trade_status'], errors='coerce').eq(1)
    if len(d.loc[trad]):
        if ((d.loc[trad, 'high'] + 1e-12) < d.loc[trad, ['open', 'close', 'low']].max(axis=1)).any():
            errors.append('high invariant')
        if ((d.loc[trad, 'low'] - 1e-12) > d.loc[trad, ['open', 'close', 'high']].min(axis=1)).any():
            errors.append('low invariant')
    if (d['volume'].dropna() < 0).any() or (d['amount'].dropna() < 0).any():
        errors.append('negative volume/amount')
    return errors

def run_shard(year, shard, shards, out_dir, chunk_size=150):
    bs = get_bs()
    login(bs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = calendar(bs, year)
    codes = codes_for_year(bs, year, cal)
    selected = [c for i, c in enumerate(codes) if i % shards == shard]
    failures = []
    totals = 0
    chunks = []
    buf = []
    security_success = 0
    for idx, code in enumerate(selected, 1):
        try:
            d = pull_one(bs, code, year)
            if len(d):
                buf.append(d)
                totals += len(d)
                security_success += 1
        except Exception as e:
            failures.append({'code': code, 'error': str(e)})
        if len(buf) >= chunk_size or idx == len(selected):
            if buf:
                x = pd.concat(buf, ignore_index=True)
                errs = audit_chunk(x)
                if errs:
                    raise RuntimeError(f'chunk audit failed: {errs}')
                p = out_dir / f'market_{year}_shard{shard}_part{len(chunks):03d}.parquet'
                x.to_parquet(p, index=False)
                chunks.append({'file': p.name, 'rows': len(x), 'sha256': sha256_file(p)})
                buf = []
        if idx % 100 == 0:
            print(json.dumps({'year': year, 'shard': shard, 'progress': idx, 'selected': len(selected), 'rows': totals, 'failures': len(failures)}))
        time.sleep(0.02)
    try:
        bs.logout()
    except Exception:
        pass
    result = {
        'year': year,
        'shard': shard,
        'shards': shards,
        'codes_total_union': len(codes),
        'codes_selected': len(selected),
        'security_success': security_success,
        'rows': totals,
        'failures': failures,
        'failure_count': len(failures),
        'chunks': chunks,
        'calendar_trading_days': int(cal['is_trading_day'].sum()) if len(cal) else 0,
        'status': 'PASS' if totals > 0 and security_success > 0 else 'FAIL',
        'semantic_status': 'REAL_BAOSTOCK_UNADJUSTED_MARKET_BOOTSTRAP_ONLY__OFFICIAL_OVERRIDE_PENDING'
    }
    (out_dir / f'summary_{year}_shard{shard}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result

def aggregate(root, out):
    root = Path(root)
    sums = []
    for p in root.rglob('summary_*_shard*.json'):
        try:
            sums.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception:
            pass
    years = range(2018, 2027)
    year_rows = {y: sum(int(x.get('rows', 0)) for x in sums if int(x.get('year', 0)) == y) for y in years}
    year_shards = {y: sum(1 for x in sums if int(x.get('year', 0)) == y and x.get('status') == 'PASS') for y in years}
    all_pass = all(year_shards[y] == 4 and year_rows[y] > 0 for y in years)
    result = {
        'years': list(years),
        'year_rows': year_rows,
        'year_pass_shards': year_shards,
        'expected_shards_per_year': 4,
        'total_rows': sum(year_rows.values()),
        'all_market_shards_pass': all_pass,
        'bf1_real_acceptance': False,
        'status': 'MARKET_BOOTSTRAP_2018_2026_PASS__OFFICIAL_OVERRIDE_PENDING' if all_pass else 'MARKET_BOOTSTRAP_INCOMPLETE_OR_FAILED',
        'blocking_next': [
            'official listing/delisting/ST/suspension override',
            'official corporate actions',
            'real survivorship audit',
            'delisting terminal values',
            'dataset-wide immutable manifest'
        ],
        'shard_summaries': sums
    }
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    x = sp.add_parser('run-shard')
    x.add_argument('--year', type=int, required=True)
    x.add_argument('--shard', type=int, required=True)
    x.add_argument('--shards', type=int, default=4)
    x.add_argument('--out-dir', required=True)
    x = sp.add_parser('aggregate')
    x.add_argument('--root', required=True)
    x.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'run-shard':
        print(json.dumps(run_shard(a.year, a.shard, a.shards, a.out_dir), ensure_ascii=False, indent=2))
    else:
        aggregate(a.root, a.out)

if __name__ == '__main__':
    main()
