from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

START = pd.Timestamp('2018-01-01')
END = pd.Timestamp('2026-08-27')
HORIZONS = (20, 60, 120)


def dataset_glob(root: Path, dataset: str) -> str:
    for base in (root / 'curated' / dataset, root / 'derived' / dataset):
        if base.exists() and any(base.rglob('*.parquet')):
            return str(base / '**' / '*.parquet')
    raise FileNotFoundError(dataset)


def file_set_hash(root: Path, dataset: str) -> str:
    files=[]
    for base in (root / 'curated' / dataset, root / 'derived' / dataset):
        if base.exists(): files += sorted(base.rglob('*.parquet'))
    h=hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(str(p.stat().st_size).encode())
        with p.open('rb') as f:
            for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()


def local_cutoff(dates: pd.Series, hour=20):
    x=pd.to_datetime(dates).dt.normalize().dt.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=hour)
    return x.dt.tz_convert('UTC')


def market_available_at(dates: pd.Series):
    x=pd.to_datetime(dates).dt.normalize().dt.tz_localize('Asia/Shanghai') + pd.Timedelta(hours=15)
    return x.dt.tz_convert('UTC')


def load_inputs(root: Path):
    con=duckdb.connect()
    b=dataset_glob(root,'daily_bars')
    i=dataset_glob(root,'instruments')
    c=dataset_glob(root,'corporate_actions')
    v=dataset_glob(root,'valuation_metrics')
    cal=dataset_glob(root,'trading_calendar')
    bars=con.execute(f"""
      select b.symbol,b.trade_date,b.open,b.high,b.low,b.close,b.volume,b.amount,b.source,b.data_version,b.fetched_at
      from read_parquet('{b}',union_by_name=true) b
      join read_parquet('{i}',union_by_name=true) i using(symbol)
      where i.asset_type='stock' and b.trade_date between date '2017-01-01' and date '2026-08-27'
    """).df()
    ca=con.execute(f"""
      select symbol,ex_date,action_type,
             coalesce(cash_dividend,0.0) cash_dividend,
             coalesce(bonus_ratio,0.0) bonus_ratio,
             coalesce(transfer_ratio,0.0) transfer_ratio,
             coalesce(allotment_ratio,0.0) allotment_ratio,
             allotment_price,source,data_version,fetched_at
      from read_parquet('{c}',union_by_name=true)
      where ex_date between date '2017-01-01' and date '2026-08-27'
    """).df()
    val=con.execute(f"""
      select symbol,trade_date,float_mv,total_mv,source,data_version,fetched_at
      from read_parquet('{v}',union_by_name=true)
      where trade_date between date '2017-01-01' and date '2026-08-27'
    """).df()
    calendar=con.execute(f"select trade_date from read_parquet('{cal}',union_by_name=true) where is_trading=true and trade_date between date '2017-01-01' and date '2027-06-30' order by trade_date").df()
    return bars,ca,val,calendar


def aggregate_actions(ca: pd.DataFrame):
    if ca.empty:
        return pd.DataFrame(columns=['symbol','trade_date','cash_dividend','bonus_ratio','transfer_ratio','allotment_ratio','allotment_price'])
    d=ca.copy();d['trade_date']=pd.to_datetime(d['ex_date'])
    for x in ['cash_dividend','bonus_ratio','transfer_ratio','allotment_ratio','allotment_price']:
        d[x]=pd.to_numeric(d[x],errors='coerce')
    # multiple action rows on same ex-date: additive per-share ratios/cash; allotment price is weighted by allotment ratio
    d['_allotment_cash']=d['allotment_ratio'].fillna(0)*d['allotment_price'].fillna(0)
    a=d.groupby(['symbol','trade_date'],as_index=False).agg(
        cash_dividend=('cash_dividend','sum'), bonus_ratio=('bonus_ratio','sum'), transfer_ratio=('transfer_ratio','sum'),
        allotment_ratio=('allotment_ratio','sum'), allotment_cash=('_allotment_cash','sum'))
    a['allotment_price']=np.where(a['allotment_ratio']>0,a['allotment_cash']/a['allotment_ratio'],np.nan)
    return a.drop(columns='allotment_cash')


def build_total_return(bars: pd.DataFrame, ca: pd.DataFrame):
    d=bars.copy();d['trade_date']=pd.to_datetime(d['trade_date'])
    for x in ['open','high','low','close','volume','amount']: d[x]=pd.to_numeric(d[x],errors='coerce')
    d=d.sort_values(['symbol','trade_date']).drop_duplicates(['symbol','trade_date'],keep='last')
    a=aggregate_actions(ca)
    d=d.merge(a,on=['symbol','trade_date'],how='left')
    for x in ['cash_dividend','bonus_ratio','transfer_ratio','allotment_ratio']: d[x]=d[x].fillna(0.0)
    d['prev_close']=d.groupby('symbol')['close'].shift(1)
    shares_multiplier=1+d['bonus_ratio']+d['transfer_ratio']+d['allotment_ratio']
    subscription_cash=d['allotment_ratio']*d['allotment_price'].fillna(0.0)
    wealth=d['close']*shares_multiplier+d['cash_dividend']-subscription_cash
    d['total_return_1d']=wealth/d['prev_close']-1
    # no prior observation => no return. Suspended placeholder bars naturally yield ~0 if close unchanged.
    d.loc[d['prev_close'].isna() | (d['prev_close']<=0),'total_return_1d']=np.nan
    d['total_return_index']=np.nan
    for sid,idx in d.groupby('symbol').groups.items():
        r=d.loc[idx,'total_return_1d'].fillna(0).clip(lower=-0.999999)
        d.loc[idx,'total_return_index']=(1+r).cumprod().values
    d['source_max_timestamp']=market_available_at(d['trade_date'])
    d['security_local_cutoff']=local_cutoff(d['trade_date'])
    return d


def build_benchmarks(tr: pd.DataFrame, val: pd.DataFrame):
    d=tr[['symbol','trade_date','total_return_1d']].copy()
    v=val.copy();v['trade_date']=pd.to_datetime(v['trade_date'])
    for c in ['float_mv','total_mv']: v[c]=pd.to_numeric(v[c],errors='coerce')
    d=d.merge(v[['symbol','trade_date','float_mv','total_mv']],on=['symbol','trade_date'],how='left')
    d=d.sort_values(['symbol','trade_date'])
    d['ff_w_raw']=d.groupby('symbol')['float_mv'].shift(1)
    d['cap_w_raw']=d.groupby('symbol')['total_mv'].shift(1)
    d['eligible_lag']=d.groupby('symbol')['total_return_1d'].shift(1).notna()
    # At date t use only information/eligibility frozen at t-1. Current return itself is of course realized at t.
    rows=[]
    for dt,g in d.groupby('trade_date'):
        g=g[g['eligible_lag'] & g['total_return_1d'].notna()].copy()
        if g.empty: continue
        ew=float(g['total_return_1d'].mean())
        def wr(col):
            x=g[[col,'total_return_1d']].dropna();x=x[x[col]>0]
            return np.nan if x.empty else float(np.average(x['total_return_1d'],weights=x[col]))
        rows.append({'trade_date':dt,'OPEN_A_EW_ret':ew,'OPEN_A_FFMCAP_ret':wr('ff_w_raw'),'OPEN_A_CAP_ret':wr('cap_w_raw'),'constituent_count':len(g)})
    b=pd.DataFrame(rows).sort_values('trade_date')
    for name in ['OPEN_A_EW','OPEN_A_FFMCAP','OPEN_A_CAP']:
        b[f'{name}_tri']=(1+b[f'{name}_ret'].fillna(0).clip(lower=-.999999)).cumprod()
    b['source_max_timestamp']=market_available_at(b['trade_date'])
    b['security_local_cutoff']=local_cutoff(b['trade_date'])
    return b


def build_labels(tr: pd.DataFrame, bench: pd.DataFrame, calendar: pd.DataFrame):
    dates=pd.DatetimeIndex(pd.to_datetime(calendar['trade_date']).sort_values().unique())
    pos={pd.Timestamp(x):i for i,x in enumerate(dates)}
    bm=bench.set_index('trade_date')['OPEN_A_FFMCAP_tri']
    rows=[]
    for sid,g in tr.groupby('symbol'):
        s=g.set_index('trade_date')['total_return_index'].sort_index()
        for dt,x0 in s.items():
            if dt<START or dt>END or dt not in pos or dt not in bm.index or not np.isfinite(x0) or x0<=0: continue
            rec={'as_of_date':dt,'ticker':sid}
            any_valid=False
            for h in HORIZONS:
                j=pos[dt]+h
                if j>=len(dates): rec[f'ret_excess_fwd_{h}']=np.nan;rec[f'label_end_timestamp_{h}']=pd.NaT;continue
                end=pd.Timestamp(dates[j])
                if end not in s.index or end not in bm.index:
                    rec[f'ret_excess_fwd_{h}']=np.nan;rec[f'label_end_timestamp_{h}']=pd.NaT;continue
                rs=float(s.loc[end]/x0-1); rb=float(bm.loc[end]/bm.loc[dt]-1)
                rec[f'ret_excess_fwd_{h}']=rs-rb
                rec[f'label_end_timestamp_{h}']=(pd.Timestamp(end.date()).tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)).tz_convert('UTC')
                any_valid=True
            if any_valid: rows.append(rec)
    return pd.DataFrame(rows)


def build_market_raw_factors(tr: pd.DataFrame, bench: pd.DataFrame):
    d=tr.copy().sort_values(['symbol','trade_date'])
    g=d.groupby('symbol',sort=False)
    d['ret60']=g['total_return_index'].pct_change(60)
    d['mom6_1']=g['total_return_index'].shift(21)/g['total_return_index'].shift(126)-1
    d['high252']=g['close'].transform(lambda x:x.rolling(252,min_periods=120).max())
    d['ma20']=g['close'].transform(lambda x:x.rolling(20,min_periods=10).mean())
    d['ma60']=g['close'].transform(lambda x:x.rolling(60,min_periods=30).mean())
    d['ma120']=g['close'].transform(lambda x:x.rolling(120,min_periods=60).mean())
    d['vol20']=g['volume'].transform(lambda x:x.rolling(20,min_periods=10).mean())
    d['vol120']=g['volume'].transform(lambda x:x.rolling(120,min_periods=60).mean())
    r1=g['total_return_index'].pct_change()
    d['downside20']=r1.where(r1<0,0).groupby(d['symbol']).transform(lambda x:x.rolling(20,min_periods=10).std())
    d['zero_amount20']=(d['amount'].fillna(0)<=0).astype(float).groupby(d['symbol']).transform(lambda x:x.rolling(20,min_periods=10).mean())
    d['amihud20']=(r1.abs()/d['amount'].replace(0,np.nan)).groupby(d['symbol']).transform(lambda x:x.rolling(20,min_periods=10).mean())
    out=d[['trade_date','symbol','source_max_timestamp','security_local_cutoff']].copy().rename(columns={'trade_date':'as_of_date','symbol':'security_id'})
    out['raw_F47']=d['mom6_1'];out['raw_F48']=d['close']/d['high252']-1
    out['raw_F49']=(d['close']/d['ma20']-1)+(d['ma20']/d['ma60']-1)+(d['ma60']/d['ma120']-1)
    out['raw_F50']=d['ret60']*(d['vol20']/d['vol120'].replace(0,np.nan))
    out['raw_F55']=d['downside20']+d['zero_amount20']+d['amihud20'].rank(pct=True)
    # Date-level factors repeated across stocks downstream.
    b=bench.copy();b['raw_F12']=b['OPEN_A_FFMCAP_tri'].pct_change(120)
    breadth=(d['close']>d['ma120']).groupby(d['trade_date']).mean().rename('raw_F13').reset_index()
    liq=d.groupby('trade_date')['amount'].sum().sort_index(); b_liq=(liq.rolling(20,min_periods=10).mean()/liq.rolling(120,min_periods=60).median()-1).rename('raw_F15').reset_index()
    date=b[['trade_date','raw_F12']].merge(breadth,on='trade_date',how='left').merge(b_liq,on='trade_date',how='left')
    out=out.merge(date,left_on='as_of_date',right_on='trade_date',how='left').drop(columns='trade_date')
    return out[(out['as_of_date']>=START)&(out['as_of_date']<=END)].reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--out-root',default=None);a=ap.parse_args()
    root=Path(a.data_root);out=Path(a.out_root) if a.out_root else root/'openpit_features';out.mkdir(parents=True,exist_ok=True)
    bars,ca,val,calendar=load_inputs(root)
    tr=build_total_return(bars,ca);bench=build_benchmarks(tr,val);labels=build_labels(tr,bench,calendar);market=build_market_raw_factors(tr,bench)
    tr[(tr.trade_date>=START)&(tr.trade_date<=END)].to_parquet(out/'pit_total_return_panel.parquet',index=False)
    bench[(bench.trade_date>=START)&(bench.trade_date<=END)].to_parquet(out/'open_a_benchmarks.parquet',index=False)
    labels.to_parquet(out/'forward_labels.parquet',index=False)
    market.to_parquet(out/'market_raw_factors.parquet',index=False)
    provenance={x:file_set_hash(root,x) for x in ['daily_bars','instruments','corporate_actions','valuation_metrics','trading_calendar']}
    summary={'rows_total_return':len(tr),'rows_benchmark':len(bench),'rows_labels':len(labels),'rows_market_raw_factors':len(market),'source_dataset_hashes':provenance,
             'total_return_policy':'RAW_CLOSE_PLUS_EFFECTIVE_CORPORATE_ACTION_WEALTH; NO VENDOR QFQ AS RAW TRUTH',
             'benchmark_policy':'PRIMARY OPEN_A_FFMCAP; t return uses t-1 float_mv weights; EW/CAP robustness retained',
             'strict_note':'Labels crossing delisting are not strict until H1 delisting terminal-value override exists.'}
    (out/'market_layer_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
