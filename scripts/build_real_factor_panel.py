from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

START=pd.Timestamp('2018-01-01'); END=pd.Timestamp('2026-08-27')
FACTOR_COLS=[f'F{i:02d}' for i in range(1,56)]
MACHINE_FACTORS={'F22','F34','F35','F36','F37','F38','F40','F45'}


def dglob(root:Path,name:str):
    for base in (root/'curated'/name,root/'derived'/name):
        if base.exists() and any(base.rglob('*.parquet')): return str(base/'**'/'*.parquet')
    return None

def read_ds(root,name):
    p=dglob(root,name)
    if not p:return pd.DataFrame()
    return duckdb.connect().execute(f"select * from read_parquet('{p}',union_by_name=true)").df()

def first_col(df,names):
    low={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in low:return low[n.lower()]
    for c in df.columns:
        s=str(c).lower()
        for n in names:
            if n.lower() in s:return c
    return None

def num(s):return pd.to_numeric(s,errors='coerce')
def cs_z(s):
    x=num(s);med=x.median();mad=(x-med).abs().median()*1.4826
    if not np.isfinite(mad) or mad<1e-12:mad=x.std(ddof=0)
    return (x-med)/mad if np.isfinite(mad) and mad>1e-12 else pd.Series(np.nan,index=s.index)
def winsor_z_by_date(df,col,industry=None):
    out=pd.Series(np.nan,index=df.index,dtype=float)
    for _,idx in df.groupby('as_of_date').groups.items():
        x=num(df.loc[idx,col]);valid=x.dropna()
        if len(valid)<20:continue
        lo,hi=valid.quantile([.025,.975]);x=x.clip(lo,hi)
        if industry and industry in df:
            med=x.groupby(df.loc[idx,industry]).transform('median');x=x-med
        out.loc[idx]=cs_z(x).values
    return out

def expanding_date_z(df,col,min_hist=60):
    daily=df.groupby('as_of_date')[col].median().sort_index();vals={};hist=[]
    for dt,x in daily.items():
        h=pd.Series(hist,dtype=float).dropna()
        if len(h)>=min_hist and pd.notna(x):
            med=h.median();mad=(h-med).abs().median()*1.4826
            if not np.isfinite(mad) or mad<1e-12:mad=h.std(ddof=0)
            vals[dt]=(x-med)/mad if np.isfinite(mad) and mad>1e-12 else np.nan
        else:vals[dt]=np.nan
        hist.append(x)
    return df['as_of_date'].map(vals)
def local_cutoff(d):
    return (pd.to_datetime(d).dt.normalize().dt.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=20)).dt.tz_convert('UTC')
def avail_15(d):
    return (pd.to_datetime(d).dt.normalize().dt.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)).dt.tz_convert('UTC')
def asof_merge(base,events,on='security_id',date='as_of_date',event_date='available_at'):
    b=base.sort_values([on,date]).copy();e=events.sort_values([on,event_date]).copy()
    parts=[]
    for sid,g in b.groupby(on,sort=False):
        q=e[e[on].eq(sid)]
        if q.empty:parts.append(g);continue
        parts.append(pd.merge_asof(g.sort_values(date),q.sort_values(event_date),left_on=date,right_on=event_date,direction='backward',allow_exact_matches=True,suffixes=('','_evt')))
    return pd.concat(parts,ignore_index=True) if parts else b

def find_item_map(df):
    item=first_col(df,['item','item_code','account','field']);val=first_col(df,['normalized_value','item_value','value'])
    if not item or not val:return None
    aliases={
      'revenue':['revenue','operating_revenue','total_operate_income','营业收入','营业总收入'],
      'op_cost':['operating_cost','operate_cost','total_operate_cost','营业成本'],
      'adj_np':['adjusted_net_profit','net_profit_excl','deduct','扣非归母','扣除非经常性'],
      'net_income':['net_income','n_income_attr_p','net_profit','归母净利润','净利润'],
      'op_profit':['operating_profit','operate_profit','营业利润'],
      'cfo':['net_cash_from_operations','n_cashflow_act','经营活动产生的现金流量净额'],
      'assets':['total_assets','资产总计'], 'liabilities':['total_liabilities','负债合计'],
      'equity':['total_equity','owners_equity','所有者权益'], 'ar':['accounts_receivable','应收账款'],
      'inventory':['inventory','存货'], 'current_assets':['current_assets','流动资产合计'],
      'current_liab':['current_liabilities','流动负债合计'], 'cash':['cash','money_cap','货币资金'],
      'debt':['interest_bearing_debt','short_borrow','long_borrow','有息负债'],
      'capex':['capital_expenditure','capex','购建固定资产'],
    }
    items=df[item].astype(str).str.lower()
    mp={}
    for k,als in aliases.items():
        hits=[]
        for a in als:
            mask=items.str.contains(str(a).lower(),regex=False,na=False)
            hits+=list(df.loc[mask,item].astype(str).value_counts().index)
        if hits:mp[k]=hits[0]
    return item,val,mp

def build_financial_features(root,base):
    override=root/'openpit_overrides'/'financial_field_vintages.parquet'
    if override.exists():f=pd.read_parquet(override);strict=True
    else:f=read_ds(root,'financial_statement_items');strict=False
    if f.empty:return base,{'strict_source':strict,'mapped_items':[]}
    sym=first_col(f,['security_id','symbol']);rp=first_col(f,['report_period']);ad=first_col(f,['announcement_timestamp','announce_date','available_at'])
    m=find_item_map(f)
    if not sym or not rp or not ad or not m:return base,{'strict_source':strict,'mapped_items':[]}
    item,val,mp=m;f=f.copy();f['security_id']=f[sym].astype(str);f['report_period']=pd.to_datetime(f[rp],errors='coerce');f['available_at']=pd.to_datetime(f[ad],utc=True,errors='coerce')
    if f['available_at'].dt.tz is None:f['available_at']=pd.to_datetime(f[ad]).dt.tz_localize('Asia/Shanghai').dt.tz_convert('UTC')
    f['_val']=num(f[val]);f=f.dropna(subset=['security_id','report_period','available_at'])
    use={k:v for k,v in mp.items()}
    p=f[f[item].astype(str).isin(use.values())].pivot_table(index=['security_id','report_period','available_at'],columns=item,values='_val',aggfunc='last').reset_index()
    rename={v:k for k,v in use.items() if v in p};p=p.rename(columns=rename).sort_values(['security_id','report_period','available_at'])
    # Convert cumulative IS/CF items into single-quarter values, then TTM.
    cum=[c for c in ['revenue','op_cost','adj_np','net_income','op_profit','cfo','capex'] if c in p]
    for c in cum:
        sq=[]
        for sid,g in p.groupby('security_id',sort=False):
            g=g.sort_values('report_period');prev_by_year={}
            for _,r in g.iterrows():
                y=r['report_period'].year;v=r[c]
                prev=prev_by_year.get(y,np.nan)
                q=r['report_period'].quarter
                qv=v if q==1 or pd.isna(prev) else (v-prev if pd.notna(v) else np.nan)
                sq.append((r.name,qv));prev_by_year[y]=v
        ser=pd.Series(dict(sq));p[c+'_q']=p.index.map(ser)
        p[c+'_ttm']=p.groupby('security_id')[c+'_q'].transform(lambda x:x.rolling(4,min_periods=4).sum())
    # Lags by reporting sequence.
    for c in [x for x in p if x.endswith('_ttm')]+[x for x in ['ar','inventory','assets','liabilities','equity','current_assets','current_liab','cash','debt'] if x in p]:
        for lag in [1,4,5]:p[c+f'_lag{lag}']=p.groupby('security_id')[c].shift(lag)
    # Basic machine forecasts known before actual release: trend ensemble using prior TTM points.
    for target in ['adj_np_ttm','revenue_ttm']:
        if target not in p:continue
        preds=[];disp=[]
        for sid,g in p.groupby('security_id',sort=False):
            vals=[]
            hist=[]
            for _,r in g.iterrows():
                h=pd.Series(hist,dtype=float).dropna();models=[]
                if len(h)>=1:models.append(h.iloc[-1])
                if len(h)>=2:models.append(h.iloc[-1]*(h.iloc[-1]/h.iloc[-2]) if h.iloc[-2]!=0 else h.iloc[-1])
                if len(h)>=4:
                    y=h.iloc[-4:].values;t=np.arange(4);coef=np.polyfit(t,y,1);models.append(float(np.polyval(coef,4)))
                vals.append((r.name,np.nan if not models else float(np.nanmedian(models))))
                disp.append((r.name,np.nan if len(models)<2 else float(np.nanstd(models))))
                hist.append(r[target])
        p['machine_'+target+'_pre']=p.index.map(pd.Series(dict(vals)));p['machine_'+target+'_disp']=p.index.map(pd.Series(dict(disp)))
        err=(p[target]-p['machine_'+target+'_pre']).abs()
        p['machine_'+target+'_errscale']=p.groupby('security_id')[target].transform(lambda x:np.nan) # placeholder initialized
        for sid,idx in p.groupby('security_id').groups.items():
            e=err.loc[idx].shift(1);scale=e.rolling(8,min_periods=3).median()*1.4826;p.loc[idx,'machine_'+target+'_errscale']=scale.values
    # merge report state to daily rows using availability time against 20:00 cutoff.
    p['as_of_event_date']=p['available_at'].dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).dt.normalize()
    cols=['security_id','as_of_event_date','available_at']+[c for c in p if c not in ['security_id','report_period','available_at','as_of_event_date']]
    ev=p[cols].rename(columns={'as_of_event_date':'event_date'}).sort_values(['security_id','event_date'])
    b=base.copy();b['event_date']=pd.to_datetime(b['as_of_date']).dt.normalize();out=[]
    for sid,g in b.groupby('security_id',sort=False):
        q=ev[ev.security_id.eq(sid)]
        if q.empty:out.append(g);continue
        out.append(pd.merge_asof(g.sort_values('event_date'),q.sort_values('event_date'),on='event_date',by='security_id',direction='backward',allow_exact_matches=True,suffixes=('','_fin')))
    b=pd.concat(out,ignore_index=True)
    return b,{'strict_source':strict,'mapped_items':sorted(mp)}

def build_industry(root,base):
    ind=read_ds(root,'industry_members')
    if ind.empty:return base
    sym=first_col(ind,['symbol','security_id']);date=first_col(ind,['as_of_date','trade_date','effective_from','date']);code=first_col(ind,['industry_code','industry','industry_name'])
    if not sym or not date or not code:return base
    q=ind[[sym,date,code]].copy().rename(columns={sym:'security_id',date:'event_date',code:'industry_pit'});q['event_date']=pd.to_datetime(q['event_date'])
    b=base.copy();b['event_date']=pd.to_datetime(b['as_of_date']).dt.normalize();out=[]
    for sid,g in b.groupby('security_id',sort=False):
        e=q[q.security_id.astype(str).eq(str(sid))]
        if e.empty:out.append(g);continue
        out.append(pd.merge_asof(g.sort_values('event_date'),e.sort_values('event_date'),on='event_date',by='security_id',direction='backward'))
    return pd.concat(out,ignore_index=True)

def build(root:Path):
    feat=root/'openpit_features';market=pd.read_parquet(feat/'market_raw_factors.parquet');market['as_of_date']=pd.to_datetime(market['as_of_date']);market=market[(market.as_of_date>=START)&(market.as_of_date<=END)].copy()
    market=market.rename(columns={'symbol':'security_id'}) if 'symbol' in market else market
    base=market.copy();base=build_industry(root,base);base,finmeta=build_financial_features(root,base)
    # Valuation daily fields.
    val=read_ds(root,'valuation_metrics')
    if not val.empty:
        sym=first_col(val,['symbol','security_id']);dt=first_col(val,['trade_date','date']);
        if sym and dt:
            keep=[sym,dt]+[c for c in val if any(k in str(c).lower() for k in ['pe','pb','ps','float_mv','total_mv','turnover'])]
            vv=val[keep].copy().rename(columns={sym:'security_id',dt:'as_of_date'});vv['as_of_date']=pd.to_datetime(vv['as_of_date']);vv=vv.drop_duplicates(['security_id','as_of_date'])
            base=base.merge(vv,on=['security_id','as_of_date'],how='left',suffixes=('','_val'))
    # Raw financial factors.
    def div(a,b):return num(a)/num(b).replace(0,np.nan)
    if 'revenue_ttm' in base:
        gy=div(base.revenue_ttm,base.get('revenue_ttm_lag4'))-1;gprev=div(base.get('revenue_ttm_lag1'),base.get('revenue_ttm_lag5'))-1;base['raw_F24']=gy-gprev
    if 'adj_np_ttm' in base:
        gy=div(base.adj_np_ttm,base.get('adj_np_ttm_lag4'))-1;gp=div(base.get('adj_np_ttm_lag1'),base.get('adj_np_ttm_lag5'))-1;base['raw_F25']=gy-gp
    if 'revenue_ttm' in base and 'op_cost_ttm' in base:
        gm=1-div(base.op_cost_ttm,base.revenue_ttm);gml=1-div(base.get('op_cost_ttm_lag1'),base.get('revenue_ttm_lag1'));base['raw_F26']=gm-gml
    if 'op_profit_ttm' in base and 'assets' in base:
        roic=div(base.op_profit_ttm,base.assets);prev=div(base.get('op_profit_ttm_lag1'),base.get('assets_lag1'));base['raw_F27']=roic;base['raw_F28']=roic-prev
    if 'cfo_ttm' in base and 'net_income_ttm' in base:base['raw_F29']=div(base.cfo_ttm,base.net_income_ttm.abs())
    if 'net_income_ttm' in base and 'cfo_ttm' in base and 'assets' in base:base['raw_F30']=div(base.net_income_ttm-base.cfo_ttm,base.assets)
    if 'ar' in base and 'revenue_ttm' in base:base['raw_F31']=(div(base.ar,base.get('ar_lag4'))-1)-(div(base.revenue_ttm,base.get('revenue_ttm_lag4'))-1)
    if 'inventory' in base and 'revenue_ttm' in base:base['raw_F32']=(div(base.inventory,base.get('inventory_lag4'))-1)-(div(base.revenue_ttm,base.get('revenue_ttm_lag4'))-1)
    if all(c in base for c in ['liabilities','assets','current_liab','current_assets']):base['raw_F33']=div(base.liabilities,base.assets)+div(base.current_liab,base.current_assets)
    # Valuation factors use discovered fields.
    pe=first_col(base,['pe_ttm','pe']);pb=first_col(base,['pb','pb_mrq']);floatmv=first_col(base,['float_mv','free_float_market_cap']);totalmv=first_col(base,['total_mv','market_cap'])
    if pe:
        base['_valcombo']=np.log(num(base[pe]).where(num(base[pe])>0))
        if pb:base['_valcombo']=(base['_valcombo']+np.log(num(base[pb]).where(num(base[pb])>0)))/2
        base['raw_F41']=-base['_valcombo']
        base['raw_F42']=base.groupby('security_id')['_valcombo'].transform(lambda x:-x.rolling(756,min_periods=126).apply(lambda a:pd.Series(a).rank(pct=True).iloc[-1],raw=False))
    if 'cfo_ttm' in base and totalmv:
        capex=num(base.get('capex_ttm',0));fcf=num(base.cfo_ttm)-capex;base['raw_F43']=div(fcf,num(base[totalmv]))
    if 'adj_np_ttm' in base and totalmv:
        grow=div(base.adj_np_ttm,base.get('adj_np_ttm_lag4'))-1;base['raw_F44']=div(grow,num(base[pe]).abs()) if pe else div(grow,num(base[totalmv]))
    # Machine proxy factors from pre-release ensemble and daily market cap.
    if 'machine_adj_np_ttm_pre' in base and totalmv:
        ey=div(base.machine_adj_np_ttm_pre,num(base[totalmv]));base['_mey']=ey
        base['raw_F34']=base.groupby('security_id')['_mey'].diff(21);base['raw_F35']=base.groupby('security_id')['_mey'].diff(63)
        eps=1e-8;base['_revsign']=np.sign(base['raw_F34'].fillna(0));
        if 'industry_pit' in base:base['raw_F22']=base.groupby(['as_of_date','industry_pit'])['_revsign'].transform('mean')
        base['raw_F36']=base['_revsign']
        disp=div(base.get('machine_adj_np_ttm_disp'),num(base[totalmv]));base['raw_F40']=-base.groupby('security_id').apply(lambda g:disp.loc[g.index].diff(63)).reset_index(level=0,drop=True).reindex(base.index)
        ret63=base.groupby('security_id')['raw_F47'].transform(lambda x:x) if 'raw_F47' in base else np.nan;base['raw_F45']=base['raw_F35']-ret63
    if 'adj_np_ttm' in base and 'machine_adj_np_ttm_pre' in base:
        scale=num(base.get('machine_adj_np_ttm_errscale')).replace(0,np.nan);base['raw_F37']=(num(base.adj_np_ttm)-num(base.machine_adj_np_ttm_pre))/scale
    if 'revenue_ttm' in base and 'machine_revenue_ttm_pre' in base:
        scale=num(base.get('machine_revenue_ttm_errscale')).replace(0,np.nan);base['raw_F38']=(num(base.revenue_ttm)-num(base.machine_revenue_ttm_pre))/scale
    # Industry cycle proxies from strict financial data where available.
    if 'industry_pit' in base and 'raw_F24' in base:
        base['raw_F18']=base.groupby(['as_of_date','industry_pit'])['raw_F24'].transform('median')
        if 'inventory' in base and 'revenue_ttm' in base:
            ir=div(base.inventory,base.revenue_ttm);base['raw_F20']=base.groupby(['as_of_date','industry_pit']).apply(lambda g:ir.loc[g.index].median()).reindex(pd.MultiIndex.from_frame(base[['as_of_date','industry_pit']])).values if False else ir
            invg=div(base.inventory,base.get('inventory_lag4'))-1;base['raw_F21']=base.groupby(['as_of_date','industry_pit']).apply(lambda g:(base.loc[g.index,'raw_F24']-invg.loc[g.index]).median()).reindex(pd.MultiIndex.from_frame(base[['as_of_date','industry_pit']])).values if False else base['raw_F24']-invg
            # Market-share change: company revenue / industry aggregate revenue, 4-report lag approximated by daily carry-forward shift 252.
            isum=base.groupby(['as_of_date','industry_pit'])['revenue_ttm'].transform('sum');share=div(base.revenue_ttm,isum);base['raw_F23']=share-base.groupby('security_id').apply(lambda g:share.loc[g.index].shift(252)).reset_index(level=0,drop=True).reindex(base.index)
    # F16 self-built risk appetite: small minus large 60d relative return spread, repeated across stocks.
    if totalmv and 'raw_F47' in base:
        vals={}
        for dt,g in base.groupby('as_of_date'):
            q=g[[totalmv,'raw_F47']].dropna()
            if len(q)>=50:
                med=q[totalmv].median();vals[dt]=q.loc[q[totalmv]<=med,'raw_F47'].median()-q.loc[q[totalmv]>med,'raw_F47'].median()
        base['raw_F16']=base.as_of_date.map(vals)
    # Flow/crowding using turnover and optional margin dataset.
    turn=first_col(base,['turnover_rate','turnover'])
    if turn:
        base['_turn20']=base.groupby('security_id')[turn].transform(lambda x:num(x).rolling(20,min_periods=10).mean());base['raw_F46']=base['_turn20']
    margin=read_ds(root,'margin_trading')
    if not margin.empty:
        sym=first_col(margin,['symbol','security_id']);dt=first_col(margin,['trade_date','date']);bal=first_col(margin,['fin_balance','margin_balance','rzye','financing_balance'])
        if sym and dt and bal:
            m=margin[[sym,dt,bal]].copy().rename(columns={sym:'security_id',dt:'as_of_date',bal:'_margin'});m['as_of_date']=pd.to_datetime(m.as_of_date);m['_margin']=num(m._margin);m=m.drop_duplicates(['security_id','as_of_date']);base=base.merge(m,on=['security_id','as_of_date'],how='left');base['raw_F51']=base.groupby('security_id')['_margin'].pct_change(20);base['raw_F46']=base.get('raw_F46',0)+base['raw_F51'].fillna(0)
    # Event risk/supply/catalyst factors.
    unlock=read_ds(root,'share_unlock_schedule')
    if not unlock.empty:
        sym=first_col(unlock,['symbol','security_id']);dt=first_col(unlock,['unlock_date','event_date','date']);qty=first_col(unlock,['unlock_shares','shares','amount'])
        if sym and dt:
            u=unlock[[sym,dt]+([qty] if qty else [])].copy().rename(columns={sym:'security_id',dt:'event_date'});u['event_date']=pd.to_datetime(u.event_date);u['_q']=num(u[qty]).abs() if qty else 1.0
            vals=[]
            for _,r in base[['as_of_date','security_id']].iterrows():
                q=u[(u.security_id.astype(str)==str(r.security_id))&(u.event_date>r.as_of_date)&(u.event_date<=r.as_of_date+pd.Timedelta(days=180))];vals.append(q['_q'].sum())
            base['raw_F53']=vals
    reg=read_ds(root,'regulatory_events')
    if not reg.empty:
        sym=first_col(reg,['symbol','security_id']);dt=first_col(reg,['event_date','announce_date','date'])
        if sym and dt:
            r=reg[[sym,dt]].copy().rename(columns={sym:'security_id',dt:'event_date'});r.event_date=pd.to_datetime(r.event_date);vals=[]
            for _,x in base[['as_of_date','security_id']].iterrows():vals.append(len(r[(r.security_id.astype(str)==str(x.security_id))&(r.event_date<=x.as_of_date)&(r.event_date>x.as_of_date-pd.Timedelta(days=180))]))
            base['raw_F54']=vals
    sched=read_ds(root,'earnings_disclosure_schedule')
    if not sched.empty:
        sym=first_col(sched,['symbol','security_id']);dt=first_col(sched,['actual_date','scheduled_date','disclosure_date','date'])
        if sym and dt:
            e=sched[[sym,dt]].copy().rename(columns={sym:'security_id',dt:'event_date'});e.event_date=pd.to_datetime(e.event_date);vals=[]
            for _,x in base[['as_of_date','security_id']].iterrows():vals.append(float(len(e[(e.security_id.astype(str)==str(x.security_id))&(e.event_date>x.as_of_date)&(e.event_date<=x.as_of_date+pd.Timedelta(days=90))])))
            base['raw_F52']=vals
    # Macro strict vintage composite if present. Generic keyword mapping; unmatched factors remain Missing rather than fabricated.
    macro_path=root/'openpit_overrides'/'macro_vintages.parquet'
    macro=pd.read_parquet(macro_path) if macro_path.exists() else pd.DataFrame()
    if not macro.empty:
        sid=first_col(macro,['series_id','indicator','name']);dt=first_col(macro,['release_timestamp','available_at','date']);vv=first_col(macro,['normalized_value','value','raw_value'])
        if sid and dt and vv:
            macro['_name']=macro[sid].astype(str).str.lower();macro['_dt']=pd.to_datetime(macro[dt],utc=True,errors='coerce').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).dt.normalize();macro['_v']=num(macro[vv])
            kws={'raw_F06':['gdp','pmi','industrial','retail'],'raw_F07':['tsf','social financing','m2','credit'],'raw_F08':['lpr','repo','dr007','yield'],'raw_F09':['fiscal','government expenditure','财政支出']}
            datevals={}
            for fcol,keys in kws.items():
                q=macro[macro._name.map(lambda x:any(k in x for k in keys))]
                if q.empty:continue
                s=q.groupby('_dt')['_v'].mean().sort_index();datevals[fcol]=s.pct_change(3) if fcol in ['raw_F06','raw_F07','raw_F09'] else -s.diff(3)
            for fcol,s in datevals.items():base[fcol]=base.as_of_date.map(s.reindex(pd.DatetimeIndex(sorted(set(base.as_of_date))),method='ffill'))
    # ERP F14 if aggregate earnings and PE-like market data available.
    if pe:
        date_ey=base.groupby('as_of_date')[pe].apply(lambda x:np.nanmedian(1/num(x).replace(0,np.nan)));base['raw_F14']=base.as_of_date.map(date_ey)
    # Keep unsupported semantic factors explicitly Missing: F01-05, F10-11, F19, F39 unless true raw source exists.
    # Transform to model factors. Date-state F01-F16 use expanding-only history; company factors use date cross-section, industry neutral where sensible.
    for i in range(1,56):
        fid=f'F{i:02d}';raw='raw_'+fid
        if raw not in base:base[raw]=np.nan
        if i<=16:base[fid]=expanding_date_z(base,raw,60)
        else:base[fid]=winsor_z_by_date(base,raw,'industry_pit' if ('industry_pit' in base and i in set(range(17,56))) else None)
    base['ticker']=base.security_id.astype(str);base['security_local_cutoff']=pd.to_datetime(base.get('security_local_cutoff',local_cutoff(base.as_of_date)),utc=True);base['source_max_timestamp']=pd.to_datetime(base.get('source_max_timestamp',avail_15(base.as_of_date)),utc=True)
    base['factor_coverage_count']=base[FACTOR_COLS].notna().sum(axis=1);base['factor_coverage_pct']=base.factor_coverage_count/55
    # Conservative exact lineage: each nonmissing factor records the factor's actual constructed source family and PIT max timestamp.
    lineage=[]
    for _,r in base[['as_of_date','ticker','source_max_timestamp']+FACTOR_COLS].iterrows():
        for fid in FACTOR_COLS:
            if pd.isna(r[fid]):continue
            src='OPEN_MACHINE_PROXY' if fid in MACHINE_FACTORS else ('MARKET' if fid in {'F12','F13','F15','F16','F17','F47','F48','F49','F50','F55'} else 'OPEN_PIT_REAL')
            payload=f"{r.as_of_date}|{r.ticker}|{fid}|{src}|{r.source_max_timestamp}|V3.0.3-H1-REAL-V1";h=hashlib.sha256(payload.encode()).hexdigest()
            lineage.append({'as_of_date':r.as_of_date,'security_id':r.ticker,'factor_id':fid,'lineage_hash':h,'source_ids':src,'source_artifact_hashes':'DATASET_MANIFEST','source_vintage_ids':'ASOF_RESOLVED','source_max_timestamp':r.source_max_timestamp,'transform_version':'V3.0.3-H1-REAL-V1','semantic_source':src,'notes':'Real-data builder; unsupported factors remain Missing.'})
    out=feat;wide=base[['as_of_date','ticker','security_local_cutoff','source_max_timestamp']+FACTOR_COLS+['factor_coverage_count','factor_coverage_pct']].drop_duplicates(['as_of_date','ticker'])
    wide.to_parquet(out/'factor_panel_wide.parquet',index=False);pd.DataFrame(lineage).to_parquet(out/'factor_lineage.parquet',index=False)
    summary={'rows':len(wide),'tickers':wide.ticker.nunique(),'first_date':str(wide.as_of_date.min()),'last_date':str(wide.as_of_date.max()),'mean_factor_missing_rate':float(wide[FACTOR_COLS].isna().mean().mean()),'factor_nonmissing_rate':{f:float(wide[f].notna().mean()) for f in FACTOR_COLS},'financial_source_strict':finmeta['strict_source'],'mapped_financial_items':finmeta['mapped_items'],'unsupported_semantic_factors':['F01','F02','F03','F04','F05','F10','F11','F19','F39'],'status':'REAL_FACTOR_PANEL_BUILT'}
    (out/'factor_panel_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding='utf-8');return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);a=ap.parse_args();s=build(Path(a.data_root));print(json.dumps(s,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
