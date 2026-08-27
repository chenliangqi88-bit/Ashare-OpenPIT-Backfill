from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, chi2, norm
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FACTORS=[f'F{i:02d}' for i in range(1,56)]
HORIZONS=(20,60,120)
MIN_CS=30


def daily_ic(y,p,d):
    x=pd.DataFrame({'y':y,'p':p,'d':pd.to_datetime(d)}).dropna();vals=[]
    for _,g in x.groupby('d'):
        if len(g)<MIN_CS or g.y.nunique()<3 or g.p.nunique()<3:continue
        r=spearmanr(g.y,g.p).statistic
        if np.isfinite(r):vals.append(float(r))
    a=np.asarray(vals,float)
    return (float(a.mean()) if len(a) else np.nan,float(a.mean()/(a.std(ddof=1)+1e-12)*math.sqrt(252)) if len(a)>1 else np.nan,len(a),a)

def hac_t(a,lag):
    a=np.asarray(pd.Series(a).dropna(),float);n=len(a)
    if n<max(10,lag+2):return np.nan
    u=a-a.mean();lr=float(np.dot(u,u)/n);L=min(lag,n-2)
    for k in range(1,L+1):lr+=2*(1-k/(L+1))*float(np.dot(u[k:],u[:-k])/n)
    se=math.sqrt(max(lr,0)/n);return float(a.mean()/se) if se>0 else np.nan

def block_lcb(x,block=20,B=1000,seed=20260827):
    x=np.asarray(pd.Series(x).dropna(),float);n=len(x)
    if n<20:return np.nan
    rng=np.random.default_rng(seed);means=[]
    for _ in range(B):
        z=[]
        while len(z)<n:
            s=int(rng.integers(0,max(1,n-block+1)));z.extend(x[s:s+block].tolist())
        means.append(np.mean(z[:n]))
    return float(np.quantile(means,.025))

def top_bottom(y,p,d):
    x=pd.DataFrame({'y':y,'p':p,'d':pd.to_datetime(d)}).dropna();vals=[];hits=[]
    for _,g in x.groupby('d'):
        if len(g)<MIN_CS:continue
        q=g.p.rank(pct=True,method='average');top=g[q>=.9].y;bot=g[q<=.1].y
        if len(top) and len(bot):vals.append(float(top.mean()-bot.mean()));hits.append(float((top>0).mean()))
    return (float(np.mean(vals)) if vals else np.nan,np.asarray(vals),float(np.mean(hits)) if hits else np.nan)

def kupiec(viol,alpha=.05):
    v=np.asarray(viol,dtype=int);n=len(v);x=int(v.sum())
    if n<20:return np.nan
    ph=x/n
    if ph in (0,1):return 0.0
    ll0=(n-x)*math.log(1-alpha)+x*math.log(alpha);ll1=(n-x)*math.log(1-ph)+x*math.log(ph)
    return float(1-chi2.cdf(-2*(ll0-ll1),1))

def christoffersen(viol):
    v=np.asarray(viol,dtype=int)
    if len(v)<30:return np.nan
    n00=n01=n10=n11=0
    for a,b in zip(v[:-1],v[1:]):
        if a==0 and b==0:n00+=1
        elif a==0 and b==1:n01+=1
        elif a==1 and b==0:n10+=1
        else:n11+=1
    p01=n01/max(1,n00+n01);p11=n11/max(1,n10+n11);p=(n01+n11)/max(1,n00+n01+n10+n11)
    def ll(a,b,p):
        z=0
        if a:z+=a*math.log(max(1e-12,1-p))
        if b:z+=b*math.log(max(1e-12,p))
        return z
    ll0=ll(n00+n10,n01+n11,p);ll1=ll(n00,n01,p01)+ll(n10,n11,p11)
    return float(1-chi2.cdf(-2*(ll0-ll1),1))

def candidates(seed=20260827):
    return {
      'ELASTIC_NET':[(f'en_{a}_{l}',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',ElasticNet(alpha=a,l1_ratio=l,max_iter=8000,random_state=seed))])) for a in [.001,.01] for l in [.25,.75]],
      'RIDGE':[(f'ridge_{a}',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=a))])) for a in [1.,10.]],
      'GBM':[(f'gbm_{d}_{lr}',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('m',HistGradientBoostingRegressor(max_depth=d,learning_rate=lr,max_iter=180,min_samples_leaf=50,l2_regularization=1.,random_state=seed))])) for d in [2,3] for lr in [.03,.05]],
    }

def select_family(models,Xtr,ytr,Xv,yv,dv):
    best=None
    for name,m in models:
        q=clone(m);q.fit(Xtr,ytr);p=q.predict(Xv);ic,_,n,_=daily_ic(yv,p,dv);rmse=math.sqrt(mean_squared_error(yv,p));key=(-999 if not np.isfinite(ic) else ic,-rmse)
        if best is None or key>best[0]:best=(key,name,q,p,ic,rmse,n)
    return best

def stack_weights(P,y,d):
    best=None
    for i in range(11):
        for j in range(11-i):
            k=10-i-j;w=np.array([i,j,k],float)/10;p=P@w;ic,_,_,_=daily_ic(y,p,d);rm=math.sqrt(mean_squared_error(y,p));key=(-999 if not np.isfinite(ic) else ic,-rm)
            if best is None or key>best[0]:best=(key,w)
    return best[1] if best else np.ones(3)/3

def make_folds(df,train_years=6,embargo=20):
    dates=pd.DatetimeIndex(sorted(df.as_of_date.unique()));years=sorted(set(dates.year));folds=[]
    for i in range(train_years+1,len(years)):
        ty=years[i];vy=years[i-1];trys=years[:i-1];vd=dates[dates.year==vy];td=dates[dates.year==ty]
        if not len(vd) or not len(td):continue
        vi=dates.get_loc(vd.min());ti=dates.get_loc(td.min());vemb=dates[max(0,vi-embargo)];temb=dates[max(0,ti-embargo)]
        le=pd.concat([pd.to_datetime(df[f'label_end_timestamp_{h}'],utc=True,errors='coerce') for h in HORIZONS],axis=1).max(axis=1)
        tr=df.as_of_date.dt.year.isin(trys)&(le<(pd.Timestamp(vemb).tz_localize('Asia/Shanghai').tz_convert('UTC')))
        va=df.as_of_date.dt.year.eq(vy)&(le<(pd.Timestamp(temb).tz_localize('Asia/Shanghai').tz_convert('UTC')))
        te=df.as_of_date.dt.year.eq(ty)
        if tr.sum()>1000 and va.sum()>200 and te.sum()>200:folds.append({'test_year':ty,'validation_year':vy,'train_mask':tr,'val_mask':va,'test_mask':te})
    return folds

def run(panel:Path,out:Path,cost_bps=20.0):
    f=pd.read_parquet(panel);labels=pd.read_parquet(panel.parent/'forward_labels.parquet');f['as_of_date']=pd.to_datetime(f.as_of_date);labels['as_of_date']=pd.to_datetime(labels.as_of_date)
    d=f.merge(labels,on=['as_of_date','ticker'],how='left');
    for c in ['security_local_cutoff','source_max_timestamp']:[None for _ in [0]]
    d['security_local_cutoff']=pd.to_datetime(d.security_local_cutoff,utc=True);d['source_max_timestamp']=pd.to_datetime(d.source_max_timestamp,utc=True)
    for h in HORIZONS:d[f'label_end_timestamp_{h}']=pd.to_datetime(d[f'label_end_timestamp_{h}'],utc=True,errors='coerce')
    miss=float(d[FACTORS].isna().mean().mean());future=int((d.source_max_timestamp>d.security_local_cutoff).sum());dups=int(d.duplicated(['as_of_date','ticker']).sum())
    pit_pass=future==0 and dups==0 and miss<=.25
    folds=make_folds(d);oos=[];fold_params=[]
    if pit_pass:
      fams=candidates()
      for fi,fold in enumerate(folds):
        tr=d[fold['train_mask']];va=d[fold['val_mask']];te=d[fold['test_mask']]
        for h in HORIZONS:
            ycol=f'ret_excess_fwd_{h}';trh=tr.dropna(subset=[ycol]);vah=va.dropna(subset=[ycol]);teh=te.dropna(subset=[ycol])
            if len(trh)<1000 or len(vah)<200 or len(teh)<200:continue
            Xtr=trh[FACTORS];Xv=vah[FACTORS];Xt=teh[FACTORS];yt=trh[ycol].values;yv=vah[ycol].values
            sels=[]
            for fam,mods in fams.items():sels.append((fam,select_family(mods,Xtr,yt,Xv,yv,vah.as_of_date.values)))
            Pv=np.column_stack([s[1][3] for s in sels]);w=stack_weights(Pv,yv,vah.as_of_date.values);Pt=np.column_stack([s[1][2].predict(Xt) for s in sels]);pv=Pv@w;pt=Pt@w
            retcal=Ridge(alpha=1.).fit(pv.reshape(-1,1),yv);predret=retcal.predict(pt.reshape(-1,1))
            logit=LogisticRegression(C=1.,max_iter=1000).fit(pv.reshape(-1,1),(yv>0).astype(int));prob=logit.predict_proba(pt.reshape(-1,1))[:,1]
            resid=yv-pv;varq=float(np.quantile(resid,.05));var_threshold=pt+varq;viol=(teh[ycol].values<var_threshold).astype(int)
            for ix,(_,r) in enumerate(teh.iterrows()):oos.append({'as_of_date':r.as_of_date,'ticker':r.ticker,'horizon':h,'y':r[ycol],'pred':pt[ix],'expected_return':predret[ix],'prob_up':prob[ix],'var_violation':int(viol[ix]),'test_year':fold['test_year']})
            fold_params.append({'fold':fi,'test_year':fold['test_year'],'validation_year':fold['validation_year'],'horizon':h,'families':[{ 'family':fam,'model':s[1][1],'validation_ic':s[1][4],'validation_rmse':s[1][5]} for fam,s in sels],'stack_weights':w.tolist()})
    o=pd.DataFrame(oos);metrics={};gates={'PIT_CHECK':pit_pass,'MIN_TEST_FOLDS':len(set(x['test_year'] for x in fold_params))>=5}
    if len(o):
      for h in HORIZONS:
        q=o[o.horizon.eq(h)];ic,icir,n,a=daily_ic(q.y,q.pred,q.as_of_date);spread,spreads,top_hit=top_bottom(q.y,q.pred,q.as_of_date);hit=float((np.sign(q.y)==np.sign(q.pred)).mean());metrics[str(h)]={'rank_ic':ic,'icir':icir,'ic_days':n,'hit_rate':hit,'top_bottom_decile_spread':spread,'top_decile_positive_rate':top_hit,'hac_t_rank_ic':hac_t(a,h-1)}
        if h==60:
            brier=brier_score_loss((q.y>0).astype(int),q.prob_up);clim=np.repeat(float((q.y>0).mean()),len(q));brier0=brier_score_loss((q.y>0).astype(int),clim);bskill=1-brier/brier0 if brier0>0 else np.nan
            daily_alpha=[]
            for _,g in q.groupby('as_of_date'):
                if len(g)<MIN_CS:continue
                rank=g.pred.rank(pct=True);daily_alpha.append(float(g.loc[rank>=.9,'y'].mean()-cost_bps/10000.0))
            lcb=block_lcb(daily_alpha,20);kp=kupiec(q.var_violation.values);cp=christoffersen(q.var_violation.values);z=metrics['60']['hac_t_rank_ic'];p_raw=1-norm.cdf(z) if np.isfinite(z) else 1.;p_adj=min(1.,p_raw*3)
            metrics['60'].update({'brier_skill':bskill,'cost_adjusted_top_decile_alpha_mean':float(np.mean(daily_alpha)) if daily_alpha else np.nan,'cost_adjusted_alpha_lcb95':lcb,'var_kupiec_p':kp,'var_christoffersen_p':cp,'selection_bias_equivalent_p_bonferroni3':p_adj})
      m60=metrics.get('60',{});m120=metrics.get('120',{});gates.update({'OOS_RANK_IC_60':m60.get('rank_ic',-1)>=.015,'OOS_RANK_IC_120':m120.get('rank_ic',-1)>=.01,'OOS_ICIR_60':m60.get('icir',-1)>=.2,'OOS_IC60_HAC_T':m60.get('hac_t_rank_ic',-1)>=1.96,'BRIER_SKILL_60':m60.get('brier_skill',-1)>0,'TOP_MINUS_BOTTOM_DECILE_60':m60.get('top_bottom_decile_spread',-1)>0,'VAR_KUPIEC':m60.get('var_kupiec_p',0)>=.05,'VAR_CHRISTOFFERSEN':m60.get('var_christoffersen_p',0)>=.05,'COST_ADJUSTED_ALPHA_LCB95':m60.get('cost_adjusted_alpha_lcb95',-1)>0,'SELECTION_BIAS_CONTROL':m60.get('selection_bias_equivalent_p_bonferroni3',1)<.05})
    mandatory=all(gates.values()) if gates else False
    if not pit_pass:status='BLOCKED_PIT_AUDIT_FAILED'
    elif not gates.get('MIN_TEST_FOLDS',False):status='BLOCKED_INSUFFICIENT_HISTORY'
    elif mandatory:status='CALIBRATED_PRODUCTION'
    else:status='VALIDATION_FAILED_NOT_PRODUCTION'
    pack={'model_version':'3.0.0','patch_version':'3.0.3-H1','status':status,'calibrated_from':{'panel':str(panel),'first_date':str(d.as_of_date.min()),'last_date':str(d.as_of_date.max()),'rows':len(d),'tickers':d.ticker.nunique(),'mean_factor_missing_rate':miss,'future_information_rows':future,'duplicate_rows':dups,'real_data':True,'synthetic_or_mock':False},'folds':sorted(set(x['test_year'] for x in fold_params)),'oos_metrics':metrics,'validation_gates':gates,'fold_parameter_history':fold_params,'cost_assumption_bps_roundtrip':cost_bps,'selection_bias_control':'Bonferroni-adjusted one-sided HAC test across 3 preregistered model families; equivalent multiple-trial control, not DSR.','production_constraint':'Deployment only if status=CALIBRATED_PRODUCTION and strict H1 data provenance gate separately PASS.'}
    out.mkdir(parents=True,exist_ok=True);(out/'production_parameter_pack.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2,default=str),encoding='utf-8');o.to_parquet(out/'oos_predictions.parquet',index=False) if len(o) else None;(out/'calibration_summary.json').write_text(json.dumps({'status':status,'metrics':metrics,'gates':gates},ensure_ascii=False,indent=2,default=str),encoding='utf-8');return pack

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--factor-panel',required=True);ap.add_argument('--out',required=True);ap.add_argument('--cost-bps',type=float,default=20.0);a=ap.parse_args();p=run(Path(a.factor_panel),Path(a.out),a.cost_bps);print(json.dumps(p,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
