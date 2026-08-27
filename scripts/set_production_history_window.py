from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

TARGETS={
 'scripts/full_real_bootstrap.py':[("START = \"2018-01-01\"","START = \"{start}\"")],
 'scripts/build_conservative_overrides.py':[("2018-01-01","{start}")],
 'scripts/build_real_market_layer.py':[("START = pd.Timestamp('2018-01-01')","START = pd.Timestamp('{start}')"),("date '2017-01-01'","date '{warmup}'")],
 'scripts/h1_real_data_audit.py':[("START = \"2018-01-01\"","START = \"{start}\""),("2018-01-10","{start_check}")],
 'scripts/build_real_factor_panel.py':[("START=pd.Timestamp('2018-01-01')","START=pd.Timestamp('{start}')")],
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',default='2008-01-01');a=ap.parse_args();start=a.start
 import pandas as pd
 warmup=str((pd.Timestamp(start)-pd.DateOffset(years=1)).date());start_check=str((pd.Timestamp(start)+pd.Timedelta(days=9)).date())
 report=[]
 for file,repls in TARGETS.items():
  p=Path(file);text=p.read_text(encoding='utf-8');before=sha(p);changed=0
  for old,new in repls:
   nn=new.format(start=start,warmup=warmup,start_check=start_check);n=text.count(old)
   if n:
    text=text.replace(old,nn);changed+=n
  if changed==0:raise SystemExit(f'No history-window replacement matched in {file}')
  p.write_text(text,encoding='utf-8');report.append({'file':file,'replacements':changed,'sha_before':before,'sha_after':sha(p)})
 print(json.dumps({'production_start':start,'warmup_start':warmup,'patches':report},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
