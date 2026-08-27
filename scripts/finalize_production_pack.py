from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def sha(path):
    h=hashlib.sha256();
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--calibration-pack',required=True);ap.add_argument('--h1-audit',required=True);ap.add_argument('--manifest');ap.add_argument('--out',required=True);a=ap.parse_args()
    cp=Path(a.calibration_pack);hp=Path(a.h1_audit);cal=json.loads(cp.read_text(encoding='utf-8'));h1=json.loads(hp.read_text(encoding='utf-8'))
    original=cal.get('status');strict=bool(h1.get('strict_pit_gate_pass'))
    if original=='CALIBRATED_PRODUCTION' and not strict:final='BLOCKED_PIT_AUDIT_FAILED'
    else:final=original
    cal['pre_h1_status']=original;cal['status']=final;cal['h1_strict_pit_gate_pass']=strict;cal['h1_audit_sha256']=sha(hp);cal['calibration_pack_pre_finalize_sha256']=sha(cp)
    if a.manifest and Path(a.manifest).exists():cal['dataset_manifest_sha256']=sha(a.manifest)
    cal['deployment_constraint']='Live deployment requires status=CALIBRATED_PRODUCTION AND h1_strict_pit_gate_pass=true.'
    cal['finalization']='V3.0.3-H1_TWO_KEY_PRODUCTION_LOCK'
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(cal,ensure_ascii=False,indent=2,default=str),encoding='utf-8');print(json.dumps({'pre_h1_status':original,'h1_strict_pit_gate_pass':strict,'final_status':final,'out':str(out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
