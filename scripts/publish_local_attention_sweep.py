#!/usr/bin/env python3
"""Versioned research artifact delivery; no claim of local project import."""
import hashlib,http.client,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from artifact_transfer_client import Client
run=Path(sys.argv[1]).resolve();state=json.loads((run/'status.json').read_text());stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=run/('handoff_'+stamp);dest.mkdir()
(dest/'HANDOFF.md').write_text(f'Remote project /home/ref/RGB_SFM_V3 on remote140165. Queue status: {state["status"]}. Source snapshots/configs and checkpoint hashes identify each experiment; unfinished variants are not results. Delivery targets verified HTTPS E:\\RGB_SFM_artifacts inbox. Local project root/access is unverified; inbox delivery is not project import. After verifying local root, import relative paths with SHA256 checks and preserve differing versions. Checkpoints remain remote.\n')
paths=[ROOT/'plan/1109_local_attention_variant_sweep.md',ROOT/'RESEARCH_CONTEXT.md',run/'status.json',dest/'HANDOFF.md']
paths+=list((run/'source_snapshot').rglob('*.py'))
paths+=[p for p in [run/'REPORT.md',run/'comparison.json'] if p.exists()]
paths+=list((ROOT/'overnight_logs/local_attention_variants_1109').glob('*checks.json'))
for v in state['variants']:
 t=Path(v['training']);e=Path(v['evaluation']);paths+=[p for p in [t/'run_config.json',t/'history.json'] if p.exists()]
 if v['status']=='complete':paths+=list(e.iterdir())
# Snapshot mutable status/history/context before hashing and sending.
files=[]
for p in paths:
 if p.is_file():
  relative=p.relative_to(ROOT);copy=dest/'payload'/relative;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes());files.append((copy,str(relative)))
manifest={'utc':stamp,'queue_status':state['status'],'files':[{'path':r,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,r in files]};mf=dest/'SHA256_manifest.json';mf.write_text(json.dumps(manifest,indent=2))
cred=Path('/root/.config/rgb-sfm-transfer');c=Client('https://100.104.52.34:40115',cred/'token',cred/'server.crt')
def connection():
 conn=http.client.HTTPSConnection('127.0.0.1',41081,context=c.context,timeout=60);conn.set_tunnel(c.host,c.port);return conn
c.connection=connection;prefix=f'notes/remote140165/documents/{stamp}_{run.name}'
for p,r in files+[(mf,'SHA256_manifest.json')]:c.upload(p,prefix+'/'+r)
(dest/'delivery.json').write_text(json.dumps({'prefix':prefix,'files':len(files)+1},indent=2));print(prefix,flush=True)
