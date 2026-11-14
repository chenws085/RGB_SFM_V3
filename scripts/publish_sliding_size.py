#!/usr/bin/env python3
"""Deliver versioned fixed-LR comparison documents to verified E: inbox."""
import hashlib,http.client,json,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from artifact_transfer_client import Client
run=Path(sys.argv[1]).resolve();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');folder=run/('handoff_'+stamp);folder.mkdir()
state=json.loads((run/'status.json').read_text());(folder/'HANDOFF.md').write_text('Remote project /home/ref/RGB_SFM_V3 on remote140165. LR scan explicitly paused. Sliding size queue status is recorded in status.json; consult each cell before treating a size as completed. RF analysis is deferred until size experiments finish. E:\\RGB_SFM_artifacts is the verified inbox, not a confirmed local project path. Verify hashes and preserve different versions before importing after local root/access is confirmed. Checkpoints remain remote.\n')
paths=[ROOT/'plan/1114_sliding_window_size_sweet_spot.md',ROOT/'RESEARCH_CONTEXT.md',run/'status.json',run/'REPORT.md',run/'results.json',folder/'HANDOFF.md']+list((run/'source_snapshot').rglob('*.py'))+list((ROOT/'overnight_logs/sliding_size_1114').glob('*checks.json'))
for c in state['cells']:
 if c['status']=='complete':paths+=list(Path(c['evaluation']).iterdir())
 if c['status'] in ['complete','reused']:
  paths+=[Path(c['training'])/'run_config.json',Path(c['training'])/'history.json']
files=[]
for p in paths:
 if p.is_file():
  rel=str(p.relative_to(ROOT));dest=folder/'payload'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());files.append((dest,rel))
mf=folder/'SHA256_manifest.json';mf.write_text(json.dumps({'status':state['status'],'files':[{'path':r,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,r in files]},indent=2));cred=Path('/root/.config/rgb-sfm-transfer');c=Client('https://100.104.52.34:40115',cred/'token',cred/'server.crt')
def connection():
 conn=http.client.HTTPSConnection('127.0.0.1',41081,context=c.context,timeout=60);conn.set_tunnel(c.host,c.port);return conn
c.connection=connection;prefix=f'notes/remote140165/documents/{stamp}_{run.name}'
for p,r in files+[(mf,'SHA256_manifest.json')]:c.upload(p,prefix+'/'+r)
(folder/'delivery.json').write_text(json.dumps({'prefix':prefix,'files':len(files)+1},indent=2));print(prefix,flush=True)
