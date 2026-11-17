"""Race-19 V3 diagnosis from owner recordings, public events and public geometry.
Use --folder and --output to reproduce evidence without committing raw logs.
"""
import json,hashlib,importlib.util,csv,sys
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--folder',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
R=Path(__file__).resolve().parents[1]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.init({});return m
mods={k:load(R/p,k) for k,p in [('main','baselines/v3/agent.py'),('experiment','experiments/v3-corner/agent.py')]}
mp=json.load(open(R/'data/racing-highland.mapdata.json'));ss=np.array([p['s_m'] for p in mp['centerline']]+[mp['length_m']]);ks=np.array([p['curvature_1pm'] for p in mp['centerline']]+[mp['centerline'][0]['curvature_1pm']]);length=mp['length_m']
frames=[];diff={k:0 for k in mods};faults=0;hash=hashlib.sha256();nb=0
p=next(args.folder.glob('training*.jsonl'))
with p.open('rb') as f:
 for line,raw in enumerate(f,1):
  r=json.loads(raw)
  if r.get('type')=='export_complete':trailer=r
  if 'observation' not in r:continue
  hash.update(raw);nb+=len(raw);faults+=bool(r.get('fault'));o=r['observation'];c=o['self'];a=r['action'];t={}
  for name,m in mods.items():
   pred=m.act(o)
   diff[name]+=any(abs(pred[k]-a[k])>1.5e-5 if isinstance(a[k],(float,int)) and not isinstance(a[k],bool) else pred[k]!=a[k] for k in a)
   t[name]=m._driver.diagnostic['target']
  main=mods['main'];road=main.plan_speed(o,main._driver.offset);k=float(np.interp(c['s']%length,ss,ks));rows=o['preview']['rows'];ex=rows[0][1]*2-rows[1][1]
  forces=c.get('drive_force_n',[0]*4);mode=a['ers_mode'];force=forces[mode];resist=c.get('drag_n',0)+c.get('rolling_resistance_n',0);grade=float(np.interp(c['s']%length,ss,np.array([p['grade_dz_ds'] for p in mp['centerline']]+[mp['centerline'][0]['grade_dz_ds']])))
  frames.append(dict(line=line,phase=o['phase'],time=o['time'],lap=c['lap'],s=c['s'],v=c['speed_mps'],n=c['n'],heading=c['heading_error'],yaw=c['yaw_rate'],steer=a['steer'],actual_steer=c['steer'],throttle=a['throttle'],brake=a['brake'],mode=mode,target=t['main'],road=road,offset=main._driver.offset,k=k,extrap_k=ex,preview_k=rows[0][1],g_lat=c['g_lat'],g_long=c['g_long'],wet=o['weather']['track_wetness'],wear=max(c['tyre_wear']),fuel=c['fuel_kg'],soc=c['ers_soc_mj'],front_slide=c['front_sliding'],rear_slide=c['rear_sliding'],rivals=len(o['rivals']),drive=force,resist=resist,mass=c['effective_mass_kg'],grade=grade,pass_seat=main._driver.pass_seat,pit=c['pit_state']))
assert trailer['sha256']==hash.hexdigest() and trailer['rows']==len(frames) and trailer['raw_bytes']==nb
out=dict(integrity=True,decisions=len(frames),faults=faults,version_action_mismatch_frames=diff,summary={},hairpins=[])
for phase in ['qualifying','race']:
 f=[x for x in frames if x['phase']==phase and not x['pit']]
 out['summary'][phase]={}
 for label,ff in [('all',f),('bend',[x for x in f if abs(x['k'])>=.002]),('straight',[x for x in f if abs(x['k'])<.002]),('free_straight',[x for x in f if abs(x['k'])<.002 and x['road']-x['target']<1])]:
  if not ff:continue
  out['summary'][phase][label]=dict(seconds=len(ff)/30,mean_kmh=float(np.mean([x['v']*3.6 for x in ff])),max_kmh=max(x['v']*3.6 for x in ff),mean_throttle=float(np.mean([x['throttle'] for x in ff])),mean_target_kmh=float(np.mean([x['target']*3.6 for x in ff])),traffic_seconds=sum(x['road']-x['target']>1 for x in ff)/30,sliding_seconds=sum(x['front_slide'] or x['rear_slide'] for x in ff)/30)
 for lap in sorted(set(x['lap'] for x in f)):
  sel=[x for x in f if x['lap']==lap and 175<x['s']<285]
  if sel:
   samples=[sel[0]]
   for x in sel:
    if x['s']-samples[-1]['s']>=8:samples.append(x)
   if samples[-1]!=sel[-1]:samples.append(sel[-1])
   out['hairpins'].append(dict(phase=phase,lap=lap,frames=samples))
# acceleration deficit on settled straight, excluding upcoming braking
ff=[x for x in frames if x['phase']=='race' and x['target']>45 and x['road']-x['target']<1 and abs(x['heading'])<.05 and abs(x['n']-x['offset'])<.5]
out['straight_high_target_examples']=sorted(ff,key=lambda x:abs(x['v']-45))[::max(1,len(ff)//8)][:8]
out['events']=[]
with next(args.folder.glob('public*.jsonl')).open() as stream:
 for line,raw in enumerate(stream,1):
  if not raw.strip():continue
  row=json.loads(raw)
  if 4 in row.get('seats',[]) and row.get('kind')!='energy_mode':
   out['events'].append(dict(line=line,**row))
with p.open() as stream:
 for raw in stream:
  row=json.loads(raw)
  if row.get('type')=='outcome':out['outcome']=row
(args.output/'diagnosis19.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
with open(args.output/'hairpin19.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=frames[0],lineterminator="\n");w.writeheader();w.writerows(x for x in frames if x['phase']=='race' and x['lap']==2 and 170<x['s']<260)
print(json.dumps({k:v for k,v in out.items() if k not in ('hairpins','straight_high_target_examples')},ensure_ascii=False,indent=2))
for h in out['hairpins']:
 print('\nHAIRPIN',h['phase'],h['lap'])
 for x in h['frames']:print('s %.1f v %.1f tgt %.1f n %.2f off %.2f head %.2f k %.4f ex %.4f pv %.4f steer %.2f actual %.2f oil %.2f brake %.2f slide %s/%s rivals %d'%(x['s'],x['v']*3.6,x['target']*3.6,x['n'],x['offset'],x['heading'],x['k'],x['extrap_k'],x['preview_k'],x['steer'],x['actual_steer'],x['throttle'],x['brake'],x['front_slide'],x['rear_slide'],x['rivals']))
