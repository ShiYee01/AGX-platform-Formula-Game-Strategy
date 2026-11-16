"""Audit exported bytes and replay V1; report observations, not simulated scores."""
import json, hashlib, importlib.util, argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

parser = argparse.ArgumentParser(description='Verify owner exports and summarize observed V1 telemetry; no simulation.')
parser.add_argument('logs', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = args.logs
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('baseline', Path(__file__).resolve().parents[1] / 'baselines/v1/agent.py')
baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)

def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), min=float(a.min()), p10=float(np.quantile(a,.1)), median=float(np.median(a)), p90=float(np.quantile(a,.9)), max=float(a.max()), mean=float(a.mean())) if len(a) else {}

all_results=[]
for folder in sorted((p for p in ROOT.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p:int(p.name)):
    source=next(folder.glob('training*.jsonl'))
    meta=outcome=complete=None
    series=defaultdict(lambda:defaultdict(list));counts=defaultdict(Counter)
    first= {}; last={}; transitions=[];prev=None; digest=hashlib.sha256();byte_count=0;decision_count=0
    mismatch=Counter();examples=[];lap_times={}; source_digest=hashlib.sha256()
    baseline.init({})
    with source.open('rb') as stream:
        for line_no,raw in enumerate(stream,1):
            source_digest.update(raw)
            row=json.loads(raw)
            if row.get('type')=='metadata':meta=row;continue
            if row.get('type')=='outcome':outcome=row;continue
            if row.get('type')=='export_complete':complete=row;continue
            if 'observation' not in row:raise ValueError((source,line_no,'unknown row'))
            digest.update(raw);byte_count+=len(raw);decision_count+=1
            o=row['observation'];c=o['self'];phase=o['phase'];a=row['action'];t=o['time']
            b=baseline.act(o); target=baseline.plan_speed(o)
            for k,v in b.items():
                if isinstance(v,(float,int)) and not isinstance(v,bool):
                    if abs(v-a.get(k,0))>0.000015:mismatch[k]+=1
                elif v!=a.get(k):mismatch[k]+=1
            first.setdefault(phase,o);last[phase]=o
            counts[phase]['decisions']+=1;counts[phase]['faults']+=bool(row.get('fault'))
            counts[phase]['pit_request']+=bool(a['pit_request'])
            counts[phase]['pit_active']+=bool(c['pit_state'])
            counts[phase]['outside3']+=abs(c['n'])>3
            counts[phase]['heading03']+=abs(c['heading_error'])>.3
            counts[phase]['heading06']+=abs(c['heading_error'])>.6
            counts[phase]['front_sliding']+=c.get('front_sliding',False)
            counts[phase]['rear_sliding']+=c.get('rear_sliding',False)
            counts[phase]['boost']+=a['ers_mode']==2
            counts[phase]['soc_full']+=c['ers_soc_mj']>=3.95
            counts[phase]['brake']+=a['brake']>.01
            counts[phase]['throttle_full']+=a['throttle']>.95
            counts[phase]['very_slow']+=c['speed_mps']<5
            if not c['pit_state']:
                for k,v in {'speed':c['speed_mps'],'n':abs(c['n']),'heading':abs(c['heading_error']),'wear':max(c['tyre_wear']),'temp':np.mean(c['tyre_temp_c']),'wet':o['weather']['track_wetness'],'target':target,'fuel':c['fuel_kg'],'soc':c['ers_soc_mj']}.items():series[phase][k].append(float(v))
            if prev and prev['phase']==phase:
                dt=t-prev['time']
                if 0<dt<.1:
                    counts[phase]['seconds']+=dt
                    if c['pit_state']:counts[phase]['pit_seconds']+=dt
                    if abs(c['n'])>3:counts[phase]['outside3_seconds']+=dt
                    if abs(c['heading_error'])>.3:counts[phase]['heading03_seconds']+=dt
                    if o['weather']['track_wetness']>.2:counts[phase]['wet_seconds']+=dt
                    if c['lap']!=prev['self']['lap']:lap_times[f'{phase}:{c["lap"]}']=round(t,4)
            if prev is None or phase!=prev['phase'] or c['pit_state']!=prev['self']['pit_state']:
                transitions.append(dict(phase=phase,t=t,elapsed=round(t-first[phase]['time'],4),pit=c['pit_state'],lap=c['lap'],s=c['s'],speed=c['speed_mps']))
            if (abs(c['n'])>3 or abs(c['heading_error'])>.3) and len(examples)<10:
                examples.append(dict(line=line_no,phase=phase,t=t,s=c['s'],n=c['n'],heading=c['heading_error'],speed=c['speed_mps'],action=a))
            prev=o
    if not meta or not outcome or not complete or row.get('type') != 'export_complete':
        raise ValueError(f'{source}: missing or misplaced completion records')
    if len(meta['participants']) != 1:
        raise ValueError('This analysis expects a single-owner-seat export')
    seat=meta['participants'][0]['seat']
    owner=next(c for c in outcome['classification'] if c['seat']==seat)
    relevant=[];event_counts=Counter();race_start=None
    events_sha=hashlib.sha256();event_rows=0
    with (folder/'public-events.jsonl').open('rb') as stream:
        for line in stream:
            events_sha.update(line)
            event=json.loads(line);event_counts[event['kind']]+=1
            if event['seq'] != event_rows:
                raise ValueError('Missing/out-of-order public event sequence')
            event_rows += 1
            if event['kind']=='lights_out':race_start=event
            if seat in event.get('seats',[]) and event['kind'] not in ('energy_mode','aero_mode','overtake'):
                relevant.append(event)
    result=dict(number=int(folder.name),job_id=meta['job_id'],seed=meta['seed'],seat=seat,version=meta['participants'][0]['version_id'],source_hash=meta['participants'][0]['source_hash'],rules_version=meta['rules_version'],track_length=first['qualifying']['track_length_m'],integrity=dict(complete=complete['complete'],rows=decision_count,rows_match=complete['rows']==decision_count,raw_bytes_match=complete['raw_bytes']==byte_count,sha256_match=complete['sha256']==digest.hexdigest(),file_sha256=source_digest.hexdigest()),classification=owner,winner=min(outcome['classification'],key=lambda x:x['position']),opponents=outcome['classification'],reason=outcome['reason'],phases={phase:dict(counts=dict(counts[phase]),stats={k:stats(v) for k,v in series[phase].items()},start=first[phase]['time'],end=last[phase]['time'],first_weather=first[phase]['weather'],last_car=last[phase]['self']) for phase in first},baseline_mismatch=dict(mismatch),transitions=transitions,owner_events=relevant,event_counts=dict(event_counts),race_start=race_start,first_instability=examples,lap_crossings=lap_times)
    result['integrity'].update(seat_calls_match=complete['seat_calls'][str(seat)]==decision_count,
                               public_events_sha256=events_sha.hexdigest(),
                               public_event_rows=event_rows,
                               public_final_tick_matches=event['tick']==outcome['steps'])
    (OUT/f'match-{folder.name}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    all_results.append(result)
    print(json.dumps(dict(n=folder.name,track=result['track_length'],rank=owner['position'],status=owner['status'],laps=owner['laps'],qual=owner['qualifying_position'],fastest=owner['fastest_lap'],pit=owner['pit_stops'],integrity=result['integrity'],mismatch=dict(mismatch),events=relevant),ensure_ascii=False),flush=True)
(OUT/'summary.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2)+'\n')

if not all_results or any(not all(r['integrity'][k] for k in ('complete','rows_match','raw_bytes_match','sha256_match','seat_calls_match','public_final_tick_matches')) or r['baseline_mismatch'] for r in all_results):
    raise SystemExit('Integrity or baseline replay verification failed')
