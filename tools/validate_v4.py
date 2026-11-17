"""Historical input contracts and explicit model sensitivity; no AGX race engine."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def validate(logs,output):
    output.mkdir(parents=True,exist_ok=True)
    v4=load(ROOT/'baselines/v4/agent.py','v4_validate')
    v3=load(ROOT/'baselines/v3/agent.py','v3_validate')
    contract=load(ROOT/'tools/validate_v3.py','action_contract')
    records=[]
    for folder in sorted((p for p in logs.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name)<=19),key=lambda p:int(p.name)):
        path=next(folder.glob('training*.jsonl'));v4.init({});v3.init({})
        counts=Counter();digest=hashlib.sha256();byte_count=0;cpu=0.;last_type=None
        with path.open('rb') as stream:
            for line_num,raw in enumerate(stream,1):
                row=json.loads(raw);last_type=row.get('type')
                if last_type=='metadata':metadata=row
                if last_type=='export_complete':trailer=row
                if 'observation' not in row:continue
                digest.update(raw);byte_count+=len(raw);counts['decisions']+=1
                o=row['observation'];start=time.process_time();a=v4.act(o);cpu+=time.process_time()-start
                contract.valid(a)
                assert a['throttle']==a['brake']==a['steer']==0. if o['self']['pit_state'] else True
                counts['map_valid']+=v4._driver.diagnostic['map_valid']
                counts['source_faults']+=bool(row.get('fault'))
                if int(folder.name)==19:
                    expected=v3.act(o)
                    for k,val in expected.items():
                        assert abs(val-row['action'][k])<=1.5e-5 if isinstance(val,(int,float)) and not isinstance(val,bool) else val==row['action'][k],(line_num,k)
                    counts['v3_recorded_action_parity']+=1
        assert last_type=='export_complete' and trailer['complete']
        assert (trailer['sha256'],trailer['rows'],trailer['raw_bytes'])==(digest.hexdigest(),counts['decisions'],byte_count)
        record=dict(folder=int(folder.name),job_id=metadata['job_id'],recorded_version_id=metadata['participants'][0]['version_id'],
                    **counts,raw_bytes=byte_count,sha256=digest.hexdigest(),integrity=True,action_contract=True,candidate_cpu_seconds=cpu)
        records.append(record);print(f"{folder.name}: {counts['decisions']} valid actions, maps {counts['map_valid']}, CPU {cpu:.3f}s",flush=True)
    feedback=load(ROOT/'tools/feedback_v4.py','v4_feedback_validate')
    scenarios=[]
    for track in ('parkland','highland','crossover'):
        length=json.loads((ROOT/f'data/racing-{track}.mapdata.json').read_text())['length_m']
        for wet in (0.,.56,1.):
            for lateral in (10.,12.):
                r=feedback.scenario(v4,track,wet,start=0,end=length,v0=20,lateral=lateral)
                r.pop('rows');assert r['completed'] and r['min_road_margin']>2.0
                scenarios.append(r)
    traffic=load(ROOT/'tests/test_traffic_v3.py','traffic_feedback_v4')
    traffic_scenarios=[]
    for kw in ({},{'initial_gap':8.,'seconds':60.},{'half_width':5.},{'curve':.0008}):
        r=traffic.traffic_scenario(v4,**kw)
        assert r['passed'] and r['returned'] and not r['collision']
        traffic_scenarios.append(dict(parameters=kw,result=r))
    result=dict(method='Recorded input action contracts and complete export hashes. These are not trajectories produced by the candidate, race results, or proof of finishing.',
      source_sha256=hashlib.sha256((ROOT/'baselines/v4/agent.py').read_bytes()).hexdigest(),
      design_sha256=hashlib.sha256((ROOT/'agent/design.json').read_bytes()).hexdigest(),
      total_decisions=sum(r['decisions'] for r in records),matches=records,
      feedback_model='Chosen force curves, tyre limits and delays with a 3.4m bicycle approximation. No rivals, damage accumulation, tyres, energy or pit engine; scenario times are not official lap predictions.',
      single_vehicle_scenarios=scenarios,traffic_scenarios=traffic_scenarios)
    (output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('logs',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();validate(args.logs,args.output)
