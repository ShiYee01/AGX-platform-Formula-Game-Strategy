"""Historical input-contract checks and declared traffic scenarios; no AGX engine."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def valid(a):
    assert set(a)=={'throttle','brake','steer','ers_mode','aero_mode','pit_request','pit_compound'}
    assert all(isinstance(a[k],(float,int)) and math.isfinite(a[k]) for k in ('throttle','brake','steer'))
    assert 0<=a['throttle']<=1 and 0<=a['brake']<=1 and -1<=a['steer']<=1
    assert type(a['ers_mode']) is int and a['ers_mode'] in range(4)
    assert type(a['aero_mode']) is int and a['aero_mode'] in (0,1)
    assert type(a['pit_request']) is bool
    assert a['pit_compound'] in ('soft','medium','hard')
    assert not(a['throttle']>0 and a['brake']>0)
    json.dumps(a,allow_nan=False)


def validate(logs,output,max_folder):
    output.mkdir(parents=True,exist_ok=True)
    v1=load(ROOT/'baselines/v1/agent.py','recorded_v1')
    v2=load(ROOT/'baselines/v2/agent.py','recorded_v2')
    v3=load(ROOT/'baselines/v3/agent.py','main_v3')
    exp=load(ROOT/'experiments/v3-corner/agent.py','corner_v3')
    records=[];corner_counts=defaultdict(Counter)
    for folder in sorted((p for p in logs.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name)<=max_folder),key=lambda p:int(p.name)):
        path=next(folder.glob('training*.jsonl'))
        v1.init({});v2.init({});v3.init({});exp.init({})
        rows=raw_bytes=qualifying_equal=extra_calls=0;cpu_main=cpu_exp=max_extra=0.0
        digest=hashlib.sha256();metadata=trailer=None;previous_phase=None
        with path.open('rb') as stream:
            for line_num,raw in enumerate(stream,1):
                row=json.loads(raw)
                if row.get('type')=='metadata':metadata=row;continue
                if row.get('type')=='export_complete':trailer=row;continue
                if 'observation' not in row:continue
                rows+=1;raw_bytes+=len(raw);digest.update(raw)
                o=row['observation'];c=o['self']
                a2=v2.act(o)
                version=metadata['participants'][0]['version_id']
                if version=='eb60e4b4-59c6-4459-9320-6e2546182862':actual=v1.act(o)
                elif version=='41ceca59-9616-4483-992b-2af98c777467':actual=a2
                else:raise ValueError('Historical verifier expects frozen V1/V2 exports')
                for k,v in actual.items():
                    if isinstance(v,(float,int)) and not isinstance(v,bool):assert abs(v-row['action'][k])<=.000015,(folder,line_num,k)
                    else:assert v==row['action'][k],(folder,line_num,k)
                start=time.process_time();a3=v3.act(o);cpu_main+=time.process_time()-start
                start=time.process_time();ae=exp.act(o);cpu_exp+=time.process_time()-start
                valid(a3);valid(ae)
                assert o['phase']!='qualifying' or a2==a3,(folder,line_num,'qualifying changed')
                qualifying_equal+=o['phase']=='qualifying'
                for k in a3:
                    if k!='throttle':assert a3[k]==ae[k],(folder,line_num,k,'experiment changed non-throttle action')
                assert v3._driver.diagnostic['target']==exp._driver.diagnostic['target']
                assert v3._driver.offset==exp._driver.offset
                extra=ae['throttle']-a3['throttle']
                assert -1e-12<=extra<=.15+1e-12
                if extra>1e-9:
                    extra_calls+=1;max_extra=max(max_extra,extra)
                    assert ae['brake']==0 and c['speed_mps']<v3._driver.diagnostic['target']
                tid=min(v3.TRACK_PITS,key=lambda p:abs(p[0]-o['track_length_m']))[0]
                key=f'{tid:.3f}/{o["phase"]}'
                corner_counts[key].update(dict(decisions=1,extra_calls=int(extra>1e-9),extra_sum=extra))
        assert trailer and trailer['complete'] and row.get('type')=='export_complete'
        assert trailer['rows']==rows and trailer['raw_bytes']==raw_bytes and trailer['sha256']==digest.hexdigest()
        records.append(dict(folder=int(folder.name),decisions=rows,recorded_version=metadata['participants'][0]['version_id'],
                            input_integrity=True,recorded_action_parity=True,main_contract=True,experimental_contract=True,
                            unchanged_qualifying_actions=qualifying_equal,corner_extra_calls=extra_calls,
                            max_extra_throttle=max_extra,main_cpu_seconds=cpu_main,experiment_cpu_seconds=cpu_exp))
        print(f'{folder.name}: {rows} actions checked; corner support on {extra_calls}; main CPU {cpu_main:.3f}s',flush=True)
    harness=load(ROOT/'tests/test_traffic_v3.py','traffic_feedback')
    scenarios={}
    for label,module,kw in [('v2-centred-leader',v2,{}),('v3-centred-leader',v3,{}),
                            ('v3-gentle-curve',v3,dict(curve=.0008)),
                            ('v3-minimum-map-width',v3,dict(half_width=5.0)),
                            ('v3-short-start-gap',v3,dict(initial_gap=8.0,seconds=60.0))]:
        result=harness.traffic_scenario(module,**kw)
        assert not result['collision'] and result['max_n']<6.5
        assert result['passed']==(module is not v2)
        scenarios[label]=result
    (output/'feedback-scenarios.json').write_text(json.dumps(dict(
        method='Chosen toy bicycle/traffic assumptions; not official dynamics or race-time evidence.',
        scenarios=scenarios),ensure_ascii=False,indent=2)+'\n')
    paths={'v2':'baselines/v2/agent.py','v3':'baselines/v3/agent.py','corner':'experiments/v3-corner/agent.py'}
    result=dict(method='Recorded input replay, immutable-parameter checks, and declared toy traffic feedback. No official engine or ranking prediction.',
                source_sha256={k:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for k,p in paths.items()},
                total_decisions=sum(r['decisions'] for r in records),matches=records,
                corner_activity={k:dict(v) for k,v in corner_counts.items()})
    (output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-folder',type=int,default=18)
    a=p.parse_args();validate(a.logs,a.output,a.max_folder)
