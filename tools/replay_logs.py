"""Check a candidate against fixed observations; this cannot predict race results."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(logs, output):
    baseline = load(ROOT/'baselines/v1/agent.py', 'baseline')
    candidate = load(ROOT/'agent/agent.py', 'candidate')
    results=[]
    for folder in sorted((p for p in logs.iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name)):
        baseline.init({});candidate.init({})
        count=0;differences=0;cpu=0.0;max_cpu=0.0;old_pit=new_pit=0
        first_old=first_new=None;race_start=None;first_actions={}
        previous=None;lane_slew_max=0.0;max_target=0.0
        with next(folder.glob('training*.jsonl')).open() as stream:
            for line_number,line in enumerate(stream,1):
                row=json.loads(line)
                if 'observation' not in row:continue
                o=row['observation'];b=baseline.act(o)
                for k,v in b.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):
                        assert abs(v-row['action'][k])<=.000015,(folder,line_number,k)
                    else:assert v==row['action'][k],(folder,line_number,k)
                start=time.process_time();a=candidate.act(o);used=time.process_time()-start
                cpu+=used;max_cpu=max(max_cpu,used);count+=1
                assert set(a)==set(b),(folder,line_number,'action fields')
                assert all(math.isfinite(a[k]) for k in ('throttle','brake','steer'))
                assert 0<=a['throttle']<=1 and 0<=a['brake']<=1 and -1<=a['steer']<=1
                assert type(a['pit_request']) is bool and type(a['ers_mode']) is int
                assert a['ers_mode'] in range(4) and a['aero_mode'] in (0,1)
                assert a['pit_compound'] in ('soft','medium','hard')
                assert not (a['throttle']>0 and a['brake']>0)
                json.dumps(a,allow_nan=False)
                differences+=any(a[k]!=b[k] for k in a)
                diag=candidate._driver.diagnostic
                max_target=max(max_target,diag['target'])
                if previous and previous['phase']==o['phase'] and not o['self']['pit_state']:
                    dt=max(1/60,min(.1,o['time']-previous['time']))
                    lane_slew_max=max(lane_slew_max,abs(diag['offset']-previous['offset'])/dt)
                previous=dict(phase=o['phase'],time=o['time'],offset=diag['offset'])
                if o['phase']=='race':
                    if race_start is None:
                        race_start=o['time'];first_actions=dict(v1=b,v2=a,grid_n=o['self']['n'],v2_offset=diag['offset'])
                    old_pit+=b['pit_request'];new_pit+=a['pit_request']
                    if b['pit_request'] and first_old is None:
                        first_old=dict(elapsed=o['time']-race_start,lap=o['self']['lap'],s=o['self']['s'],line=line_number)
                    if a['pit_request'] and first_new is None:
                        first_new=dict(elapsed=o['time']-race_start,lap=o['self']['lap'],s=o['self']['s'],line=line_number)
        results.append(dict(number=int(folder.name),decisions=count,baseline_parity=True,
                            candidate_contract=True,changed_decisions=differences,
                            candidate_cpu_seconds=cpu,max_call_cpu_ms=max_cpu*1000,
                            mean_call_cpu_us=cpu/count*1e6,max_target_mps=max_target,
                            max_lane_target_slew_mps=lane_slew_max,
                            old_pit_request_count=old_pit,new_pit_request_count=new_pit,
                            first_old_pit_request=first_old,first_new_pit_request=first_new,
                            first_race_actions=first_actions))
        print(f'Match {folder.name}: {count} actions valid; candidate CPU {cpu:.3f}s',flush=True)
    report=dict(method='Fixed-observation replay; NOT closed-loop evaluation or ranking prediction.',
                agent_sha256=__import__('hashlib').sha256((ROOT/'agent/agent.py').read_bytes()).hexdigest(),
                total_decisions=sum(r['decisions'] for r in results),matches=results)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();replay(args.logs,args.output)
