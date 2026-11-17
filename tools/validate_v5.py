"""V5 actual-input contracts, frozen V4 parity, and declared feedback scenarios."""
import argparse
from collections import Counter
import hashlib,importlib.util,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def validate(logs,output):
    output.mkdir(parents=True,exist_ok=True)
    v4=load(ROOT/'baselines/v4/agent.py','v4_validator')
    v5=load(ROOT/'agent/agent.py','v5_validator')
    contract=load(ROOT/'tools/validate_v3.py','v5_contract')
    assert v5.MAP_DATA==v4.MAP_DATA
    records=[];focus=[]
    for folder in sorted((p for p in logs.iterdir() if p.is_dir() and p.name.isdigit() and int(p.name)<=20),key=lambda p:int(p.name)):
        v4.init({});v5.init({});counts=Counter();digest=hashlib.sha256();byte_count=0;cpu=0.;meta=trailer=None
        active={};ever_yield=False
        with next(folder.glob('training*.jsonl')).open('rb') as f:
            for lineno,raw in enumerate(f,1):
                r=json.loads(raw);last_type=r.get('type')
                if last_type=='metadata':meta=r
                if last_type=='export_complete':trailer=r
                if 'observation' not in r:continue
                o=r['observation'];c=o['self'];digest.update(raw);byte_count+=len(raw);counts['decisions']+=1
                a4=v4.act(o);start=time.process_time();a5=v5.act(o);cpu+=time.process_time()-start
                contract.valid(a5)
                counts['map_valid']+=v5._driver.diagnostic['map_valid'];counts['source_faults']+=bool(r.get('fault'))
                if c['pit_state']:assert a5['throttle']==a5['brake']==a5['steer']==0.
                if int(folder.name)==20:
                    for k,x in a4.items():
                        assert abs(x-r['action'][k])<=1.5e-5 if isinstance(x,(int,float)) and not isinstance(x,bool) else x==r['action'][k],(lineno,k)
                    counts['v4_recorded_action_parity']+=1
                if o['phase']=='qualifying':
                    assert a4['steer']==a5['steer']
                    assert v4._driver.diagnostic['target']==v5._driver.diagnostic['target']
                    counts['qualifying_steer_and_target_equal']+=1
                counts['extra_accel_enabled']+=v5._driver.diagnostic['extra_accel_enabled']
                counts['higher_throttle_than_v4_on_same_input']+=a5['throttle']>a4['throttle']+1e-9
                if v5._driver.yield_seat is not None:
                    seat=v5._driver.yield_seat;key=(o['phase'],seat)
                    counts['lapping_observation_calls']+=1
                    counts['lapping_move_calls']+=v5._driver.yield_state=='move'
                    if key not in active:
                        active[key]=dict(seat=seat,first_time=o['time'],first_s=c['s'],first_line=lineno)
                    active[key].update(last_time=o['time'],last_s=c['s'])
                    if int(folder.name)==20 and len(focus)<80 and (len(focus)==0 or o['time']-focus[-1]['time']>=.5):
                        focus.append(dict(line=lineno,time=o['time'],s=c['s'],n=c['n'],recorded_v=c['speed_mps'],
                          yield_seat=seat,yield_state=v5._driver.yield_state,yield_goal=v5._driver.yield_goal,
                          offset=v5._driver.offset,target=v5._driver.diagnostic['target'],action=a5))
        assert last_type=='export_complete' and trailer['complete']
        assert (trailer['rows'],trailer['raw_bytes'],trailer['sha256'])==(counts['decisions'],byte_count,digest.hexdigest())
        record=dict(folder=int(folder.name),job_id=meta['job_id'],**counts,integrity=True,action_contract=True,
                    candidate_cpu_seconds=cpu,observed_lapping_windows=list(active.values()))
        records.append(record)
        print(f"{folder.name}: {counts['decisions']} valid actions; CPU {cpu:.3f}s; yield calls {counts['lapping_observation_calls']}",flush=True)
    f4=load(ROOT/'tools/feedback_v4.py','v5_geometry_feedback')
    f5=load(ROOT/'tools/feedback_v5.py','v5_longitudinal_feedback')
    corners=[]
    for track in ('parkland','highland','crossover'):
        length=json.loads((ROOT/f'data/racing-{track}.mapdata.json').read_text())['length_m']
        for wet in (0.,.56,1.):
            for lat in (10.,12.):
                r=f4.scenario(v5,track,wet,start=0.,end=length,v0=20.,lateral=lat)
                r.pop('rows');assert r['completed'] and r['min_road_margin']>2.,r;corners.append(r)
    delay=[]
    for track in ('parkland','highland','crossover'):
        for lag in (.06,.25):
            r=f4.scenario(v5,track,1.,lateral=10.,lag=lag)
            r.pop('rows');assert r['completed'] and r['min_road_margin']>2.,r;delay.append(r)
    acceleration=[]
    for wet in (0.,.56,1.):
        for grade in (-.05,0.,.10):
            for traction in (11.,14.):
                old=f5.acceleration_scenario(v4,wet,grade,traction_factor=traction)
                new=f5.acceleration_scenario(v5,wet,grade,traction_factor=traction)
                assert new['reached'] and new['seconds']<old['seconds']
                assert new['sliding_seconds']==0.,new
                acceleration.append(dict(parameters=dict(wet=wet,grade=grade,traction_factor=traction),v4=old,v5=new))
    lapping=[]
    for wet,width,blocked in [(0.,8.,()),(.56,8.,()),(1.,8.,()),(1.,5.,()),(1.,8.,(-3.2,))]:
        r=f5.lapping_scenario(v5,wet=wet,half_width=width,blockers=blocked)
        assert not r['collision'] and r['passed'] and r['returned'],{k:v for k,v in r.items() if k!='samples'}
        assert r['max_n']<width-1.5
        lapping.append(r)
    r=f5.lapping_scenario(v5,rival_lane=lambda t:max(-1.4,-.15*t))
    assert not r['collision'] and r['returned']
    r['opponent_motion']='Drifts left at 0.15m/s until n=-1.4m; visible before side choice.'
    lapping.append(r)
    traffic=load(ROOT/'tests/test_traffic_v3.py','v5_normal_passing_feedback')
    normal=[]
    for kw in ({},{'half_width':5.},{'initial_gap':8.,'seconds':60.},{'curve':.0008}):
        r=traffic.traffic_scenario(v5,**kw);assert not r['collision'] and r['passed'] and r['returned'];normal.append(dict(parameters=kw,result=r))
    result=dict(method='Original owner-input replay and action contracts, not V5 race trajectories. Feedback models use declared assumptions and cannot prove official finishing or rank improvement.',
                source_sha256=hashlib.sha256((ROOT/'agent/agent.py').read_bytes()).hexdigest(),
                baseline_v4_sha256=hashlib.sha256((ROOT/'baselines/v4/agent.py').read_bytes()).hexdigest(),
                design_sha256=hashlib.sha256((ROOT/'agent/design.json').read_bytes()).hexdigest(),
                total_decisions=sum(r['decisions'] for r in records),matches=records,
                full_map_sensitivity=corners,tight_corner_delay_sensitivity=delay,
                acceleration_sensitivity=acceleration,lapping_scenarios=lapping,normal_passing=normal)
    (output/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (output/'recorded20-lapping-intent.json').write_text(json.dumps(dict(
        warning='Counterfactual commands on V4 recorded positions; does not show that V5 avoided the recorded contacts.',samples=focus),ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();validate(a.logs,a.output)
