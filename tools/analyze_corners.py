"""Public-map bend catalogue and V2 corner-pedal evidence, not a physics model."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVE_THRESHOLD = 0.002
JOIN_GAP_M = 40.0


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def catalogue(path):
    raw=path.read_bytes();d=json.loads(raw)
    points=d['centerline'];s=np.array([p['s_m'] for p in points]);n=len(s)
    k=np.array([p['curvature_1pm'] for p in points]);length=d['length_m']
    ds=np.diff(np.r_[s,length])
    signs=np.where(abs(k)>=CURVE_THRESHOLD,np.sign(k),0).astype(int)
    runs=[];i=0
    while i<n:
        sign=signs[i]
        if sign==0:i+=1;continue
        j=i+1
        while j<n and signs[j]==sign:j+=1
        if runs and runs[-1]['sign']==sign and s[i]-(s[runs[-1]['indices'][-1]]+ds[runs[-1]['indices'][-1]])<=JOIN_GAP_M:
            runs[-1]['indices'].extend(range(runs[-1]['indices'][-1]+1,j))
        else:runs.append(dict(sign=int(sign),indices=list(range(i,j))))
        i=j
    if len(runs)>1 and runs[-1]['sign']==runs[0]['sign']:
        last_end=s[runs[-1]['indices'][-1]]+ds[runs[-1]['indices'][-1]]
        gap=length-last_end+s[runs[0]['indices'][0]]
        if gap<=JOIN_GAP_M:
            runs[0]['indices']=list(range(runs[-1]['indices'][0],n))+list(range(0,runs[0]['indices'][-1]+1))
            runs.pop()
    bends=[];point_bend=np.zeros(n,dtype=int)
    for number,run in enumerate(runs,1):
        idx=np.asarray(run['indices']);peak_pos=int(np.argmax(abs(k[idx])));peak=int(idx[peak_pos])
        opening=next((int(j) for j in idx[peak_pos+1:] if abs(k[j])<=.8*abs(k[peak])),None)
        radius=1.0/abs(k[peak]);point_bend[idx]=number
        bends.append(dict(track=d['track_id'],bend=number,sign=run['sign'],
                          start_s_m=float(s[idx[0]]),end_s_m=float((s[idx[-1]]+ds[idx[-1]])%length),
                          wraps=bool(idx[-1]<idx[0]),length_m=float(ds[idx].sum()),
                          peak_s_m=float(s[peak]),peak_curvature_1pm=float(k[peak]),radius_m=radius,
                          dry_reference_mps=min(55.0,math.sqrt(10.0*radius)),
                          wet_reference_mps=min(55.0,math.sqrt(5.5*radius)),
                          opening_observe_s_m=float(s[opening]) if opening is not None else None,
                          min_left_width_m=min(points[int(j)]['half_width_left_m'] for j in idx),
                          min_right_width_m=min(points[int(j)]['half_width_right_m'] for j in idx),
                          max_abs_grade_pct=100*max(abs(points[int(j)]['grade_dz_ds']) for j in idx)))
    summary=dict(track=d['track_id'],coordinate_length_m=length,official_length_m=d['official_length_m'],
                 points=n,bend_segments=len(bends),curve_threshold_1pm=CURVE_THRESHOLD,join_gap_m=JOIN_GAP_M,
                 curved_distance_m=float(ds[abs(k)>=CURVE_THRESHOLD].sum()),
                 tight_radius_below_100m_distance_m=float(ds[abs(k)>.01].sum()),
                 minimum_radius_m=float(1/max(abs(k))),
                 min_left_width_m=min(p['half_width_left_m'] for p in points),
                 min_right_width_m=min(p['half_width_right_m'] for p in points),
                 max_abs_grade_pct=100*max(abs(p['grade_dz_ds']) for p in points),
                 map_sha256=hashlib.sha256(raw).hexdigest())
    return summary,bends,dict(s=s,k=k,length=length,point_bend=point_bend)


def analyze(logs,output):
    output.mkdir(parents=True,exist_ok=True)
    all_bends=[];maps={};summary=[]
    for path in sorted((ROOT/'data').glob('*.mapdata.json')):
        s,b,m=catalogue(path);summary.append(s);all_bends.extend(b);maps[s['track']]=m
    v2=load(ROOT/'baselines/v2/agent.py','v2_corner_reference')
    support=load(ROOT/'experiments/corner_support.py','corner_support')
    groups=defaultdict(Counter);bend_groups=defaultdict(Counter);sources=[]
    for folder in sorted((p for p in logs.iterdir() if p.is_dir() and p.name.isdigit()),key=lambda p:int(p.name)):
        files=list(folder.glob('training*.jsonl'))
        if len(files)!=1:continue
        with files[0].open() as f:metadata=json.loads(next(f))
        if metadata['participants'][0]['version_id']!='41ceca59-9616-4483-992b-2af98c777467':continue
        v2.init({});previous=None;decisions=0
        with files[0].open() as stream:
            for line_number,line in enumerate(stream,1):
                row=json.loads(line)
                if 'observation' not in row:continue
                o=row['observation'];c=o['self'];driver=v2._driver
                old_offset=(c['n'] if o['phase']=='race' else 0.0) if driver.phase!=o['phase'] else driver.offset
                road_before=v2.plan_speed(o,old_offset)
                a=v2.act(o);driver=v2._driver;decisions+=1
                for key,value in a.items():
                    if isinstance(value,(float,int)) and not isinstance(value,bool):
                        assert abs(value-row['action'][key])<=.000015
                    else:assert value==row['action'][key]
                if c['pit_state']:previous=o;continue
                tid=min(maps,key=lambda key:abs(maps[key]['length']-o['track_length_m']))
                m=maps[tid];assert abs(m['length']-o['track_length_m'])<1.0
                station=c['s']%m['length'];index=int(np.searchsorted(m['s'],station,side='right')-1)%len(m['s'])
                curvature=float(np.interp(station,np.r_[m['s'],m['length']],np.r_[m['k'],m['k'][0]]))
                if abs(curvature)<CURVE_THRESHOLD:previous=o;continue
                dt=o['time']-previous['time'] if previous and previous['phase']==o['phase'] else 0.0
                assert 0<=dt<=.1
                wet=o['weather']['track_wetness'];band='low(<0.2)' if wet<.2 else ('mid(0.2-0.6)' if wet<.6 else 'high(>=0.6)')
                road_limit=min(road_before,v2.plan_speed(o,driver.offset));target=driver.diagnostic['target']
                bumped=support.corner_throttle(o,a['throttle'],a['brake'],target,driver.offset,
                         road_limit,o['time']-driver.start_time,a['ers_mode'],10*v2.grip_budget(o))
                extra=bumped-a['throttle']
                values=dict(observations=1,seconds=dt,throttle_seconds=dt*(a['throttle']>.02),
                            brake_seconds=dt*(a['brake']>.02),
                            coast_seconds=dt*(a['throttle']<=.02 and a['brake']<=.02),
                            traffic_seconds=dt*(road_limit-target>2.0),
                            experimental_support_seconds=dt*(extra>1e-9),
                            extra_throttle_time_sum=extra*dt,
                            speed_time_sum=c['speed_mps']*dt,throttle_time_sum=a['throttle']*dt,
                            wetness_time_sum=wet*dt)
                groups[(tid,o['phase'],band)].update(values)
                bend_groups[(tid,int(m['point_bend'][index]),o['phase'])].update(values)
                previous=o
        sources.append(dict(folder=int(folder.name),job_id=metadata['job_id'],decisions=decisions))
        print('Corner evidence:',folder.name,decisions,flush=True)
    def package(key,counts,names):
        result=dict(zip(names,key));result.update(dict(counts));seconds=counts['seconds']
        if seconds:
            result.update(mean_speed_mps=counts['speed_time_sum']/seconds,
                          mean_throttle=counts['throttle_time_sum']/seconds,
                          mean_wetness=counts['wetness_time_sum']/seconds,
                          mean_extra_when_active=counts['extra_throttle_time_sum']/counts['experimental_support_seconds'] if counts['experimental_support_seconds'] else 0.0)
        return result
    evidence=[package(k,v,('track','phase','wetness_band')) for k,v in sorted(groups.items())]
    by_bend=[package(k,v,('track','bend','phase')) for k,v in sorted(bend_groups.items())]
    result=dict(method='Public centreline geometry + recorded V2 pedal observations. Reference speeds are V2 policy budgets, not safe engine limits.',maps=summary,bends=all_bends,corner_evidence=evidence,bend_evidence=by_bend,sources=sources)
    (output/'corners.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (output/'corner-catalogue.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(all_bends[0]),lineterminator='\n');w.writeheader();w.writerows(all_bends)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.logs,args.output)
