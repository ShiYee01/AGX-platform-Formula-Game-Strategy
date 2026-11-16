"""Optimise a lateral line and speed profile in YOUR particle approximation.

质点模型：以公开中心线/宽度搜索行车线；所有动力学参数由本人日志标定。
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def profile(ds, curvature, lateral, brake, accel, speed_cap):
    """Apply circular acceleration and braking envelopes, including lap seam."""
    speed = np.minimum(speed_cap, np.sqrt(lateral/np.maximum(np.abs(curvature), 1e-9)))
    for _ in range(len(ds)):
        previous = speed.copy()
        for i in range(len(ds)):
            j = (i+1) % len(ds)
            speed[j] = min(speed[j], np.sqrt(speed[i]**2+2*accel*ds[i]))
        for i in range(len(ds)-1, -1, -1):
            j = (i+1) % len(ds)
            speed[i] = min(speed[i], np.sqrt(speed[j]**2+2*brake*ds[i]))
        if np.array_equal(previous, speed):
            break
    time = float(np.sum(2*ds/(speed+np.roll(speed,-1))))
    return speed, time


def optimize(path, parameters):
    """Search a smooth lateral line; report local time and estimated energy.

    Args:
        path: Public map NPZ.
        parameters: Student-fitted lateral/brake/accel/speed_cap/energy_mj_per_s,
            plus chosen safety_margin_m (geometric margin, not a platform rule).
    """
    keys = ('lateral','brake','accel','speed_cap','energy_mj_per_s','safety_margin_m')
    if any(not isinstance(parameters.get(k), (int,float)) or not math.isfinite(parameters[k]) or parameters[k] <= 0 for k in keys):
        raise ValueError('Supply finite positive student estimates for '+', '.join(keys))
    with np.load(path, allow_pickle=False) as d:
        columns = json.loads(d['centerline_columns_utf8'].tobytes())
        rows = d['centerline'].copy()
    col = lambda k: rows[:,columns.index(k)]
    xy = np.column_stack((col('x_m'),col('y_m')))
    heading = col('heading_rad')
    normal = np.column_stack((-np.sin(heading),np.cos(heading)))
    left = np.maximum(0,col('half_width_left_m')-parameters['safety_margin_m'])
    right = np.maximum(0,col('half_width_right_m')-parameters['safety_margin_m'])
    curve = col('curvature_1pm')
    smooth = sum(np.roll(curve,k) for k in range(-12,13))/25
    shape = np.tanh(smooth/max(float(np.std(smooth)),1e-9))
    candidates = []
    for scale in np.linspace(-1,1,9):
        offset = np.clip(scale*shape*np.minimum(left,right), -right, left)
        line = xy+offset[:,None]*normal
        delta = np.roll(line,-1,axis=0)-line
        ds = np.linalg.norm(delta,axis=1)
        previous = np.roll(delta,1,axis=0)
        turn = np.arctan2(previous[:,0]*delta[:,1]-previous[:,1]*delta[:,0], (previous*delta).sum(axis=1))
        curvature = turn/((ds+np.roll(ds,1))/2)
        speed, seconds = profile(ds,curvature,*(parameters[k] for k in keys[:4]))
        candidates.append((seconds,scale,offset,speed))
    seconds, scale, offset, speed = min(candidates,key=lambda c:c[0])
    return dict(approximation=True, points=len(rows), estimated_lap_seconds=seconds,
                estimated_energy_mj=seconds*parameters['energy_mj_per_s'],
                centerline_seconds=candidates[4][0], line_scale=float(scale),
                min_speed_mps=float(speed.min()), max_speed_mps=float(speed.max()),
                parameters=parameters), dict(stations=col('s_m'), offsets=offset, speed=speed)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('npz', type=Path)
    p.add_argument('--params',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('profile.npz'))
    args=p.parse_args()
    fitted=json.loads(args.params.read_text())
    result,arrays=optimize(args.npz,fitted.get('parameters',fitted))
    np.savez_compressed(args.output,**arrays)
    print(json.dumps(result,allow_nan=False))
