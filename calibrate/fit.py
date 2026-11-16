"""Estimate YOUR local-model parameters from YOUR exported training.jsonl.

从本人日志标定近似模型。No defaults are engine values; never treat fits as rules.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def fit(path, game):
    """Fit simple effective motion envelopes, grouping match/leg/seat/phase.

    Args:
        path: One participant's JSONL export.
        game: fps or racing.
    """
    groups, measurements = {}, {k: [] for k in ('speed', 'lateral', 'brake', 'accel', 'energy')}
    count = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('type') != 'decision' or row.get('fault'):
                continue
            o, a = row['observation'], row.get('applied_action') or row.get('action') or {}
            car = o.get('me' if game == 'fps' else 'self', {})
            if game == 'fps' and (not car.get('alive') or o.get('phase') != 'live'):
                continue
            tick = o.get('match_tick') if game == 'fps' else o.get('tick')
            # FPS: public match_tick/60. Racing: public time (seconds).
            t = tick/60 if game == 'fps' and tick is not None else o.get('time')
            if not isinstance(t, (int, float)) or not math.isfinite(t):
                continue
            key = (row.get('leg', 0), row.get('seat', 0), o.get('phase'), o.get('round_index'))
            before = groups.get(key)
            groups[key] = (t, car, a)
            if before is None:
                continue
            t0, prev, prev_action = before
            dt = t-t0
            if not 0 < dt < 1 or any(not isinstance(car.get(k), (int, float)) for k in ('x', 'y')):
                continue
            if game == 'fps':
                dx, dy = car['x']-prev['x'], car['y']-prev['y']
                speed = math.hypot(dx, dy)/dt
                # Exclude stationary samples and round-reset/teleport outliers.
                if speed > 0.05:
                    measurements['speed'].append(speed)
            elif car.get('pit_state') is None or not car.get('pit_state'):
                v, v0 = car.get('speed_mps'), prev.get('speed_mps')
                if not isinstance(v, (int, float)) or not isinstance(v0, (int, float)):
                    continue
                measurements['speed'].append(v)
                acc = (v-v0)/dt
                if prev_action.get('throttle', 0) > .8 and prev_action.get('brake', 0) < .05 and acc > 0:
                    measurements['accel'].append(acc)
                if prev_action.get('brake', 0) > .8 and acc < 0:
                    measurements['brake'].append(-acc)
                preview = o.get('preview', {}).get('rows', [])
                if preview and v > 1:
                    measurements['lateral'].append(v*v*abs(preview[0][1]))
                if all(isinstance(c.get('ers_soc_mj'), (int, float)) for c in (prev, car)):
                    measurements['energy'].append(max(0, prev['ers_soc_mj']-car['ers_soc_mj'])/dt)
            count += 1
    def quantile(key, q):
        values = np.asarray(measurements[key], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        return float(np.quantile(values, q)) if len(values) else None
    if count < 2:
        raise ValueError('Insufficient motion pairs; check exported observation time/fields / 有效连续移动样本不足')
    params = ({'move_speed_mps': quantile('speed', .8)} if game == 'fps' else
              dict(lateral=quantile('lateral', .9), brake=quantile('brake', .8),
                   accel=quantile('accel', .8), speed_cap=quantile('speed', .95),
                   energy_mj_per_s=quantile('energy', .8)))
    return dict(approximation=True, game=game, pairs=count, parameters=params,
                samples={k: len(v) for k, v in measurements.items()},
                note='Effective observed envelopes, not physical identification. Missing parameters stay null; collect relevant actions or supply your own estimate. / 仅为观测包络，缺失参数需补采样或自行估计。')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('training', type=Path)
    p.add_argument('--game', choices=('fps', 'racing'), required=True)
    p.add_argument('--output', type=Path, default=Path('fitted.json'))
    args = p.parse_args()
    result = fit(args.training, args.game)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
