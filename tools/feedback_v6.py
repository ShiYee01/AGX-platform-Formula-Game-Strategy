"""Whole-lap bicycle model calibrated from AGX public replays, NOT the engine.

Calibration (tools/replay_analysis.py, 21 races, ~3.8 M car frames):
- steering is kinematic: replay steer angle / (3.4 m x path curvature) is
  1.01-1.04 from 0.2 g up to 2.6 g, so a 3.4 m wheelbase with no understeer;
- lateral limit p99.5 ~2.5 g dry, ~1.4 g fully wet (x0.56 = grip_budget);
- braking p0.5 ~-2.6 to -3.0 g dry; net acceleration p99.5 ~1.5 g below
  45 m/s falling to ~0.8 g at 75-95 m/s; top speed ~92 m/s;
- drag ~0.5 v^2 N; effective mass ~1060 kg with 57 kg fuel (seat-19 log).
`lat`/`brake` are model tyre limits in m/s^2 before the same wet scaling.
The nominal case uses the measured envelope; the pessimistic case uses ~80%
of it, i.e. assumes the replay p99.5 overstates what is safely repeatable.
No collisions, kerbs, tyre/fuel/energy evolution or rival cars are modelled.
Model times are comparisons between versions, not predicted official laps.
"""
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASS = 1060.0
NET_ACCEL_G = ((0., 1.5), (37., 1.5), (52., 1.4), (67., 1.08), (85., .8), (92., 0.))
NOMINAL = dict(lat=24.5, brake=26.0, lag=.12)
PESSIMISTIC = dict(lat=20.0, brake=20.0, lag=.20)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def net_accel(v):
    for (v0, a0), (v1, a1) in zip(NET_ACCEL_G, NET_ACCEL_G[1:]):
        if v <= v1:
            return 9.81 * (a0 + (a1 - a0) * (v - v0) / (v1 - v0))
    return 0.0


def track_data(track):
    d = json.loads((ROOT / f'data/racing-{track}.mapdata.json').read_text())
    pts, length = d['centerline'], d['length_m']
    n_pts = len(pts)

    def at(key, s):
        q = (s % length) * n_pts / length
        i = int(q)
        return pts[i][key] + (pts[(i + 1) % n_pts][key] - pts[i][key]) * (q - i)
    return length, at


def drive(mod, track='highland', wet=0.0, lat=24.5, brake=26.0, lag=.12, start=0.0,
          distance=None, v0=40.0, lane=0.0, timing=(300.0, None), phase='race', launch=False):
    """Drive `distance` metres from `start`; return lap time between timing marks."""
    length, at = track_data(track)
    distance = distance if distance is not None else length + 400.0
    t0_mark, t1_mark = timing[0], timing[1] if timing[1] is not None else timing[0] + length
    grip = 1.0 - .45 * wet
    s, n, h, v, steer, yaw, t = start, lane, 0.0, v0, 0.0, 0.0, 0.0
    front = rear = False
    driver = mod.Driver()
    max_n = max_h = 0.0
    min_margin = 99.0
    slide_time = 0.0
    marks = {}
    peak_util = 0.0
    while s - start < distance and t < 900:
        k = at('curvature_1pm', s)
        grade = at('grade_dz_ds', s)
        drag, roll = .5 * v * v, 150.0
        base = MASS * net_accel(v) + drag + roll
        force = [base * .9, base * .9, base, base * 1.1]
        o = dict(phase=phase, time=100 + t, track_length_m=length, laps_total=4,
                 self=dict(n=n, heading_error=h, speed_mps=v, s=s % length, distance=s,
                           tyre_wear=[0.] * 4, tyre_temp_c=[90.] * 4, fuel_kg=40., compound='soft',
                           ers_soc_mj=3., pit_state=0, pit_stops=0, lap=1, front_sliding=front,
                           rear_sliding=rear, yaw_rate=yaw, steer=steer, g_lat=v * yaw / 9.81,
                           drag_n=drag, rolling_resistance_n=roll, effective_mass_kg=MASS,
                           drive_force_n=force, brake_force_max_n=56000.),
                 weather=dict(condition='wet' if wet > .5 else 'dry', track_wetness=wet),
                 rivals=[], preview=dict(rows=[[float(x), at('curvature_1pm', s + x), at('grade_dz_ds', s + x),
                                                at('half_width_left_m', s + x), at('half_width_right_m', s + x), 0]
                                               for x in range(15, 361, 15)]))
        if t == 0:
            driver.reset_phase(o)
            if not launch:
                driver.start_distance, driver.start_time = s - 1000, -20.0
                driver.last_unstable_time = -5.0
            driver.grid_n = driver.offset = lane
        a = driver.act(o)
        for _ in range(2):
            dt = 1 / 60
            steer += min(1, dt / lag) * (.3 * a['steer'] - steer)
            lat_cap = lat * grip
            desired = v / 3.4 * math.tan(steer)
            yaw_limit = lat_cap / max(v, 1.0)
            front = abs(desired) > yaw_limit * 1.02
            yaw += min(1, dt / .06) * (max(-yaw_limit, min(yaw_limit, desired)) - yaw)
            used = min(1.0, abs(v * yaw) / lat_cap)
            peak_util = max(peak_util, used)
            long_cap = brake * grip * math.sqrt(max(.04, 1 - used * used))
            request = (a['throttle'] * force[a['ers_mode']] - a['brake'] * 56000.) / MASS
            rear = request > long_cap * 1.02
            acc = max(-long_cap, min(long_cap, request)) - (drag + roll) / MASS - 9.81 * grade
            v = max(0.0, v + acc * dt)
            scale = max(.35, 1 - k * n)
            n += v * math.sin(h) * dt
            s += v * math.cos(h) / scale * dt
            h = math.atan2(math.sin(h + (yaw - v * k * math.cos(h) / scale) * dt),
                           math.cos(h + (yaw - v * k * math.cos(h) / scale) * dt))
            if front or rear:
                slide_time += dt
            for mark in (t0_mark, t1_mark):
                if mark not in marks and s - start >= mark:
                    marks[mark] = t + dt
            max_n, max_h = max(max_n, abs(n)), max(max_h, abs(h))
            min_margin = min(min_margin, at('half_width_left_m', s) - n, at('half_width_right_m', s) + n)
            t += dt
        if min_margin < 0 or (v < .05 and t > 10):
            break
    lap = marks[t1_mark] - marks[t0_mark] if t1_mark in marks and t0_mark in marks else None
    return dict(track=track, wet=wet, lat=lat, brake=brake, lag=lag, completed=s - start >= distance,
                lap_time=lap, seconds=t, max_n=max_n, max_heading=max_h, min_road_margin=min_margin,
                slide_fraction=slide_time / max(t, 1e-9), peak_lateral_utilisation=peak_util)


def matrix(mod, tracks=('parkland', 'highland', 'crossover'), wets=(0.0, .5, 1.0), cases=(NOMINAL, PESSIMISTIC)):
    return [drive(mod, track, wet, **case) for track in tracks for wet in wets for case in cases]


if __name__ == '__main__':
    import sys
    paths = sys.argv[1:] or ['baselines/v5/agent.py', 'agent/agent.py']
    for p in paths:
        m = load(ROOT / p, Path(p).parent.name)
        for r in matrix(m):
            print(f"{p:24s} {r['track']:9s} wet {r['wet']:.1f} lat {r['lat']:4.1f} lag {r['lag']:.2f}  "
                  f"lap {r['lap_time'] if r['lap_time'] is None else round(r['lap_time'], 1)}  "
                  f"max|n| {r['max_n']:.2f}  margin {r['min_road_margin']:.2f}  slide {r['slide_fraction']:.3f}  "
                  f"peakU {r['peak_lateral_utilisation']:.2f}  done {r['completed']}")
