"""Compare your car with every rival using AGX public replays (no login needed).

Public endpoints (see https://agx.cool/#/docs/racing-starter "对局日志与回放"):
  /api/v1/python-submissions/competitions/{competition_id}/history?limit=100&offset=N
  /api/v1/python-submissions/recordings/{job_id}/{static|events|frames}

Replays are spectator data. Use them OFFLINE to calibrate what the car can do
(grip, braking, power, ERS/aero effects); never feed them to act() at runtime.

Usage (from the repository root; needs NumPy):
  python tools/replay_analysis.py history --out replays
  python tools/replay_analysis.py download --out replays --me <display name> --last 20
  python tools/replay_analysis.py summary  --out replays --me <display name>
  python tools/replay_analysis.py envelope --out replays --me <display name>
  python tools/replay_analysis.py corners  --out replays --me <display name>
  python tools/replay_analysis.py lapsplit --out replays --me <display name> [job_id ...]

Downloaded files stay in --out (keep it out of Git and out of the submission ZIP).
"""
import argparse
import collections
import gzip
import json
import os
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SITE = 'https://agx.cool/api/v1/python-submissions/'
RACING_COMPETITION = '1eb91364-950b-5725-aa4b-3501b16860ec'  # 多人赛车 (permanent)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ['t', 'lap', 's', 'x', 'y', 'speed_mps', 'throttle', 'brake', 'steer', 'g_lat', 'g_long',
          'ers_soc_mj', 'ers_mode', 'aero_mode', 'fuel_kg', 'pit_state', 'downforce_n', 'drag_n',
          'gear', 'rpm', 'position', 'ers_power_kw']
EXTRA = ['status', 'wear', 'ttemp', 'in_pit', 'compound']
STATUS = {'waiting': 0, 'racing': 1, 'finished': 2, 'retired': 3, 'disqualified': 4}


def _ctx():
    cafile = os.environ.get('SSL_CERT_FILE')
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


def _get(url, path=None):
    with urllib.request.urlopen(url, context=_ctx(), timeout=300) as r:
        if path is None:
            return r.read()
        with open(path + '.tmp', 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    os.replace(path + '.tmp', path)


def _open(path):
    with open(path, 'rb') as f:
        magic = f.read(2)
    return gzip.open(path, 'rb') if magic == b'\x1f\x8b' else open(path, 'rb')


def cmd_history(a):
    os.makedirs(a.out, exist_ok=True)
    items = []
    for off in range(0, 100000, 100):
        page = json.loads(_get(f'{SITE}competitions/{a.competition}/history?limit=100&offset={off}'))
        items += page['items']
        if len(page['items']) < 100:
            break
    items.sort(key=lambda m: m['created_at'])
    json.dump(items, open(os.path.join(a.out, 'history.json'), 'w'), ensure_ascii=False)
    print(f'{len(items)} matches saved to {a.out}/history.json')


def load_history(out):
    return {m['id']: m for m in json.load(open(os.path.join(out, 'history.json')))}


def my_jobs(out, me):
    hist = sorted(load_history(out).values(), key=lambda m: m['created_at'])
    return [m for m in hist if me in m['participants'] and m['state'] == 'succeeded']


def extract(job, out):
    if os.path.exists(os.path.join(out, job + '.npz')):
        return job
    paths = {}
    for kind in ('static', 'events', 'frames'):
        paths[kind] = os.path.join(out, f'{job}.{kind}')
        if not os.path.exists(paths[kind]):
            _get(f'{SITE}recordings/{job}/{kind}', paths[kind])
    with _open(paths['static']) as f:
        meta = [json.loads(line) for line in f if line.strip()][-1]['metadata']
    with _open(paths['events']) as f:
        events = [json.loads(line) for line in f if line.strip()]
    rows, phase, ticks = [], [], []
    with _open(paths['frames']) as f:
        for line in f:
            if not line.strip():
                continue
            fr = json.loads(line)
            cars = sorted(fr['cars'], key=lambda c: c['seat'])
            rows.append([[float(c.get(k) or 0.0) for k in FIELDS]
                         + [STATUS.get(c['status'], 9), max(c['tyre_wear']), max(c['tyre_temp_c']),
                            1.0 if c.get('in_pit') else 0.0,
                            {'soft': 0, 'medium': 1, 'hard': 2}.get(c.get('compound'), 3)] for c in cars])
            phase.append(0 if fr['phase'] == 'qualifying' else 1)
            ticks.append([fr['tick'], fr['time']])
    np.savez_compressed(os.path.join(out, job + '.npz'), data=np.array(rows, np.float32),
                        phase=np.array(phase, np.int8), ticks=np.array(ticks, np.float64),
                        fields=np.array(FIELDS + EXTRA))
    keep = {k: meta[k] for k in ('track', 'cars', 'forecast', 'phases', 'chapters', 'race_laps')}
    keep['track'] = {k: v for k, v in meta['track'].items() if k in ('id', 'length_m', 'aid_zones', 'sectors')}
    json.dump(dict(meta=keep, events=[e for e in events if e['kind'] != 'energy_mode']),
              open(os.path.join(out, job + '.json'), 'w'), ensure_ascii=False)
    for p in paths.values():
        os.remove(p)
    return job


def cmd_download(a):
    jobs = [m['id'] for m in my_jobs(a.out, a.me)[-a.last:]] + list(a.jobs)
    with ThreadPoolExecutor(a.workers) as ex:
        for job in ex.map(lambda j: extract(j, a.out), jobs):
            print('ok', job)


class Match:
    def __init__(self, out, job, hist, me):
        self.job, self.h = job, hist[job]
        z = np.load(os.path.join(out, job + '.npz'))
        self.data, self.phase, self.ticks = z['data'], z['phase'], z['ticks']
        self.ix = {k: i for i, k in enumerate(z['fields'])}
        j = json.load(open(os.path.join(out, job + '.json')))
        self.meta, self.events = j['meta'], j['events']
        self.track = self.meta['track']['id']
        self.me = self.h['participants'].index(me)
        self.winner = self.h['ranks'].index(0)

    def car(self, seat, phase=1):
        return self.data[self.phase == phase, seat]

    def wetness(self):
        wt = [(0, self.meta['forecast']['track_wetness'])] + sorted(
            (e['tick'], e['payload']['track_wetness']) for e in self.events if e['kind'] == 'weather_change')
        return np.interp(self.ticks[:, 0], [w[0] for w in wt], [w[1] for w in wt])

    def laps(self, seat):
        """Yield (lap_time, rows) for each complete clean race lap after the first."""
        r = self.car(seat); ix = self.ix; lap = r[:, ix['lap']]
        for n in range(1, int(lap.max())):
            m = (lap == n) & (r[:, ix['status']] == 1) & (r[:, ix['pit_state']] == 0)
            if m.sum() > 100:
                t = r[m, ix['t']]
                yield t[-1] - t[0], r[m]


def matches(a):
    hist = load_history(a.out)
    jobs = a.jobs or [m['id'] for m in my_jobs(a.out, a.me)
                      if os.path.exists(os.path.join(a.out, m['id'] + '.npz'))]
    return [Match(a.out, j, hist, a.me) for j in jobs]


def cmd_summary(a):
    print('time        track     wet  rank laps  my_best win_best  myQ   bestQ | vmax me/win  glat99 me/win '
          'brake p1 me/win  fullthr me/win  aero1 win  contacts')
    for m in matches(a):
        ix = m.ix
        q = {e['seats'][0]: e['payload'] for e in m.events if e['kind'] == 'qualifying_lap'}
        contacts = collections.Counter(s for e in m.events if e['kind'] == 'incident' and e['phase'] == 'race'
                                       for s in e['seats'])
        wet = m.wetness()[m.phase == 1].mean()
        stats = {}
        for seat in (m.me, m.winner):
            r = m.car(seat); alive = r[:, ix['status']] == 1
            best = min((t for t, _ in m.laps(seat)), default=float('nan'))
            stats[seat] = (best, r[alive, ix['speed_mps']].max() * 3.6,
                           np.percentile(np.abs(r[alive, ix['g_lat']]), 99), np.percentile(r[alive, ix['g_long']], 1),
                           (r[alive, ix['throttle']] > .98).mean(), (r[alive, ix['aero_mode']] == 1).mean(),
                           int(r[:, ix['lap']].max()))
        me, w = stats[m.me], stats[m.winner]
        mq = q.get(m.me); bq = min((p['lap_time_s'] for p in q.values() if p['valid']), default=float('nan'))
        mqs = f"{mq['lap_time_s']:6.1f}" if mq and mq['valid'] else ' 无效 '
        print(f"{m.h['created_at'][5:16]} {m.track:9s} {wet:4.2f} {m.h['ranks'][m.me]+1:4d} {me[6]:4d} "
              f"{me[0]:8.1f} {w[0]:8.1f} {mqs} {bq:6.1f} | {me[1]:4.0f}/{w[1]:4.0f} {me[2]:5.2f}/{w[2]:5.2f} "
              f"{me[3]:6.2f}/{w[3]:6.2f} {me[4]:5.2f}/{w[4]:5.2f} {w[5]:5.2f} {contacts.get(m.me, 0):4d}")


def cmd_envelope(a):
    rows = []
    for m in matches(a):
        ix = m.ix; wet = m.wetness()
        for seat in range(m.data.shape[1]):
            c = m.data[:, seat]
            ok = (c[:, ix['status']] == 1) & (c[:, ix['in_pit']] == 0) & (c[:, ix['pit_state']] == 0)
            rows.append(np.column_stack([wet[ok], c[ok, ix['speed_mps']], c[ok, ix['g_lat']],
                                         c[ok, ix['g_long']], np.full(ok.sum(), seat == m.me)]))
    S = np.vstack(rows)
    wb = [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]
    vb = [(15, 30), (30, 45), (45, 60), (60, 75), (75, 95)]
    for who, mask in (('ALL cars', np.ones(len(S), bool)), ('ME', S[:, 4] == 1)):
        print(f'\n== {who}: |g_lat| p99.5 / braking g_long p0.5 / accel g_long p99.5 (g)')
        for w0, w1 in wb:
            line = f'wet {w0:.1f}-{min(w1, 1):.1f}:'
            for v0, v1 in vb:
                k = mask & (S[:, 0] >= w0) & (S[:, 0] < w1) & (S[:, 1] >= v0) & (S[:, 1] < v1)
                line += (f'  {v0}-{v1}m/s ' + ('  ---         ' if k.sum() < 200 else
                         f'{np.percentile(np.abs(S[k, 2]), 99.5):.2f}/{np.percentile(S[k, 3], .5):.2f}/'
                         f'{np.percentile(S[k, 3], 99.5):.2f}'))
            print(line)


def map_corners(track):
    d = json.load(open(os.path.join(REPO, f'data/racing-{track}.mapdata.json')))
    s = np.array([p['s_m'] for p in d['centerline']])
    k = np.array([p['curvature_1pm'] for p in d['centerline']])
    hot = np.abs(k) > 1 / 150.0
    segs, i = [], 0
    while i < len(s):
        if hot[i]:
            j = i
            while j + 1 < len(s) and hot[j + 1]:
                j += 1
            peak = np.abs(k[i:j + 1]).max()
            if segs and s[i] - segs[-1][1] < 40:
                segs[-1] = (segs[-1][0], s[j], max(segs[-1][2], peak))
            else:
                segs.append((s[i], s[j], peak))
            i = j + 1
        else:
            i += 1
    return d, s, k, segs


def cmd_corners(a):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in matches(a):
        segs = map_corners(m.track)[3]
        for who, seat in (('me', m.me), ('win', m.winner)):
            for _, r in m.laps(seat):
                s, v = r[:, m.ix['s']], r[:, m.ix['speed_mps']]
                out[m.track][who].append([v[(s >= a0) & (s <= a1)].min() * 3.6 if ((s >= a0) & (s <= a1)).any()
                                          else np.nan for a0, a1, _ in segs])
    for track, d in out.items():
        segs = map_corners(track)[3]
        me = np.nanmedian(np.array(d['me']), 0); win = np.nanmedian(np.array(d['win']), 0)
        print(f'\n### {track}: corner s-range, centreline min radius, median min speed km/h me / winner')
        for (s0, s1, k), x, y in zip(segs, me, win):
            print(f'  {s0:6.0f}-{s1:6.0f} m  R{1 / k:6.1f} m   me {x:6.1f}  win {y:6.1f}  gap {y - x:+6.1f}')


def cmd_lapsplit(a):
    for m in matches(a):
        d, sg, k, _ = map_corners(m.track)
        L = d['length_m']
        corner = np.abs(k) > 1 / 150
        approach = np.zeros_like(corner)
        for i in np.where(corner)[0]:
            approach |= (sg >= sg[i] - 150) & (sg < sg[i])
        cls = np.where(corner, 0, np.where(approach & ~corner, 1, 2))
        res = {}
        for who, seat in (('me', m.me), ('win', m.winner)):
            best = min(m.laps(seat), key=lambda x: x[0], default=None)
            if best is None:
                break
            r = best[1]; s = r[:, m.ix['s']]; o = np.argsort(s)
            v = np.interp(sg, s[o], np.maximum(r[o, m.ix['speed_mps']], 1.0))
            res[who] = (best[0], np.diff(np.append(sg, L)) / v)
        if len(res) < 2:
            continue
        print(f"\n### {m.track} {m.h['created_at'][:16]}  best lap me {res['me'][0]:.1f}s  winner {res['win'][0]:.1f}s")
        for c, name in ((0, 'corners'), (1, 'braking (150 m before)'), (2, 'straights')):
            x, y = res['me'][1][cls == c].sum(), res['win'][1][cls == c].sum()
            print(f'  {name:24s} me {x:6.1f}s  winner {y:6.1f}s  loss {x - y:6.1f}s')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['history', 'download', 'summary', 'envelope', 'corners', 'lapsplit'])
    p.add_argument('jobs', nargs='*', help='explicit job ids (default: your downloaded matches)')
    p.add_argument('--out', default='replays')
    p.add_argument('--me', help='your public display name in the match history')
    p.add_argument('--last', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--competition', default=RACING_COMPETITION)
    a = p.parse_args()
    if a.command != 'history' and not a.me:
        sys.exit('--me is required')
    globals()['cmd_' + a.command](a)


if __name__ == '__main__':
    main()
