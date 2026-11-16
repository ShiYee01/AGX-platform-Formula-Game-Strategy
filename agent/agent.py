"""V2 candidate: earlier tyre stops, gradual lane changes, distance-aware pace.

Uses public current observations and this seat's memory. No SDK/weights needed.
Platform races are still required to establish a performance improvement.
"""
import math

# Coordinate periods and pit entries, metres, from public data/*.mapdata.json.
TRACK_PITS = ((5795.843091436305, 5715.9313949727275),
              (6999.367280545016, 6609.507079216389),
              (5803.166698722436, 5355.31315011008))
TOP_SPEED = 55.0  # Candidate cap; V1 lateral/braking budgets are retained.


def clip(value, low, high):
    return max(low, min(high, value))


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def grip_budget(o):
    return ((1.0 - 0.45 * clip(o['weather']['track_wetness'], 0.0, 1.0))
            * max(0.65, 1.0 - 0.4 * max(o['self']['tyre_wear'])))


def plan_speed(o, offset=0.0):
    car, rows = o['self'], o['preview']['rows']
    lateral, braking = 10.0 * grip_budget(o), 7.0 * grip_budget(o)
    target = TOP_SPEED
    reserve = max(4.0, 0.30 * car['speed_mps'])
    # Future apex speeds constrain the approach through braking distance.
    # V1 additionally imposed the tightest of its first 3 apex speeds NOW.
    for row in rows:
        curve = abs(row[1]) / max(0.65, 1.0 - abs(offset) * abs(row[1]))
        apex_squared = lateral / max(curve, 0.0001)
        target = min(target, math.sqrt(apex_squared
                     + 2.0 * braking * max(0.0, row[0] - reserve)))
    # First preview point is ahead of the car. Extrapolate current curvature.
    current_curve = rows[0][1]
    if len(rows) > 1 and rows[1][0] > rows[0][0]:
        current_curve -= rows[0][0] * (rows[1][1] - rows[0][1]) / (rows[1][0] - rows[0][0])
    target = min(target, math.sqrt(lateral / max(abs(current_curve), 0.0001)))
    if abs(car['n'] - offset) > 2.5 or abs(angle(car['heading_error'])) > 0.30:
        target = min(target, 15.0)
    if abs(angle(car['heading_error'])) > 0.60:
        target = min(target, 8.0)
    if car['fuel_kg'] < 5.0:
        target = min(target, 27.0)
    return target


def pit_entry_distance(o):
    length = o['track_length_m']
    period, entry = min(TRACK_PITS, key=lambda pair: abs(pair[0] - length))
    if abs(period - length) > 1.0:
        return None  # Unknown circuit: never reuse another circuit's entry.
    return (entry - o['self']['s']) % length


class Driver:
    def __init__(self):
        self.phase = None
        self.last_time = None
        self.last_steer = 0.0
        self.offset = self.grid_n = 0.0
        self.start_distance = self.start_time = 0.0
        self.wet_race = False
        self.compounds = set()
        self.pass_seat = None
        self.pass_goal = 0.0
        self.diagnostic = {}

    def reset_phase(self, o):
        c = o['self']
        self.phase, self.last_time, self.last_steer = o['phase'], None, 0.0
        self.grid_n = c['n'] if self.phase == 'race' else 0.0
        self.offset = self.grid_n
        self.start_distance, self.start_time = c['distance'], o['time']
        self.wet_race, self.compounds = False, set()
        self.pass_seat, self.pass_goal = None, 0.0

    def choose_pit(self, o):
        c = o['self']
        compound = 'hard' if c['compound'] == 'medium' else 'medium'
        if o['phase'] != 'race':
            return False, compound
        self.wet_race |= o['weather']['condition'] == 'wet'
        self.compounds.add(c['compound'])
        if c['pit_state']:
            return False, compound
        required = not self.wet_race and len(self.compounds) < 2 and c['lap'] >= 1
        unhealthy = max(c['tyre_wear']) > 0.75 or max(c['tyre_temp_c']) > 130.0
        optional = unhealthy and o['laps_total'] - c['lap'] > 1
        distance = pit_entry_distance(o)
        # Start the mandatory stop on lap>=1, before a lapped car is flagged.
        return bool((required or optional) and (distance is None or distance <= 200.0)), compound

    @staticmethod
    def road_bounds(o):
        rows = o['preview']['rows'][:6]
        # Chosen geometric margins, not the official track-limit thresholds.
        return (-max(0.0, min(r[4] for r in rows) - 1.5),
                max(0.0, min(r[3] for r in rows) - 1.5))

    def clear_sweep(self, o, goal, rivals):
        c = o['self']
        low, high = sorted((c['n'], goal))
        horizon = min(3.0, abs(goal - c['n']) / 0.8 + 0.5)
        for r in rivals:
            if not low - 2.6 < c['n'] + r['dn'] < high + 2.6:
                continue
            gap = r['gap_m']
            future = gap + (r['speed_mps'] - c['speed_mps']) * horizon
            if min(gap, future) < 9.0 and max(gap, future) > -9.0:
                return False
        return True

    def traffic_plan(self, o, target, dt):
        c = o['self']
        if o['phase'] != 'race':
            self.offset = 0.0
            return target, 0.0
        rivals = [r for r in o['rivals'] if r['status'] == 'racing']
        low, high = self.road_bounds(o)
        progress = max(0.0, c['distance'] - self.start_distance)
        # Hold staggered grid lanes; blend toward centre only after 80 m.
        blend = clip((progress - 80.0) / 300.0, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        goal = clip(self.grid_n * (1.0 - blend), low, high)
        launching = progress < 380.0
        leaders = [r for r in rivals if 0.0 < r['gap_m'] < max(65.0, 2.5 * c['speed_mps'])
                   and abs(c['n'] + r['dn'] - self.offset) < 2.8]
        front = min(leaders, key=lambda r: r['gap_m'], default=None)
        straight = max(abs(row[1]) for row in o['preview']['rows'][:6]) < 0.002
        passing = next((r for r in rivals if r['seat'] == self.pass_seat), None)
        if passing is not None and passing['gap_m'] > -12.0 and straight and not launching:
            goal = clip(self.pass_goal, low, high)
        else:
            self.pass_seat = None
            if front and straight and not launching and 15.0 < front['gap_m'] < 50.0:
                for candidate in sorted((-2.5, 2.5), key=lambda n: abs(n - self.offset)):
                    if (low <= candidate <= high
                            and abs(c['n'] + front['dn'] - candidate) > 2.6
                            and self.clear_sweep(o, candidate, rivals)):
                        goal = self.pass_goal = candidate
                        self.pass_seat = front['seat']
                        break
        if abs(goal - self.offset) > 0.05 and not self.clear_sweep(o, goal, rivals):
            goal = self.offset  # Hold lane instead of merging into a car.
        self.offset = clip(self.offset + clip(goal - self.offset, -0.8 * dt, 0.8 * dt), low, high)
        for r in rivals:
            if not 0.0 < r['gap_m'] < max(65.0, 2.5 * c['speed_mps']):
                continue
            lateral = c['n'] + r['dn']
            if min(abs(lateral - c['n']), abs(lateral - self.offset)) >= 2.8:
                continue
            closing = max(0.0, c['speed_mps'] - r['speed_mps'])
            braking = 7.0 * grip_budget(o)
            gap = 5.0 + (0.55 + 0.20 * o['weather']['track_wetness']) * c['speed_mps']
            gap += closing * closing / (2.0 * braking)
            follow = max(0.0, r['speed_mps'] + 0.5 * (r['gap_m'] - gap))
            stop_cap = math.sqrt(max(0.0, r['speed_mps'] ** 2
                            + 2.0 * braking * max(0.0, r['gap_m'] - 5.0)))
            target = min(target, follow, stop_cap)
        return target, self.offset

    def act(self, o):
        if self.phase != o['phase']:
            self.reset_phase(o)
        c = o['self']
        dt = 1.0 / 30.0 if self.last_time is None else clip(o['time'] - self.last_time, 1.0 / 60.0, 0.1)
        self.last_time = o['time']
        target, offset = self.traffic_plan(o, plan_speed(o, self.offset), dt)
        target = min(target, plan_speed(o, offset))
        heading = angle(c['heading_error'])
        curve = o['preview']['rows'][0][1]
        desired = clip((3.4 * curve - 0.6 * heading - 0.04 * (c['n'] - offset)) / 0.30, -1.0, 1.0)
        steer = clip(desired, self.last_steer - 4.5 * dt, self.last_steer + 4.5 * dt)
        self.last_steer = steer
        error = target - c['speed_mps']
        throttle, brake = clip(error * 0.25, 0.0, 1.0), clip(-error * 0.15, 0.0, 1.0)
        if abs(heading) > 0.6:
            throttle = min(throttle, 0.25)
        if c['fuel_kg'] < 5.0:
            throttle = min(throttle, 0.65)
        elapsed = o['time'] - self.start_time
        if o['phase'] == 'race' and elapsed < 6.0 and o['weather']['track_wetness'] > 0.5:
            throttle = min(throttle, 0.65)
        straight = max(abs(row[1]) for row in o['preview']['rows'][:6]) < 0.002
        settled = abs(heading) < 0.10 and abs(c['n'] - offset) < 0.75
        stable = settled and not c.get('front_sliding', False) and not c.get('rear_sliding', False)
        boost = (straight and stable and (o['phase'] != 'race' or elapsed >= 6.0)
                 and error > 1.0 and c['ers_soc_mj'] > 1.5)
        pit, compound = self.choose_pit(o)
        if c['pit_state']:
            # Preserve the original engine-controlled pit transit contract.
            throttle, brake, steer, boost = 0.0, 0.0, 0.0, False
            self.last_steer = 0.0
        self.diagnostic = dict(target=target, offset=offset, wet_race=self.wet_race)
        return dict(throttle=throttle, brake=brake, steer=steer,
                    ers_mode=2 if boost else 1, aero_mode=0,
                    pit_request=bool(pit), pit_compound=compound)


_driver = Driver()


def init(context):
    global _driver
    _driver = Driver()


def act(observation):
    return _driver.act(observation)
