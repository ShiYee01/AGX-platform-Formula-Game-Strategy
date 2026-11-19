"""V6 candidate: V5 control structure with grip, braking and power budgets
raised to about 70% of the limits measured from public replays of all cars.

Uses public current observations and this seat's memory. No SDK/weights needed.
Platform races are still required to establish a performance improvement.
"""
import math

# Coordinate periods and pit entries, metres, from public data/*.mapdata.json.
TRACK_PITS = ((5795.843091436305, 5715.9313949727275),
              (6999.367280545016, 6609.507079216389),
              (5803.166698722436, 5355.31315011008))
# Public replays (21 races, all cars): dry |g_lat| ~2.5 g, braking ~2.6-3.0 g,
# top speed ~92 m/s; full wet scales grip by ~0.56, matching grip_budget().
# V6 plans at roughly 70% of those limits. These are chosen budgets, not
# official physics constants.
TOP_SPEED = 90.0  # 324 km/h candidate cap, subject to the full braking profile.
BRAKING_BUDGET = 14.0  # Chosen m/s^2 before wet/wear scaling (V5: 5.5).
ACCEL_BUDGET = 12.0  # Chosen traction-limited m/s^2 before scaling (V5: 5).
FALLBACK_BRAKING = 5.5  # Unknown map: keep the conservative V5 preview path.
_PROFILES = {}
# Online grip trust in [GRIP_SCALE_MIN, 1]; lowered by observed sliding.
GRIP_SCALE_MIN = 0.70
_GRIP_SCALE = [1.0]

# Traffic geometry only: a chosen lane must clear the existing 2.8 m guard.
TRAFFIC_CLEARANCE = 2.8
PASS_LANE_MARGIN = 0.4
PASS_LOST_HOLD_S = 1.0


def clip(value, low, high):
    return max(low, min(high, value))


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


# V5 used 0.45. Replays: in heavy rain (wetness >= 0.75) no clean lap reached
# the 0.45-scaled V6 speed in about half of the Highland/Crossover bends, so V6
# trusts less wet grip. Dry pace is unaffected.
WET_GRIP_LOSS = 0.55


def grip_budget(o):
    return ((1.0 - WET_GRIP_LOSS * clip(o['weather']['track_wetness'], 0.0, 1.0))
            * max(0.65, 1.0 - 0.4 * max(o['self']['tyre_wear'])) * _GRIP_SCALE[0])


def interpolate(values, station, length):
    q = (station % length) * len(values) / length
    i = int(q)
    return values[i] + (values[(i + 1) % len(values)] - values[i]) * (q - i)


def lateral_budget(curve):
    # Candidate budgets, about 70% of the replay envelope (~24.5 m/s^2).
    # Hairpins (radius < 20 m) keep a smaller budget: they are the bends
    # where V3 lost the car and the measured gap to the winners is smallest.
    return 13.0 if abs(curve) > 0.05 else 17.0


def map_state(o):
    length, curves, grades = min(MAP_DATA, key=lambda t: abs(t[0] - o['track_length_m']))
    if abs(length - o['track_length_m']) > 1.0:
        return None
    station = o['self']['s']
    # Refuse stale geometry when public live previews disagree with the map.
    for row in o['preview']['rows'][:3]:
        if abs(interpolate(curves, station + row[0], length) - row[1]) > 0.00012:
            return None
    if length not in _PROFILES:
        ds = length / len(curves)
        profiles = []
        for lane in range(8):
            limits = [min(1000000.0, lateral_budget(k) * max(0.35, 1.0 - lane * abs(k))
                          / max(abs(k), 0.000001)) for k in curves]
            # Two circular sweeps propagate braking constraints across start.
            for _ in range(2):
                for i in range(len(limits) - 1, -1, -1):
                    limits[i] = min(limits[i], limits[(i + 1) % len(limits)] + 2 * BRAKING_BUDGET * ds)
            profiles.append(limits)
        _PROFILES[length] = profiles
    return length, curves, grades, _PROFILES[length]


def road_plan(o, offset=0.0, state=None):
    c, rows = o['self'], o['preview']['rows']
    grip = grip_budget(o)
    if state is None:
        state = map_state(o)
    if state is not None:
        length, curves, grades, profiles = state
        # Conservative outer envelope during a lane transition, including
        # the actual position if it has not reached the requested lane yet.
        lane = min(7, int(math.ceil(max(abs(offset), abs(c['n'])))))
        profile = profiles[lane]
        reserve = max(3.0, 0.20 * c['speed_mps'])
        def squared(station):
            k = interpolate(curves, station, length)
            local = lateral_budget(k) * max(0.35, 1.0 - lane * abs(k)) / max(abs(k), 0.000001)
            return min(TOP_SPEED ** 2, grip * local,
                       grip * interpolate(profile, station + reserve, length))
        target2 = squared(c['s'])
        target = math.sqrt(target2)
        feed_accel = clip((squared(c['s'] + 4.0) - target2) / 8.0,
                          -BRAKING_BUDGET * grip, ACCEL_BUDGET * grip)
    else:
        # Unknown or changed map: keep a conservative live-preview fallback.
        target = 45.0
        for row in rows:
            k = abs(row[1]) / max(0.35, 1.0 - abs(offset) * abs(row[1]))
            target = min(target, math.sqrt(10.0 * grip / max(k, .0001)
                         + 2 * FALLBACK_BRAKING * grip * max(0.0, row[0] - 10.0)))
        target = min(target, math.sqrt(10.0 * grip / max(abs(rows[0][1]), .0001)))
        feed_accel = 0.0
    heading = abs(angle(c['heading_error']))
    error = abs(c['n'] - offset)
    if error > 1.5:
        target = min(target, max(4.0, 15.0 - 2.0 * (error - 1.5)))
        feed_accel = min(0.0, feed_accel)
    if heading > .30:
        target = min(target, 10.0)
        feed_accel = min(0.0, feed_accel)
    if heading > .60:
        target = min(target, 5.0)
    width = min(rows[0][3:5])
    if abs(c['n']) > width - 1.5:
        target = min(target, 6.0)
        feed_accel = min(0.0, feed_accel)
    if c['fuel_kg'] < 5.0:
        target = min(target, 27.0)
    return target, feed_accel


def plan_speed(o, offset=0.0):
    return road_plan(o, offset)[0]


def steer_command(o, offset, state):
    c = o['self']
    if state is not None:
        length, curves, _, _ = state
        # A short actuator lead replaces the old fixed 15 m anticipation.
        k = interpolate(curves, c['s'] + min(4.0, .10 * c['speed_mps']), length)
    else:
        k = o['preview']['rows'][0][1]
    k /= max(.35, 1.0 - offset * k)
    cross = math.atan2(1.1 * (c['n'] - offset), c['speed_mps'] + 5.0)
    return clip((math.atan(3.4 * k) - .90 * angle(c['heading_error']) - cross) / .30, -1.0, 1.0)


def pedals(o, target, feed_accel, state, mode, allow_extra=False, offset=0.0):
    c = o['self'];error = target - c['speed_mps']
    force = c.get('drive_force_n', [])
    mass = c.get('effective_mass_kg', 0.0)
    if len(force) != 4 or not mass > 0.0 or not force[mode] > 0.0:
        return clip(.25 * error, 0.0, 1.0), clip(-.15 * error, 0.0, 1.0)
    grade = interpolate(state[2], c['s'], state[0]) if state is not None else o['preview']['rows'][0][2]
    resistance = c.get('drag_n', 0.0) + c.get('rolling_resistance_n', 0.0) + mass * 9.81 * grade
    grip = grip_budget(o)
    curve = interpolate(state[1], c['s'], state[0]) if state is not None else o['preview']['rows'][0][1]
    curve /= max(.35, 1.0 - c['n'] * curve)
    lateral = max(abs(c.get('g_lat', 0.0)) * 9.81, c['speed_mps'] ** 2 * abs(curve))
    utilization = lateral / max(.1, lateral_budget(curve) * grip)
    remaining = math.sqrt(max(0.0, 1.0 - min(1.0, utilization) ** 2))
    accel_cap = ACCEL_BUDGET * grip * remaining
    # Stable, nearly straight and well below target: the replay winners spend
    # 30-70% of a lap at full throttle. Only in the conditions V5 already
    # required for its extra release (map trusted, on line, no sliding).
    full = (allow_extra and state is not None and error > 2.0
            and utilization < .5
            and abs(angle(c['heading_error'])) < .10
            and abs(c['n'] - offset) < .75
            and abs(c['n']) < min(o['preview']['rows'][0][3:5]) - 1.7
            and c['fuel_kg'] >= 5.0
            and (c['speed_mps'] > 20.0 or o['weather']['track_wetness'] < .5)
            and not c.get('front_sliding', True) and not c.get('rear_sliding', True))
    if c.get('rear_sliding', False) or c.get('front_sliding', False):
        accel_cap = min(accel_cap, .8 * grip)
    accel = clip(1.6 * error + feed_accel, -(BRAKING_BUDGET + 4.0) * grip, accel_cap)
    requested_force = mass * accel + resistance
    if full:
        throttle, brake = 1.0, 0.0
    elif requested_force >= 0.0:
        throttle, brake = clip(requested_force / force[mode], 0.0, 1.0), 0.0
    else:
        throttle = 0.0
        brake = clip(-requested_force / max(1.0, c.get('brake_force_max_n', 56000.0)), 0.0, 1.0)
    if c.get('rear_sliding', False):
        throttle = min(throttle, .25)
    # Above the requested speed, never add propulsion while recovering.
    if error < -.5 and (abs(angle(c['heading_error'])) > .30 or abs(c['n']) > min(o['preview']['rows'][0][3:5]) - 1.5):
        throttle = 0.0
        brake = max(brake, clip(-error * .04, 0.0, .35))
    return throttle, brake


def is_lapping_rival(o, rival):
    if 'lap' not in rival:
        return False
    c = o['self'];length = o['track_length_m']
    # Lap counters alone are ambiguous when one car has just crossed start.
    station = (c['s'] + rival['gap_m']) % length
    progress_ahead = (rival['lap'] - c['lap']) * length + station - c['s']
    return progress_ahead > .5 * length


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
        self.pass_seen_time = None
        self.diagnostic = {}
        self.last_unstable_time = None
        self.yield_seat = None
        self.yield_goal = 0.0
        self.yield_seen = self.yield_started = None
        self.yield_state = None
        self.yield_near = False
        self.rival_motion = {}
        self.last_brake = 0.0
        _GRIP_SCALE[0] = 1.0

    def update_grip_scale(self, o, dt):
        # If this car slides in a bend or under braking, trust less grip for a
        # while (all speed targets and budgets scale with grip_budget). Pure
        # straight-line wheelspin is already handled by the throttle limits.
        c = o['self']
        loaded = abs(c.get('g_lat', 0.0)) > .5 or self.last_brake > .05
        sliding = (c.get('front_sliding', False)
                   or (c.get('rear_sliding', False) and loaded))
        if sliding and c['speed_mps'] > 15.0 and not c['pit_state']:
            _GRIP_SCALE[0] = max(GRIP_SCALE_MIN, _GRIP_SCALE[0] - .15 * dt)
        else:
            _GRIP_SCALE[0] = min(1.0, _GRIP_SCALE[0] + .01 * dt)

    def reset_phase(self, o):
        c = o['self']
        self.phase, self.last_time, self.last_steer = o['phase'], None, 0.0
        self.grid_n = c['n'] if self.phase == 'race' else 0.0
        self.offset = self.grid_n
        self.start_distance, self.start_time = c['distance'], o['time']
        self.wet_race, self.compounds = False, set()
        self.pass_seat, self.pass_goal = None, 0.0
        self.pass_seen_time = None
        self.last_unstable_time = o['time']
        self.yield_seat = None
        self.yield_goal = self.offset
        self.yield_seen = self.yield_started = None
        self.yield_state = None
        self.yield_near = False
        self.rival_motion = {}
        self.last_brake = 0.0
        _GRIP_SCALE[0] = 1.0

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
        # Cover the full requested lane change and settling time. The target
        # still moves at 0.8 m/s, so a 3.2 m move takes over three seconds.
        horizon = abs(goal - c['n']) / 0.8 + 0.5
        for r in rivals:
            if not low - TRAFFIC_CLEARANCE < c['n'] + r['dn'] < high + TRAFFIC_CLEARANCE:
                continue
            gap = r['gap_m']
            future = gap + (r['speed_mps'] - c['speed_mps']) * horizon
            if min(gap, future) < 9.0 and max(gap, future) > -9.0:
                return False
        return True

    def observe_rivals(self, o, rivals):
        for r in rivals:
            lane = o['self']['n'] + r['dn']
            previous = self.rival_motion.get(r['seat'])
            rate = 0.0
            first_seen = o['time']
            if previous is not None:
                dt = o['time'] - previous[0]
                if 0.0 < dt < .5:
                    first_seen = previous[3] if len(previous) > 3 else previous[0]
                    measured = clip((lane - previous[1]) / dt, -2.0, 2.0)
                    rate = .75 * previous[2] + .25 * measured
            self.rival_motion[r['seat']] = (o['time'], lane, rate, first_seen)
        self.rival_motion = {seat: value for seat, value in self.rival_motion.items()
                             if o['time'] - value[0] <= 2.0}

    def clear_yield_sweep(self, o, goal, rivals):
        if not self.clear_sweep(o, goal, rivals):
            return False
        c = o['self']
        duration = abs(goal - c['n']) / .8 + .75
        low, high = sorted((c['n'], goal))
        for r in rivals:
            gap = r['gap_m']
            future_gap = gap + (r['speed_mps'] - c['speed_mps']) * duration
            if min(gap, future_gap) >= 9.0 or max(gap, future_gap) <= -9.0:
                continue
            lane = c['n'] + r['dn']
            rate = self.rival_motion.get(r['seat'], (0., 0., 0.))[2]
            future_lane = lane + clip(rate * duration, -1.0, 1.0)
            if (min(lane, future_lane) < high + TRAFFIC_CLEARANCE
                    and max(lane, future_lane) > low - TRAFFIC_CLEARANCE):
                return False
        return True

    def lapping_plan(self, o, rivals, launching):
        c = o['self']
        self.yield_near = False
        if launching or c['pit_state']:
            self.yield_seat = self.yield_state = None
            return None
        leader = next((r for r in rivals if r['seat'] == self.yield_seat), None)
        if self.yield_seat is not None:
            if leader is None:
                if self.yield_seen is not None and o['time'] - self.yield_seen <= 2.0:
                    self.yield_state = 'lost_hold'
                    return self.yield_goal
                self.yield_seat = self.yield_state = None
            elif (not is_lapping_rival(o, leader)
                    or leader['gap_m'] >= 20.0 or leader['gap_m'] < -400.0
                    or (o['time'] - self.yield_started > 20.0
                        and leader['gap_m'] < -60.0
                        and leader['speed_mps'] - c['speed_mps'] <= 1.0)):
                self.yield_seat = self.yield_state = None
                leader = None
        if self.yield_seat is None:
            candidates = []
            for r in rivals:
                closing = r['speed_mps'] - c['speed_mps']
                if (is_lapping_rival(o, r) and -400.0 < r['gap_m'] < 0.0
                        and closing > 2.0):
                    ttc = -r['gap_m'] / closing
                    if ttc < 14.0:
                        candidates.append((ttc, r['seat'], r))
            if not candidates:
                return None
            leader = min(candidates, key=lambda x: (x[0], x[1]))[2]
            self.yield_seat = leader['seat']
            self.yield_goal = self.offset
            self.yield_started = o['time']
            self.yield_state = 'watch'
        self.yield_seen = o['time']
        gap = leader['gap_m']
        closing = leader['speed_mps'] - c['speed_mps']
        ttc = -gap / closing if closing > .5 and gap < 0.0 else float('inf')
        lane = c['n'] + leader['dn']
        self.yield_near = -30.0 < gap < 20.0
        if self.yield_state in ('move', 'hold', 'lost_hold'):
            self.yield_state = 'hold' if abs(c['n'] - self.yield_goal) < .6 else 'move'
            return self.yield_goal
        # When lanes already clear, preserve that clearance instead of moving
        # unpredictably in front of a fast car. A late threat also means hold.
        if abs(lane - c['n']) >= TRAFFIC_CLEARANCE or ttc < 3.0 or gap >= -15.0:
            self.yield_state = 'hold'
            return self.yield_goal
        motion = self.rival_motion.get(leader['seat'])
        if motion is None or o['time'] - motion[3] < .25:
            return self.yield_goal  # Observe motion before choosing a side.
        state = map_state(o)
        if (state is None or abs(angle(c['heading_error'])) > .10
                or c.get('front_sliding', True) or c.get('rear_sliding', True)
                or abs(interpolate(state[1], c['s'], state[0])) >= .002):
            return self.yield_goal
        rate = self.rival_motion.get(leader['seat'], (0., 0., 0.))[2]
        projected_lane = lane + clip(rate * min(ttc, 5.0), -1.0, 1.0)
        spacing = TRAFFIC_CLEARANCE + PASS_LANE_MARGIN
        options = (min(lane, projected_lane) - spacing,
                   max(lane, projected_lane) + spacing)
        for goal in sorted(options, key=lambda n: abs(n - c['n'])):
            duration = abs(goal - c['n']) / .8 + .75
            if ttc <= duration + 1.0:
                continue
            distance = max(30.0, c['speed_mps'] * (duration + .5))
            rows = [r for r in o['preview']['rows'] if r[0] <= distance + 15.0]
            if (distance > o['preview']['rows'][-1][0] - 15.0
                    or max(abs(r[1]) for r in rows) >= .002):
                continue
            low = -max(0.0, min(r[4] for r in rows) - 1.5)
            high = max(0.0, min(r[3] for r in rows) - 1.5)
            if low <= goal <= high and self.clear_yield_sweep(o, goal, rivals):
                self.yield_goal = goal
                self.yield_state = 'move'
                break
        return self.yield_goal

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
        self.observe_rivals(o, rivals)
        yield_goal = self.lapping_plan(o, rivals, launching)
        leaders = [r for r in rivals if 0.0 < r['gap_m'] < max(65.0, 2.5 * c['speed_mps'])
                   and abs(c['n'] + r['dn'] - self.offset) < TRAFFIC_CLEARANCE]
        front = min(leaders, key=lambda r: r['gap_m'], default=None)
        straight = max(abs(row[1]) for row in o['preview']['rows'][:10]) < 0.002
        passing = next((r for r in rivals if r['seat'] == self.pass_seat), None)
        if passing is not None:
            self.pass_seen_time = o['time']
        holding_lost_target = (self.pass_seat is not None and passing is None
                               and self.pass_seen_time is not None
                               and o['time'] - self.pass_seen_time <= PASS_LOST_HOLD_S)
        if yield_goal is not None:
            goal = clip(yield_goal, low, high)
            self.pass_seat = None
            self.pass_seen_time = None
        elif (passing is not None and -12.0 < passing['gap_m'] < 65.0) or holding_lost_target:
            # Hold the side through a bend or a brief visibility gap. Returning
            # to centre waits for clearance. Abandon a target pulling far away;
            # the same sweep guard still blocks an unsafe return merge.
            goal = clip(self.pass_goal, low, high)
        else:
            self.pass_seat = None
            self.pass_seen_time = None
            if front and straight and not launching and 15.0 < front['gap_m'] < 50.0:
                front_lane = c['n'] + front['dn']
                spacing = TRAFFIC_CLEARANCE + PASS_LANE_MARGIN
                candidates = (front_lane - spacing, front_lane + spacing)
                for candidate in sorted(candidates, key=lambda n: abs(n - self.offset)):
                    if (low <= candidate <= high
                            and abs(front_lane - candidate) > TRAFFIC_CLEARANCE
                            and self.clear_sweep(o, candidate, rivals)):
                        goal = self.pass_goal = candidate
                        self.pass_seat = front['seat']
                        self.pass_seen_time = o['time']
                        break
        sweep_clear = (self.clear_yield_sweep(o, goal, rivals) if yield_goal is not None
                       else self.clear_sweep(o, goal, rivals))
        if abs(goal - self.offset) > 0.05 and not sweep_clear:
            goal = self.offset  # Hold lane instead of merging into a car.
        self.offset = clip(self.offset + clip(goal - self.offset, -0.8 * dt, 0.8 * dt), low, high)
        for r in rivals:
            if not 0.0 < r['gap_m'] < max(65.0, 2.5 * c['speed_mps']):
                continue
            lateral = c['n'] + r['dn']
            if min(abs(lateral - c['n']), abs(lateral - self.offset)) >= TRAFFIC_CLEARANCE:
                continue
            closing = max(0.0, c['speed_mps'] - r['speed_mps'])
            braking = 7.0 * grip_budget(o)
            gap = 5.0 + (0.55 + 0.20 * o['weather']['track_wetness']) * c['speed_mps']
            gap += closing * closing / (2.0 * braking)
            follow = max(0.0, r['speed_mps'] + 0.5 * (r['gap_m'] - gap))
            stop_cap = math.sqrt(max(0.0, r['speed_mps'] ** 2
                            + 2.0 * braking * max(0.0, r['gap_m'] - 5.0)))
            target = min(target, follow, stop_cap)
        # Once safely to the side, stop accelerating during the close pass.
        # This requests the current speed, never an extra braking step.
        leader = next((r for r in rivals if r['seat'] == self.yield_seat), None)
        if (leader is not None and self.yield_near
                and abs(c['n'] - self.yield_goal) < .6
                and abs(leader['dn']) >= TRAFFIC_CLEARANCE):
            target = min(target, c['speed_mps'])
        return target, self.offset

    def act(self, o):
        if self.phase != o['phase']:
            self.reset_phase(o)
        c = o['self']
        dt = 1.0 / 30.0 if self.last_time is None else clip(o['time'] - self.last_time, 1.0 / 60.0, 0.1)
        self.last_time = o['time']
        self.update_grip_scale(o, dt)
        state = map_state(o)
        initial_road, _ = road_plan(o, self.offset, state)
        target, offset = self.traffic_plan(o, initial_road, dt)
        road_target, feed_accel = road_plan(o, offset, state)
        target = min(target, road_target)
        if target < road_target - .1:
            feed_accel = 0.0
        heading = angle(c['heading_error'])
        desired = steer_command(o, offset, state)
        steer = clip(desired, self.last_steer - 4.5 * dt, self.last_steer + 4.5 * dt)
        self.last_steer = steer
        error = target - c['speed_mps']
        throttle, brake = 0.0, 0.0
        elapsed = o['time'] - self.start_time
        straight = max(abs(row[1]) for row in o['preview']['rows'][:6]) < 0.002
        settled = abs(heading) < 0.10 and abs(c['n'] - offset) < 0.75
        stable = settled and not c.get('front_sliding', False) and not c.get('rear_sliding', False)
        boost = (straight and stable and (o['phase'] != 'race' or elapsed >= 6.0)
                 and (error > .5 or feed_accel > 1.0) and c['ers_soc_mj'] > 1.5)
        mode = 2 if boost else 1
        if (not stable or state is None):
            self.last_unstable_time = o['time']
        recovery_done = (self.last_unstable_time is not None
                         and o['time'] - self.last_unstable_time >= .75)
        allow_extra = (stable and recovery_done and target >= road_target - .1
                       and (o['phase'] != 'race' or elapsed >= 6.0))
        throttle, brake = pedals(o, target, feed_accel, state, mode,
                                 allow_extra=allow_extra, offset=offset)
        if abs(heading) > .60:
            throttle = min(throttle, .20)
        if c['fuel_kg'] < 5.0:
            throttle = min(throttle, .65)
        if o['phase'] == 'race' and elapsed < 6.0 and o['weather']['track_wetness'] > .5:
            throttle = min(throttle, .65)
        pit, compound = self.choose_pit(o)
        if c['pit_state']:
            # Preserve the original engine-controlled pit transit contract.
            throttle, brake, steer, boost = 0.0, 0.0, 0.0, False
            self.last_steer = 0.0
        self.last_brake = brake
        self.diagnostic = dict(target=target, offset=offset, wet_race=self.wet_race,
                               grip_scale=_GRIP_SCALE[0],
                               road_target=road_target, feed_accel=feed_accel, map_valid=state is not None,
                               extra_accel_enabled=allow_extra,
                               yield_seat=self.yield_seat, yield_state=self.yield_state,
                               yield_goal=self.yield_goal)
        return dict(throttle=throttle, brake=brake, steer=steer,
                    ers_mode=2 if boost else 1, aero_mode=0,
                    pit_request=bool(pit), pit_compound=compound)


# BEGIN PUBLIC MAP DATA (generated by tools/embed_maps.py; curvature 1/m, grade m/m)
MAP_DATA = (
    # parkland: 2898 uniform samples; public map SHA256 e6bcbac9c86b8ebd01d37a5daa1b6984048472eb8bff842fc953e0032aa2ecc8
    (5795.843091436305,
     (
      -0.0002973,-0.0002198,-0.0003206,-0.0003003,-0.000278,-0.0003899,-0.0002598,-0.0003731,-0.0003477,-0.0003359,
      -0.0004084,-0.000282,-0.0004119,-0.0003218,-0.0004009,-0.0003244,-0.0004014,-0.0003202,-0.0004181,-0.0004081,
      -0.0003487,-0.0004478,-0.0004101,-0.0004616,-0.0004441,-0.0004621,-0.0005573,-0.0004586,-0.0006081,-0.0005589,
      -0.0006071,-0.0006119,-0.000696,-0.000654,-0.0007378,-0.0007462,-0.0007772,-0.0007944,-0.0008467,-0.0009179,
      -0.0008834,-0.0009506,-0.0010127,-0.001004,-0.001119,-0.0011254,-0.0011311,-0.0012566,-0.0011986,-0.0012998,
      -0.0012578,-0.0013315,-0.0013646,-0.0013293,-0.0012737,-0.0013891,-0.0012824,-0.0012949,-0.0012016,-0.0012627,
      -0.0011215,-0.0011561,-0.0010298,-0.0010413,-0.0009717,-0.0009397,-0.000838,-0.0008162,-0.0007306,-0.0007585,
      -0.0006209,-0.000657,-0.0005264,-0.0005382,-0.0004988,-0.0004679,-0.0003599,-0.0004827,-0.0002758,-0.0003843,
      -0.0002773,-0.0003275,-0.000219,-0.000319,-0.0001473,-0.0003014,-0.0001393,-0.0002164,-0.0001891,-8.7e-05,
      -0.0002245,-5.93e-05,-0.0001508,-8.34e-05,-0.0001277,-2.7e-06,-0.0001603,3.24e-05,-0.0001192,-3e-06,
      -1.84e-05,-7.27e-05,1.09e-05,1.33e-05,-4.9e-05,1.84e-05,-9.5e-06,4.2e-06,-7.5e-06,4.1e-05,
      -2.26e-05,6.5e-05,-8.83e-05,0.0001244,-9.64e-05,0.0001129,-7.29e-05,4.53e-05,4.03e-05,-5.74e-05,
      4.06e-05,4.36e-05,-6.58e-05,7.1e-05,-6.95e-05,5.84e-05,-1.56e-05,4.2e-06,-1.1e-06,4e-07,
      -4e-07,1.1e-06,-4.2e-06,1.57e-05,-5.86e-05,6.97e-05,-7.18e-05,6.87e-05,-5.45e-05,7e-07,
      5.14e-05,-5.72e-05,2.89e-05,-5.86e-05,5.65e-05,-1.87e-05,1.86e-05,-5.58e-05,5.57e-05,-1.85e-05,
      1.85e-05,-5.55e-05,5.46e-05,-1.44e-05,3.5e-06,4e-07,-5.3e-06,5.4e-06,-1.2e-06,-7e-07,
      4.1e-06,-1.57e-05,5.87e-05,-7.01e-05,7.31e-05,-7.38e-05,7.38e-05,-7.28e-05,6.9e-05,-5.46e-05,
      1.1e-06,5e-05,-5.21e-05,1e-05,1.17e-05,-5.68e-05,6.66e-05,-6.13e-05,3e-05,-5.93e-05,
      5.82e-05,-4.04e-05,-2.97e-05,1.02e-05,-1.14e-05,2.03e-05,-5.47e-05,3.44e-05,-6.82e-05,7.43e-05,
      -6.52e-05,2.32e-05,-1.26e-05,1.2e-05,-2.03e-05,5.39e-05,-3.12e-05,5.62e-05,-2.96e-05,4.76e-05,
      3.3e-06,8.85e-05,-5.92e-05,0.0001493,-7.65e-05,0.000143,-3.39e-05,0.0001278,-1.53e-05,9.89e-05,
      5.15e-05,9.1e-06,6.18e-05,4.16e-05,7.03e-05,-2.42e-05,4.24e-05,-1.16e-05,1.96e-05,-8.16e-05,
      9.2e-06,-9e-05,-0.0001108,-7.83e-05,-0.0001728,-0.0001247,-0.0002542,-0.0002002,-0.0003034,-0.000242,
      -0.0003852,-0.0003356,-0.0003766,-0.0004265,-0.0004681,-0.0004334,-0.0005153,-0.000503,-0.000504,-0.000496,
      -0.0005436,-0.000493,-0.0004997,-0.0005228,-0.0004412,-0.0004304,-0.0004054,-0.0003673,-0.0003777,-0.0002613,
      -0.000217,-0.0002482,-0.0001485,-7.09e-05,-1.61e-05,-1.47e-05,7.42e-05,0.0001628,0.0001846,0.0003071,
      0.0002593,0.0004421,0.0004065,0.0004991,0.0006098,0.0005389,0.0007136,0.0006622,0.0007295,0.0007918,
      0.0007738,0.0008011,0.0008579,0.0007364,0.0008695,0.0007548,0.0007993,0.0007196,0.0006968,0.0007032,
      0.0005656,0.0006488,0.0004653,0.0004936,0.0004262,0.0003693,0.000339,0.0002581,0.0002738,0.0001549,
      0.0001548,0.0001363,6.08e-05,5.73e-05,2.25e-05,2.9e-06,-3.36e-05,-1.74e-05,-4.61e-05,-9.64e-05,
      -2.9e-05,-8.74e-05,-6.91e-05,-8.44e-05,-5.43e-05,-0.0001336,-2.13e-05,-9.46e-05,-3.48e-05,-7.86e-05,
      -9.83e-05,1.06e-05,-8.18e-05,5.5e-06,-7.74e-05,-7.3e-06,-4.38e-05,3.3e-05,-8.9e-05,2.55e-05,
      -1.44e-05,1.91e-05,-4.88e-05,2.74e-05,-6.15e-05,5.67e-05,-4.1e-06,-3.94e-05,-2e-07,5.27e-05,
      -6.16e-05,3.28e-05,-5.72e-05,4.72e-05,3e-06,-4.52e-05,1.61e-05,-6.9e-06,1.16e-05,-5.26e-05,
      6.31e-05,-6.53e-05,6.32e-05,-5.26e-05,1.27e-05,-1.19e-05,4.81e-05,-4.51e-05,-2.6e-06,4.16e-05,
      -1.9e-06,-4.61e-05,5.1e-05,-2.32e-05,5.57e-05,-6.41e-05,6.59e-05,-6.47e-05,5.83e-05,-2.08e-05,
      1.24e-05,-1.56e-05,5.02e-05,-4.98e-05,1.43e-05,-8.2e-06,5.4e-06,-4e-07,-3.6e-06,1.8e-06,
      9.4e-06,-5.26e-05,6.56e-05,-6.19e-05,2.12e-05,-1.08e-05,8.7e-06,-1.1e-05,2.2e-05,-6.39e-05,
      7.21e-05,-6.36e-05,2.16e-05,-1.06e-05,7.6e-06,-6.8e-06,6.3e-06,-5.4e-06,2.2e-06,9.8e-06,
      -5.46e-05,5.99e-05,-2.43e-05,2.49e-05,-6.21e-05,6.19e-05,-2.46e-05,2.44e-05,-6e-05,5.4e-05,
      -8.4e-06,-6.1e-06,1.99e-05,-6.05e-05,6.05e-05,-2.06e-05,9.9e-06,-5.8e-06,2e-07,4.8e-06,
      -6.1e-06,6.4e-06,-6.5e-06,6.5e-06,-6.5e-06,6.5e-06,-6.4e-06,6.1e-06,-4.8e-06,-0.0,
      4.8e-06,-6.1e-06,6.4e-06,-6.5e-06,6.5e-06,-6.5e-06,6.5e-06,-6.5e-06,6.5e-06,-6.4e-06,
      6.1e-06,-4.8e-06,-0.0,4.8e-06,-6.1e-06,6.4e-06,-6.5e-06,6.5e-06,-6.5e-06,6.5e-06,
      -6.5e-06,6.5e-06,-6.4e-06,6.1e-06,-4.8e-06,-0.0,4.8e-06,-6.1e-06,6.4e-06,-6.5e-06,
      6.6e-06,-6.8e-06,7.5e-06,-1.03e-05,2.06e-05,-5.89e-05,5.36e-05,-8.3e-06,-6.1e-06,1.96e-05,
      -5.93e-05,5.61e-05,-4.7e-06,-4.93e-05,5.36e-05,-4.4e-06,-4.79e-05,6.08e-05,-6.1e-05,4.9e-05,
      -1e-06,-4.43e-05,4.28e-05,7.1e-06,-5.71e-05,5.97e-05,-2.15e-05,1.43e-05,-2.28e-05,6.37e-05,
      -7.06e-05,5.85e-05,-3.1e-06,-5.8e-05,9.97e-05,-5.83e-05,3e-07,4.28e-05,-1e-05,-1.4e-06,
      2.6e-06,4.1e-06,-1.89e-05,5.85e-05,-5.37e-05,9.4e-06,2e-06,-4.4e-06,2.6e-06,7.2e-06,
      -3.13e-05,-4.33e-05,6.81e-05,-9.51e-05,2.99e-05,-4e-05,-1.85e-05,-3.56e-05,2.47e-05,-7.73e-05,
      -1.19e-05,-3.87e-05,3.06e-05,-8.47e-05,-1.4e-06,-4.71e-05,2.73e-05,-5.02e-05,1.2e-05,8e-07,
      -2e-06,-6e-06,2.59e-05,6.4e-05,2.7e-06,8.9e-05,8.72e-05,1.04e-05,0.0001712,6.14e-05,
      0.0001821,0.0001176,0.0002444,0.0002593,0.000373,0.0005113,0.000726,0.0010831,0.0014035,0.0018686,
      0.0019299,0.0010898,-0.0012732,-0.0062488,-0.0144146,-0.0261956,-0.0397759,-0.0547988,-0.0706657,-0.08535,
      -0.0958045,-0.0996145,-0.0966503,-0.0885181,-0.0768361,-0.0634477,-0.0491819,-0.0346283,-0.0207289,-0.007163,
      0.0061437,0.0196289,0.0329901,0.0467454,0.059345,0.0703191,0.0779018,0.0814772,0.0814064,0.0788055,
      0.0748795,0.0704539,0.0659052,0.0612317,0.0560422,0.0500794,0.0433439,0.0358707,0.0278345,0.0196145,
      0.0125356,0.007595,0.0043118,0.0025853,0.0014842,0.0010408,0.000808,0.0006629,0.0005709,0.0004669,
      0.0004696,0.0004546,0.0003641,0.0003787,0.0003411,0.0003639,0.0002768,0.000347,0.0002241,0.0002838,
      0.0002016,0.0001427,0.000174,6.39e-05,3.95e-05,-8.27e-05,9e-07,-0.000261,-0.0001562,-0.0004017,
      -0.0004263,-0.0005183,-0.0006402,-0.0007869,-0.0008775,-0.0009887,-0.0012205,-0.00123,-0.0014773,-0.0015275,
      -0.0017015,-0.0018651,-0.0019432,-0.0020876,-0.0021958,-0.0023509,-0.0024164,-0.0025557,-0.0026985,-0.0027487,
      -0.0028474,-0.002992,-0.0030198,-0.0031158,-0.0032416,-0.0032553,-0.0033475,-0.0033853,-0.0034617,-0.0035186,
      -0.00351,-0.0036009,-0.003576,-0.003666,-0.0036423,-0.0036163,-0.0037348,-0.0035775,-0.0037738,-0.0035824,
      -0.0036816,-0.0036239,-0.0036037,-0.0035724,-0.0035688,-0.0034553,-0.0034608,-0.0034008,-0.0033314,-0.003217,
      -0.0031476,-0.003088,-0.0029468,-0.0028218,-0.0027126,-0.0025336,-0.0024561,-0.002197,-0.0021183,-0.0019314,
      -0.0017161,-0.0015706,-0.0014498,-0.0011973,-0.0011266,-0.0008991,-0.0008247,-0.0006236,-0.0005735,-0.0003852,
      -0.0002952,-0.0002349,-0.0001033,-0.0001081,9.63e-05,2.19e-05,0.0001327,0.0001859,0.0001968,0.0002244,
      0.000253,0.0002852,0.000263,0.0003093,0.0003064,0.0002831,0.0003573,0.00024,0.0003544,0.0002847,
      0.0003131,0.0002674,0.0002738,0.000294,0.0002033,0.0002392,0.0001958,0.0001836,0.0001136,0.0001083,
      5.9e-05,-5.03e-05,-9.1e-06,-0.0002211,-0.0001662,-0.0003178,-0.0003867,-0.0005375,-0.0006338,-0.0006962,
      -0.0009538,-0.0009244,-0.0012177,-0.0012888,-0.0014732,-0.0015598,-0.0018081,-0.0018901,-0.0021211,-0.0021403,
      -0.0024511,-0.0024226,-0.0026441,-0.0027491,-0.0028519,-0.0029498,-0.0030699,-0.0031258,-0.0032515,-0.0033054,
      -0.003308,-0.0035447,-0.0034011,-0.0036294,-0.0036189,-0.0036376,-0.0037542,-0.0037349,-0.0037607,-0.003859,
      -0.0038322,-0.003805,-0.0038702,-0.0038547,-0.0037449,-0.0038762,-0.0037628,-0.0037131,-0.0036825,-0.0037126,
      -0.0035667,-0.0036024,-0.0034597,-0.0034762,-0.003367,-0.0034059,-0.0032277,-0.003329,-0.0031144,-0.0032122,
      -0.0030852,-0.0030809,-0.0030869,-0.0029843,-0.0030586,-0.0029798,-0.0030263,-0.0030405,-0.0029994,-0.0030904,
      -0.0030915,-0.0031312,-0.0031008,-0.0032548,-0.003201,-0.0032642,-0.0033353,-0.0033285,-0.0034211,-0.003462,
      -0.0034802,-0.003571,-0.0035393,-0.003655,-0.0036317,-0.0036869,-0.0036438,-0.0037646,-0.003685,-0.0037591,
      -0.0037026,-0.0037313,-0.0037547,-0.0036744,-0.0037311,-0.0037045,-0.0036829,-0.0036756,-0.003604,-0.0036594,
      -0.0035682,-0.0035779,-0.0035772,-0.0034511,-0.0035393,-0.0034771,-0.0034273,-0.0034611,-0.0033723,-0.0034551,
      -0.0033219,-0.0034407,-0.0033051,-0.0034099,-0.0033186,-0.003377,-0.0033293,-0.0033551,-0.0032955,-0.0032983,
      -0.0033476,-0.0032505,-0.0032759,-0.0032759,-0.0032484,-0.0031521,-0.0032568,-0.0031394,-0.0031056,-0.0031371,
      -0.0030546,-0.0030443,-0.002955,-0.0030363,-0.0028911,-0.002875,-0.0028975,-0.0028213,-0.0027745,-0.0027611,
      -0.0027719,-0.0027116,-0.0026698,-0.0026901,-0.0026497,-0.0027038,-0.0026459,-0.0026504,-0.0026556,-0.0026655,
      -0.0026586,-0.0026783,-0.0027307,-0.0026229,-0.0027996,-0.0026519,-0.002783,-0.0026896,-0.0028221,-0.002667,
      -0.0028763,-0.0026473,-0.0028223,-0.0027337,-0.0027856,-0.0027184,-0.0027328,-0.0027416,-0.0026759,-0.0027419,
      -0.0026599,-0.0027229,-0.0026195,-0.0027013,-0.0026742,-0.002635,-0.0026852,-0.002587,-0.0027375,-0.0026291,
      -0.00265,-0.0026764,-0.0026847,-0.0026997,-0.0027696,-0.002669,-0.0028274,-0.0027428,-0.0028071,-0.0028546,
      -0.0027941,-0.0028507,-0.0028504,-0.0027946,-0.0028261,-0.0027532,-0.0027697,-0.0027248,-0.0026542,-0.0026406,
      -0.0025302,-0.0025591,-0.0024564,-0.0024131,-0.0024051,-0.0022886,-0.0022455,-0.0022493,-0.0021886,-0.0021409,
      -0.0021043,-0.0020703,-0.0020591,-0.0020341,-0.0020659,-0.0019619,-0.0020608,-0.002018,-0.0019828,-0.0020955,
      -0.0020039,-0.0021147,-0.0020151,-0.0021278,-0.0021345,-0.0020724,-0.0022048,-0.0021398,-0.0021222,-0.00223,
      -0.0021363,-0.002229,-0.0020948,-0.0022207,-0.0021147,-0.0021471,-0.002095,-0.0020977,-0.0019898,-0.0021019,
      -0.0019033,-0.0019747,-0.0018847,-0.0018544,-0.0017466,-0.0017652,-0.0016466,-0.0016183,-0.001572,-0.001467,
      -0.001471,-0.00141,-0.0013337,-0.0012707,-0.0012684,-0.0012069,-0.0011408,-0.0011696,-0.000967,-0.0011282,
      -0.0008573,-0.0010116,-0.0008229,-0.0008144,-0.0007384,-0.0007528,-0.0006052,-0.0005862,-0.0005231,-0.0004706,
      -0.0004452,-0.0003233,-0.0003536,-0.0002148,-0.0002946,-0.0001093,-0.0001655,-0.0001365,-3.29e-05,-4.27e-05,
      -8.4e-05,7.16e-05,-4.25e-05,8.92e-05,-4.9e-06,6.95e-05,3.75e-05,8.07e-05,8.84e-05,2.47e-05,
      0.0001012,2.97e-05,8.01e-05,9.87e-05,-1.66e-05,0.000108,4.42e-05,1.34e-05,5.29e-05,7.4e-05,
      -4.04e-05,7.78e-05,2.73e-05,-2.68e-05,8.19e-05,-1.6e-05,-6.1e-06,4.15e-05,-1.14e-05,-7.8e-06,
      5.58e-05,-6.78e-05,6.83e-05,-5.78e-05,1.46e-05,-7e-07,-2.49e-05,-3.72e-05,2.5e-05,-7.69e-05,
      -3.5e-06,-7.16e-05,3.7e-06,-0.0001066,-2.49e-05,-9.42e-05,-5.87e-05,-0.0001205,-5.64e-05,-0.0001036,
      -0.0001286,-0.0001414,-6.41e-05,-0.0001908,-9.19e-05,-0.0001787,-0.000101,-0.0001661,-0.0001434,-0.0001691,
      -6.44e-05,-0.0001852,-0.0001026,-0.0001524,-3.38e-05,-0.0001742,-3.4e-06,-0.0001355,-3.97e-05,-3.9e-06,
      -0.0001075,-1.27e-05,2.09e-05,-8.37e-05,1.67e-05,1.68e-05,-8.57e-05,2.89e-05,-4.17e-05,-5e-07,
      -0.0001078,-2.68e-05,-7.41e-05,-0.0001383,-0.0001208,-0.0001382,-0.0002258,-0.0001661,-0.0003095,-0.0002525,
      -0.0003388,-0.0003386,-0.0004244,-0.0003683,-0.0005099,-0.0004458,-0.0005625,-0.0004661,-0.0005787,-0.0005315,
      -0.000599,-0.0005243,-0.000608,-0.0005037,-0.0005307,-0.0005352,-0.0004718,-0.0004389,-0.0004691,-0.000395,
      -0.0003459,-0.0003253,-0.0003068,-0.0002434,-0.0002241,-0.0002075,-0.0001428,-0.0001268,-9.85e-05,-7.78e-05,
      -4.61e-05,-2.97e-05,8.9e-06,-6e-06,2.29e-05,5.81e-05,5.13e-05,3.62e-05,9.74e-05,2.99e-05,
      8.36e-05,8.52e-05,3.12e-05,8.32e-05,9.24e-05,-1.23e-05,0.0001156,-2.2e-06,4.29e-05,-2.04e-05,
      3.94e-05,1.16e-05,-8.74e-05,4.02e-05,-7.48e-05,-4.05e-05,-7.08e-05,-0.0001194,-5.72e-05,-0.000102,
      -0.0001279,-0.0001504,-0.0001622,-0.0001066,-0.0003135,-0.0001361,-0.0003508,-0.0002646,-0.0002366,2.11e-05,
      0.0006234,0.0020336,0.0045464,0.0089086,0.0152117,0.0241071,0.0354809,0.0472105,0.0585276,0.0673068,
      0.0710608,0.0685256,0.0610235,0.0509227,0.0403919,0.030637,0.0219868,0.0152489,0.0098426,0.0057259,
      0.0022597,-0.0007281,-0.003719,-0.0072032,-0.0113502,-0.0164519,-0.0227252,-0.0302377,-0.039105,-0.0488313,
      -0.0582235,-0.0650822,-0.0665436,-0.0615695,-0.0516714,-0.0394695,-0.0280041,-0.0181866,-0.0105927,-0.0055148,
      -0.0021439,-0.0001849,0.0007036,0.001371,0.0015576,0.0018638,0.0019476,0.0020444,0.002216,0.002189,
      0.0023036,0.0023112,0.00237,0.0023925,0.0024475,0.0023799,0.0025728,0.0023894,0.002586,0.0025116,
      0.0025831,0.002582,0.0025132,0.0026505,0.0025242,0.0025698,0.0025483,0.0024396,0.0024658,0.0023601,
      0.0022415,0.0022486,0.0020853,0.0018838,0.001956,0.0016109,0.0016065,0.001437,0.0012678,0.001168,
      0.0010207,0.0008574,0.0008066,0.0006005,0.0005797,0.0003899,0.0003687,0.0002134,0.0001961,8.67e-05,
      6.96e-05,-3.97e-05,-5.58e-05,-7.12e-05,-0.0001376,-0.0001378,-0.0001176,-0.0002023,-0.000163,-0.0001407,
      -0.000176,-0.0001023,-0.0002252,-8.22e-05,-0.0001096,-9.63e-05,-0.0001226,-7.41e-05,-5e-06,-9.93e-05,
      -7.02e-05,5.33e-05,-0.0001008,2.16e-05,-3.33e-05,1.71e-05,-8.55e-05,4.38e-05,-9.41e-05,3.2e-06,
      -6.37e-05,-8.21e-05,-8.2e-05,-6.41e-05,-0.0001397,-0.0001369,-0.0001188,-0.0001979,-0.0001814,-0.0002158,
      -0.000189,-0.000265,-0.0002196,-0.0002342,-0.0003162,-0.0002055,-0.0002865,-0.0002121,-0.0002898,-0.0001907,
      -0.0002275,-0.000177,-0.0001534,-0.0001098,-0.0001147,2.9e-06,-3.83e-05,0.0001117,0.0001141,0.0001497,
      0.0002889,0.000263,0.0004217,0.0004255,0.0005395,0.0005509,0.0006286,0.0006877,0.0007509,0.0007203,
      0.0007824,0.0007998,0.0007108,0.0007687,0.0006676,0.0005376,0.0004964,0.0003111,0.0001429,-9.19e-05,
      -0.0002656,-0.0006248,-0.0009115,-0.0013062,-0.0016218,-0.0021069,-0.002556,-0.0030752,-0.0035885,-0.0041514,
      -0.0047636,-0.0053144,-0.0059853,-0.0066723,-0.0072113,-0.007937,-0.0086152,-0.009131,-0.0098998,-0.010378,
      -0.0110054,-0.0115659,-0.0120314,-0.0124361,-0.0129512,-0.0132279,-0.0135225,-0.0138891,-0.0139813,-0.0141443,
      -0.0143793,-0.0142933,-0.0144834,-0.0143633,-0.0144377,-0.0143326,-0.0143247,-0.0142488,-0.0141682,-0.0141103,
      -0.0140232,-0.0140145,-0.0138593,-0.0139445,-0.0137476,-0.0138949,-0.0137557,-0.0138008,-0.0138334,-0.0137715,
      -0.013885,-0.0138612,-0.0139105,-0.013924,-0.0139955,-0.0140204,-0.0140468,-0.0140498,-0.0140297,-0.0140594,
      -0.0140011,-0.0139448,-0.0138627,-0.0136975,-0.0135613,-0.0133706,-0.0131131,-0.0128219,-0.0125152,-0.0121673,
      -0.011701,-0.0112538,-0.0108538,-0.0102171,-0.0097743,-0.0091455,-0.0085782,-0.0079811,-0.0073436,-0.0067676,
      -0.0061561,-0.0055807,-0.0049298,-0.0044858,-0.0037984,-0.0033761,-0.0028845,-0.0023617,-0.001965,-0.0016326,
      -0.0011134,-0.0009697,-0.0005374,-0.0003247,-0.0001153,6.45e-05,0.0003105,0.0003689,0.000465,0.0006452,
      0.0005667,0.0007065,0.0006684,0.000681,0.000681,0.000654,0.0006107,0.0005247,0.0005908,0.000418,
      0.0004516,0.0003367,0.0003027,0.0002549,0.0001811,0.0002137,2.99e-05,0.0001008,2.52e-05,-5.61e-05,
      5.18e-05,-0.0001587,-1.07e-05,-0.000115,-0.0001288,-0.0001304,-8.07e-05,-0.000176,-0.0001096,-0.0001325,
      -0.0001209,-0.0001307,-0.0001169,-0.0001487,-4.53e-05,-0.0001202,-8.48e-05,-0.0001245,-2.47e-05,-9.06e-05,
      -4.57e-05,-3.89e-05,-9.92e-05,-9e-06,-2.75e-05,-1.69e-05,-6.94e-05,1.28e-05,4.7e-06,-3.32e-05,
      -6e-06,4.56e-05,-2.87e-05,8.62e-05,-3.41e-05,6.71e-05,5.1e-05,4.11e-05,8.51e-05,6.66e-05,
      0.000112,8.31e-05,0.0001551,5.51e-05,0.0002274,9.11e-05,0.000157,0.0001928,0.0001293,0.0001881,
      0.0001775,0.0001456,0.0001505,0.0001627,9.13e-05,6.8e-05,9.9e-05,-2.14e-05,-1.52e-05,-7.06e-05,
      -0.0001557,-0.0002222,-0.0003187,-0.0003124,-0.0005471,-0.0005179,-0.0007039,-0.0007473,-0.0009775,-0.0010836,
      -0.0013181,-0.0016516,-0.002043,-0.0025343,-0.0033719,-0.0043439,-0.005801,-0.0075596,-0.0100594,-0.0128612,
      -0.0162838,-0.0199877,-0.0237625,-0.0275338,-0.0308223,-0.0336804,-0.0358456,-0.0371942,-0.0377866,-0.0372219,
      -0.03588,-0.0334955,-0.030281,-0.0265703,-0.0225469,-0.0184173,-0.0146913,-0.0112709,-0.0084824,-0.006099,
      -0.004546,-0.0031077,-0.0022432,-0.0016368,-0.0011826,-0.0008627,-0.0006518,-0.0004798,-0.0003784,-0.0003469,
      -0.0001772,-0.0002105,-0.0001269,-7.98e-05,-9.2e-05,-1.4e-05,1.93e-05,-4.2e-06,7.35e-05,4.52e-05,
      8.24e-05,9.25e-05,8.67e-05,9.96e-05,5.28e-05,0.000159,5.05e-05,0.0001064,6.2e-05,0.000113,
      2.25e-05,0.0001348,-2.81e-05,0.0001123,-1.66e-05,8.77e-05,-2.7e-06,-4.1e-06,8.01e-05,-4.39e-05,
      2.36e-05,-5.23e-05,5.79e-05,-5.16e-05,1.92e-05,-2.61e-05,1.24e-05,-8.45e-05,5.33e-05,-5.91e-05,
      -1.98e-05,9e-06,-1.69e-05,-1.49e-05,-5.56e-05,3.67e-05,-2.04e-05,-2.89e-05,6.6e-06,2.6e-06,
      -1.78e-05,-4e-06,-2.45e-05,3.09e-05,-2.8e-05,9.1e-06,-8.6e-06,2.6e-05,-2.44e-05,-2e-07,
      2.61e-05,-3.32e-05,3.56e-05,-3.81e-05,4.64e-05,-1.88e-05,-2.97e-05,6.82e-05,-4.33e-05,-2.54e-05,
      7.54e-05,-7.73e-05,3.32e-05,1.7e-05,-4.54e-05,3.6e-05,-2.75e-05,1.9e-06,2.08e-05,-1.38e-05,
      -3.9e-05,4.21e-05,3e-07,-4.5e-05,5.17e-05,-3.31e-05,2.38e-05,-4.4e-06,-6.1e-06,2.97e-05,
      -4.17e-05,6.76e-05,-2.73e-05,4.3e-05,-1.53e-05,1.91e-05,1.27e-05,6.23e-05,-6.11e-05,0.000114,
      -6.46e-05,9.01e-05,-3.69e-05,5.98e-05,1e-06,6.82e-05,-7.37e-05,0.0001,-6.89e-05,4.76e-05,
      8.7e-06,-1.14e-05,-3.68e-05,2.91e-05,-8.34e-05,-2.91e-05,-3.3e-06,-9.21e-05,-3.53e-05,-0.0001032,
      -1.67e-05,-0.00017,-4.32e-05,-0.0001259,-0.0001241,-0.0001196,-6.61e-05,-0.0001579,-0.0001023,-0.0001024,
      -9.96e-05,-9.63e-05,-5.19e-05,-2.99e-05,-3.14e-05,2.74e-05,5.56e-05,8.78e-05,0.0001361,0.0002472,
      0.0002171,0.0003768,0.0003474,0.0005286,0.000499,0.0007036,0.0006519,0.0008653,0.000859,0.0010086,
      0.0010274,0.0011966,0.0011848,0.0013274,0.0013814,0.0014103,0.0015236,0.0015015,0.0016331,0.0015876,
      0.0016401,0.0016814,0.0016678,0.0016814,0.0017204,0.0016756,0.00169,0.0016765,0.0016708,0.0016707,
      0.0016698,0.0016351,0.0016502,0.0016234,0.0016759,0.0016141,0.001602,0.0016754,0.0015992,0.0016632,
      0.0016524,0.0016315,0.0016927,0.0016235,0.00172,0.0016636,0.001741,0.0016933,0.0017461,0.0017332,
      0.0017317,0.0018133,0.0017234,0.0017944,0.0017812,0.0017628,0.0017577,0.0017715,0.0017198,0.0017027,
      0.0017062,0.0016118,0.0016397,0.0015244,0.0015369,0.0013674,0.0014047,0.0013064,0.0011399,0.0011602,
      0.0010135,0.0009467,0.0008235,0.0007652,0.0006822,0.0005342,0.0004787,0.0004113,0.000312,0.0002552,
      0.0001438,0.000127,9.27e-05,-8.08e-05,1.96e-05,-0.0001144,-8.59e-05,-0.0001937,-8.32e-05,-0.0002254,
      -0.0001863,-0.0002004,-0.0001688,-0.0001978,-0.0002114,-0.0001258,-0.0002332,-0.0001095,-0.0001903,-7.15e-05,
      -0.0001748,-7.26e-05,-5.59e-05,-0.0001437,-2e-07,-6.82e-05,-3.63e-05,-1.17e-05,-1.56e-05,-3.88e-05,
      7.53e-05,-5.38e-05,2.91e-05,5.14e-05,-2.52e-05,5.14e-05,3.12e-05,3.53e-05,3.91e-05,1.89e-05,
      -1.9e-06,8.95e-05,-3.44e-05,5.02e-05,4.54e-05,-2.22e-05,4.53e-05,5.27e-05,-4.75e-05,4.05e-05,
      -1.7e-05,2.84e-05,1.61e-05,4.9e-06,-3.65e-05,4.42e-05,-4.37e-05,3.36e-05,6.8e-06,-6.29e-05,
      3.6e-05,3.17e-05,-6.7e-05,2.65e-05,-3.98e-05,3.59e-05,-6.3e-06,-1.18e-05,-4.53e-05,8.29e-05,
      -7.86e-05,2.17e-05,-8.1e-06,1.08e-05,-3.6e-05,3.62e-05,-1.13e-05,8.8e-06,-2.47e-05,-7.6e-06,
      4.2e-05,-4.91e-05,5.77e-05,-8.58e-05,7.76e-05,-1.48e-05,-1.93e-05,-5.8e-06,2.93e-05,8e-07,
      -3.32e-05,3.51e-05,-9.4e-06,2.5e-06,-7e-07,4e-07,-8e-07,2.9e-06,-1.06e-05,4.06e-05,
      -4.07e-05,1.02e-05,-1e-07,-9.9e-06,4.07e-05,-4.16e-05,1.38e-05,-1.37e-05,4.18e-05,-4.26e-05,
      1.64e-05,-2.29e-05,7.69e-05,-7.68e-05,2.04e-05,-4.9e-06,-1e-06,8.9e-06,-3.54e-05,3.57e-05,
      -9.7e-06,3.3e-06,-3.3e-06,1e-05,-3.76e-05,4.35e-05,-3.97e-05,3.2e-06,4.21e-05,-7.57e-05,
      5.15e-05,-1.85e-05,8.2e-06,0.0,-2.34e-05,-4.2e-06,4.13e-05,-4.96e-05,4.56e-05,-3.58e-05,
      -3e-07,3.8e-05,-4.03e-05,1.1e-05,-3.5e-06,3.2e-06,-9.3e-06,3.48e-05,-3.29e-05,-1.1e-06,
      3.82e-05,-4.03e-05,1.08e-05,-2.9e-06,8e-07,-3e-07,4e-07,-1.5e-06,5.4e-06,-2e-05,
      7.63e-05,-7.67e-05,2.06e-05,-5.5e-06,1.4e-06,-2e-07,-5e-07,2.2e-06,-8.2e-06,3.14e-05,
      -5.4e-06,-2.48e-05,6.9e-06,-2.6e-06,3.5e-06,-1.15e-05,4.32e-05,-5.01e-05,4.58e-05,-3.59e-05,
      -2e-07,3.75e-05,-3.85e-05,4.2e-06,2.23e-05,4.4e-06,-2.64e-05,-1.12e-05,7.27e-05,-7.09e-05,
      7e-07,6.95e-05,-7e-05,5e-07,6.95e-05,-7e-05,3e-07,7.01e-05,-7.23e-05,8.8e-06,
      3.79e-05,-4.89e-05,4.62e-05,-3.88e-05,1.11e-05,-5.4e-06,1.07e-05,-3.81e-05,4.44e-05,-4.24e-05,
      1.31e-05,4.5e-06,-4.6e-05,8.32e-05,-7.79e-05,1.82e-05,5.1e-06,-3.93e-05,4.04e-05,-1.01e-05,
      -0.0,1.01e-05,-4.12e-05,4.33e-05,-1.95e-05,3.54e-05,-9.9e-06,-1.01e-05,6.61e-05,-5.89e-05,
      7.31e-05,-2.31e-05,2e-05,5.67e-05,-3.66e-05,7.69e-05,-4.64e-05,0.0001107,-8.87e-05,0.0001489,
      -8.71e-05,8.88e-05,-4.36e-05,7.27e-05,-2.28e-05,4.2e-06,2.1e-05,9.9e-06,-6.19e-05,2.72e-05,
      -4.84e-05,-4.54e-05,1.91e-05,-0.0001466,4.8e-05,-0.0001618,-3.44e-05,-0.0001254,-0.0001133,-4.05e-05,
      -0.000165,-0.0001459,-9.67e-05,-0.0001175,-0.0001673,-7.33e-05,-0.0001603,-4.72e-05,-0.0001738,-2.6e-06,
      -2.73e-05,-0.0001004,0.0001077,-6.1e-06,0.0001309,0.0001185,0.00026,0.0003258,0.0004589,0.0006106,
      0.0009337,0.0013135,0.0019052,0.0027817,0.0040165,0.0058553,0.0081978,0.0114451,0.0154263,0.0200741,
      0.0254022,0.0307495,0.0358199,0.0396391,0.0419402,0.0420095,0.0399796,0.0364778,0.0319956,0.0269621,
      0.0221336,0.0173806,0.0131726,0.009531,0.0064887,0.0038363,0.0019819,0.0002714,-0.0008548,-0.0018842,
      -0.0025669,-0.0031501,-0.0036098,-0.0040317,-0.0042976,-0.0045685,-0.0048911,-0.0050884,-0.0052992,-0.0055992,
      -0.0058247,-0.0061042,-0.0063876,-0.0066194,-0.0069,-0.0071968,-0.0073849,-0.0077039,-0.0079163,-0.0081593,
      -0.0083671,-0.0085634,-0.008742,-0.0090098,-0.0090457,-0.0093187,-0.0094122,-0.0095552,-0.0096133,-0.0098114,
      -0.0098366,-0.0099454,-0.010031,-0.0100844,-0.010131,-0.0102167,-0.0102615,-0.0102334,-0.0103762,-0.0103308,
      -0.0103586,-0.0103698,-0.0103483,-0.0103489,-0.0102566,-0.0102945,-0.0101365,-0.0099985,-0.0099994,-0.0097337,
      -0.0096055,-0.0093585,-0.0091372,-0.0089195,-0.0085623,-0.0082816,-0.0079385,-0.0075761,-0.0072599,-0.0067866,
      -0.0064638,-0.0060465,-0.0054967,-0.0050751,-0.0043482,-0.0036433,-0.0027048,-0.001535,-0.0001833,0.0015991,
      0.0038471,0.0064173,0.0096086,0.0132594,0.01722,0.0213674,0.0253337,0.0286884,0.031127,0.0321493,
      0.0317575,0.0300672,0.0273641,0.0240063,0.0203623,0.0168838,0.0136038,0.0106939,0.0083705,0.0063579,
      0.0048845,0.0036356,0.0028018,0.0020926,0.0016596,0.0012397,0.0009199,0.0008205,0.0004557,0.0005354,
      0.0002593,0.0002461,0.0001092,7.64e-05,1.08e-05,-0.000106,-3.38e-05,-0.0002283,-0.0001118,-0.0002209,
      -0.0002327,-0.0001877,-0.0002447,-0.0001729,-0.0002933,-0.0001394,-0.0002287,-0.0001537,-0.0001989,-0.0001088,
      -0.0001285,-0.0001398,-5.43e-05,-8.96e-05,-5.08e-05,-3.8e-06,-8.43e-05,2.75e-05,-7.6e-06,-1.53e-05,
      8.98e-05,-4.87e-05,8.92e-05,7.1e-06,5.07e-05,6.95e-05,-1.41e-05,0.0001198,-2.4e-06,5.8e-05,
      5.01e-05,5.75e-05,1.66e-05,2.51e-05,3.21e-05,1.36e-05,4.42e-05,-2.45e-05,5.5e-05,-6.68e-05,
      8.38e-05,-0.0001233,9.7e-05,-0.0001011,-7.3e-06,-1.87e-05,-6.78e-05,-6.5e-06,-5.62e-05,-6.64e-05,
      7.1e-06,-0.0001131,9e-07,-5.85e-05,-4.62e-05,-7.27e-05,4.16e-05,-9.6e-05,2.89e-05,-1e-06,
      -2.5e-05,8.5e-05,-1e-07,6.55e-05,5.51e-05,0.0001449,0.0001283,0.0001046,0.0002191,0.0002103,
      0.0001672,0.0003158,0.0002438,0.0003673,0.0002578,0.0004101,0.0003715,0.0003733,0.0003886,0.0004933,
      0.0003697,0.0004303,0.0004777,0.0003734,0.0004484,0.0003994,0.0003714,0.0003844,0.0003431,0.0003461,
      0.0002411,0.0003474,0.0001878,0.0002439,0.0001939,0.0001887,0.0001093,0.0001354,9.39e-05,8.47e-05,
      2.9e-05,8.26e-05,-3.22e-05,3.2e-05,5.34e-05,-9.96e-05,4.86e-05,-9.68e-05,4.22e-05,-7.39e-05,
      -4.52e-05,-5.9e-05,-1.59e-05,-1.12e-05,-0.0001211,6.63e-05,-0.0001474,6.33e-05,-9.24e-05,-6.6e-06,
      -3.03e-05,-2.12e-05,-3.39e-05,8.6e-06,-5e-07,-2.29e-05,-4.15e-05,4.17e-05,7.5e-06,-5.73e-05,
      5.88e-05,-1.41e-05,-2.3e-06,7.9e-06,-1.39e-05,4.86e-05,-3.23e-05,6.6e-05,-6.88e-05,6.19e-05,
      -3.06e-05,6.14e-05,-6.76e-05,6.16e-05,-3.04e-05,6.11e-05,-6.65e-05,5.75e-05,-1.54e-05,4.1e-06,
      -1.1e-06,2e-07,3e-07,-1.5e-06,5.6e-06,-5.7e-06,1.5e-06,-4e-07,1e-07,0.0,
      -3e-07,1e-06,-3.8e-06,1.42e-05,-5.39e-05,5.38e-05,-1.3e-05,1.35e-05,-5.76e-05,6.94e-05,
      -7.25e-05,7.3e-05,-7.23e-05,6.85e-05,-5.43e-05,4e-07,5.35e-05,-6.71e-05,6.72e-05,-5.43e-05,
      1.6e-06,4.88e-05,-4.94e-05,3e-07,4.91e-05,-4.91e-05,-1e-06,5.39e-05,-6.71e-05,6.71e-05,
      -5.39e-05,1e-07,5.43e-05,-7e-05,7.8e-05,-7.91e-05,7.52e-05,-7.42e-05,7.39e-05,-7.38e-05,
      7.38e-05,-7.38e-05,7.37e-05,-7.34e-05,7.24e-05,-6.87e-05,5.47e-05,-1.7e-06,-4.88e-05,4.93e-05,
      0.0,-5.01e-05,5.27e-05,-1.23e-05,1.21e-05,-5.24e-05,5e-05,9e-07,-5.43e-05,6.86e-05,
      -7.24e-05,7.34e-05,-7.37e-05,7.35e-05,-7.27e-05,6.97e-05,-5.85e-05,1.57e-05,-4.5e-06,2.2e-06,
      -4.4e-06,1.55e-05,-5.82e-05,6.96e-05,-7.26e-05,7.32e-05,-7.24e-05,6.87e-05,-5.47e-05,1.8e-06,
      4.83e-05,-6.27e-05,7.01e-05,-7e-05,6.22e-05,-3.05e-05,6.04e-05,-6.33e-05,4.5e-05,3.24e-05,
      -1.07e-05,-5.2e-06,3.22e-05,2.57e-05,2.95e-05,-1.06e-05,2.92e-05,2.76e-05,2.51e-05,2.13e-05,
      3.89e-05,-2.83e-05,7.55e-05,2.39e-05,-2.24e-05,8.25e-05,-2.59e-05,3.7e-05,2.69e-05,-1.15e-05,
      3.52e-05,2.01e-05,3.37e-05,-6.3e-06,-8.5e-06,4.09e-05,-6.5e-06,3e-07,-1.02e-05,4.12e-05,
      -5.8e-06,-2.7e-06,1.2e-06,-2e-06,6.7e-06,-9.6e-06,1.63e-05,-5.64e-05,6.12e-05,-2.44e-05,
      2.1e-05,-6.01e-05,7.16e-05,-6.31e-05,1.68e-05,-4.1e-06,-3e-07,5.3e-06,-5.5e-06,1.2e-06,
      6e-07,-3.6e-06,1.39e-05,-5.23e-05,6.28e-05,-6.62e-05,5.38e-05,-2e-07,-5.34e-05,6.58e-05,
      -6.17e-05,4.84e-05,1.3e-06,-5.43e-05,6.77e-05,-6.84e-05,5.77e-05,-1.38e-05,1.3e-05,-5.41e-05,
      5.53e-05,-1.85e-05,1.87e-05,-5.68e-05,6.03e-05,-2.03e-05,5.5e-06,-1.6e-06,1.1e-06,-2.6e-06,
      9.5e-06,-2e-05,5.57e-05,-5.45e-05,1.35e-05,6e-07,-7e-07,-1.33e-05,5.43e-05,-5.59e-05,
      2.04e-05,-1.05e-05,6e-06,-1.37e-05,4.91e-05,-3.43e-05,-4.56e-05,5.31e-05,-1.8e-05,1.89e-05,
      -5.79e-05,6.46e-05,-5.22e-05,1.09e-05,-6.7e-06,1.59e-05,-5.75e-05,6.58e-05,-5.75e-05,1.55e-05,
      -4.4e-06,2.1e-06,-4.2e-06,1.46e-05,-5.47e-05,5.59e-05,-2e-05,8.7e-06,6e-07,-1.11e-05,
      4.42e-05,-1.7e-05,3.97e-05,-8.5e-06,-5.6e-06,3.14e-05,2.94e-05,1.55e-05,4.24e-05,-2.11e-05,
      2.69e-05,7.83e-05,-4.27e-05,9.31e-05,-3.22e-05,3.61e-05,3.71e-05,-2.04e-05,2.95e-05,5.16e-05,
      -7.21e-05,7.32e-05,-5.68e-05,-1.05e-05,-3.51e-05,-1.38e-05,-5.94e-05,-4.7e-05,-6.67e-05,-0.0001188,
      -6.99e-05,-4.93e-05,-0.000197,-5.72e-05,-0.0001873,-8.86e-05,-0.0002207,-8.78e-05,-0.0001745,-0.000125,
      -0.0002207,-5.1e-05,-0.0001885,-8.93e-05,-6.64e-05,-9.25e-05,-1.02e-05,-1.55e-05,5.69e-05,0.000103,
      0.0001441,0.0002,0.000282,0.0003293,0.0003566,0.0005146,0.0004512,0.0006972,0.0005358,0.0007729,
      0.0007316,0.0008387,0.0007344,0.0008961,0.0007867,0.0007919,0.0007166,0.0007007,0.0005532,0.0003859,
      0.0003189,-7.4e-06,-0.0001421,-0.0004863,-0.0007682,-0.0011158,-0.0015493,-0.0018855,-0.0025123,-0.0028315,
      -0.0035276,-0.0040184,-0.0045911,-0.0052913,-0.0058804,-0.006536,-0.0071943,-0.0079253,-0.008446,-0.0091959,
      -0.0098056,-0.010309,-0.0109351,-0.0113792,-0.0118724,-0.0122499,-0.0126483,-0.0128481,-0.013122,-0.0132815,
      -0.0134583,-0.0134768,-0.0135669,-0.0135088,-0.0135598,-0.0134284,-0.0134402,-0.0132832,-0.0132048,-0.0131298,
      -0.0129251,-0.0128699,-0.0127188,-0.0126397,-0.0124189,-0.0124085,-0.0122032,-0.0121378,-0.0120629,-0.0118754,
      -0.0119216,-0.0117989,-0.0117132,-0.0117437,-0.0116723,-0.0116841,-0.0117013,-0.01172,-0.0117049,-0.011832,
      -0.0118482,-0.0118989,-0.0120047,-0.0120284,-0.0121562,-0.0121901,-0.0122091,-0.0123819,-0.0122286,-0.0124088,
      -0.012323,-0.0122957,-0.0121993,-0.0121881,-0.0119757,-0.0119148,-0.0116464,-0.011493,-0.0112529,-0.0110223,
      -0.0106375,-0.010514,-0.0100698,-0.0098791,-0.0095158,-0.009197,-0.0089942,-0.0086368,-0.0083615,-0.0081758,
      -0.007892,-0.0076891,-0.0075647,-0.0072791,-0.0072486,-0.0070682,-0.0069635,-0.0068555,-0.0068322,-0.0067507,
      -0.0066865,-0.0066875,-0.006617,-0.0066516,-0.0065809,-0.0066181,-0.0065412,-0.0066834,-0.0064769,-0.0066882,
      -0.0064607,-0.0066642,-0.0065023,-0.0065457,-0.0065574,-0.0064981,-0.0064583,-0.0064883,-0.0063855,-0.006383,
      -0.0063532,-0.0063489,-0.0061509,-0.0062978,-0.006107,-0.006081,-0.0060558,-0.0059797,-0.0058932,-0.0059065,
      -0.0057475,-0.0057621,-0.0056649,-0.0056491,-0.0055436,-0.0054662,-0.0055089,-0.0053499,-0.0053417,-0.0052993,
      -0.0052679,-0.0052114,-0.0051756,-0.005122,-0.005125,-0.0051313,-0.0050215,-0.0051389,-0.0049592,-0.0050749,
      -0.0049751,-0.0050381,-0.0048741,-0.0050618,-0.0048,-0.0049196,-0.0048368,-0.0047866,-0.0047357,-0.004662,
      -0.0046859,-0.0045436,-0.0044995,-0.0044815,-0.0043693,-0.0043091,-0.0042735,-0.0041771,-0.0041239,-0.0040898,
      -0.0039963,-0.0039742,-0.0039198,-0.0038276,-0.0038298,-0.0037265,-0.0037653,-0.0037118,-0.0036203,-0.0037123,
      -0.0035418,-0.0036222,-0.003581,-0.003519,-0.0035923,-0.0034734,-0.0035199,-0.0034528,-0.0034674,-0.0034481,
      -0.0033627,-0.0033733,-0.0033654,-0.0032632,-0.0033113,-0.0032215,-0.0032135,-0.0031647,-0.0031793,-0.0030212,
      -0.0031569,-0.0029375,-0.003034,-0.0028823,-0.0029012,-0.002842,-0.0027346,-0.0027751,-0.0025585,-0.0026262,
      -0.0024127,-0.0024508,-0.0022033,-0.0022561,-0.0019786,-0.0019878,-0.0017785,-0.0017022,-0.0014667,-0.0014321,
      -0.0012089,-0.0010806,-0.0009223,-0.0007789,-0.0006113,-0.0005254,-0.000288,-0.0002723,-0.0001223,1.21e-05,
      7.41e-05,0.0001417,0.0002595,0.0003205,0.0002584,0.0004476,0.0003494,0.0004067,0.0004239,0.0004477,
      0.0003351,0.0004643,0.0003574,0.0003556,0.0003225,0.0003013,0.0002749,0.0002489,0.0002323,0.0001685,
      0.0001465,0.000142,3.85e-05,0.000154,-5.46e-05,6.46e-05,-5.38e-05,7e-07,-9.89e-05,-5.48e-05,
      -0.0001317,-0.0001684,-9.77e-05,-0.0001875,-0.0002052,-0.0001915,-0.0002287,-0.0002434,
     ),
     (
      0.00507,0.00475,0.00443,0.00452,0.0045,0.0045,0.0045,0.00451,0.00448,0.00459,
      0.00418,0.00419,0.00456,0.00456,0.0042,0.00416,0.00468,0.00414,0.00428,0.00424,
      0.00425,0.00426,0.00423,0.00434,0.00391,0.00403,0.00397,0.00409,0.00367,0.00375,
      0.00384,0.00341,0.00353,0.00348,0.00356,0.00327,0.00285,0.00334,0.00329,0.003,
      0.0027,0.00268,0.00306,0.00306,0.00269,0.00268,0.00308,0.003,0.00292,0.00333,
      0.00325,0.00318,0.00355,0.00363,0.00443,0.00416,0.00444,0.00508,0.00524,0.00544,
      0.00599,0.00659,0.00667,0.00725,0.00783,0.00794,0.00841,0.0094,0.00948,0.0097,
      0.01024,0.01084,0.0109,0.01156,0.01184,0.01158,0.01235,0.01254,0.01251,0.01242,
      0.01281,0.01282,0.0124,0.01259,0.01225,0.01192,0.01207,0.01182,0.01118,0.01098,
      0.0109,0.00994,0.00935,0.00965,0.00904,0.0082,0.00768,0.00758,0.00698,0.0065,
      0.00603,0.0054,0.00536,0.00466,0.0045,0.00434,0.00366,0.00352,0.00325,0.003,
      0.00277,0.00243,0.0025,0.00258,0.00217,0.00225,0.00234,0.00191,0.00202,0.00202,
      0.00192,0.00231,0.00234,0.00185,0.00227,0.00256,0.00248,0.0025,0.00252,0.00243,
      0.00275,0.00306,0.003,0.00294,0.00324,0.00359,0.00339,0.00383,0.00378,0.00357,
      0.00445,0.00414,0.0045,0.00484,0.00465,0.00505,0.00515,0.00584,0.00597,0.00627,
      0.00644,0.00698,0.00761,0.00757,0.00861,0.00897,0.00952,0.00994,0.01073,0.01161,
      0.01182,0.01311,0.01373,0.01447,0.01536,0.01557,0.01686,0.01747,0.01825,0.01901,
      0.01972,0.02058,0.02097,0.02154,0.02189,0.02291,0.02299,0.02315,0.0239,0.02374,
      0.02365,0.02415,0.02374,0.0234,0.02365,0.02299,0.0224,0.02241,0.02148,0.02068,
      0.02032,0.01955,0.01851,0.01744,0.01676,0.01605,0.01507,0.01369,0.01269,0.01207,
      0.01106,0.0097,0.00866,0.00817,0.00667,0.00568,0.00511,0.00389,0.00334,0.00225,
      0.00118,0.00055,-0.00037,-0.00057,-0.00183,-0.00256,-0.00292,-0.00376,-0.00455,-0.00501,
      -0.00541,-0.00635,-0.0067,-0.00734,-0.00744,-0.00791,-0.0089,-0.00898,-0.00919,-0.00975,
      -0.01032,-0.01048,-0.01075,-0.011,-0.01125,-0.01151,-0.01173,-0.01207,-0.012,-0.01194,
      -0.01225,-0.01256,-0.0125,-0.01242,-0.01281,-0.01282,-0.01242,-0.0125,-0.01258,-0.01217,
      -0.01225,-0.01232,-0.01198,-0.01178,-0.01143,-0.01151,-0.01154,-0.01132,-0.01069,-0.01043,
      -0.01058,-0.01023,-0.01,-0.00975,-0.0095,-0.00925,-0.009,-0.00876,-0.00847,-0.00837,
      -0.00758,-0.00783,-0.0076,-0.00681,-0.00718,-0.00648,-0.00594,-0.00576,-0.00554,-0.0051,
      -0.00409,-0.00406,-0.00366,-0.00232,-0.00209,-0.00132,-0.00015,0.00042,0.00143,0.00281,
      0.00377,0.0046,0.00628,0.00773,0.00924,0.01076,0.01217,0.01401,0.01574,0.01749,
      0.01927,0.02092,0.02302,0.02497,0.02706,0.02873,0.03047,0.03234,0.03363,0.03559,
      0.03697,0.03851,0.03992,0.04029,0.04191,0.04254,0.04295,0.04365,0.04346,0.0435,
      0.04355,0.04331,0.04272,0.04231,0.04156,0.0405,0.0395,0.03855,0.03733,0.03567,
      0.03453,0.03324,0.03205,0.0306,0.0286,0.02753,0.02632,0.02476,0.02319,0.02202,
      0.02076,0.0195,0.01828,0.01693,0.01605,0.01491,0.01434,0.01326,0.01218,0.01155,
      0.01065,0.01036,0.00941,0.00902,0.00851,0.00799,0.00753,0.00692,0.00678,0.00647,
      0.00635,0.00566,0.00552,0.00525,0.005,0.00476,0.00449,0.00428,0.00391,0.00408,
      0.00375,0.00341,0.00359,0.00323,0.00302,0.0027,0.00271,0.00293,0.00208,0.00227,
      0.00233,0.00191,0.00202,0.002,0.00198,0.00208,0.00169,0.00168,0.00209,0.00198,
      0.002,0.00202,0.00193,0.00226,0.0025,0.00272,0.00309,0.00291,0.00326,0.00352,
      0.00366,0.00431,0.00458,0.00436,0.00499,0.00565,0.00542,0.00568,0.00633,0.00648,
      0.00675,0.00701,0.00722,0.00758,0.00747,0.00753,0.00741,0.00783,0.00774,0.00772,
      0.00785,0.00739,0.0076,0.00723,0.007,0.00675,0.0065,0.00626,0.00597,0.00584,
      0.00517,0.005,0.00482,0.00425,0.00369,0.00351,0.00326,0.00298,0.00284,0.00219,
      0.00191,0.00216,0.00143,0.00116,0.00143,0.00066,0.00046,0.00049,0.00056,0.00025,
      -6e-05,-1e-05,8e-05,-0.00033,-0.00024,-0.00024,-0.00028,-0.00015,-0.0006,-0.00045,
      -0.00059,-0.00019,-0.00015,-0.00067,-0.00017,-0.00019,-0.00056,-0.00057,-0.00018,-0.00025,
      -0.00028,-0.00013,-0.00068,-0.00017,-0.00018,-0.00059,-0.00048,-0.00051,-0.00049,-0.00052,
      -0.00041,-0.00083,-0.00075,-0.00068,-0.001,-0.0013,-0.00127,-0.00115,-0.00161,-0.0014,
      -0.00178,-0.00197,-0.00233,-0.00218,-0.00247,-0.00288,-0.00254,-0.00344,-0.00317,-0.0034,
      -0.00418,-0.00389,-0.00427,-0.0045,-0.00472,-0.00508,-0.00494,-0.00516,-0.00589,-0.00577,
      -0.00557,-0.00642,-0.00621,-0.00625,-0.00626,-0.00624,-0.00624,-0.0063,-0.00603,-0.00563,
      -0.00591,-0.00523,-0.00468,-0.00456,-0.00408,-0.00317,-0.00276,-0.00232,-0.00147,-0.00084,
      0.00031,0.00103,0.00156,0.00316,0.00374,0.00434,0.00583,0.00675,0.00763,0.00917,
      0.00966,0.01068,0.01206,0.01305,0.0137,0.01464,0.01617,0.01664,0.01774,0.01884,
      0.01939,0.02059,0.02125,0.02192,0.02306,0.02384,0.02408,0.02534,0.02608,0.02634,
      0.02759,0.02831,0.02867,0.0295,0.03034,0.03064,0.0316,0.032,0.03243,0.03334,
      0.03373,0.03427,0.03473,0.03533,0.03548,0.03574,0.03609,0.03589,0.03634,0.03628,
      0.036,0.03572,0.03558,0.03495,0.03452,0.03396,0.03356,0.03279,0.03167,0.03098,
      0.03031,0.02923,0.02816,0.02758,0.02646,0.02552,0.02438,0.02386,0.02262,0.02204,
      0.02123,0.02042,0.02009,0.01915,0.01877,0.01823,0.01777,0.01717,0.01702,0.01674,
      0.01648,0.01633,0.01567,0.01544,0.01556,0.0153,0.01469,0.01441,0.01466,0.01395,
      0.01345,0.01324,0.01305,0.01256,0.01163,0.01137,0.01036,0.01011,0.00919,0.0085,
      0.0078,0.00667,0.00599,0.00531,0.00422,0.00318,0.0025,0.00173,0.00105,1e-05,
      -0.00114,-0.0015,-0.00185,-0.00318,-0.00352,-0.00375,-0.00403,-0.00415,-0.00492,-0.00469,
      -0.00476,-0.0048,-0.00449,-0.00424,-0.00401,-0.00372,-0.00356,-0.00305,-0.00217,-0.00174,
      -0.00133,-0.0004,2e-05,0.00034,0.00169,0.00197,0.00244,0.00333,0.00376,0.00417,
      0.00509,0.00556,0.00568,0.00626,0.00683,0.00698,0.00726,0.0075,0.00774,0.00809,
      0.00791,0.00827,0.00856,0.00848,0.00851,0.00849,0.00853,0.00841,0.00885,0.00874,
      0.00867,0.00911,0.0089,0.00928,0.00951,0.00967,0.01034,0.01052,0.0106,0.01162,
      0.01199,0.01243,0.01336,0.01368,0.01445,0.01559,0.01625,0.01695,0.01801,0.01908,
      0.01975,0.02045,0.02151,0.02256,0.02332,0.0237,0.02443,0.02564,0.02607,0.0261,
      0.02709,0.02763,0.0274,0.02777,0.02806,0.02799,0.02799,0.02805,0.02782,0.02716,
      0.02701,0.02681,0.02623,0.02573,0.02532,0.02447,0.02375,0.02299,0.02223,0.02155,
      0.02049,0.0194,0.01883,0.01775,0.01657,0.01643,0.01517,0.01429,0.01315,0.0125,
      0.01183,0.01057,0.01036,0.00945,0.00878,0.00791,0.00751,0.00707,0.00616,0.00574,
      0.00533,0.00442,0.00393,0.00385,0.00314,0.00307,0.00257,0.00159,0.00153,0.0013,
      0.00072,0.00027,-0.00033,-0.0005,-0.00069,-0.00128,-0.00173,-0.00235,-0.00241,-0.00302,
      -0.00358,-0.00367,-0.00428,-0.00473,-0.00535,-0.00542,-0.00596,-0.00681,-0.00735,-0.00734,
      -0.00829,-0.00907,-0.00941,-0.01036,-0.01068,-0.01144,-0.01262,-0.01316,-0.01428,-0.01533,
      -0.01593,-0.01703,-0.01802,-0.01894,-0.02027,-0.02157,-0.02251,-0.02345,-0.02476,-0.0261,
      -0.0269,-0.02837,-0.02923,-0.03026,-0.03134,-0.03193,-0.03302,-0.03407,-0.03478,-0.03534,
      -0.03541,-0.03603,-0.03654,-0.03679,-0.03682,-0.0364,-0.03658,-0.03629,-0.03572,-0.0353,
      -0.03457,-0.03338,-0.03285,-0.03169,-0.0308,-0.02956,-0.02783,-0.02706,-0.02587,-0.02389,
      -0.02246,-0.02123,-0.02003,-0.01857,-0.01656,-0.0156,-0.01398,-0.01235,-0.01157,-0.01032,
      -0.00858,-0.00777,-0.00681,-0.00541,-0.00451,-0.0035,-0.00242,-0.00176,-0.00101,-0.00016,
      0.00017,0.00099,0.0019,0.00192,0.00243,0.00341,0.00346,0.00376,0.00402,0.00417,
      0.00482,0.00507,0.00491,0.00529,0.00547,0.00583,0.00574,0.00568,0.00608,0.006,
      0.00592,0.00632,0.00633,0.00582,0.00637,0.00622,0.00624,0.00635,0.00582,0.00636,
      0.00628,0.00601,0.00568,0.00576,0.00582,0.00542,0.0055,0.00559,0.00513,0.00535,
      0.00498,0.00467,0.00485,0.00438,0.00462,0.00415,0.00424,0.00442,0.00357,0.00378,
      0.00382,0.00341,0.00353,0.00348,0.00356,0.00327,0.00285,0.00334,0.0033,0.00293,
      0.003,0.00308,0.00268,0.00267,0.00318,0.00263,0.00277,0.00282,0.00241,0.00252,
      0.00249,0.00251,0.00247,0.00259,0.00215,0.00228,0.00224,0.00224,0.00232,0.00193,
      0.00194,0.00232,0.00233,0.00185,0.00226,0.00264,0.00218,0.00211,0.00292,0.00274,
      0.00261,0.00335,0.00353,0.00353,0.00334,0.00413,0.00421,0.00452,0.00476,0.0049,
      0.00567,0.00541,0.0057,0.00632,0.00656,0.00644,0.00667,0.0074,0.00725,0.00709,
      0.00793,0.00771,0.00769,0.00806,0.00807,0.00766,0.00778,0.00775,0.00771,0.00792,
      0.00707,0.00727,0.00739,0.00666,0.00644,0.00656,0.00631,0.00568,0.00544,0.00556,
      0.00534,0.00457,0.00486,0.0045,0.00409,0.00467,0.00424,0.00385,0.00436,0.00423,
      0.00419,0.00454,0.00466,0.00534,0.00552,0.00559,0.00664,0.00689,0.00778,0.00853,
      0.00908,0.0107,0.01115,0.01222,0.01354,0.01466,0.01637,0.01741,0.01904,0.02052,
      0.02193,0.02385,0.02523,0.02679,0.02819,0.03004,0.03174,0.03356,0.0351,0.0361,
      0.03808,0.03966,0.04032,0.04211,0.04332,0.04414,0.04567,0.04621,0.04703,0.04773,
      0.04857,0.04905,0.04923,0.04957,0.04949,0.04947,0.04965,0.04895,0.0485,0.04804,
      0.0473,0.04623,0.04521,0.0444,0.04264,0.04148,0.04039,0.03838,0.03699,0.03563,
      0.0334,0.03168,0.03033,0.02846,0.02674,0.02501,0.02315,0.02184,0.01992,0.0184,
      0.01743,0.01533,0.01418,0.01343,0.01156,0.01079,0.00975,0.00865,0.00812,0.00684,
      0.0065,0.00616,0.00483,0.00448,0.00423,0.00406,0.00351,0.00285,0.00308,0.00281,
      0.00216,0.00205,0.00165,0.00183,0.00155,0.00092,0.00077,0.00048,0.00033,-0.0003,
      -0.00066,-0.0016,-0.00199,-0.00244,-0.00329,-0.00393,-0.00503,-0.006,-0.00703,-0.00793,
      -0.00927,-0.01052,-0.01166,-0.01337,-0.01441,-0.01603,-0.01753,-0.0189,-0.02093,-0.02191,
      -0.02346,-0.02532,-0.0268,-0.02801,-0.02918,-0.03083,-0.03206,-0.03297,-0.03408,-0.03474,
      -0.03548,-0.03639,-0.03647,-0.03673,-0.03714,-0.03671,-0.0365,-0.0363,-0.0358,-0.03499,
      -0.03423,-0.03357,-0.03246,-0.03157,-0.03022,-0.029,-0.02775,-0.02647,-0.02536,-0.02356,
      -0.02286,-0.02148,-0.02017,-0.01935,-0.0179,-0.01702,-0.01598,-0.01502,-0.01392,-0.01326,
      -0.01251,-0.01166,-0.01133,-0.01048,-0.0097,-0.00918,-0.00906,-0.00857,-0.0076,-0.00752,
      -0.00733,-0.00663,-0.00662,-0.00589,-0.00576,-0.00559,-0.00483,-0.00509,-0.0048,-0.00418,
      -0.00393,-0.00409,-0.00372,-0.00351,-0.00325,-0.00294,-0.003,-0.00306,-0.00276,-0.00235,
      -0.00285,-0.00279,-0.00244,-0.00245,-0.00276,-0.00307,-0.00298,-0.00302,-0.00295,-0.00319,
      -0.00382,-0.00407,-0.00391,-0.00428,-0.00452,-0.00465,-0.00544,-0.00514,-0.00545,-0.00616,
      -0.00591,-0.00618,-0.00691,-0.00674,-0.0066,-0.00741,-0.0073,-0.00684,-0.00735,-0.0073,
      -0.00692,-0.00701,-0.00705,-0.00679,-0.00623,-0.00575,-0.00524,-0.00475,-0.0042,-0.00389,
      -0.00269,-0.00174,-0.00082,0.00064,0.00134,0.00308,0.00451,0.00596,0.00781,0.00942,
      0.01166,0.01309,0.0156,0.01777,0.01996,0.02265,0.0246,0.02767,0.02999,0.03256,
      0.03503,0.03748,0.04028,0.04309,0.04559,0.04771,0.05029,0.05283,0.05508,0.05702,
      0.05901,0.06111,0.06273,0.06459,0.06606,0.06727,0.06849,0.06984,0.07078,0.07158,
      0.07198,0.07249,0.07314,0.07295,0.07306,0.07282,0.07214,0.07207,0.07158,0.07052,
      0.06924,0.06789,0.06711,0.06559,0.06336,0.06181,0.05979,0.05735,0.05565,0.05291,
      0.05047,0.04804,0.04514,0.04272,0.04028,0.03747,0.03464,0.0323,0.02948,0.02656,
      0.02458,0.02197,0.01933,0.01755,0.0153,0.01256,0.01078,0.00872,0.00667,0.00499,
      0.00321,0.00156,-0.0006,-0.00231,-0.00378,-0.0052,-0.00703,-0.00883,-0.01027,-0.01171,
      -0.01352,-0.01535,-0.01668,-0.01855,-0.02028,-0.02196,-0.02403,-0.02608,-0.02779,-0.02937,
      -0.0319,-0.03373,-0.0358,-0.03778,-0.03972,-0.04204,-0.04426,-0.04659,-0.04853,-0.05046,
      -0.05278,-0.05509,-0.05701,-0.05903,-0.06103,-0.06301,-0.0651,-0.06675,-0.06851,-0.07034,
      -0.07174,-0.07332,-0.07459,-0.07539,-0.07692,-0.07753,-0.07798,-0.07866,-0.07838,-0.07883,
      -0.07886,-0.0782,-0.0778,-0.07704,-0.07596,-0.07505,-0.07376,-0.07229,-0.07049,-0.06863,
      -0.06737,-0.06526,-0.0629,-0.06098,-0.05905,-0.05668,-0.05456,-0.05195,-0.04942,-0.04721,
      -0.04505,-0.04243,-0.03999,-0.03747,-0.03489,-0.03283,-0.03012,-0.028,-0.02571,-0.02348,
      -0.02123,-0.0189,-0.01701,-0.01489,-0.01326,-0.01146,-0.00975,-0.00791,-0.00644,-0.00526,
      -0.0039,-0.00307,-0.00174,-0.00035,0.00022,0.00101,0.00182,0.00224,0.00276,0.00326,
      0.00374,0.0043,0.00458,0.00439,0.00486,0.00471,0.00477,0.00474,0.00472,0.00493,
      0.00404,0.00439,0.00394,0.00376,0.00357,0.0029,0.00284,0.00222,0.00176,0.00122,
      0.00084,-0.0001,-0.00051,-0.00086,-0.00212,-0.00275,-0.00343,-0.00462,-0.00516,-0.00628,
      -0.00735,-0.00786,-0.00929,-0.01059,-0.01142,-0.0128,-0.014,-0.01527,-0.01653,-0.01769,
      -0.01933,-0.0206,-0.02136,-0.02303,-0.02466,-0.0254,-0.02681,-0.02798,-0.02935,-0.03022,
      -0.03129,-0.03223,-0.03333,-0.03406,-0.03442,-0.03533,-0.03581,-0.03598,-0.03627,-0.03648,
      -0.03682,-0.03679,-0.03649,-0.03625,-0.03598,-0.03584,-0.03511,-0.03517,-0.0342,-0.03341,
      -0.03316,-0.03188,-0.03126,-0.03055,-0.02946,-0.02852,-0.02738,-0.02691,-0.02547,-0.02408,
      -0.02366,-0.02221,-0.02091,-0.02007,-0.01873,-0.0174,-0.0166,-0.01514,-0.01423,-0.0134,
      -0.01155,-0.01078,-0.0098,-0.00841,-0.00749,-0.00655,-0.00523,-0.00392,-0.00301,-0.00196,
      -0.00109,0.00038,0.00115,0.00254,0.00383,0.00469,0.00603,0.00726,0.00853,0.00968,
      0.01136,0.01248,0.01377,0.01502,0.0162,0.01777,0.01933,0.02051,0.02169,0.02336,
      0.02449,0.02576,0.0271,0.02791,0.02935,0.03031,0.03094,0.03202,0.03306,0.03381,
      0.03424,0.03476,0.03524,0.03581,0.03604,0.03601,0.03592,0.03631,0.03637,0.03567,
      0.03542,0.03566,0.03494,0.0345,0.03408,0.03311,0.03293,0.03161,0.031,0.0304,
      0.02879,0.02835,0.02727,0.02594,0.0249,0.0244,0.02297,0.02158,0.02118,0.01963,
      0.01869,0.01807,0.01696,0.01601,0.01493,0.01418,0.0138,0.01307,0.01186,0.01143,
      0.01141,0.01039,0.00994,0.00984,0.00915,0.009,0.00883,0.00815,0.00803,0.00773,
      0.0075,0.00727,0.00689,0.00719,0.00636,0.0063,0.00597,0.00575,0.00557,0.00489,
      0.00485,0.00415,0.00401,0.0038,0.00325,0.00265,0.00259,0.00197,0.00144,0.00126,
      0.00098,0.00083,0.00017,-5e-05,1e-05,-1e-05,1e-05,-5e-05,0.00018,0.00085,
      0.00094,0.00138,0.00263,0.00319,0.00413,0.00588,0.00695,0.00841,0.01058,0.01244,
      0.01482,0.01695,0.01951,0.02222,0.02531,0.02831,0.03116,0.03483,0.03828,0.04182,
      0.0452,0.04914,0.05253,0.05597,0.05988,0.06326,0.06685,0.07009,0.07301,0.0761,
      0.07882,0.08133,0.08357,0.08555,0.08739,0.08849,0.08973,0.09121,0.09152,0.09172,
      0.09213,0.09178,0.09123,0.09078,0.09011,0.0887,0.08749,0.08628,0.08479,0.08296,
      0.08123,0.07952,0.07756,0.07512,0.07331,0.07104,0.06832,0.06655,0.06435,0.06138,
      0.05892,0.05681,0.05418,0.0518,0.04896,0.04616,0.04375,0.04119,0.03882,0.03587,
      0.03349,0.03102,0.02822,0.02542,0.02289,0.0209,0.01785,0.01549,0.01304,0.01014,
      0.00772,0.00531,0.00238,-3e-05,-0.00238,-0.00564,-0.00827,-0.01095,-0.01411,-0.01679,
      -0.0194,-0.02281,-0.02609,-0.02901,-0.03205,-0.03499,-0.0382,-0.04195,-0.04475,-0.04772,
      -0.05112,-0.05402,-0.057,-0.06018,-0.06248,-0.06504,-0.06757,-0.06982,-0.07183,-0.07348,
      -0.07539,-0.07655,-0.0775,-0.07854,-0.07938,-0.07946,-0.0798,-0.07986,-0.07923,-0.07872,
      -0.07838,-0.07725,-0.07606,-0.07447,-0.07294,-0.07172,-0.06907,-0.06733,-0.06501,-0.06246,
      -0.06004,-0.05722,-0.05445,-0.0518,-0.04869,-0.04571,-0.0428,-0.03938,-0.03649,-0.03346,
      -0.03047,-0.02749,-0.0244,-0.02174,-0.01898,-0.01616,-0.01375,-0.01121,-0.00876,-0.00608,
      -0.00425,-0.00232,0.00018,0.00172,0.00352,0.00533,0.00675,0.00827,0.00975,0.01132,
      0.01256,0.0135,0.01451,0.01551,0.0165,0.01756,0.01832,0.01868,0.0195,0.02039,
      0.02049,0.02067,0.02135,0.02146,0.02181,0.02181,0.0214,0.0216,0.02122,0.02099,
      0.02081,0.02023,0.01973,0.01934,0.01841,0.01799,0.01764,0.01638,0.01578,0.01498,
      0.01425,0.01349,0.01275,0.01199,0.01125,0.01049,0.00974,0.00901,0.00816,0.00783,
      0.00698,0.00618,0.00576,0.00525,0.00472,0.00433,0.00342,0.00294,0.0028,0.00231,
      0.00143,0.00093,0.00085,0.00014,7e-05,-0.00043,-0.00141,-0.00147,-0.0017,-0.00227,
      -0.00276,-0.00324,-0.00382,-0.004,-0.00419,-0.00478,-0.00523,-0.00582,-0.006,-0.00617,
      -0.00684,-0.00699,-0.0072,-0.00775,-0.00832,-0.00848,-0.00876,-0.00901,-0.00919,-0.00976,
      -0.0103,-0.01057,-0.01043,-0.0107,-0.01131,-0.01159,-0.01133,-0.01211,-0.0123,-0.01218,
      -0.01252,-0.01273,-0.01307,-0.01297,-0.01303,-0.01291,-0.01333,-0.01331,-0.01292,-0.01303,
      -0.01298,-0.01307,-0.01275,-0.01242,-0.01257,-0.01231,-0.01166,-0.01153,-0.01124,-0.011,
      -0.01075,-0.01048,-0.01033,-0.00966,-0.0095,-0.00933,-0.00866,-0.00852,-0.00827,-0.0079,
      -0.00815,-0.00748,-0.00686,-0.00708,-0.00681,-0.00616,-0.00603,-0.00572,-0.00557,-0.00498,
      -0.00444,-0.00424,-0.00406,-0.0035,-0.0029,-0.00292,-0.0019,-0.00142,-0.00142,-0.00037,
      -4e-05,0.00052,0.001,0.00149,0.00209,0.00216,0.00278,0.00323,0.00383,0.00398,
      0.00427,0.00448,0.00482,0.00477,0.00456,0.0055,0.00497,0.00459,0.00518,0.00469,
      0.00452,0.00421,0.0041,0.0034,0.00327,0.00305,0.00248,0.00201,0.00141,0.00134,
      0.00072,0.00025,-0.00026,-0.00075,-0.00126,-0.00174,-0.00232,-0.0025,-0.00269,-0.00325,
      -0.00382,-0.00398,-0.00426,-0.0045,-0.00473,-0.00509,-0.00491,-0.00526,-0.00556,-0.00548,
      -0.00551,-0.00548,-0.00559,-0.00518,-0.00517,-0.00567,-0.00513,-0.00527,-0.00533,-0.00491,
      -0.00502,-0.005,-0.005,-0.00502,-0.00493,-0.00525,-0.00558,-0.00543,-0.0057,-0.00629,
      -0.00666,-0.00759,-0.00801,-0.00837,-0.00957,-0.01041,-0.01185,-0.01276,-0.01365,-0.01521,
      -0.01706,-0.01862,-0.01951,-0.02189,-0.02351,-0.02512,-0.02762,-0.02948,-0.03154,-0.03344,
      -0.03579,-0.038,-0.04027,-0.04251,-0.04477,-0.04702,-0.04923,-0.05167,-0.05318,-0.05519,
      -0.05767,-0.05922,-0.06101,-0.06283,-0.06423,-0.06582,-0.06706,-0.06798,-0.06906,-0.06983,
      -0.07016,-0.07106,-0.07162,-0.07145,-0.07159,-0.07121,-0.07105,-0.07057,-0.06963,-0.0694,
      -0.06827,-0.06697,-0.06581,-0.06423,-0.06273,-0.06129,-0.05955,-0.05745,-0.05558,-0.05315,
      -0.05125,-0.0493,-0.04694,-0.04485,-0.04206,-0.04033,-0.03806,-0.03534,-0.03351,-0.03155,
      -0.02923,-0.02695,-0.02491,-0.02334,-0.02116,-0.01943,-0.01809,-0.01614,-0.01478,-0.01318,
      -0.01194,-0.01101,-0.00998,-0.00901,-0.00794,-0.0072,-0.00673,-0.00636,-0.00532,-0.0053,
      -0.005,-0.00467,-0.00485,-0.0044,-0.00454,-0.00442,-0.00476,-0.00506,-0.00502,-0.00487,
      -0.0055,-0.00617,-0.00582,-0.00654,-0.00707,-0.00717,-0.00778,-0.00825,-0.00876,-0.00925,
      -0.00977,-0.01018,-0.01101,-0.01181,-0.01225,-0.01269,-0.01351,-0.01433,-0.01469,-0.01545,
      -0.01657,-0.01732,-0.01767,-0.01853,-0.01927,-0.0199,-0.02117,-0.02147,-0.02196,-0.02275,
      -0.02356,-0.02407,-0.02418,-0.02476,-0.02532,-0.02548,-0.02575,-0.02605,-0.02605,-0.02575,
      -0.02542,-0.02557,-0.02529,-0.02474,-0.02423,-0.02382,-0.02298,-0.02223,-0.02158,-0.02041,
      -0.01973,-0.01914,-0.01765,-0.01669,-0.01607,-0.01497,-0.01401,-0.01296,-0.0121,-0.01058,
      -0.01,-0.0094,-0.00781,-0.0073,-0.00648,-0.00572,-0.0051,-0.00384,-0.0035,-0.00316,
      -0.00181,-0.00156,-0.00097,-0.00051,2e-05,0.0005,0.001,0.00157,0.00173,0.00203,
      0.00214,0.00294,0.00263,0.00303,0.00332,0.00315,0.00359,0.00348,0.00351,0.0035,
      0.00348,0.00357,0.00325,0.00292,0.00307,0.00281,0.00217,0.002,0.00184,0.00113,
      0.0011,0.00046,-1e-05,-0.00043,-0.00134,-0.00175,-0.00217,-0.00309,-0.00352,-0.00384,
      -0.00518,-0.00549,-0.00587,-0.0071,-0.00779,-0.00826,-0.00869,-0.00951,-0.01031,-0.01076,
      -0.01116,-0.0121,-0.01248,-0.013,-0.01358,-0.01367,-0.01426,-0.01482,-0.01498,-0.01527,
      -0.01548,-0.01583,-0.01574,-0.01568,-0.01606,-0.01607,-0.01567,-0.01573,-0.01592,-0.01505,
      -0.01536,-0.01504,-0.01442,-0.01427,-0.01398,-0.01382,-0.01323,-0.01275,-0.01225,-0.01172,
      -0.01133,-0.01042,-0.00994,-0.00981,-0.00929,-0.00852,-0.00759,-0.0076,-0.00703,-0.00625,
      -0.00545,-0.00491,-0.00492,-0.00387,-0.00354,-0.00298,-0.0025,-0.00201,-0.00141,-0.00133,
      -0.00074,-0.00019,0.0,0.00017,0.00084,0.00098,0.00124,0.00159,0.00141,0.00176,
      0.00209,0.0019,0.00232,0.00233,0.00183,0.00235,0.00228,0.00201,0.00169,0.00173,
      0.00193,0.00105,0.00135,0.00107,0.00033,0.00059,0.00029,-0.00027,-0.00073,-0.00135,
      -0.00141,-0.00201,-0.0026,-0.0026,-0.00353,-0.00431,-0.00476,-0.00518,-0.00603,-0.00675,
      -0.00748,-0.00836,-0.0086,-0.00976,-0.01092,-0.01108,-0.01229,-0.01331,-0.01399,-0.01478,
      -0.01542,-0.0166,-0.01723,-0.01799,-0.01884,-0.01917,-0.01999,-0.02091,-0.0209,-0.0215,
      -0.02216,-0.02188,-0.02234,-0.02231,-0.02193,-0.02198,-0.02214,-0.02145,-0.02101,-0.0205,
      -0.01996,-0.01965,-0.01838,-0.01778,-0.01699,-0.01623,-0.01557,-0.01447,-0.01352,-0.01242,
      -0.01175,-0.01105,-0.01,-0.00893,-0.00827,-0.00749,-0.00674,-0.00602,-0.00515,-0.00486,
      -0.00389,-0.00353,-0.00299,-0.00249,-0.00204,-0.00132,-0.0017,-0.0009,-0.00067,-0.00094,
      -7e-05,-0.00028,-0.00032,9e-05,-2e-05,1e-05,-0.0,1e-05,-2e-05,7e-05,
      -0.00026,-0.00056,-0.0005,-0.00043,-0.00077,-0.001,-0.00123,-0.00159,-0.00141,-0.00178,
      -0.00199,-0.00226,-0.0025,-0.00273,-0.00308,-0.00294,-0.00316,-0.00393,-0.00364,-0.00401,
      -0.00435,-0.00408,-0.00485,-0.00504,-0.00498,-0.00502,-0.00494,-0.00523,-0.00565,-0.00516,
      -0.00518,-0.00565,-0.00523,-0.00492,-0.00509,-0.0047,-0.00458,-0.00399,-0.00342,-0.00332,
      -0.0028,-0.00198,-0.00126,-0.00048,0.0002,0.00121,0.00252,0.00375,0.00502,0.00619,
      0.00777,0.00927,0.01071,0.01244,0.01459,0.01625,0.01797,0.01994,0.02232,0.02434,
      0.02587,0.02826,0.03066,0.03215,0.03427,0.03635,0.03788,0.04018,0.04146,0.043,
      0.04458,0.0457,0.04715,0.04772,0.0485,0.04931,0.04977,0.05013,0.04971,0.0495,
      0.0493,0.04879,0.04803,0.04706,0.04571,0.04456,0.04304,0.04124,0.03948,0.03782,
      0.03573,0.03371,0.0319,0.02914,0.027,0.02482,0.02215,0.02,0.0178,0.01522,
      0.01275,0.01023,0.00776,0.00518,0.00297,0.00089,-0.0021,-0.00457,-0.00665,-0.00937,
      -0.01142,-0.01399,-0.01668,-0.01834,-0.02102,-0.02366,-0.02541,-0.02777,-0.03007,-0.032,
      -0.03399,-0.03609,-0.03772,-0.03958,-0.04099,-0.04249,-0.04412,-0.04506,-0.04566,-0.04682,
      -0.04761,-0.04773,-0.04798,-0.04837,-0.04805,-0.04741,-0.04731,-0.04684,-0.0458,-0.04442,
      -0.04351,-0.04252,-0.04139,-0.03939,-0.038,-0.03658,-0.03464,-0.03334,-0.03146,-0.02977,
      -0.02792,-0.0265,-0.02507,-0.02318,-0.02168,-0.02057,-0.01901,-0.01737,-0.01652,-0.01555,
      -0.01427,-0.01286,-0.01228,-0.01151,-0.01066,-0.01033,-0.00951,-0.00859,-0.0086,-0.00801,
      -0.00735,-0.00759,-0.00731,-0.00668,-0.00645,-0.00652,-0.00647,-0.00659,-0.00619,-0.00616,
      -0.00667,-0.00616,-0.00619,-0.00658,-0.0065,-0.00641,-0.00684,-0.00674,-0.00669,-0.007,
      -0.00731,-0.00725,-0.00717,-0.00758,-0.0075,-0.00742,-0.00784,-0.00775,-0.00767,-0.00808,
      -0.008,-0.00792,-0.00833,-0.00825,-0.00815,-0.00867,-0.00818,-0.00811,-0.0089,-0.00881,
      -0.00835,-0.00877,-0.00906,-0.009,-0.00894,-0.00925,-0.00958,-0.00942,-0.00975,-0.01008,
      -0.00992,-0.01025,-0.01058,-0.01041,-0.01077,-0.011,-0.01123,-0.01159,-0.01142,-0.01175,
      -0.01209,-0.01189,-0.01234,-0.01225,-0.01217,-0.0126,-0.01245,-0.0126,-0.01216,-0.01226,
      -0.01231,-0.01198,-0.01175,-0.0115,-0.01126,-0.01097,-0.01084,-0.01014,-0.01009,-0.0095,
      -0.00889,-0.00892,-0.00791,-0.00744,-0.00734,-0.0067,-0.00634,-0.00543,-0.00493,-0.00483,
      -0.00423,-0.00375,-0.00325,-0.00274,-0.00227,-0.00166,-0.00158,-0.001,-0.00039,-0.00043,
      0.0006,0.00104,0.00125,0.00146,0.00191,0.0029,0.003,0.0031,0.00411,0.00447,
      0.00501,0.0055,0.006,0.00652,0.0069,0.00787,0.00814,0.0091,0.00949,0.00994,
      0.01077,0.0115,0.01225,0.013,0.01375,0.0145,0.01527,0.01591,0.01709,0.01775,
      0.01841,0.01961,0.02016,0.02124,0.0224,0.02267,0.02344,0.02458,0.02523,0.026,
      0.0268,0.02732,0.02742,0.02801,0.02855,0.02881,0.02874,0.02874,0.0288,0.02855,
      0.02801,0.0274,0.02739,0.02654,0.02543,0.02473,0.02416,0.02264,0.02176,0.02081,
      0.01948,0.01828,0.01691,0.01609,0.01472,0.0135,0.01226,0.01093,0.01002,0.00899,
      0.008,0.007,0.00599,0.00501,0.00395,0.00318,0.00281,0.00208,0.00087,0.00043,
      0.00042,-0.00061,-0.00097,-0.00149,-0.00209,-0.00217,-0.00276,-0.00332,-0.00348,-0.00377,
      -0.00393,-0.0045,-0.00509,-0.00514,-0.00584,-0.00599,-0.00621,-0.00668,-0.00757,-0.00807,
      -0.00816,-0.00879,-0.00918,-0.01001,-0.01081,-0.01125,-0.01169,-0.01251,-0.01327,-0.01391,
      -0.01509,-0.01572,-0.01653,-0.01718,-0.01825,-0.01934,-0.01991,-0.02103,-0.02197,-0.02309,
      -0.02367,-0.02476,-0.02583,-0.02642,-0.02751,-0.02854,-0.02932,-0.02967,-0.03051,-0.03129,
      -0.03183,-0.0319,-0.03258,-0.0328,-0.03272,-0.03285,-0.0324,-0.03256,-0.03236,-0.03149,
      -0.03068,-0.0303,-0.02964,-0.02815,-0.02723,-0.0264,-0.02464,-0.02353,-0.02222,-0.02107,
      -0.01951,-0.01789,-0.01692,-0.01493,-0.01334,-0.01269,-0.01088,-0.00975,-0.0086,-0.00684,
      -0.00602,-0.00507,-0.00368,-0.00268,-0.0021,-0.0009,-0.00029,0.00057,0.001,0.00141,
      0.00235,0.00271,0.00332,0.0035,0.00367,0.00432,0.00454,0.00451,0.00443,0.00476,
      0.00504,0.00507,0.00469,0.00468,0.00509,0.00495,0.0051,0.00466,0.00477,0.00475,
      0.00472,0.00486,0.00433,0.00484,0.00483,0.00433,0.00484,0.00483,0.00436,0.00475,
      0.00515,0.00466,0.00471,0.00499,0.00531,0.00525,0.00517,0.00558,0.0055,0.00542,
      0.00583,0.00575,0.00567,0.00608,0.006,0.00592,0.00633,0.00626,0.00614,0.00667,
      0.00616,0.00619,0.00658,0.00648,0.00651,0.0065,0.0065,0.0065,0.0065,0.0065,
      0.0065,0.0065,0.0065,0.0065,0.00651,0.00648,0.00659,0.00616,0.00627,0.00625,
      0.00623,0.00634,0.00591,0.00602,0.006,0.00598,0.00609,0.00566,0.00576,0.00581,
      0.0055,0.00519,0.00524,0.00534,0.00491,0.00502,0.005,0.00498,
     ),
    ),
    # highland: 3500 uniform samples; public map SHA256 d75ffc483e1874ed4746453d003c1f99173b577c233bc477a1f9e0ec21e1b555
    (6999.367280545016,
     (
      1.26e-05,-4.77e-05,4.76e-05,-1.25e-05,2.4e-06,2.9e-06,-1.41e-05,5.34e-05,-6.91e-05,9.27e-05,
      -9.7e-05,9.07e-05,-6.12e-05,7.99e-05,-5.41e-05,6e-06,3.01e-05,3.8e-06,-4.53e-05,4.71e-05,
      -1.26e-05,3.4e-06,-1.2e-06,1.3e-06,-4e-06,1.45e-05,2e-05,-3.85e-05,3.5e-06,2.43e-05,
      -2.64e-05,7.1e-06,-2.1e-06,1.4e-06,-3.5e-06,1.26e-05,-4.7e-05,4.51e-05,-3.1e-06,-3.26e-05,
      3.3e-06,1.94e-05,-2.46e-05,2.3e-05,-6.76e-05,4.28e-05,2.64e-05,-7.42e-05,6.58e-05,-5.9e-05,
      4.02e-05,-2.75e-05,-4.1e-06,4.39e-05,-4.14e-05,-8.5e-06,7.52e-05,-8.81e-05,7.28e-05,1e-06,
      -2.06e-05,2.53e-05,4.96e-05,-1.93e-05,2.78e-05,3.83e-05,2.35e-05,-2e-06,5.9e-05,2.66e-05,
      3.93e-05,2.1e-05,7.3e-06,0.0001544,-8.57e-05,0.0001147,3.58e-05,7.74e-05,-1.03e-05,9.42e-05,
      4.23e-05,7.14e-05,6.9e-06,0.000106,3.4e-05,9.3e-05,3.2e-06,0.0001547,-8.3e-06,0.0001394,
      6.44e-05,6.82e-05,0.0001278,3.46e-05,0.0001988,4.4e-05,0.0001656,0.000168,0.000167,0.0001685,
      0.0002384,-4.27e-05,-0.0003454,-0.0014553,-0.0034984,-0.0074364,-0.0137847,-0.0229925,-0.0352011,-0.0487793,
      -0.0604182,-0.0691377,-0.0748343,-0.0780814,-0.0808315,-0.0837485,-0.0866687,-0.0887991,-0.0883038,-0.0845045,
      -0.0778458,-0.069012,-0.058148,-0.0459457,-0.0335823,-0.0221106,-0.0125723,-0.0060719,-0.0023703,-0.000429,
      0.000263,0.0004599,0.0003131,0.0001498,2.26e-05,-0.0001121,-0.0001911,-0.0003339,-0.0002871,-0.0004124,
      -0.0004108,-0.0004292,-0.0004959,-0.000481,-0.0004596,-0.0005798,-0.0005739,-0.0005517,-0.0006509,-0.0006704,
      -0.0006281,-0.0007837,-0.000668,-0.0008468,-0.0007747,-0.0008242,-0.000899,-0.0008823,-0.0009403,-0.0009305,
      -0.0009779,-0.0010641,-0.0010081,-0.00108,-0.0011846,-0.0010896,-0.0012399,-0.001197,-0.0012885,-0.0012304,
      -0.0013772,-0.0013104,-0.0012983,-0.0014174,-0.0012798,-0.0013166,-0.0012994,-0.0012019,-0.0012066,-0.0010784,
      -0.0010493,-0.0008934,-0.000806,-0.0007125,-0.0005613,-0.0004588,-0.0003499,-0.0002168,-0.0001162,1.89e-05,
      4.46e-05,0.000144,0.0002588,0.0001602,0.0003149,0.0002547,0.000208,0.0001932,0.0001527,-1.01e-05,
      -4.49e-05,-0.0001531,-0.0003588,-0.0003688,-0.0005967,-0.0007377,-0.0008315,-0.0010468,-0.0010335,-0.0013459,
      -0.00131,-0.0015559,-0.0015325,-0.0017382,-0.0017342,-0.0018823,-0.0019063,-0.0019979,-0.0020175,-0.0021052,
      -0.0020934,-0.0021831,-0.002118,-0.0022134,-0.0021801,-0.0022187,-0.0021771,-0.0022238,-0.0021138,-0.0022218,
      -0.00206,-0.0021897,-0.0020336,-0.0020757,-0.0019373,-0.0020363,-0.0018582,-0.0019365,-0.0017962,-0.0018298,
      -0.0017119,-0.0017344,-0.0016704,-0.001633,-0.0016689,-0.0015279,-0.00155,-0.0015742,-0.0014515,-0.0015017,
      -0.0014206,-0.001461,-0.0013779,-0.0013678,-0.0013744,-0.0013334,-0.0013037,-0.0013485,-0.0012906,-0.0012678,
      -0.0013241,-0.001311,-0.0013078,-0.0013543,-0.0013633,-0.0013961,-0.0014508,-0.0014263,-0.0015713,-0.0015488,
      -0.0016942,-0.0016679,-0.0017426,-0.0018829,-0.0018793,-0.001978,-0.0019984,-0.0021343,-0.0021005,-0.0022049,
      -0.0022442,-0.0022965,-0.002258,-0.0023622,-0.0022938,-0.002362,-0.0022573,-0.0022937,-0.0022509,-0.0021723,
      -0.0022147,-0.0020136,-0.0020361,-0.0019382,-0.0018766,-0.0018115,-0.0016918,-0.0016187,-0.0015884,-0.0013744,
      -0.0014094,-0.0012672,-0.0011651,-0.0010814,-0.0009957,-0.0008719,-0.000833,-0.0006872,-0.0006619,-0.0004824,
      -0.0004975,-0.0003868,-0.0002934,-0.0002432,-0.0002288,-0.0001061,-9.34e-05,-5.52e-05,7.8e-06,9.2e-06,
      8.82e-05,4.42e-05,6.13e-05,0.0001367,0.0001402,4.87e-05,0.0001891,5.4e-05,0.000117,0.0001292,
      9.7e-05,1.77e-05,0.0001602,-2.85e-05,5.29e-05,4.37e-05,-3.44e-05,-4.1e-06,-4.83e-05,-3.09e-05,
      -0.0001408,-3.89e-05,-0.0001451,-0.0001133,-0.0001511,-0.0001451,-0.0002145,-0.0001675,-0.0001902,-0.0002284,
      -0.0001808,-0.0002208,-0.0002054,-0.0002443,-0.0001996,-0.0002265,-0.0001789,-0.0002127,-0.0001578,-0.0002143,
      -0.0001546,-0.0001265,-0.0001863,-8.56e-05,-0.0001057,-0.0001439,-4.74e-05,-0.0001076,-4.04e-05,-5.66e-05,
      -4.07e-05,-8.4e-06,-2.25e-05,2.4e-06,-4.5e-06,1.67e-05,5.32e-05,-1.9e-05,0.0001408,-2.79e-05,
      8.8e-05,9.97e-05,4.96e-05,0.0001086,7.06e-05,0.0001298,6.07e-05,0.0001492,9.02e-05,0.0001244,
      6.28e-05,0.0001465,9.9e-05,9.11e-05,7.3e-05,3.92e-05,9.67e-05,-5.5e-06,2.29e-05,2.74e-05,
      -0.0001366,-1.62e-05,-0.00011,-0.0001984,-0.0001539,-0.0002463,-0.0002638,-0.0003937,-0.0002924,-0.0004583,
      -0.0004299,-0.0004939,-0.0005746,-0.0005106,-0.0005736,-0.0006156,-0.0005746,-0.0006012,-0.0005342,-0.0005652,
      -0.000409,-0.0004473,-0.000255,-0.000239,-4.86e-05,0.0001136,0.0002344,0.0004573,0.0007334,0.0008621,
      0.0012689,0.0014303,0.0018616,0.0021048,0.0025099,0.0028753,0.0032373,0.0036406,0.0040081,0.0044547,
      0.0048431,0.0051653,0.0056049,0.0058481,0.0062363,0.0064577,0.0066706,0.0068943,0.0069614,0.0070366,
      0.0070385,0.0069903,0.0068184,0.0067593,0.006371,0.0062442,0.0058202,0.0055123,0.0050662,0.0046649,
      0.0041729,0.0037177,0.003199,0.0026087,0.0021685,0.0015579,0.0009921,0.0004452,-9.51e-05,-0.0007076,
      -0.0012662,-0.0017887,-0.0023844,-0.0028989,-0.0034883,-0.003937,-0.0045485,-0.0050214,-0.0054271,-0.0060399,
      -0.0063927,-0.0068265,-0.0072748,-0.0075679,-0.0079974,-0.0082524,-0.0085259,-0.0088188,-0.0089089,-0.0091655,
      -0.0091846,-0.0093726,-0.0092396,-0.0094041,-0.0092323,-0.009205,-0.0091152,-0.0089672,-0.0088134,-0.0086443,
      -0.0085251,-0.0082688,-0.0081171,-0.0079039,-0.0077631,-0.0075309,-0.007422,-0.0071727,-0.0071076,-0.0069377,
      -0.0068007,-0.0067182,-0.0065921,-0.0065181,-0.0064361,-0.006386,-0.0063809,-0.0061971,-0.0063372,-0.0061341,
      -0.0061837,-0.0061357,-0.0060858,-0.006061,-0.0059804,-0.0058717,-0.0058329,-0.0056492,-0.0055503,-0.0054064,
      -0.0051324,-0.005009,-0.0046551,-0.0044611,-0.004055,-0.0037964,-0.0033865,-0.0030144,-0.0025728,-0.0021658,
      -0.001602,-0.0012542,-0.000636,-0.0001072,0.0003203,0.0009901,0.0014702,0.0020835,0.0026505,0.0032271,
      0.0037492,0.004418,0.0048898,0.0054432,0.0060197,0.0064207,0.0069493,0.0072913,0.0077481,0.0079748,
      0.0082576,0.0085247,0.0085199,0.0086406,0.0086822,0.0084995,0.0084705,0.0082332,0.0079351,0.00771,
      0.0073903,0.0069365,0.0065822,0.0061727,0.0057587,0.0052824,0.0048043,0.0044396,0.0039209,0.003611,
      0.0031039,0.0027416,0.0023591,0.0020432,0.0017334,0.0013315,0.001171,0.0007976,0.0006251,0.0004817,
      0.0001437,0.0001471,-0.0001152,-0.0001884,-0.0002588,-0.0004029,-0.0003768,-0.0005074,-0.0005068,-0.0005033,
      -0.0005183,-0.0005432,-0.0004712,-0.0005287,-0.0004053,-0.0004734,-0.0004036,-0.0003257,-0.0003302,-0.0003085,
      -0.0001821,-0.0002522,-0.0001417,-0.0001351,-6.42e-05,-6.79e-05,-7.82e-05,4.94e-05,6.8e-06,4.3e-06,
      0.0001043,-8.2e-06,0.0001376,4.34e-05,0.0001036,8.28e-05,0.000106,3.29e-05,9.71e-05,0.0001194,
      -3.63e-05,0.0001543,-4.26e-05,9.82e-05,-1.85e-05,-2.47e-05,3.72e-05,-4.47e-05,-6.62e-05,-2.34e-05,
      -4.79e-05,-0.0001197,-1.29e-05,-0.0001644,-7.65e-05,-7.01e-05,-0.000104,-0.0001358,-9.97e-05,-5.12e-05,
      -0.0001105,-4.65e-05,-3.64e-05,-1.45e-05,-3.2e-05,6.38e-05,6.52e-05,0.0001369,0.0001358,0.0001961,
      0.0002899,0.0002673,0.000392,0.0004104,0.0005104,0.0004601,0.0006904,0.000564,0.0007611,0.000719,
      0.0008171,0.0007988,0.0008566,0.0008512,0.0009409,0.0007925,0.0008828,0.0008354,0.0007672,0.0007544,
      0.0006239,0.0006594,0.0003962,0.0004591,0.0002197,0.0001975,-5.65e-05,-5.87e-05,-0.0003791,-0.0003825,
      -0.0006768,-0.0007846,-0.000973,-0.0011538,-0.0013235,-0.001462,-0.0016962,-0.0018262,-0.0019454,-0.0021321,
      -0.0022214,-0.0024253,-0.0024864,-0.002612,-0.0026848,-0.0028101,-0.0029041,-0.0029407,-0.003012,-0.0031569,
      -0.0031176,-0.0032442,-0.0033166,-0.0032938,-0.0034465,-0.003457,-0.0035277,-0.0035005,-0.0037418,-0.0036066,
      -0.0037718,-0.003792,-0.003884,-0.0038283,-0.0039724,-0.0039406,-0.0039849,-0.004055,-0.0039475,-0.0040582,
      -0.003946,-0.0039512,-0.0039357,-0.0037974,-0.0037413,-0.0036793,-0.0035032,-0.0033682,-0.0031836,-0.003064,
      -0.0027771,-0.0026213,-0.0023524,-0.0021628,-0.001808,-0.001667,-0.0013477,-0.0011609,-0.0008454,-0.0007302,
      -0.0004128,-0.0002723,-5.97e-05,4.03e-05,0.0002302,0.0003169,0.0004664,0.0004762,0.0006169,0.0005832,
      0.000656,0.0006483,0.0006854,0.0006092,0.0006684,0.000571,0.0006124,0.0004389,0.0005795,0.0003731,
      0.0004005,0.0003526,0.0002691,0.0002394,0.0001948,0.0002177,6.66e-05,9.24e-05,3.73e-05,8.82e-05,
      -6.19e-05,-2.63e-05,2.33e-05,-0.0001094,-1.76e-05,-0.0001092,-6.07e-05,-8.05e-05,-9.12e-05,-2.83e-05,
      -0.0001261,-4.36e-05,-7.09e-05,-0.0001056,2.02e-05,-0.0001204,-1.18e-05,-1.77e-05,-6.22e-05,-2.17e-05,
      5e-06,-3.99e-05,1.06e-05,-2.6e-06,-3e-07,3.8e-06,-1.48e-05,5.57e-05,-2.33e-05,-3.6e-06,
      7.95e-05,-6.82e-05,9.12e-05,-8.7e-06,-1.52e-05,2.87e-05,4.46e-05,-2.19e-05,4.34e-05,-7.9e-06,
      -1.19e-05,5.61e-05,-2.73e-05,1.21e-05,2.04e-05,9.2e-06,-1.62e-05,1.44e-05,-2e-07,-1.37e-05,
      1.37e-05,1e-07,-1.4e-05,1.48e-05,-4.1e-06,1.4e-06,-4.3e-05,6.81e-05,-8.58e-05,9.02e-05,
      -9.04e-05,8.67e-05,-7.18e-05,1.52e-05,1.09e-05,-1.79e-05,1.94e-05,-1.88e-05,1.44e-05,-3.89e-05,
      3.86e-05,-1.28e-05,1.25e-05,-3.75e-05,3.48e-05,8e-07,-3.84e-05,5.03e-05,-6.03e-05,4.73e-05,
      1.53e-05,-6.77e-05,7.04e-05,-7.04e-05,6.74e-05,-1.41e-05,-5.26e-05,8.07e-05,-8.52e-05,7.52e-05,
      -7.18e-05,6.81e-05,-1.56e-05,-4.71e-05,6.02e-05,-5e-05,3.73e-05,3.8e-06,-5.26e-05,6.29e-05,
      -1.38e-05,-4.92e-05,6.7e-05,-3.36e-05,2.61e-05,-2.98e-05,5.18e-05,-3.37e-05,-1.97e-05,7.17e-05,
      -8.18e-05,7.06e-05,-1.52e-05,-9.7e-06,1.27e-05,2e-07,-1.35e-05,1.27e-05,3.9e-06,-2.85e-05,
      6.89e-05,-6.18e-05,3.43e-05,-7.55e-05,8.27e-05,-7.03e-05,1.32e-05,1.77e-05,-4.3e-05,-3.13e-05,
      2.42e-05,-6.56e-05,5.31e-05,-0.0001056,4e-05,-5.47e-05,-6.7e-06,-2.17e-05,-9.21e-05,6.08e-05,
      -0.0001103,5.09e-05,-9.35e-05,-6.5e-06,-2.49e-05,-3.84e-05,-7.1e-06,-3.64e-05,-3.27e-05,-1.81e-05,
      2e-06,-3.13e-05,-2.12e-05,-2.83e-05,-9.9e-06,2.67e-05,-5.58e-05,1.12e-05,1.08e-05,-5.46e-05,
      6.35e-05,-5.54e-05,1.39e-05,-3e-07,-1.27e-05,5.11e-05,-4.76e-05,-5.1e-06,6.8e-05,-8.18e-05,
      7.37e-05,-6.9e-05,5.81e-05,-1.91e-05,1.82e-05,-1.27e-05,-8.5e-06,4.66e-05,-3.37e-05,-1.5e-05,
      5.26e-05,-1e-05,-5.39e-05,8.13e-05,-8.61e-05,7.76e-05,-3.89e-05,3.7e-05,-6.79e-05,4.93e-05,
      1.5e-05,-6.83e-05,7.29e-05,-3.8e-05,3.8e-05,-7.29e-05,6.81e-05,-1.41e-05,-5.27e-05,8.07e-05,
      -8.47e-05,7.28e-05,-2.13e-05,-2.87e-05,3.29e-05,4e-07,-3.43e-05,3.37e-05,2.6e-06,-2.8e-06,
      -3.24e-05,2.92e-05,1.88e-05,-6.3e-05,4.81e-05,1.49e-05,-6.67e-05,6.64e-05,-1.36e-05,-5.31e-05,
      8.17e-05,-8.83e-05,8.63e-05,-7.17e-05,1.55e-05,9.8e-06,-1.35e-05,3.2e-06,8e-07,-6.5e-06,
      2.51e-05,-5.28e-05,1e-06,4.87e-05,-5.16e-05,1.38e-05,-3.7e-06,9e-07,0.0,-1e-06,
      4.1e-06,-1.53e-05,5.7e-05,-6.86e-05,7.35e-05,-8.12e-05,0.0001074,-6.02e-05,3.06e-05,4.08e-05,
      -8.6e-06,-6.4e-06,3.4e-05,1.44e-05,5.28e-05,-4.02e-05,6.69e-05,-4.24e-05,0.0001028,-3.97e-05,
      1.5e-05,2.07e-05,4.62e-05,-6.14e-05,5.55e-05,-1.65e-05,1.06e-05,-2.61e-05,-5.04e-05,4.23e-05,
      -7.77e-05,-6.05e-05,-9.9e-06,-4.43e-05,-0.0001014,-6.49e-05,-7.18e-05,-0.0001217,-5.95e-05,-0.0001144,
      -0.0001009,-0.0001413,-5.57e-05,-0.0001512,-0.0001019,-5.97e-05,-0.0001333,-2.5e-05,-9.67e-05,-2.13e-05,
      -3.6e-06,3.55e-05,4.66e-05,6.67e-05,0.00016,0.0001994,0.0001751,0.0003368,0.0003316,0.0003768,
      0.0004895,0.0005081,0.0005685,0.0006379,0.0006716,0.0007141,0.0008401,0.0007672,0.0008329,0.0008445,
      0.0008613,0.0008399,0.0008087,0.0007677,0.0007203,0.0005766,0.0005814,0.000374,0.000296,0.0001526,
      -9.9e-05,-0.0001836,-0.0004894,-0.0006567,-0.000962,-0.0011393,-0.0015479,-0.0017751,-0.0021252,-0.0024363,
      -0.0027969,-0.0031871,-0.0034907,-0.0039896,-0.0044148,-0.0049806,-0.0056371,-0.0063423,-0.0072409,-0.008199,
      -0.0094131,-0.0107891,-0.0123393,-0.0139568,-0.0158058,-0.0175667,-0.0193143,-0.0208501,-0.0222175,-0.0232301,
      -0.0240276,-0.0245496,-0.024864,-0.0251445,-0.0251983,-0.025373,-0.0253915,-0.0254334,-0.0253286,-0.0250789,
      -0.0247336,-0.0238642,-0.0229217,-0.021367,-0.0195378,-0.0173882,-0.0149688,-0.0125037,-0.0099203,-0.0074157,
      -0.0051201,-0.0028243,-0.0008306,0.0011443,0.0028598,0.0046469,0.0061362,0.0078771,0.0091601,0.0108724,
      0.0122257,0.0136492,0.0151372,0.0163637,0.0176609,0.0187818,0.019813,0.0207124,0.0214828,0.0220239,
      0.0227029,0.0230129,0.0234344,0.0235969,0.0237521,0.0238151,0.0236042,0.0232702,0.0226737,0.0217928,
      0.020717,0.0193674,0.0178305,0.0161569,0.0144833,0.0128826,0.0111743,0.0098039,0.0085583,0.0073279,
      0.006453,0.005532,0.0049645,0.0042366,0.0038325,0.0034107,0.0029772,0.0026602,0.002428,0.0021106,
      0.0019375,0.0015639,0.0015578,0.0011919,0.0010969,0.0009335,0.0007737,0.0007364,0.0005034,0.0004873,
      0.0003886,0.0002481,0.0001844,0.0001062,-3e-05,-9.19e-05,-0.00026,-0.0003946,-0.0004906,-0.0007356,
      -0.0009354,-0.0011279,-0.0013568,-0.0016687,-0.0020108,-0.0022234,-0.002704,-0.0030074,-0.0034606,-0.0037838,
      -0.0044457,-0.0048278,-0.0054884,-0.0062026,-0.0069247,-0.0077795,-0.0088471,-0.0098126,-0.0111027,-0.0122555,
      -0.0137274,-0.0149868,-0.0163677,-0.017684,-0.0188061,-0.0198928,-0.0207823,-0.0214195,-0.0220132,-0.0223253,
      -0.022424,-0.022446,-0.0221951,-0.0218027,-0.0212675,-0.0205332,-0.0196862,-0.0186252,-0.0175992,-0.0162887,
      -0.0151678,-0.0137509,-0.0125935,-0.011303,-0.0102239,-0.009085,-0.0081087,-0.0072496,-0.0064001,-0.0057137,
      -0.005071,-0.0044231,-0.0039705,-0.0034296,-0.002958,-0.0025945,-0.0021844,-0.0017316,-0.0014576,-0.0010909,
      -0.0007522,-0.0005592,-0.0002552,-4.17e-05,0.0001312,0.0002251,0.0005085,0.0004454,0.0006231,0.0006406,
      0.000685,0.0007386,0.0006877,0.0007135,0.0006612,0.0006362,0.0005799,0.0005829,0.0004569,0.0005023,
      0.000323,0.0003687,0.0002419,0.0002861,0.000113,0.0001347,9.78e-05,1.48e-05,5.19e-05,-9.94e-05,
      1.38e-05,-7.87e-05,-0.0001167,-8.12e-05,-0.0001365,-8.5e-05,-0.0001874,-3.83e-05,-0.0002004,-0.0001185,
      -7.51e-05,-0.0001217,-0.000102,-9.68e-05,-5.16e-05,-0.0001522,3.4e-05,-0.0001073,-2.25e-05,-1.15e-05,
      -5.48e-05,2.2e-05,-3.32e-05,2.53e-05,1.73e-05,-8.9e-06,5.62e-05,-7.2e-06,5.82e-05,2.09e-05,
      2.9e-05,2.42e-05,4.51e-05,4.2e-05,-4.2e-06,6.04e-05,9.1e-06,-1.14e-05,3.64e-05,-1.09e-05,
      7.2e-06,-1.78e-05,-5.93e-05,4.61e-05,-0.0001253,3.79e-05,-0.0001496,2e-05,-0.0001394,-3.2e-06,
      -0.0001799,-2.67e-05,-0.0001311,-0.000113,-4.3e-05,-0.0001705,-2.42e-05,-0.0001502,-3.91e-05,-0.0001109,
      -5.82e-05,1.13e-05,-7.22e-05,3.06e-05,7.3e-05,9.2e-06,0.0001839,4.25e-05,0.0002722,0.0001584,
      0.0002994,0.0003514,0.0002494,0.0004824,0.0004013,0.0004932,0.0004526,0.0005723,0.0005026,0.0005394,
      0.0005842,0.0004543,0.0005116,0.0003764,0.0003555,0.0002776,0.0001214,-1.23e-05,-0.0001584,-0.0003474,
      -0.000578,-0.0007924,-0.0010814,-0.0013323,-0.0016987,-0.0019857,-0.002425,-0.0026672,-0.0032149,-0.0037347,
      -0.0042272,-0.0049874,-0.0057002,-0.0066349,-0.0076591,-0.0089098,-0.0102311,-0.0117732,-0.0133188,-0.0150695,
      -0.01661,-0.0182295,-0.0195434,-0.020712,-0.0215099,-0.022098,-0.0223366,-0.0224311,-0.0222885,-0.0220193,
      -0.0217068,-0.0213437,-0.0210079,-0.0206917,-0.0204683,-0.0201651,-0.0200683,-0.0199838,-0.0199631,-0.019946,
      -0.0200517,-0.0201293,-0.0202619,-0.0204467,-0.0205097,-0.0207351,-0.0207433,-0.0208945,-0.0208786,-0.0209203,
      -0.0208066,-0.020842,-0.020553,-0.0206402,-0.0203726,-0.0202488,-0.0201061,-0.0199184,-0.0198152,-0.0195345,
      -0.0194944,-0.0192305,-0.0192199,-0.0189391,-0.0190008,-0.0188388,-0.0188662,-0.0189125,-0.0190459,-0.0190994,
      -0.019414,-0.0196735,-0.0199658,-0.0204795,-0.0208439,-0.0213309,-0.0217879,-0.022267,-0.0225484,-0.0229377,
      -0.0228974,-0.0228644,-0.0225618,-0.0218203,-0.0210773,-0.0198238,-0.0185441,-0.0169129,-0.015323,-0.0136072,
      -0.0119781,-0.0103614,-0.0090265,-0.0076502,-0.0065972,-0.0055876,-0.0047894,-0.0040822,-0.0033813,-0.0029829,
      -0.0024478,-0.0020749,-0.0016975,-0.0014052,-0.0010898,-0.0008474,-0.0005585,-0.0003658,-0.0001867,3.4e-05,
      5.19e-05,0.0003011,0.0002871,0.0003793,0.0004068,0.0003621,0.0004351,0.0003142,0.0003431,0.0001887,
      0.0001896,5.35e-05,8.8e-06,-0.0001698,-0.0002051,-0.0003467,-0.0003648,-0.000565,-0.0005401,-0.0006465,
      -0.0007073,-0.0006915,-0.0007236,-0.0007094,-0.0006842,-0.0005923,-0.0004445,-0.0004596,-0.0002109,-3.25e-05,
      0.000137,0.0004395,0.0006543,0.0009857,0.0012729,0.0017206,0.0019606,0.0024314,0.0028848,0.0032241,
      0.003866,0.004412,0.0051219,0.0058979,0.0068383,0.0078483,0.0091005,0.0104637,0.0119466,0.0135711,
      0.0153273,0.0168752,0.0186172,0.02003,0.0212553,0.0223097,0.0231137,0.0235977,0.0239861,0.0240875,
      0.0242018,0.024076,0.0240073,0.0237283,0.0235123,0.0230835,0.0225871,0.0218924,0.0211427,0.0200716,
      0.0187588,0.0174392,0.0158603,0.0142125,0.0125921,0.0110247,0.009488,0.0082506,0.0069765,0.0060015,
      0.0051068,0.0043667,0.0037565,0.0031421,0.0027389,0.0023104,0.0019142,0.0016988,0.0012954,0.0010382,
      0.000857,0.0005127,0.0003603,0.000181,-4.64e-05,-0.0001391,-0.0003483,-0.0003664,-0.00056,-0.0005735,
      -0.0006588,-0.0007295,-0.0007416,-0.0007157,-0.0008565,-0.0007438,-0.0007702,-0.0007768,-0.0006761,-0.0007891,
      -0.000625,-0.0006014,-0.0006221,-0.0005618,-0.0004038,-0.0005267,-0.000334,-0.00032,-0.0002832,-0.0002557,
      -0.0001616,-0.0001436,-6.91e-05,-5.4e-05,-4.59e-05,9.55e-05,-5.05e-05,0.0001533,5.46e-05,0.0001032,
      0.0001503,5.55e-05,0.0001482,0.0001118,6.71e-05,4.84e-05,7.08e-05,-9e-07,-6.8e-05,-5.91e-05,
      -0.0001711,-0.0002057,-0.0002416,-0.0004409,-0.0003212,-0.0006482,-0.0005498,-0.0007606,-0.0008212,-0.0009857,
      -0.0010277,-0.0012142,-0.0013301,-0.0013983,-0.0016136,-0.0016468,-0.0017743,-0.0019323,-0.0019962,-0.0020978,
      -0.0022058,-0.0022905,-0.0023163,-0.0023804,-0.0024261,-0.0024332,-0.0025059,-0.0023876,-0.0024264,-0.0023558,
      -0.0023073,-0.0022459,-0.0020673,-0.0020475,-0.0018339,-0.0017969,-0.0015396,-0.0015179,-0.0012435,-0.0011562,
      -0.0009868,-0.0008396,-0.0006857,-0.0005626,-0.0003906,-0.0003164,-0.0001677,-7.21e-05,-1.27e-05,0.0001235,
      0.0001362,0.0002447,0.0002407,0.0003224,0.0003212,0.0003389,0.0003494,0.0003571,0.000369,0.0003138,
      0.0003483,0.0002926,0.0002789,0.0002963,0.0002129,0.0002345,0.0002038,0.0001575,0.0001063,0.0001824,
      7.6e-05,0.0001031,9.4e-06,0.0001263,-4.49e-05,8.15e-05,-1.42e-05,-2.48e-05,-6.1e-06,2.11e-05,
      -7.87e-05,-1.1e-06,-6.45e-05,-3.57e-05,-0.0001154,2.74e-05,-0.0001421,-4.87e-05,-0.0001331,-3.63e-05,
      -0.0001643,-9.91e-05,-0.0001765,-0.0001071,-0.0001878,-0.000174,-0.0002038,-0.0002458,-0.0001677,-0.000291,
      -0.0002251,-0.0003113,-0.0002072,-0.0003899,-0.0002325,-0.0003306,-0.0002968,-0.000307,-0.0003003,-0.0003169,
      -0.0002572,-0.0003318,-0.0002668,-0.0002521,-0.0002543,-0.0002603,-0.0002081,-0.0002892,-0.0001647,-0.0002859,
      -0.0001699,-0.0002937,-0.0001593,-0.0002774,-0.0002858,-0.0002312,-0.0002936,-0.0002968,-0.0003443,-0.0002742,
      -0.0004091,-0.0003832,-0.0003026,-0.0004286,-0.0003756,-0.0003625,-0.0003197,-0.000308,-0.0002975,-0.000155,
      -0.0001433,-3.64e-05,0.0001167,0.0002094,0.0003785,0.0005934,0.0007256,0.001042,0.0012219,0.0015187,
      0.0018285,0.0021168,0.0025042,0.0028408,0.0031993,0.0036612,0.0039851,0.0045275,0.0048846,0.0053369,
      0.00586,0.006189,0.0067928,0.0071138,0.0075738,0.0080012,0.0083869,0.0087415,0.0090457,0.0093522,
      0.0095993,0.0098542,0.010012,0.0101575,0.0102812,0.0103258,0.0104665,0.0103928,0.0104357,0.0104184,
      0.0103525,0.0103524,0.0102075,0.0102722,0.0101166,0.0100991,0.0100257,0.0100407,0.0098939,0.0100094,
      0.0099034,0.0100181,0.0099222,0.0101517,0.0100613,0.0102915,0.0102608,0.0104497,0.0105546,0.0107084,
      0.0107414,0.0109437,0.0109946,0.0111057,0.0111634,0.0111377,0.0112593,0.0110505,0.0111112,0.010898,
      0.0107592,0.0104983,0.0101638,0.0099376,0.0094594,0.0090639,0.0085803,0.0080872,0.0075626,0.0070092,
      0.0064298,0.0058258,0.0052982,0.0047483,0.0041445,0.0035927,0.0032068,0.0025848,0.0022658,0.0017599,
      0.0015477,0.0010707,0.0009484,0.0006941,0.0004988,0.0004644,0.0003591,0.0003549,0.0004662,0.0004949,
      0.0007173,0.000836,0.0010571,0.0013913,0.0016487,0.0019319,0.0023392,0.0026815,0.0029962,0.0035061,
      0.0036885,0.0042504,0.0045175,0.0048314,0.0052603,0.0054826,0.0058186,0.0060681,0.0062456,0.0065137,
      0.006637,0.0068432,0.0068199,0.0070803,0.0070241,0.007087,0.0071293,0.0071152,0.0071088,0.0070832,
      0.0070474,0.0070208,0.006971,0.0069421,0.0068545,0.0068545,0.0067838,0.0067246,0.0067391,0.0066475,
      0.0066754,0.0065816,0.0065618,0.0065371,0.0066307,0.0064384,0.006566,0.0064881,0.0065372,0.0065461,
      0.0065303,0.0065943,0.0065088,0.0066721,0.0066077,0.0066754,0.0067018,0.0066753,0.0067941,0.0067609,
      0.0067838,0.0068613,0.0068479,0.0068558,0.0068752,0.0069494,0.0068742,0.0069519,0.0069949,0.0069208,
      0.0070514,0.006995,0.0070904,0.0070784,0.0071395,0.0071356,0.0072285,0.00721,0.0072173,0.0072737,
      0.0072222,0.007302,0.0072277,0.0071475,0.0071576,0.0070059,0.0069724,0.0068003,0.0066514,0.0065276,
      0.0062723,0.0061069,0.0058489,0.0056179,0.0053757,0.0050244,0.004857,0.0044682,0.0042433,0.0038645,
      0.0036584,0.0032577,0.0030558,0.0026646,0.0024826,0.0021227,0.0018738,0.0016134,0.0013696,0.0011853,
      0.0008844,0.0007959,0.0005303,0.0004284,0.0002652,0.0001284,3.07e-05,-5.53e-05,-0.0001426,-0.0001812,
      -0.0002742,-0.0002815,-0.0002711,-0.0003889,-0.0002617,-0.0003757,-0.0003242,-0.000278,-0.0003191,-0.0002566,
      -0.0002709,-0.0002192,-0.0001903,-0.0001624,-0.000164,-0.0001285,-7.36e-05,-0.0001078,-2.53e-05,-6.93e-05,
      -3.2e-05,5.79e-05,-6.15e-05,5e-05,-1e-07,8.95e-05,-2.37e-05,6.28e-05,4.99e-05,7.19e-05,
      -3.2e-06,8.01e-05,1.68e-05,4.86e-05,6.63e-05,2.04e-05,-8.4e-06,7.02e-05,5.1e-06,4.88e-05,
      -5e-06,2.81e-05,-2.49e-05,7.17e-05,-6.7e-05,5.83e-05,-2.79e-05,5.35e-05,-4.78e-05,-7e-07,
      5e-05,-6.08e-05,5.5e-05,-2.09e-05,2.91e-05,-3.9e-05,7.07e-05,-4.9e-05,6.96e-05,-3.47e-05,
      6.98e-05,-4.98e-05,7.34e-05,-4.92e-05,0.0001241,-0.0001137,0.0001369,-0.0001012,0.0001302,-8.68e-05,
      2.33e-05,-7.1e-06,5.1e-06,-1.34e-05,-9.01e-05,4.01e-05,-7.15e-05,-8.76e-05,-5.11e-05,-4.3e-05,
      -0.0001115,-0.0001223,-6.78e-05,-0.0001368,-0.0001355,-0.0001289,-0.0001566,-0.0001091,-0.0001583,-0.0001219,
      -0.0001622,-0.0001188,-8.93e-05,-0.0001365,-3.37e-05,-6.44e-05,-4.31e-05,9.76e-05,-1.42e-05,0.0001555,
      0.0001408,0.0002273,0.0002869,0.0002963,0.0004205,0.0004435,0.0005625,0.0005363,0.0007185,0.0006267,
      0.0007586,0.0007669,0.0008256,0.0007775,0.0007728,0.0007844,0.0007426,0.0006197,0.0004865,0.0004741,
      0.0001824,8.31e-05,-0.0001772,-0.0004567,-0.0007468,-0.0010622,-0.0014597,-0.0017825,-0.0023223,-0.0027067,
      -0.0031943,-0.0037535,-0.0041557,-0.0048855,-0.005304,-0.0059072,-0.0064928,-0.0070278,-0.0076504,-0.008188,
      -0.0087127,-0.0092745,-0.0097815,-0.0103057,-0.010708,-0.0112249,-0.0116017,-0.0119184,-0.0123796,-0.012579,
      -0.0128746,-0.013153,-0.0132566,-0.0134952,-0.0136227,-0.0136135,-0.0138642,-0.0137366,-0.0138355,-0.0138235,
      -0.0138105,-0.013767,-0.0137753,-0.0136321,-0.0136453,-0.0135815,-0.0135406,-0.0134471,-0.0134966,-0.0133508,
      -0.0134798,-0.01337,-0.0133807,-0.0134905,-0.013439,-0.013558,-0.0135066,-0.0137291,-0.0136742,-0.0137596,
      -0.0139165,-0.0138764,-0.0139998,-0.014035,-0.0140289,-0.01403,-0.0139914,-0.0139698,-0.0138743,-0.0136324,
      -0.0135293,-0.0132586,-0.0129301,-0.0126128,-0.0121146,-0.0116739,-0.0111036,-0.0104703,-0.0098095,-0.00903,
      -0.0082973,-0.0073879,-0.0065444,-0.0056078,-0.0046508,-0.0036493,-0.0026424,-0.001626,-0.000576,0.0004088,
      0.0015426,0.0025572,0.0034899,0.0046624,0.0055473,0.0065298,0.0075401,0.0083333,0.0092367,0.0100534,
      0.0107984,0.0114666,0.012094,0.0126103,0.0130716,0.0135259,0.0137728,0.014083,0.0142073,0.0143771,
      0.0144161,0.014465,0.0144302,0.0143955,0.0143542,0.0142479,0.0141893,0.014078,0.0140351,0.0138713,
      0.0139713,0.013729,0.0138325,0.0137274,0.0137733,0.013683,0.013819,0.0137323,0.0138122,0.0137909,
      0.0139169,0.0138377,0.0139292,0.0138502,0.0138932,0.013807,0.0137108,0.013641,0.0134173,0.0132594,
      0.0130393,0.0127325,0.0123823,0.0120559,0.0115847,0.0112153,0.0106164,0.010223,0.0094766,0.0091156,
      0.0082799,0.0078244,0.0071583,0.0064919,0.0058664,0.0053185,0.0046663,0.0040755,0.0035914,0.0029963,
      0.0025782,0.0020154,0.0017273,0.0011961,0.0009071,0.0006364,0.0002585,8.55e-05,-0.0001831,-0.0003023,
      -0.0004816,-0.0005987,-0.0006992,-0.000769,-0.0007561,-0.000913,-0.0007906,-0.000839,-0.0008468,-0.0007676,
      -0.0007441,-0.0007096,-0.0006636,-0.0005492,-0.0005939,-0.0004467,-0.0004512,-0.0003282,-0.0003193,-0.0002259,
      -0.0001935,-8.31e-05,-9.9e-05,2e-05,1.75e-05,0.000117,0.000179,0.0001224,0.0003308,0.0002565,
      0.0003093,0.0004178,0.0004327,0.0004296,0.0004716,0.0005949,0.0005182,0.0005778,0.000667,0.0006194,
      0.0006865,0.0006725,0.0007426,0.0006825,0.0007736,0.0008014,0.0006801,0.0008525,0.000695,0.0008215,
      0.0007255,0.0007299,0.0007286,0.0006019,0.0005711,0.0005356,0.0004022,0.0002766,0.0001661,1.43e-05,
      -0.0002195,-0.0004178,-0.0006506,-0.0008854,-0.001256,-0.0015415,-0.0018875,-0.0023496,-0.0026497,-0.003231,
      -0.0035143,-0.0041729,-0.0045376,-0.0052508,-0.005697,-0.0064962,-0.0071909,-0.0080181,-0.0088494,-0.0099372,
      -0.0108297,-0.0121366,-0.0131361,-0.0143524,-0.0155738,-0.0166409,-0.0176757,-0.0186286,-0.0193657,-0.0200728,
      -0.0204992,-0.0209409,-0.0210687,-0.0213023,-0.0213042,-0.0213799,-0.0214021,-0.0213063,-0.0214205,-0.0214205,
      -0.021448,-0.0215184,-0.0216416,-0.0216027,-0.0217912,-0.0216187,-0.0215988,-0.0213287,-0.0208483,-0.0202033,
      -0.0194716,-0.0181843,-0.0171547,-0.0154841,-0.0142182,-0.0124854,-0.0111654,-0.0096214,-0.0084526,-0.0072783,
      -0.0062511,-0.0053956,-0.0046781,-0.0039907,-0.0034236,-0.0030096,-0.0025265,-0.0021356,-0.0018439,-0.0015679,
      -0.0012208,-0.0009352,-0.0007449,-0.0004694,-0.0002925,-7.43e-05,4.6e-05,0.0002175,0.0003712,0.0004254,
      0.0006263,0.000577,0.0007263,0.0008065,0.0007569,0.0009082,0.0008594,0.0009045,0.0008935,0.0008931,
      0.0009057,0.0008556,0.0008348,0.0007587,0.0007529,0.0006426,0.0005501,0.0004896,0.0002834,0.0002474,
      -1.69e-05,-0.0001393,-0.0004177,-0.0005569,-0.0008904,-0.00116,-0.0014567,-0.0017966,-0.0021272,-0.0025588,
      -0.0029041,-0.0033676,-0.0037793,-0.0041907,-0.0046858,-0.0051489,-0.0055428,-0.0060368,-0.0064902,-0.0068203,
      -0.0073476,-0.0076383,-0.0080463,-0.0083354,-0.0086772,-0.0089412,-0.0091654,-0.0094274,-0.0096053,-0.009775,
      -0.0099159,-0.0100363,-0.0101639,-0.0102088,-0.010291,-0.0103846,-0.0104159,-0.0104264,-0.0104818,-0.0105364,
      -0.0105104,-0.0105719,-0.0105778,-0.0105715,-0.0106585,-0.0106055,-0.0106539,-0.0106393,-0.0107114,-0.0106279,
      -0.0106815,-0.0106922,-0.0105844,-0.0107033,-0.0105597,-0.01063,-0.0104726,-0.0105575,-0.0104052,-0.0103685,
      -0.0103018,-0.0102578,-0.0100326,-0.0100872,-0.0098187,-0.009706,-0.0095875,-0.009364,-0.0091251,-0.0089454,
      -0.0086766,-0.008372,-0.0080691,-0.007761,-0.0074182,-0.0070043,-0.0066366,-0.0062353,-0.0058151,-0.0053615,
      -0.0049234,-0.0045151,-0.0040673,-0.0035635,-0.0032756,-0.0027989,-0.0023816,-0.0020646,-0.0017114,-0.0014054,
      -0.00112,-0.0008639,-0.000697,-0.0004459,-0.0003676,-0.0002304,-0.0001353,-0.0001779,-0.0001008,-0.0001424,
      -0.000222,-0.000304,-0.0004547,-0.0004921,-0.0007511,-0.0008719,-0.0010782,-0.0013293,-0.0014271,-0.0017621,
      -0.0018817,-0.0022122,-0.0023039,-0.0025552,-0.0027951,-0.0029232,-0.0031013,-0.0032671,-0.0034313,-0.0034774,
      -0.0036578,-0.0036311,-0.0037561,-0.0037511,-0.003717,-0.0037905,-0.0036062,-0.0036637,-0.0035666,-0.0034191,
      -0.0033215,-0.003265,-0.003049,-0.0029662,-0.0027904,-0.0027023,-0.0025124,-0.002408,-0.0022994,-0.0021363,
      -0.0020578,-0.0019934,-0.0019132,-0.0017979,-0.0018412,-0.0017798,-0.0018181,-0.0017653,-0.0018993,-0.0018319,
      -0.0020068,-0.0019618,-0.0021269,-0.002067,-0.0023008,-0.0022594,-0.0024033,-0.0023767,-0.0025514,-0.0025078,
      -0.0025376,-0.002708,-0.0025622,-0.0026496,-0.0027234,-0.0025754,-0.0026519,-0.0025865,-0.0025433,-0.0024743,
      -0.0024665,-0.002338,-0.0023252,-0.0021799,-0.0021404,-0.0020178,-0.0020012,-0.0017667,-0.0018757,-0.0015598,
      -0.0016592,-0.0014624,-0.0014082,-0.0012936,-0.0012731,-0.0011482,-0.0011414,-0.0009718,-0.0010224,-0.0008792,
      -0.0008748,-0.0008216,-0.0008264,-0.0006465,-0.0007286,-0.0006815,-0.0005775,-0.0006177,-0.0005585,-0.0005443,
      -0.0004481,-0.0005291,-0.0004061,-0.0004998,-0.000352,-0.0004282,-0.000375,-0.0004062,-0.0003339,-0.0003811,
      -0.0003699,-0.0003659,-0.0003953,-0.0003858,-0.0003952,-0.0004741,-0.0004646,-0.0005324,-0.0005859,-0.0006242,
      -0.0007337,-0.0007885,-0.0008848,-0.0009768,-0.0011427,-0.0011371,-0.0014137,-0.0013759,-0.0016786,-0.0016714,
      -0.0019271,-0.0019768,-0.0021715,-0.0022778,-0.002429,-0.0025661,-0.0027129,-0.0027521,-0.0029706,-0.0029355,
      -0.0031689,-0.0031511,-0.0032047,-0.0033342,-0.0033235,-0.0033582,-0.0033484,-0.0033812,-0.0033795,-0.0033528,
      -0.0033171,-0.0032448,-0.003308,-0.0031268,-0.0031656,-0.0031026,-0.0030246,-0.0029029,-0.0030091,-0.0027457,
      -0.0028699,-0.0026316,-0.0027175,-0.0025287,-0.0025829,-0.0023469,-0.0024417,-0.0021949,-0.0022399,-0.0020773,
      -0.0019605,-0.0019751,-0.0016964,-0.0017816,-0.0015239,-0.0014568,-0.0013813,-0.0012112,-0.0011653,-0.0009707,
      -0.0009322,-0.0008094,-0.0006633,-0.0006368,-0.0004819,-0.0004591,-0.0003034,-0.0002754,-0.0002099,-0.0001604,
      -2.42e-05,-8.19e-05,1.63e-05,1.42e-05,0.0001274,7.58e-05,0.0001076,0.000164,0.0001082,0.000208,
      0.0001331,0.000198,0.0001491,0.0001437,0.0002181,0.0001221,0.0001684,0.0001407,0.0001422,0.0001637,
      7.62e-05,0.0001351,0.0001225,4.61e-05,0.0001628,4.08e-05,7.53e-05,6.29e-05,7.8e-05,2.41e-05,
      2.95e-05,6.04e-05,-5.7e-06,3.35e-05,4.3e-06,1.94e-05,-1.86e-05,-7.6e-06,-2.06e-05,-4.25e-05,
      -1.15e-05,-4.52e-05,-9.8e-06,-0.0001181,1.57e-05,-8.02e-05,-9.77e-05,3.2e-06,-0.0001194,8e-06,
      -0.0001171,-5.9e-06,-6.38e-05,-4.4e-06,1.04e-05,-3.77e-05,0.0001405,9.3e-06,0.0001594,0.0001533,
      0.0002321,0.0003263,0.0003385,0.0004627,0.0005586,0.0006519,0.0007213,0.0008884,0.001022,0.0011243,
      0.0012527,0.0014491,0.0015282,0.0017715,0.0017818,0.0021127,0.0021132,0.0023252,0.0024963,0.002606,
      0.0026681,0.0029257,0.0029862,0.0030325,0.0032001,0.0033065,0.0033369,0.0034445,0.0034777,0.0036123,
      0.0035811,0.0036584,0.0037571,0.0037033,0.0037883,0.0037971,0.003767,0.0038485,0.0037325,0.0037899,
      0.0037406,0.0036722,0.0036455,0.0036071,0.0035199,0.0035209,0.003341,0.0034293,0.0032376,0.0032853,
      0.0031594,0.0031941,0.0030609,0.0031384,0.0030602,0.0030848,0.003058,0.0031419,0.0030435,0.00325,
      0.0031547,0.0032321,0.0033348,0.0033773,0.0034371,0.0034685,0.0036289,0.0035769,0.0037534,0.0037622,
      0.003749,0.003945,0.003804,0.0039887,0.0038549,0.0040025,0.0038688,0.0039879,0.0038816,0.0038942,
      0.0038599,0.0038106,0.0038706,0.0036783,0.0038069,0.0036037,0.0036705,0.0035646,0.0035123,0.003492,
      0.0033882,0.0033626,0.003207,0.003242,0.0030492,0.0030455,0.0028991,0.0027634,0.0027681,0.0025464,
      0.0024894,0.0023977,0.002216,0.0021675,0.0020002,0.0019822,0.0017495,0.001763,0.001535,0.0015672,
      0.0013904,0.0013342,0.0011991,0.0011177,0.0010507,0.000929,0.0008765,0.0007348,0.0007587,0.0005292,
      0.0006211,0.0004137,0.0004126,0.0003598,0.0002988,0.0002611,0.0001375,0.0001988,0.0001478,8.06e-05,
      6.62e-05,6.66e-05,6.37e-05,1.42e-05,1.83e-05,4.96e-05,-1.93e-05,2.97e-05,3.64e-05,-3.82e-05,
      4.25e-05,3e-06,2.21e-05,-3.03e-05,2.44e-05,7.2e-06,-5.24e-05,6.2e-06,-4.97e-05,-6.46e-05,
      -0.0001014,-6.19e-05,-0.0001238,-0.0001113,-0.0002371,-7.95e-05,-0.0003164,-0.0001265,-0.0002626,-0.0002217,
      -0.0002663,-0.0001899,-0.0001809,-0.0002295,-4.33e-05,-7.61e-05,1.12e-05,0.0001633,0.000197,0.0004589,
      0.0005006,0.0007604,0.0008782,0.0012149,0.0013743,0.001668,0.0019446,0.0022956,0.0025053,0.0029121,
      0.0032608,0.003526,0.003955,0.0042824,0.004583,0.0050385,0.0052321,0.0056634,0.0059636,0.0062357,
      0.00647,0.0068398,0.0069964,0.0072673,0.0074008,0.0076592,0.0077104,0.0079241,0.0080716,0.0080246,
      0.0082504,0.0082115,0.008326,0.0082667,0.0084556,0.0082941,0.0084057,0.008383,0.0083299,0.0084113,
      0.0082833,0.0082963,0.0082377,0.0082159,0.0081047,0.0080367,0.0079335,0.0078562,0.0076447,0.0075764,
      0.0073184,0.0071543,0.0069151,0.0067265,0.0063721,0.0061559,0.005775,0.0055947,0.005103,0.0048713,
      0.0044738,0.0041589,0.0037247,0.0035037,0.0030123,0.0027896,0.0024088,0.0021193,0.0018115,0.0015237,
      0.0012785,0.0010694,0.0007285,0.0006876,0.000375,0.0003057,0.000112,-1.99e-05,-7.82e-05,-0.0001365,
      -0.0003182,-0.0002529,-0.000371,-0.0003485,-0.0003784,-0.0003664,-0.0003854,-0.0003747,-0.000346,-0.0003326,
      -0.0002748,-0.000326,-0.0001768,-0.0002538,-0.0001424,-0.0002149,-5.22e-05,-0.0001432,-4.14e-05,-8.13e-05,
      -1.3e-06,-2.25e-05,4e-06,5.5e-06,6.01e-05,-1.2e-06,3.38e-05,0.0001109,-5.83e-05,0.0001273,
      1.75e-05,-6e-07,0.000126,-3.32e-05,6.49e-05,-9e-07,4.68e-05,3.98e-05,-1.21e-05,6.33e-05,
      -1.61e-05,5.63e-05,-1.73e-05,0.0001011,-3.6e-06,5.65e-05,5.63e-05,5.12e-05,0.0001252,6.6e-06,
      0.0001849,5.6e-05,0.0001219,0.0002082,4.39e-05,0.0002349,0.0001516,0.0001089,0.0002225,0.000139,
      0.0001719,0.0001211,0.0001531,7.41e-05,8.16e-05,2.06e-05,3.25e-05,-0.0001483,-9.91e-05,-0.0002082,
      -0.0002581,-0.0004548,-0.0003901,-0.0006328,-0.0006327,-0.0008695,-0.0008642,-0.0010764,-0.0012267,-0.0012234,
      -0.0015134,-0.0015396,-0.0017162,-0.0017881,-0.0019417,-0.0020115,-0.0021052,-0.0022275,-0.0022609,-0.0023402,
      -0.0023703,-0.0024073,-0.0024686,-0.0024723,-0.0024992,-0.002474,-0.0025643,-0.0024538,-0.0025439,-0.0025134,
      -0.0024692,-0.0024951,-0.0024709,-0.0024677,-0.0023959,-0.002434,-0.002386,-0.0023616,-0.0023271,-0.0023426,
      -0.0022493,-0.0022039,-0.0022661,-0.0020603,-0.0022113,-0.0019497,-0.0020633,-0.0018603,-0.0019229,-0.0017484,
      -0.0017722,-0.0016716,-0.0015609,-0.0015627,-0.0014402,-0.0014879,-0.0012904,-0.0013774,-0.0012636,-0.0013167,
      -0.0011711,-0.0013033,-0.0011862,-0.0012328,-0.0012863,-0.001194,-0.0013648,-0.0011931,-0.0014217,-0.001299,
      -0.0013687,-0.0013925,-0.0013763,-0.0013861,-0.0013954,-0.0013168,-0.0013429,-0.0013129,-0.0012305,-0.0011745,
      -0.0011382,-0.0010677,-0.0009265,-0.0009529,-0.0008329,-0.0006947,-0.0007574,-0.000501,-0.0005532,-0.0004524,
      -0.0003473,-0.0002836,-0.0003178,-0.0001038,-0.000186,-5e-05,-6.91e-05,1.4e-05,1.98e-05,4.36e-05,
      0.0001139,9.52e-05,0.0001148,0.0001807,8e-05,0.0002386,0.0001671,0.0001537,0.0002654,0.000138,
      0.0002454,0.0002224,0.0002226,0.0002332,0.0002138,0.000258,0.0002595,0.0001995,0.0003141,0.0001848,
      0.0003086,0.0002337,0.0002662,0.0001965,0.0003202,0.0001633,0.0002415,0.0002268,0.0001966,0.0001972,
      0.0002231,0.0001192,0.0002012,0.000133,0.0001657,0.0001138,0.0001287,0.0001318,9.14e-05,0.0001024,
      0.00011,5.56e-05,0.0001314,1.45e-05,0.0001139,-2.41e-05,-1.11e-05,-7.88e-05,-0.0002648,-0.0005038,
      -0.0011573,-0.002198,-0.0042398,-0.0075944,-0.0130355,-0.0204178,-0.0300992,-0.0409187,-0.050723,-0.0589401,
      -0.0654788,-0.070326,-0.0732112,-0.0738875,-0.0722641,-0.0677071,-0.0608503,-0.0518685,-0.042049,-0.0319595,
      -0.0225724,-0.0144778,-0.0077207,-0.0025381,0.0009116,0.0029719,0.0037157,0.0039838,0.004456,0.0057023,
      0.0081889,0.0122326,0.0177779,0.0247663,0.0329041,0.0419151,0.0512943,0.0599873,0.0669373,0.0704716,
      0.0706698,0.0683333,0.065195,0.0616887,0.0589352,0.056862,0.0556238,0.0549089,0.0536654,0.0509634,
      0.0457934,0.037807,0.0273589,0.0172688,0.0086298,0.0024583,-0.0015904,-0.003578,-0.0042074,-0.0042263,
      -0.0036798,-0.0033125,-0.0027035,-0.0024384,-0.0021046,-0.0019264,-0.0016363,-0.0015608,-0.0014585,-0.001299,
      -0.0012697,-0.0010823,-0.0010631,-0.0009158,-0.0007794,-0.0007485,-0.0005784,-0.0005099,-0.0004697,-0.0002893,
      -0.0003264,-0.0001528,-0.0001408,-8.36e-05,-6.35e-05,7.79e-05,1.27e-05,7.65e-05,0.0001473,7.82e-05,
      0.0002104,8.45e-05,0.0001962,0.0001353,0.0001371,0.0001912,0.0001025,0.0001433,0.0001242,0.000104,
      0.0001297,4.66e-05,9.36e-05,4.37e-05,6.65e-05,2.5e-05,3.83e-05,2.64e-05,-1.4e-05,2.97e-05,
      -3.04e-05,1.72e-05,-3.88e-05,8e-06,-6.8e-05,4e-06,-2.28e-05,-4.3e-05,-1e-05,-4.74e-05,
      -5e-06,-6.31e-05,-2.19e-05,-3.48e-05,1.23e-05,-7.03e-05,-1e-05,-1.98e-05,-4.12e-05,-1.99e-05,
      -9.3e-06,-1.74e-05,-5.15e-05,1.9e-05,-2.47e-05,-5.05e-05,2.21e-05,-3.81e-05,1e-07,-3.69e-05,
      1.76e-05,-3.36e-05,-1.33e-05,1.25e-05,-3.68e-05,4.6e-06,1.85e-05,-7.88e-05,9.24e-05,-8.65e-05,
      4.93e-05,-3.63e-05,2.17e-05,-5.05e-05,5.02e-05,-2.02e-05,3.08e-05,-2.88e-05,9.9e-06,-1.08e-05,
      3.34e-05,-4.85e-05,8.62e-05,-9.21e-05,7.78e-05,-1.44e-05,-2.02e-05,2.08e-05,1.14e-05,8.1e-06,
      1.18e-05,-5.55e-05,8.01e-05,-6.06e-05,8.78e-05,-8.64e-05,5.31e-05,-5.17e-05,7.95e-05,-6.19e-05,
      -3.65e-05,7.81e-05,-7.13e-05,2.4e-06,-1.27e-05,-7.3e-06,-3.26e-05,7.3e-06,-7.13e-05,1.74e-05,
      -7.28e-05,1.35e-05,-5.57e-05,-5.13e-05,-1.83e-05,-5.9e-06,-8.84e-05,-4.97e-05,2.67e-05,-0.0001315,
      3.44e-05,-8.04e-05,-4.76e-05,1.01e-05,-6.74e-05,-7.55e-05,3.44e-05,-6.23e-05,1.03e-05,-0.0001091,
      9.12e-05,-0.0001253,7.51e-05,-0.0001009,4.98e-05,-2.39e-05,-2.85e-05,7.4e-06,-1.1e-06,-2.9e-06,
     ),
     (
      0.0105,0.01027,0.00991,0.01008,0.00976,0.00939,0.00967,0.00892,0.00867,0.00892,
      0.00816,0.00795,0.00805,0.00784,0.00708,0.00736,0.007,0.00666,0.00686,0.00639,
      0.00659,0.00624,0.00596,0.00593,0.00631,0.00634,0.00585,0.00628,0.00655,0.00653,
      0.00634,0.0071,0.00724,0.00743,0.00802,0.00849,0.00903,0.0094,0.01036,0.01067,
      0.01146,0.01251,0.01351,0.01444,0.01574,0.0171,0.01785,0.01952,0.02109,0.02211,
      0.02395,0.02608,0.02775,0.02943,0.03152,0.0335,0.0355,0.03749,0.03953,0.04141,
      0.04384,0.04574,0.04771,0.04991,0.05115,0.05299,0.05489,0.05595,0.05732,0.05829,
      0.05903,0.05959,0.05962,0.06047,0.06005,0.05937,0.0595,0.05818,0.05732,0.05606,
      0.05447,0.05309,0.0512,0.04966,0.04722,0.04502,0.04276,0.04049,0.03832,0.03575,
      0.0332,0.031,0.02882,0.02626,0.02368,0.02154,0.01919,0.01724,0.01536,0.01285,
      0.01127,0.00958,0.00744,0.00568,0.00436,0.00241,0.00102,-0.00048,-0.00209,-0.00316,
      -0.00476,-0.00631,-0.0075,-0.00869,-0.01027,-0.01176,-0.01323,-0.01485,-0.01591,-0.01755,
      -0.01902,-0.02044,-0.02236,-0.02374,-0.0253,-0.02668,-0.02861,-0.03003,-0.03136,-0.03369,
      -0.03505,-0.03619,-0.03779,-0.03925,-0.04082,-0.04208,-0.04294,-0.04425,-0.04568,-0.04613,
      -0.04735,-0.04809,-0.04833,-0.04968,-0.05004,-0.05019,-0.05077,-0.0513,-0.05158,-0.05142,
      -0.05179,-0.052,-0.05225,-0.05257,-0.05251,-0.05243,-0.05278,-0.05299,-0.05327,-0.05349,
      -0.05378,-0.05393,-0.05451,-0.0551,-0.05511,-0.05601,-0.05693,-0.05682,-0.05783,-0.05848,
      -0.05926,-0.06011,-0.06034,-0.06163,-0.06223,-0.063,-0.06385,-0.06415,-0.06512,-0.06549,
      -0.06594,-0.06683,-0.06731,-0.06749,-0.06775,-0.06807,-0.06799,-0.06801,-0.06799,-0.06806,
      -0.0678,-0.06723,-0.06676,-0.06623,-0.0658,-0.06506,-0.06392,-0.06323,-0.06265,-0.06113,
      -0.06027,-0.05929,-0.05798,-0.05674,-0.0555,-0.05423,-0.053,-0.05173,-0.05052,-0.04914,
      -0.04835,-0.04696,-0.04569,-0.04477,-0.04365,-0.0431,-0.04189,-0.0413,-0.04041,-0.04002,
      -0.03952,-0.03884,-0.03913,-0.03867,-0.03869,-0.03912,-0.03883,-0.03956,-0.04,-0.04045,
      -0.04129,-0.04194,-0.04303,-0.04403,-0.04492,-0.04635,-0.04728,-0.0481,-0.04995,-0.05072,
      -0.05172,-0.05301,-0.05432,-0.05533,-0.05594,-0.05702,-0.05809,-0.05873,-0.05957,-0.0601,
      -0.06007,-0.06118,-0.06131,-0.06108,-0.06192,-0.06181,-0.06134,-0.06183,-0.06187,-0.06117,
      -0.06095,-0.06107,-0.06082,-0.06017,-0.06001,-0.05981,-0.05926,-0.05867,-0.05857,-0.05808,
      -0.05708,-0.0571,-0.05655,-0.05566,-0.0553,-0.05465,-0.0546,-0.05396,-0.05351,-0.05303,
      -0.05235,-0.05261,-0.05223,-0.05194,-0.05201,-0.05202,-0.05194,-0.05226,-0.05258,-0.05243,
      -0.05271,-0.05327,-0.05376,-0.05425,-0.05478,-0.05519,-0.05602,-0.05682,-0.05725,-0.05772,
      -0.05844,-0.0596,-0.06025,-0.06094,-0.06208,-0.06284,-0.06311,-0.06427,-0.06541,-0.06563,
      -0.06661,-0.06699,-0.06744,-0.06833,-0.06881,-0.06899,-0.06925,-0.06958,-0.06947,-0.06958,
      -0.06923,-0.069,-0.06882,-0.06823,-0.06776,-0.06723,-0.06682,-0.06597,-0.06526,-0.0645,
      -0.06373,-0.06308,-0.06191,-0.06126,-0.06057,-0.05943,-0.05869,-0.05834,-0.05746,-0.05678,
      -0.05591,-0.05553,-0.05501,-0.05441,-0.05436,-0.05365,-0.05355,-0.05317,-0.05326,-0.05332,
      -0.05294,-0.05293,-0.05334,-0.05325,-0.05317,-0.05362,-0.05339,-0.05385,-0.05377,-0.05359,
      -0.05444,-0.0542,-0.05425,-0.05435,-0.05383,-0.05436,-0.05429,-0.05399,-0.05377,-0.05342,
      -0.05358,-0.05329,-0.05276,-0.05216,-0.05209,-0.0515,-0.0509,-0.05093,-0.04988,-0.04952,
      -0.04907,-0.04816,-0.04778,-0.04722,-0.04684,-0.04591,-0.0455,-0.0451,-0.04407,-0.04412,
      -0.04348,-0.04292,-0.04285,-0.04216,-0.04201,-0.04183,-0.04118,-0.04093,-0.04111,-0.04066,
      -0.04074,-0.04092,-0.04009,-0.04022,-0.04058,-0.04048,-0.04053,-0.04041,-0.04085,-0.04074,
      -0.04067,-0.04112,-0.04089,-0.04135,-0.04124,-0.04118,-0.04159,-0.04148,-0.04151,-0.0415,
      -0.04151,-0.04148,-0.04157,-0.04126,-0.0409,-0.04116,-0.04047,-0.03993,-0.03984,-0.03921,
      -0.03882,-0.038,-0.03714,-0.03691,-0.03569,-0.03477,-0.03374,-0.03274,-0.03181,-0.03047,
      -0.02927,-0.0279,-0.02709,-0.02572,-0.02448,-0.02332,-0.02167,-0.02043,-0.01955,-0.01831,
      -0.01665,-0.01552,-0.01423,-0.013,-0.01172,-0.01056,-0.00899,-0.00741,-0.00633,-0.0047,
      -0.00329,-0.00155,0.0006,0.00225,0.00397,0.00595,0.00831,0.01044,0.01304,0.01554,
      0.01788,0.02106,0.02403,0.02698,0.0302,0.03388,0.03694,0.04049,0.04429,0.04802,
      0.0518,0.05543,0.05965,0.06316,0.06738,0.07102,0.07469,0.07894,0.08224,0.0858,
      0.08927,0.09283,0.09611,0.09891,0.10243,0.10507,0.10744,0.11035,0.11281,0.11508,
      0.117,0.11908,0.12082,0.12224,0.12382,0.12509,0.12592,0.12733,0.12838,0.12873,
      0.12925,0.12984,0.12997,0.13034,0.13024,0.13023,0.13042,0.12962,0.12962,0.12897,
      0.12851,0.12807,0.12724,0.12651,0.12576,0.12501,0.12423,0.12359,0.12244,0.1217,
      0.12131,0.12058,0.11938,0.11893,0.11894,0.11782,0.11779,0.11759,0.11685,0.11704,
      0.11703,0.11691,0.11737,0.11718,0.11745,0.11811,0.11817,0.11877,0.11934,0.11945,
      0.11993,0.12093,0.1209,0.12151,0.12214,0.12197,0.12202,0.12199,0.12207,0.12179,
      0.1213,0.12057,0.11947,0.11856,0.11728,0.1158,0.11403,0.11206,0.10971,0.10754,
      0.10509,0.10205,0.09866,0.09573,0.09287,0.0892,0.08575,0.08221,0.07882,0.07491,
      0.07145,0.06822,0.06507,0.06143,0.0581,0.05562,0.05234,0.04994,0.04785,0.04509,
      0.04323,0.04147,0.03985,0.03911,0.03768,0.03666,0.0362,0.03603,0.03568,0.03573,
      0.03594,0.03652,0.03703,0.03737,0.03854,0.03952,0.04044,0.04176,0.04309,0.04394,
      0.0452,0.04682,0.0481,0.04884,0.05061,0.05182,0.05266,0.0541,0.05498,0.05602,
      0.05699,0.05809,0.05873,0.05952,0.06023,0.06109,0.06149,0.062,0.06258,0.06272,
      0.0631,0.06292,0.06325,0.06364,0.06322,0.06301,0.06276,0.06246,0.06241,0.0614,
      0.06099,0.06065,0.05939,0.05879,0.05796,0.05735,0.05615,0.05553,0.05476,0.05392,
      0.0536,0.05267,0.05222,0.05194,0.05203,0.05197,0.05213,0.05305,0.05371,0.05463,
      0.05632,0.05765,0.05966,0.06233,0.06463,0.06775,0.07099,0.07442,0.07848,0.08281,
      0.08696,0.09149,0.09624,0.10121,0.10663,0.11144,0.11677,0.12218,0.1267,0.13169,
      0.13725,0.14152,0.14583,0.14982,0.15356,0.15712,0.16011,0.1626,0.16462,0.16608,
      0.16719,0.16725,0.16736,0.1669,0.16557,0.16386,0.16151,0.1591,0.15608,0.1526,
      0.14852,0.14429,0.13976,0.13511,0.12971,0.12449,0.11928,0.11378,0.10799,0.10216,
      0.0968,0.09107,0.08483,0.07951,0.07406,0.06814,0.06277,0.05718,0.05191,0.04708,
      0.04165,0.03668,0.03203,0.02709,0.023,0.01883,0.01409,0.01021,0.00651,0.00268,
      -0.00082,-0.00398,-0.00733,-0.01029,-0.01308,-0.01548,-0.01807,-0.02031,-0.02224,-0.02432,
      -0.02604,-0.02757,-0.02873,-0.03007,-0.03104,-0.03182,-0.03222,-0.03281,-0.03306,-0.03298,
      -0.03306,-0.0328,-0.03224,-0.03174,-0.0313,-0.03055,-0.02949,-0.02849,-0.02756,-0.02625,
      -0.02492,-0.02406,-0.02281,-0.02116,-0.02003,-0.01872,-0.01759,-0.01592,-0.01469,-0.01383,
      -0.01247,-0.01126,-0.00995,-0.00893,-0.00832,-0.00727,-0.00607,-0.00594,-0.00467,-0.00385,
      -0.00394,-0.00288,-0.00253,-0.002,-0.00144,-0.00124,-0.00108,-0.00043,-0.00016,-0.00042,
      0.00036,0.00049,0.00067,0.00132,0.00156,0.00146,0.00162,0.00258,0.00306,0.00316,
      0.00379,0.00417,0.00502,0.00575,0.0065,0.00726,0.00796,0.00893,0.01034,0.01125,
      0.01218,0.01354,0.01466,0.01634,0.0175,0.0187,0.02025,0.02183,0.02294,0.02442,
      0.02642,0.02742,0.02893,0.03091,0.03198,0.0332,0.03475,0.03632,0.03749,0.03874,
      0.04007,0.04101,0.04192,0.04335,0.04423,0.04527,0.04623,0.04733,0.04801,0.04865,
      0.04993,0.05016,0.05095,0.05206,0.05282,0.05318,0.05401,0.05484,0.05517,0.05601,
      0.05684,0.05717,0.05801,0.05882,0.05923,0.05978,0.06016,0.0611,0.06149,0.06198,
      0.06262,0.06256,0.06369,0.06373,0.06391,0.06468,0.0644,0.06475,0.06515,0.06467,
      0.06468,0.06514,0.0648,0.06418,0.06398,0.06391,0.06292,0.06243,0.06238,0.06155,
      0.06042,0.05978,0.05898,0.05832,0.05726,0.05614,0.05568,0.05414,0.05328,0.05225,
      0.05123,0.05037,0.04882,0.04837,0.04722,0.04625,0.0453,0.04406,0.04396,0.04263,
      0.04203,0.04126,0.04043,0.04002,0.03949,0.03903,0.0384,0.03836,0.03765,0.03755,
      0.03718,0.03724,0.03736,0.03682,0.03739,0.03715,0.03753,0.03777,0.03791,0.03861,
      0.03868,0.03921,0.03999,0.04083,0.04119,0.04193,0.0431,0.04368,0.0447,0.04604,
      0.04716,0.04886,0.04993,0.05145,0.05329,0.05491,0.05711,0.0587,0.06062,0.06335,
      0.06552,0.06761,0.07059,0.07309,0.0751,0.07805,0.08073,0.08357,0.08602,0.08841,
      0.09142,0.09347,0.09575,0.09808,0.09998,0.10207,0.1038,0.10527,0.10666,0.10715,
      0.10831,0.10918,0.10903,0.10875,0.10852,0.10821,0.10667,0.10564,0.1038,0.10169,
      0.09996,0.09703,0.09399,0.09106,0.08779,0.0843,0.08056,0.07649,0.07251,0.06848,
      0.0646,0.06017,0.05625,0.05236,0.04785,0.04429,0.04053,0.03662,0.03353,0.03028,
      0.02688,0.02422,0.02177,0.01921,0.0169,0.01521,0.01377,0.01221,0.0109,0.01022,
      0.00975,0.00931,0.00854,0.00905,0.00928,0.00935,0.01032,0.01089,0.01213,0.01261,
      0.01395,0.01558,0.01674,0.01796,0.01943,0.02134,0.02273,0.02423,0.02584,0.02692,
      0.02851,0.03007,0.03124,0.03249,0.03379,0.03484,0.03538,0.03667,0.03696,0.0375,
      0.03805,0.0383,0.03824,0.03824,0.03831,0.03805,0.0375,0.03698,0.03658,0.03571,
      0.0351,0.03392,0.03325,0.03259,0.0314,0.03084,0.02977,0.02862,0.02829,0.02775,
      0.02725,0.02679,0.02611,0.02628,0.02628,0.02611,0.02679,0.02725,0.0277,0.02844,
      0.02953,0.03042,0.03178,0.03294,0.03445,0.03627,0.03799,0.03978,0.0414,0.04361,
      0.04515,0.04727,0.04925,0.05123,0.05333,0.05495,0.05685,0.05815,0.06007,0.06156,
      0.0627,0.06417,0.06463,0.06581,0.06663,0.0667,0.06709,0.06697,0.06707,0.06679,
      0.0663,0.06554,0.06456,0.06325,0.06198,0.06089,0.059,0.05716,0.05589,0.05381,
      0.05143,0.04954,0.04748,0.0456,0.04317,0.04127,0.03928,0.03716,0.03564,0.03335,
      0.03202,0.0306,0.02861,0.02748,0.02651,0.02552,0.02445,0.02369,0.02329,0.02268,
      0.02252,0.02226,0.02195,0.02195,0.02226,0.0225,0.02274,0.02304,0.02309,0.02408,
      0.02456,0.02467,0.02527,0.02574,0.02626,0.02674,0.0273,0.02757,0.02742,0.02775,
      0.02807,0.02798,0.02801,0.02799,0.02805,0.02783,0.02715,0.02709,0.02651,0.02589,
      0.02594,0.02487,0.0246,0.02375,0.02292,0.02259,0.02173,0.02101,0.02025,0.0195,
      0.01874,0.01803,0.01715,0.01686,0.01591,0.01552,0.01502,0.01441,0.01433,0.01376,
      0.01313,0.01324,0.01342,0.0126,0.0127,0.0131,0.01292,0.01325,0.0136,0.01337,
      0.01393,0.01492,0.01491,0.01544,0.01635,0.01666,0.01752,0.01827,0.01891,0.02009,
      0.02074,0.02144,0.0225,0.02359,0.02417,0.02525,0.02634,0.0269,0.02809,0.02876,
      0.02939,0.03068,0.0309,0.03174,0.03266,0.03264,0.03329,0.03371,0.0344,0.03422,
      0.03424,0.03435,0.03388,0.03416,0.03349,0.03288,0.033,0.03163,0.031,0.03041,
      0.0289,0.02801,0.02707,0.02573,0.02451,0.02323,0.02206,0.02053,0.01883,0.01817,
      0.0165,0.01483,0.01419,0.01241,0.01118,0.01038,0.00882,0.00834,0.00732,0.00587,
      0.0052,0.00483,0.004,0.00318,0.00279,0.00218,0.002,0.00183,0.00118,0.00096,
      0.00099,0.00109,0.00066,0.00077,0.00077,0.00066,0.00109,0.00098,0.00101,0.001,
      0.001,0.00101,0.00098,0.00107,0.00073,0.00051,0.00023,7e-05,-0.00051,-0.00104,
      -0.00134,-0.00263,-0.00317,-0.00421,-0.00551,-0.00676,-0.00795,-0.00944,-0.01129,-0.01292,
      -0.01504,-0.01692,-0.01929,-0.02143,-0.024,-0.02661,-0.02859,-0.03155,-0.03425,-0.03698,
      -0.03985,-0.04216,-0.04503,-0.04773,-0.05056,-0.05306,-0.05525,-0.0575,-0.0598,-0.06181,
      -0.06348,-0.06532,-0.06679,-0.06805,-0.06904,-0.06981,-0.07023,-0.07079,-0.07113,-0.07071,
      -0.07057,-0.07006,-0.06923,-0.06855,-0.06762,-0.066,-0.06439,-0.06348,-0.06122,-0.05916,
      -0.05766,-0.05521,-0.05302,-0.05073,-0.04857,-0.04601,-0.04342,-0.04134,-0.03875,-0.03619,
      -0.03402,-0.03174,-0.02952,-0.0272,-0.02518,-0.0236,-0.02143,-0.0197,-0.01828,-0.01668,
      -0.01551,-0.01429,-0.01284,-0.01235,-0.01126,-0.01012,-0.00976,-0.00933,-0.00841,-0.00802,
      -0.00751,-0.00693,-0.00675,-0.00655,-0.00603,-0.00532,-0.00568,-0.00497,-0.00445,-0.00425,
      -0.00406,-0.0035,-0.00294,-0.00274,-0.00258,-0.00192,-0.00174,-0.00161,-0.00082,-0.00113,
      -0.00067,-0.00068,-0.00109,-0.00094,-0.00114,-0.002,-0.00284,-0.00313,-0.00415,-0.00577,
      -0.00727,-0.00865,-0.01061,-0.01338,-0.01535,-0.01823,-0.02123,-0.02436,-0.02832,-0.03184,
      -0.0363,-0.04043,-0.04496,-0.04974,-0.0546,-0.06034,-0.06552,-0.07059,-0.0766,-0.082,
      -0.08739,-0.09343,-0.09838,-0.10406,-0.1094,-0.11386,-0.11916,-0.1235,-0.12785,-0.13162,
      -0.13469,-0.13813,-0.14083,-0.1431,-0.14483,-0.14613,-0.14671,-0.14758,-0.14801,-0.14694,
      -0.14631,-0.14537,-0.14377,-0.14213,-0.1398,-0.13724,-0.13481,-0.13211,-0.12882,-0.12519,
      -0.12199,-0.11892,-0.11491,-0.11151,-0.1081,-0.10417,-0.10078,-0.09727,-0.09371,-0.09043,
      -0.08761,-0.08418,-0.08121,-0.07854,-0.07568,-0.07329,-0.0707,-0.06846,-0.06651,-0.06451,
      -0.06245,-0.0607,-0.05929,-0.05768,-0.05652,-0.05526,-0.05395,-0.05295,-0.05226,-0.05152,
      -0.05068,-0.05029,-0.04967,-0.04954,-0.04918,-0.04927,-0.04927,-0.04916,-0.04963,-0.04933,
      -0.05005,-0.05049,-0.05103,-0.05144,-0.05225,-0.05311,-0.05335,-0.05452,-0.05558,-0.05617,
      -0.05726,-0.05831,-0.05901,-0.05969,-0.06076,-0.06183,-0.06247,-0.06335,-0.06367,-0.06451,
      -0.06533,-0.06572,-0.06633,-0.06648,-0.06676,-0.06701,-0.06724,-0.06758,-0.06746,-0.0676,
      -0.06716,-0.06729,-0.06723,-0.06732,-0.06701,-0.06667,-0.06687,-0.06638,-0.06662,-0.06616,
      -0.06626,-0.06633,-0.06593,-0.06596,-0.06625,-0.06657,-0.0665,-0.06646,-0.06668,-0.06734,
      -0.06749,-0.06772,-0.06818,-0.06909,-0.0695,-0.06994,-0.07077,-0.0715,-0.07226,-0.073,
      -0.07378,-0.07442,-0.07559,-0.07625,-0.07694,-0.07803,-0.07899,-0.08006,-0.08082,-0.08119,
      -0.08195,-0.08308,-0.0838,-0.08425,-0.08476,-0.08526,-0.08574,-0.08633,-0.08649,-0.08674,
      -0.08708,-0.08696,-0.0871,-0.08667,-0.08677,-0.0868,-0.08655,-0.08604,-0.08533,-0.08567,
      -0.08505,-0.08417,-0.08378,-0.08323,-0.08283,-0.08198,-0.08128,-0.08044,-0.07998,-0.07966,
      -0.07841,-0.07772,-0.07724,-0.07684,-0.07592,-0.07551,-0.07509,-0.07416,-0.0738,-0.07318,
      -0.073,-0.07284,-0.07218,-0.07196,-0.072,-0.07206,-0.07177,-0.07137,-0.07178,-0.07206,
      -0.072,-0.07196,-0.07219,-0.07284,-0.07301,-0.07317,-0.07384,-0.074,-0.07419,-0.07478,
      -0.07524,-0.07582,-0.07601,-0.07617,-0.07685,-0.07698,-0.07726,-0.07751,-0.07775,-0.07805,
      -0.07808,-0.07767,-0.07775,-0.07785,-0.07738,-0.07767,-0.07696,-0.0765,-0.07609,-0.07517,
      -0.07477,-0.0743,-0.07357,-0.07242,-0.07176,-0.07106,-0.07,-0.06894,-0.06826,-0.06756,
      -0.0665,-0.06544,-0.06477,-0.06399,-0.06327,-0.06246,-0.06191,-0.06195,-0.06084,-0.06072,
      -0.06083,-0.06049,-0.06023,-0.06012,-0.06083,-0.06109,-0.06085,-0.06154,-0.06202,-0.06242,
      -0.06335,-0.06373,-0.06428,-0.0647,-0.06544,-0.06657,-0.06731,-0.06774,-0.06828,-0.06869,
      -0.06949,-0.07039,-0.07049,-0.07068,-0.07132,-0.07158,-0.07141,-0.07183,-0.07182,-0.07141,
      -0.07158,-0.07129,-0.07076,-0.07017,-0.07008,-0.06955,-0.06875,-0.06799,-0.06731,-0.06628,
      -0.06508,-0.06492,-0.06377,-0.06252,-0.06119,-0.06025,-0.05932,-0.05798,-0.05678,-0.05542,
      -0.05457,-0.05332,-0.05167,-0.0505,-0.04934,-0.04767,-0.04651,-0.04531,-0.04376,-0.04217,
      -0.04106,-0.03959,-0.03758,-0.03659,-0.03508,-0.03308,-0.03209,-0.03056,-0.02867,-0.02727,
      -0.02574,-0.02425,-0.02275,-0.02126,-0.0197,-0.01843,-0.01758,-0.01626,-0.01486,-0.01428,
      -0.01351,-0.01267,-0.0123,-0.01162,-0.01172,-0.01203,-0.01219,-0.01272,-0.01344,-0.01455,
      -0.01539,-0.0169,-0.01902,-0.02103,-0.02288,-0.02547,-0.02827,-0.03097,-0.03388,-0.03753,
      -0.04104,-0.04434,-0.04861,-0.05224,-0.05594,-0.06002,-0.06401,-0.06799,-0.07209,-0.07572,
      -0.07958,-0.083,-0.08648,-0.09013,-0.09304,-0.09575,-0.09852,-0.10123,-0.10262,-0.10482,
      -0.10663,-0.1077,-0.10912,-0.10986,-0.10998,-0.11026,-0.11055,-0.11061,-0.11005,-0.10923,
      -0.10859,-0.10747,-0.10657,-0.10531,-0.10374,-0.10227,-0.10076,-0.09926,-0.09775,-0.09628,
      -0.09466,-0.09359,-0.09203,-0.09034,-0.08963,-0.08817,-0.08721,-0.08652,-0.08576,-0.08498,
      -0.08436,-0.0846,-0.08425,-0.08394,-0.08404,-0.08396,-0.08418,-0.08486,-0.08492,-0.08551,
      -0.08608,-0.08619,-0.0867,-0.08756,-0.0881,-0.08809,-0.0891,-0.08957,-0.08967,-0.09026,
      -0.09081,-0.09102,-0.09115,-0.09193,-0.09165,-0.09201,-0.09234,-0.09215,-0.09258,-0.09257,
      -0.09218,-0.09225,-0.09234,-0.09192,-0.09202,-0.09204,-0.09185,-0.09109,-0.09134,-0.0911,
      -0.09031,-0.09069,-0.08998,-0.08943,-0.08935,-0.08872,-0.08833,-0.08751,-0.08668,-0.08632,
      -0.08558,-0.08441,-0.08383,-0.08282,-0.08143,-0.0805,-0.0796,-0.07815,-0.07734,-0.07601,
      -0.07467,-0.07386,-0.07241,-0.07153,-0.07051,-0.06946,-0.06868,-0.06834,-0.0675,-0.06671,
      -0.0662,-0.06601,-0.06577,-0.06543,-0.06552,-0.06552,-0.06543,-0.0658,-0.06591,-0.06658,
      -0.06677,-0.06685,-0.06786,-0.06823,-0.06875,-0.06928,-0.06966,-0.07059,-0.07101,-0.07142,
      -0.07232,-0.07282,-0.07293,-0.0735,-0.07409,-0.07415,-0.07482,-0.07508,-0.07491,-0.07532,
      -0.07535,-0.07482,-0.07542,-0.07505,-0.07443,-0.07426,-0.07405,-0.07358,-0.07268,-0.07224,
      -0.07188,-0.07077,-0.06959,-0.06941,-0.06832,-0.06684,-0.06635,-0.0653,-0.06401,-0.06269,
      -0.06175,-0.06084,-0.05941,-0.05853,-0.0575,-0.05649,-0.05559,-0.05418,-0.05321,-0.05249,
      -0.05184,-0.05067,-0.05002,-0.04927,-0.04843,-0.04803,-0.04747,-0.04712,-0.04607,-0.04614,
      -0.04541,-0.04525,-0.04511,-0.04435,-0.04453,-0.04456,-0.04426,-0.04394,-0.044,-0.04408,
      -0.04369,-0.04368,-0.04409,-0.04398,-0.04401,-0.044,-0.04402,-0.04394,-0.04425,-0.04457,
      -0.04448,-0.04452,-0.04444,-0.04475,-0.04509,-0.04489,-0.04534,-0.04526,-0.04516,-0.04561,
      -0.0454,-0.04582,-0.04584,-0.04534,-0.04584,-0.04582,-0.04541,-0.04558,-0.04531,-0.0447,
      -0.04442,-0.04463,-0.04405,-0.04318,-0.04276,-0.0423,-0.04158,-0.04042,-0.03976,-0.03907,
      -0.03798,-0.03702,-0.03598,-0.03509,-0.03368,-0.03273,-0.03193,-0.03007,-0.0293,-0.02824,
      -0.02725,-0.02626,-0.02524,-0.0243,-0.02309,-0.02286,-0.02199,-0.02118,-0.02079,-0.02018,
      -0.02,-0.01984,-0.01918,-0.01898,-0.01893,-0.01933,-0.01925,-0.01918,-0.01954,-0.01966,
      -0.02033,-0.02052,-0.0206,-0.02158,-0.02209,-0.02208,-0.02311,-0.0235,-0.02391,-0.02487,
      -0.02514,-0.0261,-0.02648,-0.027,-0.0275,-0.02799,-0.02852,-0.02895,-0.02968,-0.02933,
      -0.03001,-0.03063,-0.03049,-0.03042,-0.03084,-0.03073,-0.03076,-0.03074,-0.03079,-0.0306,
      -0.02983,-0.03008,-0.02988,-0.02893,-0.02844,-0.02832,-0.0278,-0.02699,-0.02625,-0.02551,
      -0.02473,-0.02409,-0.02292,-0.02226,-0.02157,-0.02049,-0.01951,-0.0185,-0.0175,-0.0165,
      -0.0155,-0.0145,-0.01351,-0.0125,-0.01153,-0.01042,-0.00983,-0.00878,-0.00758,-0.00742,
      -0.00625,-0.0051,-0.00485,-0.004,-0.00317,-0.00284,-0.002,-0.00118,-0.00078,-0.00022,
      0.00016,0.00106,0.00157,0.00166,0.00227,0.00274,0.00325,0.00375,0.00425,0.00476,
      0.00523,0.00584,0.00592,0.00649,0.00711,0.00707,0.00809,0.00854,0.00876,0.00893,
      0.00952,0.00998,0.01056,0.01077,0.01088,0.01168,0.01141,0.01169,0.01231,0.01255,
      0.01249,0.0125,0.01252,0.01242,0.01281,0.01282,0.01242,0.0125,0.01257,0.01223,
      0.01203,0.01166,0.01184,0.01149,0.01125,0.01101,0.01074,0.01052,0.01021,0.01012,
      0.00932,0.00959,0.00931,0.00869,0.00844,0.00856,0.00832,0.00769,0.00743,0.00759,
      0.00723,0.007,0.00677,0.00643,0.0065,0.00656,0.00626,0.00593,0.00602,0.006,
      0.00598,0.00608,0.00569,0.00568,0.00608,0.006,0.00592,0.00632,0.00627,0.0061,
      0.00683,0.00706,0.00694,0.00718,0.00783,0.008,0.00819,0.00874,0.00933,0.00942,
      0.00999,0.01059,0.01066,0.01128,0.01172,0.01234,0.01242,0.013,0.01357,0.01373,
      0.01401,0.01424,0.01451,0.01473,0.01508,0.01496,0.0151,0.01467,0.01475,0.01482,
      0.01448,0.01426,0.01398,0.01383,0.01323,0.01278,0.01217,0.01206,0.01158,0.01064,
      0.01037,0.00941,0.00902,0.00851,0.00797,0.0076,0.00666,0.00628,0.00574,0.00528,
      0.00466,0.00459,0.00399,0.00349,0.00305,0.00232,0.00267,0.002,0.00135,0.00158,
      0.00131,0.00067,0.00052,0.00025,-0.0,-0.00025,-0.0005,-0.00075,-0.001,-0.00127,
      -0.00141,-0.00208,-0.00226,-0.0024,-0.00311,-0.00315,-0.00377,-0.00424,-0.00475,-0.00525,
      -0.00574,-0.00627,-0.00668,-0.0075,-0.00831,-0.00874,-0.00921,-0.0099,-0.01117,-0.0114,
      -0.01225,-0.01309,-0.0134,-0.01428,-0.01497,-0.01582,-0.01624,-0.01673,-0.01734,-0.01742,
      -0.018,-0.01857,-0.01873,-0.01902,-0.01921,-0.01966,-0.01917,-0.01917,-0.01966,-0.01921,
      -0.019,-0.01877,-0.01841,-0.01857,-0.0183,-0.01776,-0.01718,-0.01702,-0.01673,-0.01656,
      -0.01603,-0.01533,-0.01567,-0.01501,-0.01433,-0.01469,-0.01392,-0.01366,-0.01392,-0.01316,
      -0.01294,-0.01307,-0.01278,-0.01233,-0.01292,-0.0125,-0.0121,-0.01259,-0.01256,-0.01219,
      -0.0122,-0.0125,-0.01281,-0.01276,-0.01266,-0.01311,-0.01291,-0.01325,-0.01358,-0.01344,
      -0.01366,-0.01442,-0.01414,-0.01452,-0.01475,-0.01499,-0.01529,-0.01537,-0.0162,-0.01582,
      -0.0165,-0.01714,-0.01694,-0.0171,-0.01816,-0.01824,-0.01841,-0.01911,-0.01915,-0.01978,
      -0.02023,-0.02082,-0.021,-0.02118,-0.02176,-0.02229,-0.0226,-0.02233,-0.02308,-0.02332,
      -0.02315,-0.02359,-0.0235,-0.02343,-0.02379,-0.02391,-0.02308,-0.02327,-0.02332,-0.02296,
      -0.02285,-0.02217,-0.02201,-0.02182,-0.02125,-0.02069,-0.02049,-0.02034,-0.01967,-0.0195,
      -0.01933,-0.01869,-0.01843,-0.01858,-0.01825,-0.01795,-0.01796,-0.01824,-0.01858,-0.01843,
      -0.0187,-0.01926,-0.01974,-0.02026,-0.0207,-0.02143,-0.02258,-0.02325,-0.02393,-0.02501,
      -0.02602,-0.02691,-0.02833,-0.02925,-0.03016,-0.03158,-0.0325,-0.03342,-0.03481,-0.03582,
      -0.03642,-0.0375,-0.03856,-0.03924,-0.03998,-0.04085,-0.04115,-0.04206,-0.0426,-0.04256,
      -0.04368,-0.04371,-0.04401,-0.04427,-0.04445,-0.04495,-0.0443,-0.04487,-0.04473,-0.04474,
      -0.04482,-0.04451,-0.04417,-0.04434,-0.04399,-0.04372,-0.04362,-0.04282,-0.0431,-0.0428,
      -0.04224,-0.04175,-0.04128,-0.04067,-0.04057,-0.04005,-0.03926,-0.03845,-0.03798,-0.03764,
      -0.03649,-0.03545,-0.03476,-0.03405,-0.03308,-0.03167,-0.03078,-0.02974,-0.02882,-0.02752,
      -0.02616,-0.02535,-0.02397,-0.02279,-0.02141,-0.02061,-0.01917,-0.01826,-0.01732,-0.016,
      -0.01471,-0.01368,-0.01309,-0.012,-0.01095,-0.01024,-0.00959,-0.00842,-0.00777,-0.00702,
      -0.00616,-0.00583,-0.00502,-0.0041,-0.00408,-0.00357,-0.00266,-0.0023,-0.00166,-0.00159,
      -0.001,-0.00043,-0.00028,3e-05,0.00016,0.00082,0.00106,0.00094,0.00118,0.00182,
      0.00204,0.00201,0.00192,0.00232,0.00228,0.00208,0.0029,0.00282,0.00233,0.00286,
      0.00273,0.00274,0.00278,0.00263,0.00318,0.00264,0.00277,0.00278,0.00263,0.00318,
      0.00266,0.00271,0.003,0.00331,0.00326,0.00317,0.00354,0.00365,0.00433,0.0045,
      0.00465,0.00537,0.00534,0.00625,0.00711,0.00732,0.00861,0.00924,0.00993,0.01101,
      0.01201,0.01293,0.01424,0.01558,0.01643,0.0177,0.01924,0.02081,0.022,0.02318,
      0.02477,0.02622,0.02784,0.02892,0.03049,0.03209,0.03314,0.03484,0.036,0.03714,
      0.0389,0.03974,0.04067,0.04209,0.04298,0.04399,0.04507,0.04574,0.04649,0.04732,
      0.04774,0.04824,0.04882,0.04899,0.04926,0.04949,0.04982,0.04975,0.0497,0.04994,
      0.04906,0.04935,0.04905,0.04849,0.04802,0.04747,0.04711,0.04614,0.04585,0.04498,
      0.04426,0.04351,0.04276,0.04199,0.04131,0.04028,0.0391,0.03884,0.03807,0.03692,
      0.03628,0.0355,0.03475,0.03403,0.03319,0.03276,0.03232,0.03151,0.03069,0.03027,
      0.02975,0.02926,0.02873,0.02834,0.02742,0.027,0.02659,0.02567,0.02526,0.02481,
      0.02401,0.02318,0.02278,0.02223,0.02182,0.02101,0.02016,0.01986,0.01892,0.01849,
      0.01812,0.01708,0.01708,0.01659,0.01558,0.0156,0.01502,0.01434,0.01461,0.01423,
      0.014,0.01377,0.01345,0.01343,0.01383,0.01375,0.01368,0.01402,0.01424,0.01452,
      0.01467,0.01527,0.01574,0.01627,0.01668,0.01749,0.01831,0.01874,0.01921,0.01992,
      0.02108,0.02173,0.02249,0.02327,0.02391,0.02506,0.02584,0.02608,0.02732,0.0281,
      0.0283,0.02967,0.02999,0.03039,0.03142,0.03142,0.03192,0.0329,0.03299,0.03317,
      0.03384,0.03398,0.03425,0.03451,0.03473,0.03507,0.035,0.03494,0.03525,0.03557,
      0.03548,0.03551,0.0355,0.0355,0.03553,0.03542,0.03583,0.03577,0.03564,0.03618,
      0.03567,0.03566,0.03618,0.03565,0.03575,0.03585,0.03539,0.0356,0.03524,0.03498,
      0.03484,0.03418,0.03398,0.03391,0.03291,0.0325,0.0321,0.03114,0.03087,0.02993,
      0.02948,0.02918,0.02784,0.0275,0.02718,0.02584,0.02552,0.02509,0.02416,0.02378,
      0.02325,0.02273,0.02234,0.02142,0.02101,0.02053,0.01991,0.01985,0.01923,0.01876,
      0.01827,0.01769,0.01749,0.01734,0.01667,0.01652,0.01625,0.01599,0.01578,0.01541,
      0.01558,0.01527,0.01485,0.01533,0.01532,0.01492,0.01502,0.01502,0.01492,0.01533,
      0.01526,0.01515,0.01562,0.01539,0.01583,0.01578,0.01557,0.01642,0.01623,0.01618,
      0.01652,0.01673,0.01703,0.01715,0.01784,0.01798,0.01826,0.01846,0.01892,0.01983,
      0.02024,0.02071,0.02141,0.0226,0.02316,0.02426,0.02525,0.02622,0.0273,0.02808,
      0.02984,0.03099,0.03216,0.03383,0.03497,0.03627,0.03741,0.03907,0.04027,0.04137,
      0.04318,0.04389,0.04525,0.04656,0.04748,0.04851,0.04948,0.05057,0.05124,0.05198,
      0.05282,0.05323,0.05376,0.05423,0.05481,0.05502,0.05515,0.0559,0.05574,0.05567,
      0.0561,0.05598,0.05602,0.05598,0.05609,0.05568,0.05574,0.05585,0.05541,0.05555,
      0.05542,0.05581,0.05584,0.05537,0.0557,0.05635,0.05641,0.05703,0.05746,0.05815,
      0.05937,0.05986,0.06115,0.06298,0.06486,0.06604,0.06845,0.07103,0.07335,0.07645,
      0.07975,0.08293,0.08643,0.0902,0.09413,0.09858,0.10241,0.10665,0.1113,0.11546,
      0.11971,0.12403,0.12803,0.13172,0.13549,0.13918,0.14168,0.14451,0.14717,0.14876,
      0.15027,0.15166,0.15211,0.15194,0.1522,0.15135,0.15003,0.14865,0.14653,0.14439,
      0.14158,0.1385,0.13564,0.13216,0.12799,0.12412,0.11979,0.1155,0.11146,0.10646,
      0.10202,0.09773,0.09238,0.08806,0.08363,0.07872,0.07425,0.07004,0.06582,0.06146,
      0.0576,0.0534,0.05005,0.04657,0.04289,0.04006,0.03706,0.03389,0.03156,0.02898,
      0.0267,0.02481,0.0227,0.02098,0.01951,0.01806,0.01639,0.01546,0.01482,0.01379,
      0.01261,0.01229,0.01174,0.01128,0.01068,0.01053,0.01019,0.01023,0.01035,0.00991,
      0.01002,0.01002,0.00992,0.01033,0.01026,0.01018,0.01051,0.0108,0.01077,0.01065,
      0.01109,0.01098,0.01101,0.011,0.01098,0.01109,0.01068,0.01074,0.01083,0.01048,
      0.01027,0.00999,0.00977,0.00948,0.00932,0.00876,0.00817,0.00808,0.0075,0.007,
      0.00652,0.00597,0.00561,0.00464,0.00434,0.00352,0.00265,0.00243,0.00117,0.00047,
      -5e-05,-0.00168,-0.00215,-0.00317,-0.00457,-0.00549,-0.00643,-0.00774,-0.00901,-0.01017,
      -0.01174,-0.0133,-0.0145,-0.01567,-0.01725,-0.01873,-0.02024,-0.02172,-0.0233,-0.02448,
      -0.02571,-0.02708,-0.02791,-0.02924,-0.03055,-0.03152,-0.03233,-0.03265,-0.03356,-0.03406,
      -0.03423,-0.0345,-0.03477,-0.03488,-0.03424,-0.03369,-0.03355,-0.03314,-0.03196,-0.03109,
      -0.02976,-0.0285,-0.02733,-0.02577,-0.02417,-0.02167,-0.01982,-0.01767,-0.01472,-0.01213,
      -0.00898,-0.00616,-0.00259,0.00126,0.00483,0.00917,0.01372,0.01816,0.02284,0.02816,
      0.0337,0.03922,0.04458,0.05062,0.05704,0.06287,0.06912,0.07576,0.08195,0.0881,
      0.09476,0.101,0.10692,0.11298,0.11883,0.1239,0.12933,0.13401,0.13843,0.14299,
      0.14591,0.14923,0.15248,0.15425,0.156,0.15767,0.15832,0.1586,0.15836,0.15756,
      0.15653,0.15543,0.1534,0.15074,0.1484,0.1454,0.14182,0.13813,0.13401,0.13018,
      0.1256,0.1208,0.11604,0.11143,0.10608,0.10071,0.09592,0.09053,0.08535,0.07996,
      0.07517,0.06974,0.06477,0.06001,0.05554,0.0511,0.04641,0.04254,0.03872,0.03532,
      0.03175,0.02843,0.02574,0.02331,0.02072,0.0185,0.01643,0.01494,0.01388,0.01213,
      0.0112,0.01063,0.00938,0.00894,0.00886,0.00816,0.00804,0.00769,0.00776,0.00776,
      0.00773,0.00777,0.00773,0.00779,0.00767,0.00801,0.0083,0.00827,0.00815,0.00859,
      0.00849,0.00844,0.00876,0.009,0.00924,0.00952,0.00967,0.01027,0.01074,0.01126,
      0.0117,0.01241,0.01358,0.01419,0.01512,0.01674,0.01831,0.01942,0.02094,0.02269,
      0.0247,0.02685,0.02832,0.03074,0.03305,0.03492,0.03715,0.0398,0.04196,0.04423,
      0.04647,0.04876,0.05082,0.05236,0.05462,0.05605,0.05713,0.05885,0.05988,0.06013,
      0.06111,0.06139,0.06088,0.06113,0.06065,0.0594,0.05888,0.05766,0.05566,0.05435,
      0.05259,0.05048,0.04866,0.04606,0.04336,0.04168,0.03909,0.0362,0.03383,0.03119,
      0.0291,0.02656,0.02389,0.02205,0.02009,0.01775,0.01562,0.01439,0.01245,0.01096,
      0.0098,0.00845,0.00747,0.00676,0.00603,0.0052,0.00471,0.0045,0.0043,0.00384,
      0.00434,0.00426,0.00417,0.00453,0.00473,0.00503,0.00516,0.00581,0.00607,0.00592,
      0.00624,0.00657,0.00646,0.00657,0.00625,0.00596,0.0059,0.00498,0.00425,0.00355,
      0.00264,0.00099,-0.00044,-0.00216,-0.00421,-0.00638,-0.00903,-0.01132,-0.01443,-0.01772,
      -0.02091,-0.0244,-0.02817,-0.03213,-0.03647,-0.0407,-0.04489,-0.04944,-0.05398,-0.05834,
      -0.06329,-0.06765,-0.07225,-0.07651,-0.08042,-0.0845,-0.08827,-0.09165,-0.09538,-0.09808,
      -0.10019,-0.10252,-0.10462,-0.10586,-0.10644,-0.10735,-0.10764,-0.10713,-0.10596,-0.10516,
      -0.10355,-0.10186,-0.09967,-0.09674,-0.09412,-0.0911,-0.08779,-0.08457,-0.08121,-0.07698,
      -0.07325,-0.06986,-0.06613,-0.062,-0.05825,-0.05481,-0.05131,-0.04776,-0.04447,-0.04164,
      -0.03823,-0.03525,-0.0325,-0.03003,-0.02758,-0.0249,-0.02298,-0.02136,-0.01924,-0.01742,
      -0.01616,-0.0141,-0.01306,-0.01176,-0.01052,-0.00923,-0.00818,-0.0076,-0.00651,-0.00546,
      -0.00475,-0.0041,-0.00296,-0.00217,-0.00192,-0.0007,8e-05,0.00039,0.00128,0.00195,
      0.00284,0.00316,0.00399,0.00481,0.00523,0.00574,0.00627,0.00666,0.00756,0.00805,
      0.00825,0.00845,0.00896,0.00966,0.00941,0.00969,0.01031,0.01057,0.01042,0.01074,
      0.01109,0.01092,0.01125,0.01159,0.01142,0.01175,0.01209,0.01192,0.01225,0.01259,
      0.01241,0.01277,0.01299,0.01326,0.01348,0.01384,0.01367,0.014,0.01434,0.01414,
      0.0146,0.01448,0.01451,0.01451,0.01448,0.0146,0.01415,0.01432,0.01408,0.01333,
      0.0136,0.01328,0.01276,0.01216,0.01209,0.01149,0.01093,0.01077,0.01047,0.01034,
      0.00966,0.00951,0.00929,0.00884,0.00935,0.00926,0.00912,0.00979,0.01024,0.01077,
      0.0112,0.01195,0.01303,0.01395,0.0152,0.01676,0.01827,0.01968,0.02154,0.0232,
      0.02518,0.02761,0.02941,0.03178,0.03402,0.03617,0.03883,0.04107,0.04293,0.04526,
      0.04758,0.04948,0.05157,0.05331,0.05474,0.05625,0.0578,0.05911,0.05979,0.06025,
      0.06073,0.06136,0.06136,0.06073,0.06025,0.05979,0.05913,0.05772,0.0565,0.0553,
      0.05381,0.05198,0.05026,0.04848,0.04682,0.04473,0.04276,0.04075,0.03874,0.03679,
      0.03459,0.03334,0.03156,0.02943,0.0277,0.02626,0.02476,0.0232,0.02196,0.02098,
      0.02011,0.01859,0.01804,0.01728,0.01634,0.01635,0.01576,0.01512,0.01528,0.01527,
      0.01516,0.01561,0.01542,0.01571,0.01624,0.01684,0.01693,0.01746,0.01825,0.01906,
      0.01953,0.01983,0.02118,0.02148,0.02192,0.02284,0.02323,0.02376,0.02423,0.02483,
      0.02496,0.02535,0.02515,0.02557,0.0256,0.02503,0.02578,0.02487,0.02474,0.02468,
      0.02356,0.0236,0.02307,0.02214,0.02186,0.02092,0.02048,0.02018,0.01882,0.01853,
      0.01805,0.01725,0.01643,0.01601,0.01552,0.01491,0.01484,0.01422,0.01377,0.0132,
      0.01294,0.01306,0.01281,0.01219,0.01195,0.01202,0.01198,0.01207,0.01175,0.01143,
      0.01152,0.0115,0.01148,0.01157,0.01125,0.01093,0.01102,0.01098,0.01107,0.01073,
     ),
    ),
    # crossover: 2902 uniform samples; public map SHA256 3d387be6e4e12659047511d3a3bc136fd2e3eaad4520fdc47f8fad6cddfa321f
    (5803.166698722436,
     (
      -6e-05,6.61e-05,-8.87e-05,7.76e-05,-1.03e-05,-3.64e-05,4.03e-05,-9.2e-06,-3.5e-06,3.4e-06,
      9.6e-06,-4.18e-05,4.22e-05,-1.13e-05,3.2e-06,-1.3e-06,2.2e-06,-7.3e-06,7.3e-06,-2e-06,
      7e-07,-7e-07,2e-06,-7.5e-06,8.1e-06,-4.9e-06,1.16e-05,-4.16e-05,3.91e-05,7e-07,
      -4.18e-05,5.08e-05,-4.61e-05,1.81e-05,-6.7e-06,-1.12e-05,5.16e-05,-7.98e-05,5.63e-05,-5e-05,
      4.82e-05,-4.7e-05,4.44e-05,-3.5e-05,-0.0,3.5e-05,-4.42e-05,4.62e-05,-4.5e-05,3.83e-05,
      -1.25e-05,1.16e-05,-1.42e-05,2.55e-05,2.77e-05,-4.07e-05,3.94e-05,-2.12e-05,6.54e-05,-4.93e-05,
      3.62e-05,1.99e-05,-1.99e-05,5.99e-05,-8.7e-06,-2.49e-05,1.27e-05,6.97e-05,-8.05e-05,4.15e-05,
      2.97e-05,-6.44e-05,1.71e-05,-4.2e-06,-1e-07,-9.08e-05,3.7e-05,-5.74e-05,-1.84e-05,-0.0001001,
      1.63e-05,-8.09e-05,-0.0001147,1.8e-06,-0.0001043,-0.0001223,-4e-05,-0.0001405,-5.16e-05,-0.0001712,
      -1.26e-05,-0.0002011,-4.75e-05,-0.0001273,-9.71e-05,-0.0001179,-6.47e-05,-0.0001616,-3.8e-05,-0.0001092,
      -6.34e-05,-5.99e-05,-0.0001192,-1.3e-06,-8.72e-05,-7.19e-05,4.73e-05,-0.000118,3e-06,-1.06e-05,
      -5.57e-05,2.24e-05,-3.42e-05,-1.3e-06,3.93e-05,-3.99e-05,4.7e-06,2.08e-05,2.79e-05,-3.71e-05,
      2.57e-05,2.89e-05,-2.55e-05,7.31e-05,-5.63e-05,3.68e-05,2.49e-05,-2.02e-05,3.56e-05,-6.3e-06,
      -1.03e-05,4.76e-05,-6.41e-05,9.35e-05,-9.92e-05,9.3e-05,-6.25e-05,6.26e-05,-9.31e-05,9.94e-05,
      -9.41e-05,6.67e-05,-5.76e-05,4.82e-05,-4.05e-05,1.9e-05,-3.59e-05,2.94e-05,1.28e-05,-8.05e-05,
      9.84e-05,-0.0001031,0.0001037,-0.0001014,9.17e-05,-5.52e-05,1.38e-05,2.05e-05,-9e-07,-1.69e-05,
      6.83e-05,-4.58e-05,2.03e-05,5.91e-05,-4.6e-05,3.06e-05,1.84e-05,1.19e-05,2.92e-05,-1.3e-05,
      2.29e-05,1.62e-05,2.81e-05,-1.28e-05,2.8e-06,2.22e-05,3.5e-06,-3.58e-05,2.42e-05,-4.07e-05,
      2.29e-05,-5.12e-05,-2.85e-05,-4.59e-05,-1.99e-05,-6.51e-05,-4.63e-05,-7.68e-05,-6.87e-05,-7.09e-05,
      -7.01e-05,-9.18e-05,-7.99e-05,-0.0001269,-4.55e-05,-0.000135,-4.73e-05,-9.88e-05,-7.43e-05,-4.76e-05,
      -6.3e-05,-6.6e-06,-2.76e-05,2.2e-05,3.4e-05,5.27e-05,8.21e-05,0.0001567,0.0001347,0.0002647,
      0.0001881,0.000271,0.000321,0.0003652,0.0003499,0.0004626,0.0003737,0.0004823,0.0004621,0.000455,
      0.0003693,0.0005276,0.0002857,0.0003496,0.0002375,0.0001042,7.86e-05,-0.0002054,-0.0002135,-0.0005331,
      -0.000627,-0.0010087,-0.0011912,-0.0014639,-0.0018045,-0.0021846,-0.0024551,-0.0029374,-0.0032893,-0.0036748,
      -0.0041503,-0.004562,-0.0050198,-0.00542,-0.0059769,-0.0063326,-0.0068043,-0.0072642,-0.0076626,-0.0080891,
      -0.0085028,-0.0089027,-0.0092763,-0.0096936,-0.0099178,-0.0104249,-0.0106121,-0.0109749,-0.0112413,-0.0114937,
      -0.01169,-0.0119228,-0.0121018,-0.0122205,-0.0123461,-0.0123962,-0.0124014,-0.0124753,-0.012332,-0.0122751,
      -0.0121498,-0.0119533,-0.0116981,-0.0115074,-0.011152,-0.0108291,-0.0104172,-0.0100511,-0.0096209,-0.0091069,
      -0.0086788,-0.0081766,-0.0076665,-0.0072001,-0.0067153,-0.0061161,-0.0058251,-0.0052065,-0.0048732,-0.0044225,
      -0.0041225,-0.0037684,-0.0034605,-0.0032948,-0.0030678,-0.0029487,-0.0028898,-0.0027683,-0.0029291,-0.0028793,
      -0.0031084,-0.0032356,-0.0034956,-0.003749,-0.004102,-0.0044214,-0.0048971,-0.0053468,-0.0058852,-0.0065426,
      -0.0072345,-0.0080275,-0.0089606,-0.0099292,-0.0109602,-0.0122634,-0.0134169,-0.0146936,-0.0159399,-0.0172168,
      -0.0182072,-0.0191947,-0.0199452,-0.0203689,-0.0207184,-0.0207036,-0.0206569,-0.0203245,-0.0199862,-0.0194593,
      -0.0191107,-0.0184721,-0.0181573,-0.0176308,-0.0173919,-0.0170352,-0.0167877,-0.0166194,-0.0165369,-0.0164435,
      -0.0163483,-0.0164551,-0.0163874,-0.0165369,-0.0165101,-0.0166404,-0.0167284,-0.0167756,-0.0168782,-0.0169999,
      -0.0170539,-0.0170775,-0.0172154,-0.0172433,-0.0172468,-0.0173187,-0.0172163,-0.0172337,-0.0171685,-0.0169791,
      -0.016897,-0.0166014,-0.016434,-0.016012,-0.0156796,-0.0151863,-0.0147718,-0.0141451,-0.013618,-0.0128933,
      -0.0122758,-0.0114857,-0.0108747,-0.0099613,-0.0093522,-0.0084397,-0.0077644,-0.006943,-0.0062317,-0.0054711,
      -0.0047677,-0.0040991,-0.0034827,-0.0027492,-0.0023079,-0.0016395,-0.0012277,-0.000816,-0.0002737,-5.12e-05,
      0.0002687,0.0005933,0.0007758,0.0009586,0.0010992,0.0011788,0.0012614,0.0013138,0.0012282,0.0013915,
      0.0011569,0.0012665,0.0010612,0.0011071,0.0009205,0.0008739,0.0007865,0.0006421,0.0005593,0.0004499,
      0.0003908,0.0002772,8.2e-05,0.0001888,-8.84e-05,-4.05e-05,-0.0002077,-0.0002091,-0.0003268,-0.0003561,
      -0.0004119,-0.0004925,-0.0005293,-0.0005535,-0.0005833,-0.00061,-0.000681,-0.0005779,-0.0006679,-0.0006623,
      -0.0005151,-0.0006043,-0.0004855,-0.0003665,-0.0003402,-0.0002297,-3.14e-05,9.82e-05,0.0003037,0.0004329,
      0.0007921,0.0009733,0.0012596,0.0016012,0.0019845,0.0022732,0.0026917,0.0030987,0.0035903,0.0040457,
      0.0044823,0.0050345,0.0055969,0.0060585,0.0067081,0.0071943,0.0078571,0.0083983,0.0090055,0.0095715,
      0.0101599,0.0107044,0.0113102,0.011744,0.0123222,0.0127274,0.0132416,0.0135564,0.0139476,0.0142891,
      0.0144595,0.0147154,0.0148467,0.0148808,0.0149566,0.0148304,0.0147639,0.0144508,0.0142736,0.0138739,
      0.0134986,0.0129625,0.0124329,0.0118413,0.0111638,0.0103669,0.0097276,0.0088579,0.0079771,0.0071758,
      0.0062349,0.0053289,0.0043334,0.0034811,0.0024724,0.0015122,0.0005028,-0.0003561,-0.0014651,-0.0023184,
      -0.0033557,-0.0042048,-0.0051932,-0.0060991,-0.0069317,-0.0078526,-0.0086297,-0.0093921,-0.0102187,-0.0108132,
      -0.0114922,-0.0120987,-0.0125623,-0.0131265,-0.013503,-0.0138423,-0.0141658,-0.0144495,-0.0146119,-0.0147842,
      -0.0149442,-0.0149598,-0.0150409,-0.0150167,-0.0151005,-0.0149793,-0.0149106,-0.0149372,-0.014724,-0.0146692,
      -0.0144921,-0.0143985,-0.0141823,-0.0140187,-0.0137136,-0.0135736,-0.0132679,-0.0129765,-0.0126452,-0.0123612,
      -0.011881,-0.0116069,-0.0111192,-0.010735,-0.0102365,-0.0097944,-0.0092691,-0.0087597,-0.0083664,-0.007741,
      -0.0072685,-0.0067634,-0.0063037,-0.0057506,-0.0052961,-0.0048211,-0.0043846,-0.0038629,-0.0035426,-0.003081,
      -0.0026293,-0.0023474,-0.0019241,-0.0015703,-0.0011902,-0.0009171,-0.0005676,-0.0002028,0.0001358,0.0004187,
      0.0008043,0.0011605,0.0014631,0.0019689,0.0022267,0.0027284,0.0030808,0.0036128,0.0040024,0.0045442,
      0.0049765,0.0055488,0.0059894,0.0065482,0.0070464,0.0075874,0.008116,0.008536,0.0091103,0.0095283,
      0.0099315,0.0104108,0.0107574,0.0111123,0.0114756,0.011696,0.0120428,0.0121762,0.0125383,0.0124964,
      0.0128015,0.0128555,0.0130099,0.0130246,0.0131445,0.0131303,0.0132847,0.0131679,0.0132759,0.0132487,
      0.0132756,0.0132417,0.0131851,0.0132587,0.0131017,0.013108,0.0130385,0.0129311,0.0128175,0.0127607,
      0.0125104,0.0123246,0.0122557,0.0118235,0.0116903,0.0113324,0.0109372,0.0106531,0.0101433,0.0097847,
      0.0092124,0.0087809,0.0082135,0.0076372,0.0070965,0.0064637,0.0058738,0.0052238,0.0046432,0.0039907,
      0.0032775,0.00272,0.0020592,0.001358,0.0007717,6.77e-05,-0.0005813,-0.0011823,-0.001882,-0.0025325,
      -0.0031308,-0.0038473,-0.0044328,-0.0051385,-0.0056388,-0.006419,-0.0068794,-0.0074741,-0.0081242,-0.0085193,
      -0.0090625,-0.0095673,-0.0098393,-0.0103363,-0.010578,-0.0108249,-0.0111174,-0.0111736,-0.0114022,-0.0114481,
      -0.0114721,-0.0115252,-0.011503,-0.011423,-0.0114278,-0.011314,-0.0112362,-0.011155,-0.0110918,-0.0109724,
      -0.01087,-0.0108459,-0.0107247,-0.010695,-0.0106299,-0.0106433,-0.0105047,-0.0106341,-0.010586,-0.0106018,
      -0.0106679,-0.0107121,-0.0108238,-0.010927,-0.0110226,-0.0112053,-0.0113259,-0.0115532,-0.0116905,-0.0119347,
      -0.0121141,-0.0123789,-0.0125736,-0.0127939,-0.0130174,-0.0132587,-0.0134734,-0.0136557,-0.0138358,-0.0140049,
      -0.0141397,-0.0142907,-0.0143659,-0.0144153,-0.0144378,-0.0144222,-0.0143732,-0.0142889,-0.0141325,-0.0139281,
      -0.0137666,-0.013389,-0.0131539,-0.0127488,-0.0123558,-0.0119393,-0.011347,-0.0110146,-0.0103221,-0.0098565,
      -0.0092717,-0.008671,-0.0080241,-0.0075533,-0.006826,-0.006343,-0.005721,-0.0051329,-0.0046489,-0.0040966,
      -0.0035579,-0.0031048,-0.0026982,-0.0021294,-0.0018587,-0.0012794,-0.0011097,-0.0005298,-0.0003443,0.0001482,
      0.0004148,0.00075,0.0011504,0.0014407,0.0018305,0.0022647,0.0025057,0.0030045,0.0033843,0.0038222,
      0.0042542,0.0046884,0.0051738,0.0056397,0.0061378,0.0066243,0.0070791,0.0076502,0.008029,0.0085567,
      0.0089769,0.009446,0.0098498,0.0101857,0.0106284,0.010934,0.0111989,0.0114844,0.0117617,0.0119247,
      0.0121355,0.0123054,0.0123317,0.0125872,0.0125264,0.0127052,0.0127214,0.0126997,0.0128045,0.0128,
      0.0128029,0.0128171,0.0128321,0.0128378,0.0128292,0.0128049,0.0128608,0.0128697,0.012803,0.0128263,
      0.0128736,0.0127926,0.0127761,0.0127552,0.0126969,0.0126333,0.0125213,0.0124731,0.0122516,0.0121514,
      0.0119205,0.0117104,0.0114815,0.0111367,0.0108791,0.0105228,0.0100684,0.0098326,0.0091958,0.0089389,
      0.0083945,0.0079372,0.0074691,0.0070161,0.0065257,0.0060802,0.0056218,0.0052354,0.0048201,0.0043606,
      0.0041243,0.0037095,0.0034554,0.0031405,0.0029097,0.0028043,0.0025182,0.0024271,0.0023413,0.002241,
      0.002256,0.0022328,0.0022497,0.0022657,0.002458,0.0024196,0.0026591,0.0027421,0.0029214,0.003123,
      0.0032506,0.0035429,0.0036127,0.0039898,0.0040609,0.0042998,0.0045029,0.0047292,0.0048562,0.0049787,
      0.0052197,0.0052396,0.005354,0.0055177,0.0054728,0.0055236,0.0055861,0.0055107,0.0055389,0.005526,
      0.0053747,0.0053189,0.0052932,0.0051082,0.0050656,0.004912,0.0047873,0.0046566,0.0045068,0.0044457,
      0.0042289,0.0042348,0.0039545,0.0040043,0.0037819,0.003843,0.0036768,0.0036697,0.0037054,0.00357,
      0.0037734,0.0036906,0.0038204,0.0038344,0.0040976,0.0040885,0.0043198,0.0044643,0.0047028,0.0048449,
      0.0050883,0.0053495,0.0054585,0.005837,0.0059158,0.0062108,0.0063302,0.0065308,0.0066381,0.0067065,
      0.006851,0.0067908,0.006835,0.0067034,0.0067098,0.0064918,0.0063841,0.0061033,0.0059427,0.0056743,
      0.0053379,0.0051491,0.0048228,0.0044729,0.0042671,0.0039617,0.0036575,0.0034261,0.0032336,0.0030103,
      0.0028744,0.0026357,0.0026432,0.0024999,0.0024719,0.0024518,0.0024865,0.0025032,0.0026002,0.0026605,
      0.0027933,0.0029384,0.0030976,0.0031881,0.0034063,0.0035873,0.0037232,0.0039454,0.0041281,0.0042708,
      0.0043931,0.0046571,0.0047135,0.0048929,0.0049282,0.0051521,0.0051301,0.0052455,0.0052681,0.0052707,
      0.0053401,0.0052386,0.0052165,0.005256,0.0050665,0.0050135,0.0049468,0.0048059,0.0046878,0.0045201,
      0.004449,0.0042507,0.0041379,0.0039399,0.00388,0.0036752,0.0035988,0.0033636,0.0033216,0.0031915,
      0.0030166,0.0028912,0.002826,0.0026917,0.002541,0.002448,0.0023246,0.0021748,0.0020505,0.0019623,
      0.0017896,0.0016393,0.0015361,0.0013818,0.0012404,0.0011141,0.0010113,0.0007996,0.0008082,0.0005609,
      0.0005641,0.0003734,0.0003096,0.0002915,0.0001199,9.72e-05,3.09e-05,-7.9e-06,-0.0001135,-7.42e-05,
      -0.0001079,-0.0002427,-9.11e-05,-0.0002594,-0.0001369,-0.0002505,-0.0001474,-0.000237,-0.0001714,-0.0001545,
      -0.0001523,-0.0001012,-0.0001928,2.75e-05,-0.0001315,-3.78e-05,6.96e-05,-3.15e-05,5.75e-05,0.0001263,
      6.96e-05,0.0002292,8.71e-05,0.0002688,0.0002182,0.0002415,0.0003138,0.0002123,0.0004321,0.0001889,
      0.0004094,0.000303,0.000301,0.0002989,0.0003294,0.0001888,0.0003005,0.0001075,0.0001176,-3.76e-05,
      -0.0001742,-0.0004324,-0.0007593,-0.0013222,-0.002136,-0.0033047,-0.0050656,-0.0074576,-0.0107282,-0.0149582,
      -0.0199348,-0.0259186,-0.0316734,-0.0370955,-0.0403414,-0.0410493,-0.0385261,-0.0337293,-0.0277063,-0.0212895,
      -0.0157699,-0.0108561,-0.0072636,-0.0044858,-0.0025822,-0.0013401,-0.0005103,-0.0001014,0.0003063,0.0003752,
      0.0004454,0.0005456,0.0005347,0.0004804,0.0005478,0.0004925,0.0004859,0.0004299,0.0004991,0.0002899,
      0.0004494,0.0003161,0.000257,0.0003016,0.0002067,0.0002183,0.0001293,0.0001624,0.0001309,6.19e-05,
      8.42e-05,4.98e-05,1.6e-05,3.62e-05,-1.12e-05,9.3e-06,-1.29e-05,2.92e-05,-0.0001037,8.82e-05,
      -0.0001016,2.12e-05,1.53e-05,-8.25e-05,1.71e-05,1.28e-05,-8.15e-05,2.88e-05,-3.5e-05,-5.06e-05,
      -6.07e-05,7.6e-06,-0.000133,-8.46e-05,-0.0001134,-0.0002215,-0.0003561,-0.0004563,-0.0008301,-0.0013346,
      -0.0022392,-0.0036763,-0.0058906,-0.0092248,-0.0137389,-0.0197435,-0.0268945,-0.0345809,-0.0416285,-0.0468278,
      -0.049493,-0.050252,-0.0494708,-0.0485411,-0.0475291,-0.0463538,-0.04479,-0.0419913,-0.0378548,-0.0322224,
      -0.0256236,-0.0192367,-0.0132936,-0.0086499,-0.0052224,-0.0029551,-0.0015014,-0.0006941,-0.0003142,-2.2e-05,
      0.0001067,3.66e-05,0.0001885,0.0001498,0.0001244,0.0001175,0.0001705,0.0001128,0.0001436,7.8e-05,
      0.0001623,3.77e-05,0.0001288,9.37e-05,5.62e-05,3.42e-05,0.0001016,1e-06,7.08e-05,9.9e-06,
      3.68e-05,-1e-05,3.25e-05,-2.1e-06,5.1e-06,-4.72e-05,6.59e-05,-9.85e-05,3.42e-05,-9.4e-06,
      -2.58e-05,-3.44e-05,-1.28e-05,-3.24e-05,-4.8e-06,2.24e-05,-8.49e-05,2.32e-05,-8.1e-06,9.3e-06,
      -2.92e-05,-3.95e-05,4e-05,-2.7e-06,-0.0,2.8e-06,-1.12e-05,4.18e-05,-9.1e-06,-5.3e-06,
      3.04e-05,3.08e-05,-6.3e-06,2.36e-05,2.98e-05,3.35e-05,-1.67e-05,3.33e-05,3.05e-05,-8e-06,
      3.09e-05,2.4e-06,-1.14e-05,4.31e-05,-4.32e-05,1.21e-05,-3.44e-05,7.7e-06,-2.57e-05,-5.21e-05,
      -6.02e-05,-1.5e-06,-0.0001103,-2.75e-05,-7.43e-05,-0.0001168,-7.6e-05,-4.99e-05,-0.000166,-5.08e-05,
      -0.000102,-0.0001587,-2.8e-05,-0.0001716,-5e-05,-9.93e-05,-2.33e-05,-0.0001024,-8.8e-06,-1.04e-05,
      2.13e-05,7.2e-05,1.37e-05,0.0001965,8.27e-05,0.0002665,0.0002041,0.0002999,0.0002732,0.0004606,
      0.0003255,0.0005328,0.0004256,0.0005307,0.000511,0.0005155,0.0006042,0.0004516,0.0005036,0.0004473,
      0.0003273,0.000362,7.97e-05,8.65e-05,-0.000131,-0.000325,-0.0005367,-0.0007629,-0.0009987,-0.0013592,
      -0.0016815,-0.0020033,-0.002394,-0.0027749,-0.0032754,-0.00363,-0.0041283,-0.0046401,-0.0051034,-0.0055577,
      -0.0061219,-0.0065344,-0.0071195,-0.0075004,-0.0080293,-0.0083449,-0.0089161,-0.0091294,-0.0094825,-0.0098048,
      -0.0099793,-0.0101987,-0.0102681,-0.0103094,-0.0103506,-0.0102805,-0.0101084,-0.0099964,-0.0097681,-0.0094666,
      -0.0091518,-0.0088449,-0.0084304,-0.0080181,-0.007635,-0.0071432,-0.0066907,-0.0062499,-0.005724,-0.0053334,
      -0.0047385,-0.0043906,-0.0037327,-0.0034817,-0.0028366,-0.0025169,-0.0020771,-0.0015991,-0.0013424,-0.0008854,
      -0.0006967,-0.0003563,-0.0001032,3.06e-05,0.0003133,0.0003261,0.0005924,0.0005917,0.0007284,0.0007264,
      0.000861,0.0007953,0.0008626,0.0009192,0.0008369,0.0008989,0.0008074,0.0009145,0.0008355,0.0008474,
      0.0008781,0.0008202,0.0009444,0.0009763,0.0013344,0.0018758,0.0028545,0.0047725,0.0076935,0.012154,
      0.0181459,0.0253993,0.0335244,0.0407454,0.0464965,0.0489777,0.0495573,0.0489244,0.0478999,0.0470684,
      0.046866,0.0469141,0.0473368,0.0478855,0.0482364,0.0482747,0.0480324,0.0474823,0.0470482,0.0465184,
      0.0461924,0.0460526,0.0457831,0.0456386,0.0454173,0.0449704,0.0441753,0.0430213,0.0408208,0.0377515,
      0.0337812,0.0292985,0.0246189,0.0198053,0.0153267,0.0112866,0.007936,0.0052651,0.0034826,0.0020272,
      0.0011602,0.0006425,0.0001626,-4.13e-05,-0.0001852,-0.0003816,-0.0003749,-0.0005417,-0.0004948,-0.0005688,
      -0.0006025,-0.0006504,-0.0006301,-0.0007428,-0.0006437,-0.0007243,-0.0007559,-0.0007033,-0.0006763,-0.0007106,
      -0.000647,-0.0006125,-0.0006073,-0.0004534,-0.0004887,-0.0003393,-0.0003151,-0.0002266,-0.0001103,-7.8e-05,
      1.14e-05,0.0001154,6.85e-05,0.0002827,0.000216,0.0002702,0.0003271,0.0002497,0.0003447,0.0001982,
      0.0002779,0.0001016,0.0001088,-4.82e-05,-0.0001267,-0.0002773,-0.0004399,-0.0005928,-0.0007739,-0.0009793,
      -0.0011852,-0.0013646,-0.00165,-0.001778,-0.0021102,-0.0022299,-0.0025455,-0.0026155,-0.0028969,-0.0030742,
      -0.0032496,-0.0033272,-0.003469,-0.0036285,-0.003719,-0.0037027,-0.0038596,-0.0038766,-0.0039271,-0.0039196,
      -0.004088,-0.003934,-0.0041542,-0.004039,-0.0041683,-0.0041725,-0.0042252,-0.0043127,-0.0043337,-0.0044687,
      -0.0044324,-0.0046504,-0.0046609,-0.0048247,-0.0049051,-0.0050548,-0.005137,-0.0052974,-0.0054515,-0.0055351,
      -0.0057111,-0.0058012,-0.005936,-0.0061237,-0.0061034,-0.0064092,-0.006311,-0.0066207,-0.0065926,-0.006757,
      -0.0069163,-0.0069502,-0.0071644,-0.007213,-0.0074291,-0.0074996,-0.007638,-0.0077944,-0.0078235,-0.0080249,
      -0.0080678,-0.0081204,-0.008114,-0.0081325,-0.0081146,-0.0080704,-0.0079172,-0.0078006,-0.0077083,-0.007449,
      -0.0073177,-0.006988,-0.0068643,-0.0064484,-0.0062476,-0.0059002,-0.0056276,-0.0051837,-0.0050332,-0.0045172,
      -0.0043693,-0.0039254,-0.0037296,-0.0034062,-0.0032022,-0.0029653,-0.0027583,-0.002582,-0.0024516,-0.0023743,
      -0.0022158,-0.002168,-0.0022077,-0.0020952,-0.0021981,-0.0022078,-0.0021955,-0.0023974,-0.0023313,-0.0025636,
      -0.0025885,-0.0027005,-0.0028286,-0.0029494,-0.0030915,-0.0031433,-0.0032438,-0.0033917,-0.0034466,-0.0035309,
      -0.0035899,-0.0035675,-0.0037611,-0.0036056,-0.0037422,-0.0036633,-0.0036934,-0.0036587,-0.0036382,-0.0036179,
      -0.0035695,-0.0035117,-0.0034941,-0.0034883,-0.0033771,-0.0034365,-0.0032909,-0.0033677,-0.0032668,-0.0032771,
      -0.0032339,-0.0032246,-0.0032055,-0.0031434,-0.0031427,-0.0030596,-0.0031276,-0.0029358,-0.0030488,-0.0028483,
      -0.0029228,-0.0028571,-0.0027174,-0.0027942,-0.0026247,-0.0026452,-0.0025867,-0.0025476,-0.0023944,-0.0025076,
      -0.0022454,-0.0023401,-0.0021585,-0.0021315,-0.0020316,-0.0019408,-0.0018292,-0.0017495,-0.0016602,-0.0015701,
      -0.0015005,-0.0013936,-0.0013208,-0.0012452,-0.0012415,-0.0011421,-0.0011164,-0.0011768,-0.0010564,-0.0010984,
      -0.0011933,-0.0011527,-0.0012216,-0.0012695,-0.0013939,-0.0014118,-0.0016018,-0.0016304,-0.0018015,-0.0018956,
      -0.00207,-0.0021718,-0.0023126,-0.002456,-0.0026053,-0.0026725,-0.0029244,-0.002909,-0.0030553,-0.0032152,
      -0.0032343,-0.0033193,-0.0034333,-0.0034883,-0.0034886,-0.0036283,-0.0035386,-0.0036909,-0.003572,-0.0036995,
      -0.0036423,-0.0036805,-0.003596,-0.003694,-0.0036058,-0.0036576,-0.0036116,-0.0036388,-0.0036264,-0.0036268,
      -0.00364,-0.0036669,-0.0036257,-0.0037202,-0.0036728,-0.0037285,-0.0037241,-0.0038098,-0.0037628,-0.0038855,
      -0.0038045,-0.0039135,-0.0039183,-0.0039421,-0.0039347,-0.0040335,-0.0040217,-0.0039913,-0.0040839,-0.0040355,
      -0.0041532,-0.0040082,-0.0041497,-0.0040386,-0.0041125,-0.0040367,-0.0040411,-0.0039892,-0.0039823,-0.003952,
      -0.003868,-0.0039203,-0.0037932,-0.0038382,-0.0037857,-0.0037475,-0.0037527,-0.0037697,-0.0037243,-0.0037763,
      -0.0037433,-0.0038225,-0.0038115,-0.0038502,-0.0038695,-0.0039654,-0.0039164,-0.0040468,-0.0040239,-0.0041047,
      -0.0040798,-0.0041363,-0.0041431,-0.0041351,-0.0042025,-0.0041464,-0.004101,-0.0041345,-0.0040433,-0.0040594,
      -0.0039571,-0.0039144,-0.0038528,-0.0037865,-0.0036317,-0.0036482,-0.0034585,-0.0033192,-0.0033333,-0.0030722,
      -0.0029626,-0.0029248,-0.0026487,-0.002581,-0.0023964,-0.0022622,-0.0020394,-0.0019328,-0.0017751,-0.0015819,
      -0.0013558,-0.0013264,-0.0010595,-0.0009543,-0.0007563,-0.0006673,-0.0005328,-0.0003516,-0.0002854,-0.0001206,
      -9.81e-05,4.15e-05,0.0001302,0.0001874,0.0002635,0.0002958,0.0003072,0.0004264,0.0003316,0.0004739,
      0.0003929,0.0004373,0.0003402,0.0004277,0.0002937,0.0003479,0.0002039,0.000237,0.0001271,6.11e-05,
      -3.72e-05,-5.01e-05,-0.0002362,-0.0002085,-0.0004089,-0.0003814,-0.0006098,-0.0005474,-0.0006893,-0.0007343,
      -0.0008069,-0.0008057,-0.0008167,-0.0008329,-0.0007565,-0.0007109,-0.0006347,-0.0004535,-0.0003084,-0.0001231,
      0.000134,0.0004557,0.0007469,0.0011306,0.0015904,0.001988,0.0024997,0.0030699,0.0036288,0.0042512,
      0.0048883,0.0055961,0.0062199,0.0070407,0.0077645,0.0084915,0.0093871,0.0100994,0.0109256,0.0117664,
      0.0124745,0.0133545,0.0139429,0.0146863,0.0153345,0.0158535,0.0162724,0.0167912,0.0169995,0.0172097,
      0.0173828,0.0173746,0.0173386,0.0172023,0.0169523,0.0167123,0.0163358,0.0160043,0.0154684,0.0151088,
      0.0145778,0.0140271,0.0136189,0.0129573,0.0125413,0.0119523,0.0114488,0.010997,0.0103602,0.0099786,
      0.0094418,0.0090084,0.0084957,0.0080692,0.0077109,0.0071484,0.0069028,0.0065038,0.0061027,0.0058798,
      0.0055314,0.0053777,0.0049913,0.004933,0.0046709,0.0046001,0.0044161,0.0043475,0.0042301,0.0043216,
      0.0041599,0.0042501,0.0042462,0.0043713,0.0043717,0.0045676,0.004636,0.0048172,0.0050342,0.005222,
      0.0054487,0.0056648,0.0060706,0.0062464,0.0066014,0.0069766,0.0072534,0.0077628,0.0081571,0.0087152,
      0.0092054,0.0099747,0.0105617,0.0114324,0.0123013,0.0133329,0.0142349,0.0153624,0.0163783,0.0173168,
      0.0182532,0.0189768,0.0194827,0.019857,0.0200546,0.0200172,0.0198653,0.0196068,0.0192201,0.0188084,
      0.0184691,0.0179513,0.0176642,0.0172347,0.0169498,0.016649,0.0164642,0.0161996,0.0160087,0.015921,
      0.0156712,0.0156059,0.0154329,0.0153344,0.0151497,0.01502,0.014845,0.0147335,0.0144878,0.0143469,
      0.0141189,0.0139379,0.0136558,0.013483,0.0132459,0.0129586,0.0127979,0.0124556,0.0123638,0.0119213,
      0.0118162,0.0114729,0.0112287,0.0109648,0.0106901,0.0103534,0.0101129,0.0098302,0.0094649,0.0092034,
      0.0088761,0.0085962,0.0082146,0.0079547,0.0076166,0.007307,0.0069724,0.0067441,0.0063307,0.0061411,
      0.0057851,0.0055784,0.0052429,0.0050805,0.0047847,0.0045293,0.0043621,0.0041175,0.0038855,0.0036808,
      0.0034637,0.0032728,0.0030839,0.0028678,0.0026614,0.0025198,0.0023582,0.0020882,0.0020466,0.0018635,
      0.00166,0.0016186,0.0014495,0.0013225,0.0012509,0.0011253,0.0010874,0.0008988,0.0009867,0.0008004,
      0.0008256,0.0006996,0.000758,0.0006497,0.0006968,0.000616,0.0006821,0.0006158,0.0006981,0.000645,
      0.0006831,0.0007951,0.0007288,0.0007906,0.0008208,0.0009405,0.0008786,0.0010077,0.000973,0.0011026,
      0.0010385,0.0011658,0.00117,0.0011469,0.0012947,0.0011754,0.0013284,0.0012214,0.0013208,0.0013274,
      0.0012352,0.0013544,0.001213,0.0013292,0.0012136,0.001265,0.0011368,0.0012194,0.0011018,0.0010727,
      0.001059,0.0009737,0.0009968,0.0008643,0.000872,0.0008075,0.000766,0.0007058,0.0006957,0.0006736,
      0.0006038,0.0005718,0.0006061,0.0005828,0.0005156,0.0006009,0.0005784,0.0005384,0.00064,0.0006068,
      0.0006385,0.0007063,0.0007018,0.000733,0.0007731,0.000802,0.0008865,0.0007729,0.0009373,0.0008845,
      0.000901,0.0009673,0.0008912,0.0009231,0.001,0.000866,0.0009916,0.000957,0.0008928,0.0010037,
      0.0009339,0.0009738,0.00096,0.0009751,0.0009824,0.0010139,0.0010102,0.0009936,0.001064,0.0010032,
      0.001102,0.0009716,0.0010596,0.001027,0.001026,0.001034,0.0009577,0.0010385,0.000935,0.0009933,
      0.0009348,0.0009684,0.0009018,0.0009229,0.0009138,0.0009283,0.0008722,0.0009501,0.000833,0.0009522,
      0.000859,0.0009146,0.0008499,0.0009199,0.0008382,0.0008937,0.0008209,0.0009224,0.00079,0.0008156,
      0.0009137,0.0007625,0.0008677,0.0007988,0.0009039,0.0007526,0.0009176,0.0008088,0.000885,0.0008829,
      0.0008859,0.0008761,0.0009123,0.0009134,0.0009361,0.0009171,0.0008971,0.0009331,0.0009457,0.0009214,
      0.0009443,0.0008771,0.0009861,0.0009521,0.0009183,0.0009505,0.0009931,0.000988,0.0009654,0.0010606,
      0.0010572,0.0010145,0.0011507,0.0011003,0.0011652,0.001175,0.0012675,0.0012223,0.0013104,0.0012814,
      0.0013097,0.0014205,0.0012825,0.0014196,0.001313,0.0013534,0.0012988,0.0012836,0.0011708,0.001198,
      0.001114,0.0010371,0.0009549,0.0008863,0.0008195,0.0007202,0.0006699,0.0005361,0.0005565,0.0003673,
      0.0003966,0.0002767,0.0002526,0.0001361,0.000152,6.31e-05,7e-05,-5.98e-05,2.77e-05,-5.11e-05,
      -0.0001065,-4.67e-05,-0.0001315,-9.29e-05,-0.0001128,-0.0001216,-6.65e-05,-0.0001786,-7.58e-05,-8.45e-05,
      -0.0001106,-8.91e-05,-5.72e-05,-0.0001066,9.4e-06,-0.0001226,6.6e-06,-4.56e-05,-1.56e-05,-3.34e-05,
      7.6e-06,2.7e-06,-1.85e-05,2.13e-05,-1.68e-05,4.58e-05,-2.52e-05,5.49e-05,-3.2e-06,4.95e-05,
      -3.4e-06,1.44e-05,3.75e-05,2.71e-05,-4.2e-06,-1.02e-05,9.5e-05,-8.7e-05,6.21e-05,2.95e-05,
      -3.86e-05,3.36e-05,-4.4e-06,-1.61e-05,6.87e-05,-6.74e-05,9.8e-06,2.79e-05,-7.16e-05,6.72e-05,
      -5.61e-05,1.59e-05,-7.6e-06,1.44e-05,-4.99e-05,4.38e-05,-3.39e-05,5e-07,-1.82e-05,-1.92e-05,
      4.49e-05,-1.9e-05,-5.99e-05,6.74e-05,-6.87e-05,6.62e-05,-5.51e-05,1.32e-05,2e-06,-2.12e-05,
      3.3e-05,-6.08e-05,6.91e-05,-7.45e-05,3.82e-05,-2.85e-05,2.59e-05,-2.54e-05,2.6e-05,-2.89e-05,
      3.96e-05,-7.98e-05,8.86e-05,-8.37e-05,5.53e-05,3.4e-06,-1.89e-05,2.22e-05,-2.02e-05,8.8e-06,
      3.5e-05,-7.3e-06,-5.6e-06,2.95e-05,2.86e-05,-2.4e-06,3.12e-05,1.89e-05,3.48e-05,3.34e-05,
      -2.68e-05,7.41e-05,6.27e-05,-4.18e-05,0.0001052,-4.67e-05,0.0001321,-5.79e-05,0.0001002,-1.08e-05,
      8.49e-05,3.8e-06,4.19e-05,2e-05,1.99e-05,4.21e-05,3.1e-06,8.72e-05,-6.91e-05,9.83e-05,
      -4.18e-05,6.97e-05,-4.61e-05,6.56e-05,-2.55e-05,3.7e-05,-3.07e-05,-5.6e-06,5.3e-05,-6.51e-05,
      6.64e-05,-5.97e-05,3.16e-05,-6.69e-05,4.52e-05,-6.49e-05,2.36e-05,-3.01e-05,-4.44e-05,1.66e-05,
      -2.24e-05,-6.83e-05,1.3e-05,-3.4e-05,-1.83e-05,-3.44e-05,-3.53e-05,3.4e-05,-0.000101,3.79e-05,
      -5.14e-05,2.64e-05,-5.47e-05,5.13e-05,-5.93e-05,9.47e-05,-3.74e-05,5.56e-05,5.8e-06,6.3e-05,
      7.45e-05,6.33e-05,5.5e-05,0.0001413,4.46e-05,0.0001551,9.17e-05,0.0001438,0.0001398,0.0001043,
      0.0002008,9.01e-05,0.0001974,6.86e-05,0.0001944,5.27e-05,0.0001196,8.48e-05,1.61e-05,4.25e-05,
      -4.43e-05,-5.53e-05,-0.0001591,-0.0001141,-0.0002409,-0.0002948,-0.0003348,-0.0004043,-0.0004684,-0.0005673,
      -0.0005815,-0.0006181,-0.00079,-0.0006808,-0.0008072,-0.0008323,-0.0007979,-0.000822,-0.0007977,-0.0006929,
      -0.0007001,-0.000546,-0.0004376,-0.00027,-0.0001013,0.0001445,0.0003786,0.0007063,0.000969,0.0013466,
      0.0017505,0.0021798,0.0026216,0.0030788,0.0036536,0.0041674,0.0046223,0.0053759,0.0057956,0.0064676,
      0.007027,0.0075614,0.0082038,0.0087028,0.0091829,0.0097001,0.0100772,0.0105167,0.0107205,0.0111213,
      0.0111123,0.0114059,0.0113183,0.0113243,0.0112121,0.0110599,0.0107658,0.0105319,0.0101656,0.009757,
      0.0093106,0.0089477,0.0083659,0.0078995,0.0074234,0.0068816,0.0064481,0.0059135,0.0054618,0.0050423,
      0.0045541,0.0042031,0.0038488,0.0034763,0.0031814,0.0029765,0.0026155,0.0025578,0.0022907,0.0021892,
      0.002098,0.0019965,0.0020189,0.0018878,0.0019684,0.0019685,0.0019761,0.0021374,0.0020647,0.0022818,
      0.0022616,0.0024894,0.0024337,0.0026526,0.0027522,0.0028063,0.0029632,0.0030695,0.0031592,0.0032282,
      0.0033963,0.0033875,0.0035206,0.0035318,0.0036219,0.0035848,0.0036458,0.0035786,0.0036472,0.003517,
      0.0035052,0.0034927,0.0033505,0.0033365,0.0032208,0.0032111,0.0031319,0.0029693,0.0030599,0.0029384,
      0.0028412,0.0029262,0.0028078,0.0028639,0.0027979,0.0027997,0.0028567,0.0027962,0.0028602,0.0028727,
      0.0028467,0.0028175,0.0029589,0.0028093,0.0028498,0.0028974,0.002786,0.002739,0.002826,0.0025936,
      0.002713,0.0025368,0.0024925,0.002421,0.002292,0.0022475,0.0021302,0.0019946,0.0019205,0.0018013,
      0.0016182,0.0015993,0.0013441,0.0013155,0.0011476,0.0010017,0.0009049,0.0007924,0.000602,0.0006285,
      0.000388,0.0003689,0.0002663,0.0001556,7.25e-05,9.04e-05,-0.0001184,-3.71e-05,-0.0001573,-0.0001799,
      -0.0001824,-0.0002599,-0.0002607,-0.0002909,-0.0002715,-0.000327,-0.0002186,-0.0003939,-0.0002219,-0.0003152,
      -0.0002129,-0.0003185,-0.0002166,-0.0001908,-0.0003015,-8.61e-05,-0.0002057,-0.0001501,-5.21e-05,-0.0001688,
      -1.79e-05,-7.53e-05,-3.5e-06,-1.35e-05,4.77e-05,4.27e-05,9.43e-05,0.0001132,8.81e-05,0.0001702,
      7.79e-05,0.0002652,0.0001302,0.0001741,0.0002329,0.0001651,0.0001764,0.0001809,0.0001685,0.0001041,
      0.0001542,2.52e-05,6.13e-05,-4.98e-05,-7.98e-05,-0.0001552,-0.000255,-0.0003057,-0.0004361,-0.0004903,
      -0.000678,-0.000719,-0.0009132,-0.0010327,-0.0011079,-0.0013344,-0.0014107,-0.0015168,-0.0017626,-0.0017437,
      -0.0019876,-0.0020007,-0.0021471,-0.0022633,-0.0023194,-0.0024033,-0.0024533,-0.0025924,-0.0025231,-0.0026865,
      -0.0026145,-0.0027901,-0.0026718,-0.0027892,-0.002855,-0.0028029,-0.0030077,-0.0029567,-0.0031997,-0.0032668,
      -0.003628,-0.0041004,-0.0049649,-0.0063841,-0.0088939,-0.0125957,-0.0179315,-0.0247194,-0.0331644,-0.0416219,
      -0.0497755,-0.0573069,-0.0634002,-0.0665782,-0.0656557,-0.0600679,-0.0509555,-0.0398926,-0.0288073,-0.0193086,
      -0.0120524,-0.0067961,-0.0032097,-0.0009875,0.0003454,0.0013915,0.0021525,0.0034749,0.0052818,0.0080003,
      0.0115156,0.0161057,0.0214683,0.027332,0.0333593,0.0390568,0.0437489,0.047315,0.0494383,0.0503154,
      0.050388,0.0499813,0.0489537,0.04729,0.0445394,0.04034,0.0344249,0.0274256,0.0207954,0.0145668,
      0.0094844,0.0053169,0.0018907,-0.0005068,-0.0024426,-0.0038678,-0.004816,-0.0056657,-0.0059984,-0.0066108,
      -0.0067188,-0.0071109,-0.0072075,-0.0074381,-0.0075757,-0.0076674,-0.0079808,-0.0079816,-0.0082455,-0.0083984,
      -0.0086299,-0.0087041,-0.0089844,-0.0091543,-0.0092394,-0.0095209,-0.0095189,-0.0098142,-0.0098211,-0.0099762,
      -0.0099902,-0.0101639,-0.0099962,-0.0101658,-0.0100032,-0.0100282,-0.0098593,-0.0097121,-0.0096201,-0.0094213,
      -0.0091819,-0.008984,-0.0087988,-0.0085583,-0.0082185,-0.0081216,-0.007715,-0.0075833,-0.0072973,-0.0070449,
      -0.0068333,-0.0065796,-0.0064666,-0.00613,-0.0061079,-0.0058155,-0.00581,-0.0055673,-0.0055418,-0.005422,
      -0.0054106,-0.0052986,-0.0053521,-0.0052917,-0.0053763,-0.005345,-0.0053864,-0.005508,-0.0055096,-0.0055905,
      -0.0056885,-0.0057212,-0.0057799,-0.0058768,-0.0059148,-0.0059359,-0.0060427,-0.006055,-0.0060349,-0.0061027,
      -0.0060501,-0.0061031,-0.0060115,-0.006016,-0.0059237,-0.005929,-0.0057629,-0.0057376,-0.0055844,-0.0055461,
      -0.0053105,-0.0053521,-0.0050888,-0.005104,-0.0049094,-0.0048115,-0.0047453,-0.0046418,-0.0045525,-0.0044241,
      -0.0044385,-0.004253,-0.0042249,-0.0042326,-0.0040187,-0.0040803,-0.0040099,-0.003897,-0.0039311,-0.0037859,
      -0.0038815,-0.0037033,-0.0037825,-0.0037088,-0.0036198,-0.0036743,-0.0035455,-0.0035636,-0.0035036,-0.0034802,
      -0.0034381,-0.0033877,-0.0033083,-0.0033531,-0.0032426,-0.003216,-0.0032123,-0.0030349,-0.0031132,-0.0029728,
      -0.0029251,-0.0028316,-0.002829,-0.0026151,-0.0026258,-0.0024797,-0.0024164,-0.0021849,-0.0022181,-0.0020014,
      -0.0018866,-0.0018056,-0.0016288,-0.0015596,-0.0013641,-0.0013941,-0.0011317,-0.0010929,-0.0009885,-0.0009086,
      -0.0007955,-0.0006929,-0.000696,-0.0005581,-0.0004886,-0.0004653,-0.0003991,-0.0003957,-0.0002642,-0.000354,
      -0.0002223,-0.0002546,-0.0002572,-0.0001964,-0.0002441,-0.0002106,-0.000182,-0.0002334,-0.000171,-0.0002547,
      -0.0001933,-0.0002402,-0.0001324,-0.0002871,-0.0001207,-0.0002871,-0.0001135,-0.0002197,-0.0001793,-0.0001194,
      -0.0002078,-0.0001053,-0.0001201,-0.0001638,-6.97e-05,-9.56e-05,-8.6e-05,-7.85e-05,-4.22e-05,-5.97e-05,
      -4.58e-05,-6.41e-05,7.14e-05,-0.0001062,2.68e-05,-1e-06,-2.7e-06,-7.9e-06,3.44e-05,-1.43e-05,
      2.3e-05,1.8e-05,2.07e-05,1.49e-05,1.53e-05,1.95e-05,2.22e-05,7.3e-06,4.42e-05,-6.86e-05,
      0.0001149,-6.45e-05,2.78e-05,6.91e-05,-9.3e-05,9.2e-05,-6.39e-05,4.81e-05,-1.29e-05,3.4e-06,
      -9e-07,0.0,7e-07,-2.9e-06,1.1e-05,-4.1e-05,3.76e-05,6e-06,-6.18e-05,3e-05,
      3.74e-05,-6.43e-05,8.6e-06,3e-05,-3.29e-05,6.1e-06,8.7e-06,-4.08e-05,3.88e-05,9e-07,
      -4.25e-05,5.35e-05,-5.6e-05,5.5e-05,-4.83e-05,2.27e-05,-2.27e-05,4.84e-05,-5.53e-05,5.71e-05,
      -5.78e-05,5.83e-05,
     ),
     (
      -0.00929,-0.00909,-0.00986,-0.00997,-0.01027,-0.01048,-0.01084,-0.01217,-0.01249,-0.01288,
      -0.01401,-0.01508,-0.01569,-0.01668,-0.01809,-0.01898,-0.02002,-0.02096,-0.02217,-0.0239,
      -0.02475,-0.02561,-0.02734,-0.02857,-0.02943,-0.03076,-0.03206,-0.03301,-0.03392,-0.03532,
      -0.03632,-0.03692,-0.03801,-0.03905,-0.03983,-0.04017,-0.04101,-0.04182,-0.04224,-0.04276,
      -0.04326,-0.04374,-0.04433,-0.04449,-0.04476,-0.04501,-0.04525,-0.04551,-0.04574,-0.04608,
      -0.04598,-0.04603,-0.04591,-0.04635,-0.04624,-0.04624,-0.04635,-0.0459,-0.0461,-0.04576,
      -0.04542,-0.04559,-0.04523,-0.04501,-0.04476,-0.0445,-0.04428,-0.04392,-0.04409,-0.04376,
      -0.04342,-0.04359,-0.04325,-0.04294,-0.04302,-0.043,-0.04301,-0.04299,-0.04307,-0.04276,
      -0.04244,-0.0425,-0.04257,-0.04225,-0.04198,-0.04185,-0.04115,-0.0411,-0.04049,-0.03999,
      -0.03957,-0.03876,-0.03792,-0.03759,-0.03674,-0.03598,-0.03537,-0.0341,-0.03378,-0.03331,
      -0.03251,-0.03168,-0.03129,-0.03068,-0.03052,-0.03027,-0.02993,-0.03002,-0.03001,-0.02995,
      -0.03019,-0.03079,-0.03116,-0.03208,-0.03252,-0.03287,-0.03402,-0.03508,-0.03567,-0.03677,
      -0.03778,-0.03864,-0.04018,-0.04067,-0.04167,-0.04315,-0.04374,-0.04442,-0.0456,-0.04621,
      -0.04708,-0.04749,-0.04799,-0.04858,-0.04874,-0.049,-0.0493,-0.04933,-0.04891,-0.04908,
      -0.04883,-0.04816,-0.04807,-0.04758,-0.04665,-0.04635,-0.04549,-0.04474,-0.0441,-0.04292,
      -0.04226,-0.04157,-0.04049,-0.03951,-0.0385,-0.03753,-0.03642,-0.03584,-0.03474,-0.03375,
      -0.03281,-0.03157,-0.03143,-0.03025,-0.02911,-0.02883,-0.02809,-0.02685,-0.02653,-0.02603,
      -0.02539,-0.02545,-0.02435,-0.02419,-0.02442,-0.02366,-0.02347,-0.02349,-0.02358,-0.02319,
      -0.02318,-0.02358,-0.0235,-0.02344,-0.02375,-0.02407,-0.024,-0.02394,-0.02427,-0.0245,
      -0.02473,-0.02508,-0.02498,-0.02503,-0.02492,-0.02532,-0.02532,-0.02492,-0.02501,-0.02506,
      -0.02477,-0.02439,-0.02468,-0.0239,-0.02375,-0.0236,-0.02289,-0.02285,-0.02225,-0.02169,
      -0.02152,-0.02125,-0.021,-0.02078,-0.02041,-0.02061,-0.02016,-0.02027,-0.02028,-0.02015,
      -0.02064,-0.02032,-0.0211,-0.02127,-0.02134,-0.02236,-0.02274,-0.02319,-0.02401,-0.02477,
      -0.02541,-0.02659,-0.02725,-0.02793,-0.02904,-0.02991,-0.03133,-0.03225,-0.03316,-0.03459,
      -0.03549,-0.03647,-0.03766,-0.0394,-0.04024,-0.04116,-0.04262,-0.04339,-0.04484,-0.04574,
      -0.04673,-0.04785,-0.0484,-0.04958,-0.0503,-0.05074,-0.05125,-0.05176,-0.05222,-0.05289,
      -0.05276,-0.05263,-0.05326,-0.05239,-0.05225,-0.05215,-0.05122,-0.05051,-0.0498,-0.04885,
      -0.04739,-0.04667,-0.045,-0.04341,-0.0424,-0.04056,-0.03844,-0.03675,-0.0351,-0.0329,
      -0.03135,-0.02924,-0.02725,-0.02531,-0.02308,-0.02193,-0.01976,-0.0176,-0.01638,-0.01442,
      -0.013,-0.01161,-0.00962,-0.00846,-0.00758,-0.00623,-0.00502,-0.0037,-0.0027,-0.002,
      -0.00134,-0.00017,0.00049,0.00121,0.00212,0.00232,0.00359,0.00429,0.00475,0.0052,
      0.00592,0.00708,0.00773,0.00849,0.00929,0.00984,0.01132,0.01233,0.01284,0.01429,
      0.01549,0.01675,0.01799,0.01926,0.02043,0.022,0.02352,0.02491,0.02684,0.02824,
      0.02969,0.0315,0.03331,0.03475,0.0362,0.03797,0.03991,0.0409,0.0425,0.0441,
      0.04513,0.04689,0.0478,0.04843,0.04949,0.05062,0.05104,0.05125,0.0515,0.05178,
      0.05191,0.05114,0.05107,0.05063,0.04948,0.04852,0.0475,0.04654,0.0454,0.04344,
      0.04193,0.04089,0.03905,0.03695,0.03519,0.03382,0.03207,0.02994,0.02821,0.02675,
      0.02534,0.02342,0.02202,0.02052,0.01893,0.0178,0.0164,0.01561,0.01418,0.01318,
      0.01261,0.01141,0.01077,0.01004,0.0091,0.00909,0.00856,0.00769,0.0072,0.00701,
      0.00675,0.00649,0.00628,0.0059,0.00611,0.00567,0.00574,0.00586,0.00532,0.00586,
      0.00574,0.00568,0.00603,0.00622,0.00659,0.00643,0.00668,0.00734,0.00747,0.00777,
      0.00794,0.00849,0.00909,0.00915,0.0098,0.01016,0.01106,0.01159,0.0116,0.01252,
      0.01331,0.01374,0.01425,0.01477,0.01518,0.016,0.01682,0.01721,0.01784,0.01794,
      0.01841,0.0194,0.01949,0.01965,0.02043,0.02015,0.02048,0.02091,0.0204,0.02051,
      0.02056,0.02024,0.01999,0.01983,0.01923,0.01876,0.01825,0.01776,0.01723,0.01684,
      0.01593,0.01548,0.01517,0.01385,0.01343,0.01342,0.01241,0.01195,0.01177,0.01147,
      0.01136,0.01059,0.01078,0.01077,0.01066,0.0111,0.01095,0.01111,0.0121,0.0125,
      0.01293,0.01377,0.01451,0.01521,0.01617,0.0176,0.01841,0.01974,0.0211,0.02185,
      0.02351,0.0251,0.02608,0.02808,0.02959,0.03058,0.0326,0.034,0.03541,0.03734,
      0.03873,0.04026,0.04175,0.04325,0.04475,0.04625,0.04773,0.04932,0.05051,0.05167,
      0.05332,0.05455,0.05549,0.05651,0.05751,0.05849,0.05955,0.06033,0.06068,0.06149,
      0.06239,0.06249,0.06268,0.06332,0.06356,0.0635,0.06351,0.06352,0.06347,0.06364,
      0.063,0.06242,0.06235,0.06172,0.06133,0.06052,0.05966,0.05941,0.05825,0.05717,
      0.0566,0.05547,0.05459,0.05324,0.05201,0.05079,0.0494,0.04867,0.04699,0.04545,
      0.04426,0.04306,0.04154,0.03986,0.03909,0.03784,0.03611,0.03528,0.03429,0.03309,
      0.03288,0.03193,0.03145,0.03128,0.03095,0.03095,0.03127,0.03147,0.03186,0.0331,
      0.03376,0.03439,0.03569,0.03735,0.03841,0.04002,0.04152,0.0429,0.04487,0.04614,
      0.0481,0.04949,0.05098,0.05258,0.05371,0.05508,0.05599,0.05698,0.05812,0.05857,
      0.05866,0.05931,0.05962,0.05925,0.05891,0.05915,0.05854,0.05775,0.05702,0.05624,
      0.05556,0.05458,0.0532,0.05219,0.0516,0.05048,0.04953,0.04844,0.04775,0.04709,
      0.04594,0.04521,0.04477,0.04425,0.04377,0.04321,0.04294,0.04306,0.04284,0.04213,
      0.04218,0.04268,0.04215,0.04228,0.04225,0.04225,0.04226,0.04226,0.04224,0.04232,
      0.04202,0.04166,0.04185,0.04148,0.04127,0.04099,0.04081,0.04032,0.03945,0.03894,
      0.03882,0.0383,0.03752,0.03667,0.03634,0.03551,0.03465,0.03443,0.03315,0.0325,
      0.03187,0.03056,0.03043,0.02926,0.02809,0.02793,0.02671,0.02576,0.02478,0.02366,
      0.0231,0.02198,0.02102,0.01998,0.01909,0.01769,0.01668,0.01611,0.0149,0.01432,
      0.01334,0.01185,0.01127,0.01058,0.00942,0.00875,0.00811,0.00686,0.00649,0.00618,
      0.00482,0.00455,0.00399,0.0035,0.00301,0.00248,0.00205,0.00132,0.00168,0.00098,
      0.00043,0.0003,-0.00012,0.00018,-0.00061,-0.00073,-0.00097,-0.00137,-0.00107,-0.00184,
      -0.00206,-0.00192,-0.00227,-0.0025,-0.00274,-0.00302,-0.00316,-0.00382,-0.00406,-0.00393,
      -0.0042,-0.00475,-0.00529,-0.00557,-0.00543,-0.0057,-0.00625,-0.00679,-0.00707,-0.00694,
      -0.00719,-0.00781,-0.00807,-0.00792,-0.00827,-0.0085,-0.00873,-0.00908,-0.00898,-0.00903,
      -0.00891,-0.00934,-0.00924,-0.00922,-0.00937,-0.00883,-0.00934,-0.00931,-0.00892,-0.009,
      -0.00908,-0.00867,-0.00875,-0.00884,-0.00839,-0.00859,-0.00825,-0.00793,-0.00802,-0.00798,
      -0.00805,-0.00782,-0.00719,-0.00694,-0.00706,-0.00682,-0.00615,-0.00607,-0.00557,-0.00467,
      -0.00426,-0.0038,-0.00306,-0.00199,-0.00102,4e-05,0.00086,0.00252,0.00402,0.0054,
      0.00736,0.00863,0.0106,0.01342,0.01518,0.01736,0.02034,0.02273,0.0252,0.02794,
      0.031,0.03402,0.03691,0.04033,0.04325,0.04616,0.04959,0.05247,0.05551,0.0585,
      0.06149,0.06455,0.06731,0.06972,0.07233,0.07447,0.07681,0.07881,0.08047,0.08232,
      0.08379,0.08507,0.08598,0.08707,0.08779,0.08832,0.0885,0.08875,0.08906,0.08907,
      0.08873,0.08858,0.08802,0.08743,0.08733,0.0868,0.08603,0.08517,0.08487,0.08393,
      0.08349,0.08318,0.08186,0.08145,0.08141,0.08049,0.0797,0.07928,0.07875,0.07827,
      0.07774,0.07734,0.07644,0.07595,0.07582,0.07533,0.07443,0.07403,0.07351,0.07298,
      0.07261,0.07165,0.07135,0.07051,0.06968,0.06935,0.06849,0.06775,0.06708,0.066,
      0.065,0.06407,0.06277,0.0614,0.06067,0.05898,0.05749,0.05611,0.05413,0.05293,
      0.05073,0.04874,0.04686,0.0444,0.04261,0.04024,0.03802,0.03574,0.0336,0.03093,
      0.02875,0.02663,0.02382,0.02214,0.01967,0.01775,0.01586,0.01335,0.01179,0.01002,
      0.00819,0.00677,0.00527,0.00369,0.00252,0.00127,-7e-05,-0.00099,-0.00196,-0.00316,
      -0.0034,-0.00425,-0.00507,-0.00548,-0.00599,-0.00656,-0.00677,-0.00688,-0.00769,-0.00737,
      -0.00784,-0.00775,-0.00766,-0.0081,-0.00796,-0.00807,-0.00775,-0.00744,-0.0075,-0.00756,
      -0.00726,-0.00692,-0.00707,-0.00682,-0.00617,-0.00602,-0.00575,-0.0055,-0.00525,-0.005,
      -0.00476,-0.00448,-0.00434,-0.00368,-0.00344,-0.00356,-0.00331,-0.00269,-0.00243,-0.00258,
      -0.00223,-0.002,-0.00175,-0.00149,-0.00128,-0.00091,-0.00108,-0.00075,-0.00042,-0.00058,
      -0.00025,6e-05,-0.0,-6e-05,0.00024,0.00059,0.00039,0.00084,0.00075,0.00066,
      0.00109,0.00098,0.00101,0.001,0.001,0.00101,0.00098,0.00109,0.00067,0.00075,
      0.00082,0.00047,0.00028,-9e-05,6e-05,-0.00016,-0.00092,-0.00067,-0.00092,-0.00161,
      -0.00163,-0.00234,-0.00249,-0.00269,-0.00324,-0.00383,-0.00394,-0.0044,-0.00542,-0.0054,
      -0.00599,-0.00661,-0.00657,-0.00761,-0.00798,-0.00848,-0.00909,-0.00916,-0.00977,-0.01025,
      -0.01073,-0.01134,-0.0114,-0.01208,-0.01226,-0.01238,-0.01318,-0.01289,-0.01326,-0.01357,
      -0.01348,-0.01351,-0.0135,-0.01351,-0.01348,-0.01357,-0.01324,-0.013,-0.01275,-0.0125,
      -0.01224,-0.01206,-0.01152,-0.01091,-0.01084,-0.01023,-0.00975,-0.00928,-0.00867,-0.00858,
      -0.00801,-0.00741,-0.00736,-0.00666,-0.00652,-0.00627,-0.00591,-0.00611,-0.00566,-0.00576,
      -0.00577,-0.00567,-0.00605,-0.00615,-0.00685,-0.00693,-0.00742,-0.00835,-0.00867,-0.00945,
      -0.01051,-0.01149,-0.01249,-0.01351,-0.01445,-0.01567,-0.01732,-0.0185,-0.01968,-0.02126,
      -0.02274,-0.02425,-0.02574,-0.02725,-0.02872,-0.03033,-0.03142,-0.03299,-0.03457,-0.03572,
      -0.03703,-0.03814,-0.03988,-0.04083,-0.04131,-0.04293,-0.04347,-0.04423,-0.04509,-0.04546,
      -0.04609,-0.04621,-0.04658,-0.04651,-0.04641,-0.04688,-0.04659,-0.04582,-0.04616,-0.04558,
      -0.04459,-0.0446,-0.04405,-0.04325,-0.04251,-0.04175,-0.04102,-0.04023,-0.03961,-0.0384,
      -0.03784,-0.03678,-0.03559,-0.03541,-0.03432,-0.03287,-0.03225,-0.03165,-0.03018,-0.02917,
      -0.02868,-0.02717,-0.02619,-0.02559,-0.02449,-0.02351,-0.0225,-0.02151,-0.0205,-0.01952,
      -0.01844,-0.01775,-0.01709,-0.01592,-0.01527,-0.01451,-0.01373,-0.01311,-0.01186,-0.01149,
      -0.01118,-0.00984,-0.00951,-0.00915,-0.00943,-0.01013,-0.01006,-0.01111,-0.01148,-0.01199,
      -0.01253,-0.0129,-0.01384,-0.01422,-0.01476,-0.01525,-0.01575,-0.01625,-0.01675,-0.01725,
      -0.01773,-0.01832,-0.01851,-0.01867,-0.01932,-0.01957,-0.01942,-0.01977,-0.01999,-0.0203,
      -0.02033,-0.01991,-0.02003,-0.01998,-0.02005,-0.01982,-0.0192,-0.01892,-0.01915,-0.01849,
      -0.01792,-0.01784,-0.01723,-0.01674,-0.01634,-0.01544,-0.01493,-0.01484,-0.01421,-0.01384,
      -0.01293,-0.01243,-0.01234,-0.01173,-0.01125,-0.01077,-0.01019,-0.01,-0.00984,-0.00916,
      -0.00902,-0.00875,-0.0085,-0.00828,-0.00791,-0.00809,-0.00775,-0.00744,-0.00752,-0.0075,
      -0.00748,-0.00759,-0.00718,-0.00718,-0.00759,-0.00748,-0.0075,-0.00752,-0.00741,-0.00784,
      -0.00775,-0.00767,-0.00807,-0.00806,-0.0077,-0.00766,-0.00815,-0.00773,-0.00743,-0.00757,
      -0.00729,-0.00676,-0.00617,-0.00606,-0.00557,-0.00465,-0.00433,-0.00354,-0.00251,-0.00142,
      -0.00081,0.00017,0.0016,0.0024,0.00377,0.00499,0.00625,0.0075,0.00874,0.01002,
      0.01118,0.01275,0.01431,0.0155,0.01668,0.01827,0.01975,0.02125,0.02275,0.02425,
      0.02575,0.02725,0.02874,0.03028,0.03166,0.03361,0.03491,0.03675,0.0386,0.03989,
      0.04187,0.04313,0.04512,0.04642,0.04823,0.05016,0.05115,0.05276,0.05432,0.05549,
      0.05674,0.05808,0.05897,0.06009,0.06072,0.06158,0.062,0.06247,0.06316,0.06295,
      0.06309,0.06276,0.06244,0.06254,0.06246,0.06267,0.06339,0.06331,0.06291,0.06308,
      0.06281,0.06225,0.06175,0.06132,0.06051,0.0597,0.05926,0.0588,0.05808,0.05695,
      0.05619,0.05583,0.05503,0.05411,0.05409,0.05357,0.05269,0.05221,0.05201,0.05178,
      0.05142,0.05159,0.05125,0.05094,0.05102,0.05103,0.05092,0.05134,0.05126,0.05117,
      0.0516,0.05148,0.05151,0.0515,0.05151,0.05148,0.0516,0.05116,0.05132,0.0511,
      0.05034,0.05058,0.05038,0.04944,0.04892,0.04891,0.04799,0.04718,0.04684,0.04598,
      0.04526,0.04451,0.04375,0.04301,0.04224,0.04157,0.04053,0.03935,0.0391,0.03827,
      0.03739,0.0372,0.03584,0.03546,0.03534,0.03473,0.03427,0.03371,0.03343,0.03361,
      0.03316,0.03327,0.03329,0.0331,0.03384,0.03408,0.03388,0.03445,0.03533,0.03573,
      0.03627,0.03671,0.03743,0.03859,0.03923,0.04,0.04078,0.04141,0.04259,0.04324,
      0.04399,0.04484,0.04518,0.04598,0.04692,0.04688,0.0476,0.04774,0.04797,0.0484,
      0.04797,0.04777,0.04749,0.04731,0.04678,0.04608,0.0449,0.04433,0.0433,0.04197,
      0.04082,0.03923,0.03776,0.03623,0.03482,0.03297,0.03125,0.02949,0.02775,0.02599,
      0.02425,0.02249,0.02074,0.019,0.01717,0.01575,0.01426,0.01265,0.01158,0.01,
      0.00836,0.00754,0.00645,0.00561,0.00407,0.00355,0.00275,0.00191,0.0016,0.00065,
      0.00029,-0.00033,-0.0005,-0.00067,-0.00132,-0.00155,-0.00148,-0.00153,-0.00141,-0.00185,
      -0.00172,-0.00174,-0.00183,-0.00141,-0.00153,-0.00148,-0.00157,-0.00124,-0.00094,-0.00099,
      -0.00109,-0.00066,-0.00076,-0.00083,-0.00039,-0.00062,-0.00014,-0.00028,-0.00023,-0.00026,
      -0.00024,-0.00025,-0.00027,-0.00016,-0.00061,-0.00041,-0.00076,-0.00108,-0.00091,-0.00127,
      -0.00151,-0.00169,-0.00226,-0.00281,-0.00302,-0.0031,-0.00409,-0.00458,-0.0046,-0.00553,
      -0.00631,-0.00673,-0.00727,-0.00771,-0.00841,-0.00969,-0.00989,-0.01077,-0.01158,-0.01192,
      -0.01278,-0.01348,-0.01435,-0.01464,-0.01561,-0.01598,-0.01649,-0.01709,-0.01715,-0.01785,
      -0.01798,-0.01825,-0.01857,-0.01849,-0.01848,-0.0186,-0.01814,-0.01833,-0.01806,-0.01743,
      -0.01725,-0.01707,-0.01648,-0.01598,-0.01561,-0.01457,-0.01459,-0.01406,-0.01314,-0.01286,
      -0.01192,-0.01142,-0.01139,-0.0105,-0.0096,-0.00961,-0.00899,-0.00842,-0.00834,-0.00772,
      -0.00727,-0.00667,-0.00652,-0.00625,-0.00598,-0.00583,-0.00517,-0.00496,-0.005,-0.00504,
      -0.00484,-0.00409,-0.00427,-0.00433,-0.00389,-0.00411,-0.00366,-0.00373,-0.00392,-0.00307,
      -0.00328,-0.00333,-0.00289,-0.00312,-0.00263,-0.00285,-0.00249,-0.00215,-0.00242,-0.00166,
      -0.00145,-0.00156,-0.00131,-0.00069,-0.00041,-0.00068,0.00012,0.00024,0.00042,0.00111,
      0.00113,0.00187,0.0019,0.00252,0.00306,0.00326,0.00344,0.004,0.00461,0.00458,
      0.00557,0.00615,0.00582,0.00655,0.00699,0.0075,0.00806,0.00825,0.00844,0.00901,
      0.00957,0.00973,0.01003,0.01017,0.01082,0.01107,0.01092,0.01125,0.01158,0.01142,
      0.01174,0.01215,0.01167,0.01168,0.01215,0.01173,0.01142,0.01158,0.01126,0.01089,
      0.01118,0.01038,0.01027,0.01007,0.00942,0.00925,0.00906,0.0085,0.00791,0.00785,
      0.00716,0.00698,0.00692,0.00582,0.00578,0.00557,0.0049,0.00483,0.00427,0.0036,
      0.00384,0.00355,0.00292,0.00277,0.0025,0.00225,0.002,0.00175,0.0015,0.00125,
      0.00098,0.00084,0.00016,1e-05,-0.00019,-0.00075,-0.00131,-0.00152,-0.00159,-0.00261,
      -0.00299,-0.00342,-0.00436,-0.00466,-0.00553,-0.00624,-0.00701,-0.00775,-0.0085,-0.00927,
      -0.00993,-0.01103,-0.01198,-0.01307,-0.01375,-0.01444,-0.0155,-0.01657,-0.01723,-0.01801,
      -0.01876,-0.01949,-0.02032,-0.02074,-0.02124,-0.02182,-0.02199,-0.02224,-0.02258,-0.02247,
      -0.02258,-0.02224,-0.02198,-0.02185,-0.02112,-0.02117,-0.02023,-0.0194,-0.01918,-0.01789,
      -0.01726,-0.01656,-0.0155,-0.01444,-0.01373,-0.01315,-0.01166,-0.01068,-0.0101,-0.00889,
      -0.00834,-0.00727,-0.00607,-0.00595,-0.00462,-0.00403,-0.00326,-0.00241,-0.00211,-0.00113,
      -0.00086,0.0001,0.0005,0.0009,0.00192,0.00191,0.00242,0.00342,0.00339,0.00401,
      0.0046,0.00458,0.00557,0.00614,0.00585,0.00644,0.0074,0.00748,0.00769,0.00827,
      0.00873,0.00932,0.0095,0.00967,0.01034,0.01048,0.01076,0.01101,0.01123,0.0116,
      0.01139,0.01185,0.01174,0.01168,0.01207,0.01207,0.01167,0.01176,0.01181,0.01148,
      0.01126,0.01098,0.01083,0.01019,0.00992,0.01015,0.00947,0.00895,0.00875,0.00856,
      0.008,0.00741,0.00737,0.00663,0.00662,0.0059,0.00576,0.00558,0.00491,0.00477,
      0.00449,0.00427,0.00391,0.00409,0.00375,0.00341,0.00361,0.00315,0.00328,0.00324,
      0.00325,0.00327,0.00316,0.00359,0.00347,0.00353,0.00342,0.00378,0.00398,0.00431,
      0.00428,0.00407,0.00495,0.00463,0.00501,0.00533,0.00516,0.00553,0.00573,0.00607,
      0.006,0.00594,0.00625,0.00658,0.00642,0.00674,0.00715,0.00669,0.0066,0.00743,
      0.0072,0.00726,0.00727,0.00716,0.00759,0.00747,0.00752,0.00744,0.00772,0.00818,
      0.00757,0.00803,0.00833,0.00814,0.00861,0.00843,0.00869,0.00931,0.00959,0.00935,
      0.01003,0.01056,0.01075,0.01096,0.01141,0.01242,0.01241,0.01294,0.01386,0.01414,
      0.01509,0.01552,0.01585,0.01709,0.01781,0.01817,0.01902,0.01978,0.0204,0.02166,
      0.02201,0.02234,0.02368,0.02398,0.02443,0.02532,0.0258,0.02602,0.02615,0.02691,
      0.02673,0.02668,0.02707,0.02705,0.02675,0.02647,0.02641,0.02542,0.02493,0.02488,
      0.02405,0.02292,0.02228,0.02147,0.02083,0.01973,0.01875,0.01778,0.01664,0.01618,
      0.01466,0.01368,0.01311,0.01188,0.01137,0.01015,0.00952,0.00877,0.00789,0.00769,
      0.00634,0.00596,0.00583,0.00522,0.00477,0.00418,0.00402,0.00374,0.00353,0.00316,
      0.00333,0.00302,0.00258,0.00317,0.00274,0.00235,0.00284,0.00279,0.00251,0.00218,
      0.00225,0.00232,0.00198,0.00176,0.00148,0.00132,0.00073,0.00024,-0.00019,-0.00099,
      -0.00185,-0.00211,-0.00321,-0.00458,-0.00549,-0.00648,-0.00761,-0.00961,-0.011,-0.01242,
      -0.01438,-0.01562,-0.01769,-0.02017,-0.02169,-0.02361,-0.02642,-0.02825,-0.03012,-0.03284,
      -0.03507,-0.03693,-0.03926,-0.0416,-0.04341,-0.04584,-0.0478,-0.0495,-0.05128,-0.05297,
      -0.05493,-0.05589,-0.05757,-0.05892,-0.0593,-0.06093,-0.06155,-0.06192,-0.06284,-0.06329,
      -0.06357,-0.06349,-0.06351,-0.06351,-0.06349,-0.06356,-0.06331,-0.06274,-0.06228,-0.06167,
      -0.06158,-0.06107,-0.06017,-0.05978,-0.05925,-0.05877,-0.05823,-0.05785,-0.05691,-0.05653,
      -0.05602,-0.05542,-0.05536,-0.05466,-0.05453,-0.05427,-0.05391,-0.05412,-0.05366,-0.05376,
      -0.05384,-0.05342,-0.05353,-0.05351,-0.05348,-0.0536,-0.05317,-0.05324,-0.05341,-0.05264,
      -0.05254,-0.05223,-0.05206,-0.05156,-0.05074,-0.05,-0.04927,-0.04847,-0.04789,-0.04649,
      -0.04516,-0.04441,-0.04271,-0.04125,-0.03982,-0.038,-0.03617,-0.03482,-0.03307,-0.03092,
      -0.02925,-0.02757,-0.02548,-0.02352,-0.02143,-0.01975,-0.01806,-0.01599,-0.01396,-0.01215,
      -0.01092,-0.00866,-0.00692,-0.00567,-0.0034,-0.0017,-0.00026,0.00127,0.00268,0.00451,
      0.0063,0.00782,0.00894,0.01044,0.01235,0.01371,0.01535,0.01642,0.01801,0.01959,
      0.02065,0.02233,0.02357,0.02443,0.02576,0.02708,0.02797,0.02908,0.02975,0.03047,
      0.03142,0.03138,0.03208,0.03233,0.03213,0.03267,0.03222,0.03198,0.03189,0.03097,
      0.03025,0.02957,0.02849,0.02747,0.02664,0.02499,0.0234,0.02242,0.02044,0.01883,
      0.01676,0.01465,0.01316,0.0107,0.00851,0.00623,0.00405,0.00155,-0.00127,-0.00397,
      -0.00686,-0.0091,-0.01228,-0.01533,-0.01794,-0.02094,-0.02434,-0.02723,-0.03026,-0.03326,
      -0.03624,-0.03933,-0.04199,-0.04475,-0.04756,-0.05004,-0.05232,-0.05424,-0.05629,-0.05816,
      -0.05911,-0.06096,-0.06161,-0.06164,-0.06239,-0.06237,-0.0617,-0.06137,-0.06036,-0.05874,
      -0.05723,-0.05588,-0.05379,-0.05148,-0.04933,-0.04672,-0.04432,-0.04151,-0.03868,-0.03632,
      -0.03357,-0.03042,-0.02777,-0.02501,-0.02218,-0.01977,-0.01724,-0.01477,-0.01217,-0.01004,
      -0.00768,-0.00574,-0.00385,-0.00135,0.00024,0.0019,0.00416,0.00546,0.00699,0.00859,
      0.00966,0.01126,0.01279,0.01408,0.01491,0.01628,0.01749,0.0188,0.01985,0.02033,
      0.02184,0.02282,0.0234,0.02459,0.02526,0.0259,0.02716,0.02749,0.02792,0.02885,
      0.02921,0.02983,0.02999,0.03023,0.0306,0.0304,0.03083,0.0308,0.03049,0.03026,
      0.02998,0.02983,0.02922,0.02883,0.02801,0.02717,0.02682,0.02608,0.0249,0.02435,
      0.02323,0.02226,0.02126,0.02022,0.01937,0.01782,0.01736,0.01625,0.01514,0.01469,
      0.0131,0.01244,0.01217,0.01091,0.0102,0.00978,0.00918,0.009,0.00883,0.00818,
      0.00795,0.00803,0.00793,0.00827,0.00851,0.00868,0.00926,0.00976,0.01018,0.01102,
      0.01176,0.01245,0.01345,0.01474,0.01609,0.01691,0.01829,0.01942,0.02102,0.0225,
      0.024,0.02552,0.02693,0.02875,0.03056,0.03201,0.03342,0.03534,0.03673,0.03825,
      0.0398,0.04108,0.0419,0.04333,0.04431,0.04497,0.04583,0.04623,0.04679,0.04713,
      0.04673,0.04649,0.04633,0.04573,0.04529,0.04465,0.04313,0.04235,0.04099,0.03972,
      0.03867,0.03667,0.0352,0.03406,0.03258,0.03067,0.02928,0.02775,0.02625,0.02477,
      0.02321,0.02191,0.02118,0.01941,0.01821,0.01727,0.01624,0.01528,0.01416,0.01359,
      0.0125,0.01144,0.01077,0.00998,0.00932,0.00824,0.00725,0.00627,0.00521,0.00442,
      0.00263,0.00158,8e-05,-0.00186,-0.00313,-0.00512,-0.00786,-0.00992,-0.01246,-0.01521,
      -0.01819,-0.02152,-0.02471,-0.02812,-0.03227,-0.03625,-0.04019,-0.04445,-0.049,-0.05354,
      -0.05784,-0.06308,-0.06782,-0.07215,-0.07709,-0.08149,-0.08597,-0.09065,-0.09445,-0.09857,
      -0.10228,-0.10584,-0.10889,-0.11115,-0.11256,-0.11368,-0.11379,-0.11377,-0.11371,-0.11251,
      -0.11137,-0.10961,-0.10728,-0.10488,-0.10181,-0.09852,-0.09523,-0.09071,-0.08659,-0.08206,
      -0.0773,-0.07235,-0.06692,-0.0621,-0.05682,-0.05128,-0.04571,-0.04053,-0.03529,-0.02994,
      -0.02505,-0.01995,-0.01523,-0.0107,-0.00656,-0.00214,0.00154,0.00494,0.00863,0.01151,
      0.0143,0.01676,0.01912,0.02073,0.02245,0.02444,0.0253,0.02589,0.02714,0.02754,
      0.02772,0.02808,0.02797,0.02806,0.02783,0.02716,0.02707,0.0266,0.02559,0.02559,
      0.02507,0.02415,0.02386,0.02293,0.02245,0.02228,0.02197,0.02187,0.02109,0.02129,
      0.02127,0.02115,0.02163,0.02136,0.02195,0.02283,0.02324,0.02371,0.02444,0.02551,
      0.0265,0.0275,0.02852,0.02943,0.03076,0.032,0.03323,0.03457,0.03548,0.03651,
      0.0375,0.03849,0.03955,0.04032,0.0407,0.0414,0.04123,0.04122,0.0414,0.04071,
      0.04032,0.03955,0.03853,0.03738,0.0355,0.03367,0.03238,0.03038,0.02767,0.02552,
      0.02333,0.02075,0.01825,0.01583,0.01301,0.01022,0.00769,0.00558,0.00304,0.00035,
      -0.00137,-0.00382,-0.0058,-0.00744,-0.00941,-0.01038,-0.01205,-0.01338,-0.01394,-0.01484,
      -0.01519,-0.0159,-0.01572,-0.01574,-0.01582,-0.01551,-0.01515,-0.01391,-0.01324,-0.01265,
      -0.0112,-0.01008,-0.0085,-0.00696,-0.00568,-0.00338,-0.00185,0.00025,0.00229,0.00406,
      0.00693,0.00867,0.01088,0.01377,0.01502,0.01615,0.01785,0.01893,0.02042,0.02233,
      0.02372,0.02528,0.02666,0.02856,0.03007,0.03117,0.03275,0.03431,0.03551,0.03667,
      0.03832,0.03955,0.04049,0.04149,0.04255,0.04332,0.04368,0.04448,0.04541,0.04539,
      0.04607,0.04633,0.04614,0.04662,0.0464,0.04683,0.04683,0.04643,0.04651,0.04659,
      0.04617,0.04627,0.04626,0.04623,0.04635,0.04592,0.04601,0.04609,0.04568,0.04575,
      0.04583,0.04549,0.04526,0.04501,0.04474,0.04457,0.044,0.04348,0.04311,0.04215,
      0.04183,0.04108,0.03991,0.03935,0.03825,0.03724,0.03635,0.03492,0.03402,0.03307,
      0.03177,0.03043,0.02958,0.02832,0.0267,0.02546,0.0245,0.02358,0.02225,0.02101,
      0.01978,0.01843,0.01753,0.0165,0.01551,0.01451,0.0135,0.01253,0.01141,0.01086,
      0.00968,0.00899,0.00835,0.00715,0.00656,0.00566,0.00534,0.00451,0.00366,0.00336,
      0.00242,0.002,0.00159,0.00068,0.00019,6e-05,-0.00042,-0.00134,-0.00172,-0.00228,
      -0.00266,-0.00356,-0.00407,-0.00415,-0.00479,-0.00515,-0.00608,-0.0065,-0.00691,-0.00783,
      -0.00824,-0.00869,-0.00949,-0.01032,-0.01072,-0.01128,-0.01166,-0.01257,-0.01303,-0.01333,
      -0.01464,-0.01507,-0.01508,-0.0161,-0.01648,-0.01697,-0.01759,-0.01766,-0.01826,-0.01881,
      -0.01901,-0.01917,-0.01981,-0.02007,-0.01992,-0.02025,-0.02057,-0.02048,-0.02051,-0.0205,
      -0.02053,-0.02042,-0.02081,-0.02083,-0.02041,-0.02053,-0.02047,-0.02059,-0.02017,-0.02027,
      -0.02026,-0.02024,-0.02027,-0.02022,-0.02035,-0.01991,-0.02003,-0.02,-0.01998,-0.02009,
      -0.01967,-0.01975,-0.01984,-0.01942,-0.0195,-0.01959,-0.01915,-0.01932,-0.0191,-0.01833,
      -0.01862,-0.01822,-0.01801,-0.01776,-0.01748,-0.01734,-0.01666,-0.01652,-0.01625,-0.01598,
      -0.01584,-0.01518,-0.01494,-0.01508,-0.01475,-0.01442,-0.01459,-0.01423,-0.014,-0.01377,
      -0.01341,-0.01359,-0.01324,-0.01294,-0.01301,-0.01305,-0.01282,-0.01219,-0.01194,-0.01206,
      -0.01182,-0.01117,-0.011,-0.01083,-0.01017,-0.01,-0.00983,-0.00917,-0.009,-0.00883,
      -0.00816,-0.00802,-0.00776,-0.00743,-0.00752,-0.0075,-0.00749,-0.00754,-0.00736,-0.00802,
      -0.00859,-0.00861,-0.00948,-0.01051,-0.01152,-0.01245,-0.01372,-0.01519,-0.01704,-0.0187,
      -0.0207,-0.02306,-0.02511,-0.02804,-0.03078,-0.03339,-0.03671,-0.04036,-0.04346,-0.04688,
      -0.05112,-0.05476,-0.05845,-0.06254,-0.06649,-0.07061,-0.07418,-0.07826,-0.08243,-0.08563,
      -0.08967,-0.09282,-0.09567,-0.09915,-0.10185,-0.10407,-0.10597,-0.10813,-0.10961,-0.11052,
      -0.1114,-0.11147,-0.11179,-0.11196,-0.11095,-0.11034,-0.10929,-0.10806,-0.10655,-0.1048,
      -0.10283,-0.10043,-0.09849,-0.09664,-0.09398,-0.09145,-0.08925,-0.08709,-0.08441,-0.08228,
      -0.08001,-0.07769,-0.07577,-0.07377,-0.07171,-0.06993,-0.06861,-0.06665,-0.0653,-0.06367,
      -0.06254,-0.06118,-0.06025,-0.05936,-0.05783,-0.05737,-0.05624,-0.05519,-0.05453,-0.05373,
      -0.05307,-0.052,-0.05094,-0.05025,-0.04959,-0.0484,-0.04785,-0.04675,-0.04567,-0.04509,
      -0.044,-0.04292,-0.04235,-0.04123,-0.04025,-0.03928,-0.03814,-0.03768,-0.03616,-0.03519,
      -0.03459,-0.03347,-0.03253,-0.03142,-0.03082,-0.02984,-0.02833,-0.02786,-0.02674,-0.02567,
      -0.02511,-0.02391,-0.02326,-0.02258,-0.02143,-0.02069,-0.02033,-0.0195,-0.01866,-0.01835,
      -0.01743,-0.01694,-0.01683,-0.01625,-0.01569,-0.0155,-0.01533,-0.01469,-0.01441,-0.01467,
      -0.01391,-0.01367,-0.01392,-0.01316,-0.01294,-0.01308,-0.01275,-0.01242,-0.01259,-0.01224,
      -0.01194,-0.012,-0.01206,-0.01175,-0.01143,-0.01152,-0.0115,-0.0115,-0.01152,-0.01144,
      -0.01175,-0.01208,-0.01194,-0.01219,-0.01283,-0.01299,-0.01321,-0.01368,-0.01458,-0.015,
      -0.01541,-0.01636,-0.01666,-0.01752,-0.01827,-0.01891,-0.02009,-0.02075,-0.02145,-0.02248,
      -0.02366,-0.02392,-0.02469,-0.02584,-0.02646,-0.02736,-0.02764,-0.0286,-0.02898,-0.02949,
      -0.03007,-0.03023,-0.03051,-0.03073,-0.03108,-0.03096,-0.03108,-0.03073,-0.03051,-0.03026,
      -0.02997,-0.02989,-0.02901,-0.02809,-0.02815,-0.02732,-0.02607,-0.02593,-0.02475,-0.02359,
      -0.0234,-0.02231,-0.02086,-0.02028,-0.01956,-0.01851,-0.01742,-0.01683,-0.01577,-0.01459,
      -0.01436,-0.01348,-0.01275,-0.01204,-0.01109,-0.01111,-0.01049,-0.00994,-0.00976,-0.00952,
      -0.00918,-0.00926,
     ),
    ),
)
# END PUBLIC MAP DATA

_driver = Driver()


def init(context):
    global _driver
    _driver = Driver()


def act(observation):
    return _driver.act(observation)
