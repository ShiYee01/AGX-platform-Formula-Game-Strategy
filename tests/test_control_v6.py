"""Current V6 (agent/): raised grip/braking/power budgets, full throttle on
stable straights, online grip trust, and everything else kept from V5."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


v5 = load(ROOT / 'baselines/v5/agent.py', 'v5_for_v6_tests')
v6 = load(ROOT / 'agent/agent.py', 'v6_current_tests')
fixture = load(ROOT / 'tests/test_control_v4.py', 'v4_fixture_for_v6_tests')
model = load(ROOT / 'tools/feedback_v6.py', 'feedback_v6_tests')
TRACKS = ('parkland', 'highland', 'crossover')


def observation(station=400., speed=25., wet=1., track=0):
    return fixture.observation(station, speed, wet, track=track)


def flat_state(o):
    st = v6.map_state(o)
    return st[0], (0.,) * len(st[1]), (0.,) * len(st[2]), st[3]


def prepared(o):
    d = v6.Driver()
    d.reset_phase(o)
    d.start_distance = o['self']['distance'] - 1000
    d.start_time = o['time'] - 20
    d.last_unstable_time = o['time'] - 5
    return d


def functions(path):
    tree = ast.parse((ROOT / path).read_text())
    out = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for c in tree.body:
        if isinstance(c, ast.ClassDef):
            out.update({f'{c.name}.{n.name}': n for n in c.body if isinstance(n, ast.FunctionDef)})
    return out


class Contract(unittest.TestCase):
    def setUp(self):
        v6._GRIP_SCALE[0] = 1.0

    def test_action_fields_types_ranges_on_many_states(self):
        for track in range(3):
            length = v6.MAP_DATA[track][0]
            for station in [i * length / 40 for i in range(40)]:
                for speed in (0., 12., 40., 85.):
                    for wet in (0., .5, 1.):
                        o = observation(station, speed, wet, track)
                        a = v6.Driver().act(o)
                        self.assertEqual(set(a), {'throttle', 'brake', 'steer', 'ers_mode', 'aero_mode',
                                                  'pit_request', 'pit_compound'})
                        for k in ('throttle', 'brake', 'steer'):
                            self.assertIsInstance(a[k], float)
                            self.assertTrue(math.isfinite(a[k]))
                        self.assertTrue(0 <= a['throttle'] <= 1 and 0 <= a['brake'] <= 1 and -1 <= a['steer'] <= 1)
                        self.assertFalse(a['throttle'] > 0 and a['brake'] > 0)
                        self.assertIn(a['ers_mode'], (1, 2))
                        self.assertEqual(a['aero_mode'], 0)
                        self.assertIs(type(a['pit_request']), bool)

    def test_source_is_stdlib_only_and_small(self):
        tree = ast.parse((ROOT / 'agent/agent.py').read_text())
        imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imports |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imports, {'math'})
        self.assertLess((ROOT / 'agent/agent.py').stat().st_size, 256 * 1024)

    def test_design_unchanged_from_v5(self):
        self.assertEqual((ROOT / 'agent/design.json').read_bytes(), (ROOT / 'baselines/v5/design.json').read_bytes())


class KeptFromV5(unittest.TestCase):
    def test_unchanged_functions(self):
        old, new = functions('baselines/v5/agent.py'), functions('agent/agent.py')
        for name in ('clip', 'angle', 'interpolate', 'map_state', 'plan_speed', 'steer_command',
                     'is_lapping_rival', 'pit_entry_distance', 'Driver.choose_pit', 'Driver.road_bounds',
                     'Driver.clear_sweep', 'Driver.observe_rivals', 'Driver.clear_yield_sweep',
                     'Driver.lapping_plan', 'Driver.traffic_plan'):
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        self.assertEqual(v5.MAP_DATA, v6.MAP_DATA)

    def test_unknown_map_fallback_identical_to_v5(self):
        for wet in (0., 1.):
            o = observation(wet=wet)
            o['track_length_m'] = 4000.
            self.assertEqual(v5.road_plan(o), v6.road_plan(o))
            self.assertLessEqual(v6.plan_speed(o), 45.)

    def test_pit_lane_zero_pedals(self):
        o = observation()
        o['self']['pit_state'] = 1
        a = v6.Driver().act(o)
        self.assertEqual((a['throttle'], a['brake'], a['steer']), (0., 0., 0.))

    def test_off_line_spin_and_low_fuel_recovery_caps_kept(self):
        for update, cap in (({'n': 3.}, 15.), ({'heading_error': .4}, 10.), ({'heading_error': .7}, 5.),
                            ({'fuel_kg': 3.}, 27.)):
            o = observation(1500., 60., 0., track=1)
            o['self'].update(update)
            self.assertLessEqual(v6.plan_speed(o), cap, update)


class Budgets(unittest.TestCase):
    def setUp(self):
        v6._GRIP_SCALE[0] = 1.0

    def test_chosen_budgets(self):
        self.assertEqual((v6.TOP_SPEED, v6.BRAKING_BUDGET, v6.ACCEL_BUDGET), (90., 14., 12.))
        self.assertEqual([v6.lateral_budget(k) for k in (0., .01, .049, .051, -.09)], [17., 17., 17., 13., 13.])
        # Planned peaks stay below the clean top-3 replay 90th percentiles:
        # dry (<0.3, ~0.1) 17.2 and wet (>=0.7, ~0.85) 11.4 m/s^2.
        self.assertEqual(v6.WET_GRIP_LOSS, .55)
        self.assertLessEqual(17. * (1 - .55 * .1), 17.2)
        self.assertLessEqual(17. * (1 - .55 * .85), 11.4)

    def test_v6_plan_never_slower_than_v5_and_respects_top_speed(self):
        for track in range(3):
            length = v6.MAP_DATA[track][0]
            for station in [i * length / 300 for i in range(300)]:
                for wet in (0., 1.):
                    o = observation(station, 30., wet, track)
                    new = v6.plan_speed(o)
                    self.assertGreaterEqual(new + 1e-9, v5.plan_speed(o))
                    self.assertLessEqual(new, 90. + 1e-9)

    def test_full_throttle_on_stable_straight_below_target(self):
        for wet in (0., 1.):
            o = observation(speed=40., wet=wet)
            self.assertEqual(v6.pedals(o, 60., 0., flat_state(o), 2, allow_extra=True), (1.0, 0.0))

    def test_no_full_throttle_when_unstable_or_loaded(self):
        cases = ({'rear_sliding': True}, {'front_sliding': True}, {'heading_error': .11}, {'n': 1.},
                 {'fuel_kg': 3.}, {'g_lat': 12. * .55 / 9.81})
        for update in cases:
            o = observation(speed=40.)
            o['self'].update(update)
            self.assertLess(v6.pedals(o, 60., 0., flat_state(o), 2, allow_extra=True)[0], 1.0, update)
        o = observation(speed=40.)
        self.assertLess(v6.pedals(o, 60., 0., flat_state(o), 2, allow_extra=False)[0], 1.0)
        o = observation(speed=40.)
        del o['self']['rear_sliding']
        self.assertLess(v6.pedals(o, 60., 0., flat_state(o), 2, allow_extra=True)[0], 1.0)

    def test_no_full_throttle_wet_low_speed_or_near_target(self):
        o = observation(speed=15., wet=1.)
        self.assertLess(v6.pedals(o, 40., 0., flat_state(o), 2, allow_extra=True)[0], 1.0)
        o = observation(speed=40., wet=0.)
        self.assertLess(v6.pedals(o, 41.5, 0., flat_state(o), 2, allow_extra=True)[0], 1.0)

    def test_rear_slide_still_caps_throttle(self):
        o = observation(speed=40.)
        o['self']['rear_sliding'] = True
        self.assertLessEqual(v6.pedals(o, 70., 0., flat_state(o), 2, allow_extra=True)[0], .25)

    def test_braking_is_stronger_than_v5_for_large_overspeed(self):
        for wet in (0., 1.):
            o = observation(speed=60., wet=wet)
            s = flat_state(o)
            old = v5.pedals(o, 30., -5., s, 1)[1]
            new = v6.pedals(o, 30., -5., s, 1)[1]
            self.assertGreater(new, old)
            decel = new * 56000. / o['self']['effective_mass_kg']
            self.assertLessEqual(decel, (14. + 4.) * v6.grip_budget(o) + 1.)


class ReplayEvidence(unittest.TestCase):
    """analysis/v6-validation/corner-check.json: fastest clean replay laps."""

    def test_planned_corner_speed_within_demonstrated_clean_laps(self):
        import json
        data = json.loads((ROOT / 'analysis/v6-validation/corner-check.json').read_text())
        index = {'parkland': 0, 'highland': 1, 'crossover': 2}
        wet = {'dry': .15, 'wet': .85}
        for c in data['corners']:
            stations = [c['s_start'] + i * (c['s_end'] - c['s_start']) / 11 for i in range(12)]
            plan = min(v6.plan_speed(observation(x, 30., wet[c['bucket']], index[c['track']]))
                       for x in stations) * 3.6
            self.assertLessEqual(plan, c['clean_max_kmh'] + 1.5, c)


class GripTrust(unittest.TestCase):
    def setUp(self):
        v6._GRIP_SCALE[0] = 1.0

    def test_corner_slide_lowers_targets_and_recovers(self):
        o = observation(990., 30., 0., track=1)  # Highland R106 bend: grip-limited
        d = prepared(o)
        base = d.act(copy.deepcopy(o)) and d.diagnostic['road_target']
        o['self'].update(front_sliding=True, g_lat=1.0)
        for _ in range(30):
            o['time'] += 1 / 30
            d.act(o)
        self.assertAlmostEqual(d.diagnostic['grip_scale'], 1 - .15, places=2)
        self.assertLess(d.diagnostic['road_target'], base)
        o['self'].update(front_sliding=False)
        for _ in range(300):
            o['time'] += 1 / 30
            d.act(o)
        self.assertAlmostEqual(d.diagnostic['grip_scale'], .95, places=2)

    def test_floor_and_reset(self):
        o = observation(1500., 50., 0., track=1)
        d = prepared(o)
        o['self'].update(front_sliding=True, g_lat=1.0)
        for _ in range(600):
            o['time'] += 1 / 30
            d.act(o)
        self.assertAlmostEqual(d.diagnostic['grip_scale'], v6.GRIP_SCALE_MIN)
        o['phase'] = 'qualifying'
        o['self'].update(front_sliding=False, g_lat=0.)
        d.act(o)
        self.assertGreater(d.diagnostic['grip_scale'], .99)
        v6._GRIP_SCALE[0] = .8
        v6.init({})
        self.assertEqual(v6._GRIP_SCALE[0], 1.0)

    def test_straight_wheelspin_does_not_lower_trust(self):
        o = observation(speed=40.)
        d = prepared(o)
        o['self'].update(rear_sliding=True, g_lat=0.)
        for _ in range(30):
            o['time'] += 1 / 30
            d.act(o)
        self.assertEqual(d.diagnostic['grip_scale'], 1.0)


class CalibratedModel(unittest.TestCase):
    """tools/feedback_v6.py: replay-calibrated bicycle model, not the engine."""

    def test_full_laps_stay_on_road_and_beat_v5(self):
        for track in TRACKS:
            for wet in (0., 1.):
                for case in (model.NOMINAL, model.PESSIMISTIC):
                    new = model.drive(v6, track, wet, **case)
                    old = model.drive(v5, track, wet, **case)
                    self.assertTrue(new['completed'], (track, wet, case))
                    self.assertGreater(new['min_road_margin'], 4.0, (track, wet, case))
                    self.assertLess(new['slide_fraction'], .05, (track, wet, case))
                    self.assertLess(new['lap_time'], old['lap_time'] * .90, (track, wet, case))

    def test_wet_standing_start_from_both_grid_columns(self):
        for track, start in (('parkland', 5794.8), ('highland', 6998.5), ('crossover', 5802.2)):
            for lane in (-3.5, 3.5):
                r = model.drive(v6, track, 1., start=start, distance=800, v0=.5, lane=lane, launch=True,
                                timing=(0, 800), **model.PESSIMISTIC)
                self.assertTrue(r['completed'], (track, lane))
                self.assertGreater(r['min_road_margin'], 4.0)


if __name__ == '__main__':
    unittest.main()
