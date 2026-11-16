"""Regression cases for the failures seen in the V1 race exports."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('racing_agent', ROOT / 'agent/agent.py')
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def observation():
    return dict(phase='race', time=0.0, track_length_m=5795.843, laps_total=4,
                self=dict(n=0.0, heading_error=0.0, speed_mps=30.0,
                          fuel_kg=60.0, tyre_wear=[0.0]*4,
                          tyre_temp_c=[80.0]*4, compound='soft',
                          ers_soc_mj=3.0, pit_state=0, pit_stops=0,
                          lap=0, distance=-64.0, s=5731.843,
                          front_sliding=False, rear_sliding=False),
                preview=dict(rows=[[float(i*15), 0.0, 0.0, 8.0, 8.0, 0]
                                   for i in range(1,25)]),
                weather=dict(condition='dry', track_wetness=0.0), rivals=[])


class DriverTests(unittest.TestCase):
    def test_straight_cap(self):
        self.assertEqual(agent.plan_speed(observation()), 55.0)

    def test_future_apex_is_not_imposed_immediately(self):
        o = observation()
        o['preview']['rows'][2][1] = .025
        target = agent.plan_speed(o)
        self.assertGreater(target, math.sqrt(10/.025))
        self.assertLessEqual(target, math.sqrt(10/.025 + 2*7*(45-9)))

    def test_full_horizon_sees_bend_beyond_old_180m(self):
        o = observation(); o['self']['speed_mps'] = 55.0
        o['preview']['rows'][12][1] = .10
        self.assertLess(agent.plan_speed(o), 55.0)

    def test_wet_and_worn_reduce_corner_speed(self):
        o = observation()
        for row in o['preview']['rows']: row[1] = .02
        dry = agent.plan_speed(o)
        o['weather']['track_wetness'] = 1.0
        wet = agent.plan_speed(o)
        o['self']['tyre_wear'] = [.8]*4
        self.assertLess(wet, dry)
        self.assertLess(agent.plan_speed(o), wet)

    def test_low_fuel_is_still_conservative(self):
        o = observation(); o['self']['fuel_kg'] = 3.0
        self.assertLessEqual(agent.plan_speed(o), 27.0)

    def test_required_dry_stop_on_lap_one(self):
        o = observation(); o['self'].update(lap=1, s=5600.0)
        d = agent.Driver(); d.act(o)
        self.assertEqual(d.choose_pit(o), (True, 'medium'))

    def test_request_only_in_pit_approach_window(self):
        o = observation(); o['self'].update(lap=1, s=1000.0)
        d = agent.Driver(); d.act(o)
        self.assertFalse(d.choose_pit(o)[0])

    def test_wet_waiver_persists_after_drying(self):
        o = observation(); o['self'].update(lap=1, s=5600.0)
        o['weather'].update(condition='wet', track_wetness=1.0)
        d = agent.Driver(); self.assertFalse(d.act(o)['pit_request'])
        o['time'] += 1/30; o['weather'].update(condition='dry', track_wetness=0.0)
        self.assertFalse(d.act(o)['pit_request'])

    def test_qualifying_rain_does_not_waive_dry_race(self):
        o = observation(); o['phase'] = 'qualifying'
        o['weather']['condition'] = 'wet'
        d = agent.Driver(); d.act(o)
        o['phase'] = 'race'; o['weather']['condition'] = 'dry'
        o['self'].update(lap=1, s=5600.0)
        self.assertTrue(d.act(o)['pit_request'])

    def test_completed_compounds_prevent_repeat_stop(self):
        o = observation(); o['self'].update(lap=1, s=5600.0)
        d = agent.Driver(); d.act(o)
        o['time'] += 1/30; o['self'].update(compound='medium', pit_stops=1)
        self.assertFalse(d.act(o)['pit_request'])

    def test_worn_wet_tyres_still_allow_health_stop(self):
        o = observation(); o['self'].update(lap=1, s=5600.0, tyre_wear=[.8]*4)
        o['weather']['condition'] = 'wet'
        self.assertTrue(agent.Driver().act(o)['pit_request'])

    def test_pit_transit_keeps_engine_contract(self):
        o = observation(); o['self']['pit_state'] = 1
        a = agent.Driver().act(o)
        self.assertEqual((a['throttle'],a['brake'],a['steer']), (0.0,0.0,0.0))
        self.assertFalse(a['pit_request'])

    def test_grid_lanes_not_forced_to_centre(self):
        for lane in (-3.5,3.5):
            o = observation(); o['self'].update(n=lane,speed_mps=0.0)
            d = agent.Driver(); a = d.act(o)
            self.assertEqual(d.offset,lane)
            self.assertAlmostEqual(a['steer'],0.0)

    def test_lane_target_changes_gradually(self):
        o = observation(); o['self'].update(n=3.5,speed_mps=20.0)
        d = agent.Driver(); d.act(o)
        o['time'] += 1/30; o['self']['distance'] += 400.0
        d.act(o)
        self.assertLessEqual(abs(d.offset-3.5), .8/30 + 1e-9)

    def test_adjacent_car_blocks_merge(self):
        o = observation(); d = agent.Driver()
        o['rivals'] = [dict(seat=1,status='racing',gap_m=0.0,dn=-3.5,speed_mps=30.0)]
        o['self']['n'] = 3.5
        self.assertFalse(d.clear_sweep(o,0.0,o['rivals']))

    def test_early_following_guard(self):
        o = observation(); o['self']['speed_mps'] = 40.0
        o['rivals'] = [dict(seat=1,status='racing',gap_m=25.0,dn=0.0,speed_mps=20.0)]
        a = agent.Driver().act(o)
        self.assertGreater(a['brake'],0.0)
        self.assertEqual(a['throttle'],0.0)

    def test_car_behind_does_not_stop_leader(self):
        o = observation()
        o['rivals'] = [dict(seat=1,status='racing',gap_m=-4.0,dn=0.0,speed_mps=35.0)]
        self.assertGreater(agent.Driver().act(o)['throttle'],0.0)

    def test_spin_angles_are_wrapped(self):
        self.assertAlmostEqual(agent.angle(-3.94), -3.94+2*math.pi)

    def test_phase_reset_clears_lane_and_memory(self):
        o = observation(); o['self']['n']=3.5
        d=agent.Driver();d.act(o)
        o['phase']='qualifying';o['self']['n']=0.0;d.act(o)
        self.assertEqual(d.offset,0.0)
        self.assertEqual(d.compounds,set())

    def test_native_serializable_action_and_bounds(self):
        o = observation();o['self']['heading_error']=7.2
        a=agent.Driver().act(o)
        json.dumps(a,allow_nan=False)
        self.assertIs(type(a['pit_request']),bool)
        self.assertIs(type(a['ers_mode']),int)
        self.assertTrue(0<=a['throttle']<=1 and 0<=a['brake']<=1 and -1<=a['steer']<=1)


if __name__ == '__main__':
    unittest.main()
