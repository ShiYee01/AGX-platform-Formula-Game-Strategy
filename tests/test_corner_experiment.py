"""Corner experiment may add throttle only inside the unchanged envelope."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corner_helper',ROOT/'experiments/corner_support.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def case():
    o=dict(self=dict(n=0.0,heading_error=.01,speed_mps=20.0,pit_state=0,
                    fuel_kg=60.0,front_sliding=False,rear_sliding=False,
                    drive_force_n=[10000.0]*4,drag_n=400.0,
                    rolling_resistance_n=100.0,g_lat=.1),
           preview=dict(rows=[[15,.006],[30,.004],[45,.003]]))
    return o


class CornerTests(unittest.TestCase):
    def apply(self,o=None,**changes):
        params=dict(throttle=.4,brake=0.0,target=22.0,offset=0.0,
                    road_limit=22.0,elapsed=20.0,ers_mode=1,lateral_budget=10.0)
        params.update(changes)
        return helper.corner_throttle(o or case(),**params)

    def test_opening_bend_adds_resistance_support(self):
        self.assertAlmostEqual(self.apply(),.45)

    def test_no_support_during_braking(self):
        self.assertEqual(self.apply(throttle=0.0,brake=.1),0.0)

    def test_no_support_when_traffic_caps_the_target(self):
        self.assertEqual(self.apply(road_limit=30.0),.4)

    def test_no_support_above_or_at_target(self):
        self.assertEqual(self.apply(target=20.0),.4)

    def test_no_support_in_tightening_bend(self):
        o=case();o['preview']['rows'][1][1]=.01
        self.assertEqual(self.apply(o),.4)

    def test_no_support_in_direction_reversal(self):
        o=case();o['preview']['rows'][1][1]=-.004
        self.assertEqual(self.apply(o),.4)

    def test_reversal_at_third_preview_point_also_blocks_support(self):
        o=case();o['preview']['rows'][2][1]=-.003
        self.assertEqual(self.apply(o),.4)

    def test_no_support_on_straight(self):
        o=case()
        for row in o['preview']['rows']:row[1]=0.0
        self.assertEqual(self.apply(o),.4)

    def test_no_support_near_lateral_budget(self):
        o=case();o['self']['g_lat']=1.0
        self.assertEqual(self.apply(o),.4)

    def test_preview_load_also_preserves_lateral_reserve(self):
        o=case();o['self']['speed_mps']=40.0
        self.assertEqual(self.apply(o,target=45.0,road_limit=45.0),.4)

    def test_no_support_when_sliding(self):
        for key in ('front_sliding','rear_sliding'):
            o=case();o['self'][key]=True
            self.assertEqual(self.apply(o),.4)

    def test_no_support_when_off_line(self):
        o=case();o['self']['n']=1.0
        self.assertEqual(self.apply(o),.4)

    def test_no_support_during_launch_or_pits_or_low_fuel(self):
        self.assertEqual(self.apply(elapsed=3.0),.4)
        o=case();o['self']['pit_state']=2
        self.assertEqual(self.apply(o),.4)
        o=case();o['self']['fuel_kg']=4.0
        self.assertEqual(self.apply(o),.4)

    def test_missing_telemetry_falls_back(self):
        o=case();del o['self']['drive_force_n']
        self.assertEqual(self.apply(o),.4)

    def test_nonfinite_telemetry_falls_back(self):
        o=case();o['self']['drag_n']=float('nan')
        self.assertEqual(self.apply(o),.4)

    def test_extra_throttle_is_bounded(self):
        o=case();o['self']['drag_n']=10000.0
        self.assertAlmostEqual(self.apply(o),.55)
        self.assertEqual(self.apply(o,throttle=.95),1.0)

    def test_support_fades_near_target(self):
        self.assertAlmostEqual(self.apply(target=20.5,road_limit=20.5),.425)

    def test_input_observation_is_not_mutated(self):
        o=case();before=copy.deepcopy(o);self.apply(o)
        self.assertEqual(o,before)


if __name__=='__main__':unittest.main()
