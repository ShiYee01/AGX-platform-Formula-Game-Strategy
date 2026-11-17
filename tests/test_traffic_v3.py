"""Positive passing scenarios, collision guards, and frozen V2 parameters."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v2 = load(ROOT/'baselines/v2/agent.py', 'v2_frozen')
v3 = load(ROOT/'agent/agent.py', 'v3_current')


def observation():
    return dict(phase='race', time=20.0, track_length_m=5795.843, laps_total=4,
                self=dict(n=0.0, heading_error=0.0, speed_mps=20.0,
                          fuel_kg=60.0, tyre_wear=[0.0]*4,
                          tyre_temp_c=[80.0]*4, compound='soft',
                          ers_soc_mj=3.0, pit_state=0, pit_stops=0,
                          lap=0, distance=1000.0, s=1000.0,
                          front_sliding=False, rear_sliding=False),
                preview=dict(rows=[[float(i*15), 0.0, 0.0, 8.0, 8.0, 0]
                                   for i in range(1,25)]),
                weather=dict(condition='dry', track_wetness=0.0),
                rivals=[dict(seat=1,status='racing',gap_m=35.0,dn=0.0,speed_mps=20.0)])


def prepared(module, o):
    d = module.Driver()
    initial = copy.deepcopy(o)
    initial['time'] = 0.0
    initial['self']['distance'] = 0.0
    d.reset_phase(initial)
    return d


def traffic_scenario(module, curve=0.0, initial_gap=35.0, seconds=45.0, half_width=8.0):
    """Toy bicycle/traffic integration; not AGX physics or a lap-time model.

    Acceleration, steering lag and rectangular collision dimensions are chosen
    test assumptions. This checks that a lane change can finish in a feedback
    loop, rather than assuming that an intended lane change already happened.
    """
    o = observation()
    o['self']['distance'] = o['self']['s'] = 0.0
    d = prepared(module,o)
    d.start_distance = -500.0  # Test begins after the launch zone.
    n = heading = station = steer = 0.0
    speed = 20.0
    dt = 1.0/30.0
    max_n = 0.0
    collision = False
    passed = False
    samples = []
    for step in range(round(seconds/dt)):
        elapsed = step*dt
        gap = initial_gap+20.0*elapsed-station
        o['time'] = 20.0+elapsed
        o['self'].update(n=n,heading_error=heading,speed_mps=speed,
                         distance=station,s=station%o['track_length_m'])
        for row in o['preview']['rows']:
            row[1]=curve;row[3]=row[4]=half_width
        o['rivals']=[dict(seat=1,status='racing',gap_m=gap,dn=-n,speed_mps=20.0)]
        action=d.act(o)
        if abs(gap)<4.5 and abs(n)<2.0: collision=True
        if gap < -12.0: passed=True
        if step%15==0:
            samples.append(dict(t=round(elapsed,3),gap=gap,n=n,speed=speed,
                                offset=d.offset,pass_seat=d.pass_seat))
        speed=max(0.0,speed+dt*(8.0*action['throttle']-12.0*action['brake']-.4))
        steer+=min(1.0,dt/.12)*(.30*action['steer']-steer)
        heading+=dt*(speed/3.4*math.tan(steer)-speed*curve/max(.5,1.0-curve*n))
        n+=dt*speed*math.sin(heading)
        station+=dt*speed*math.cos(heading)/max(.5,1.0-curve*n)
        max_n=max(max_n,abs(n))
        if passed and abs(n)<.2 and gap < -15.0:
            return dict(passed=True,returned=True,collision=collision,max_n=max_n,
                        steps=step+1,samples=samples)
    return dict(passed=passed,returned=False,collision=collision,max_n=max_n,
                steps=step+1,samples=samples)


class TrafficTests(unittest.TestCase):
    def test_centre_leader_has_a_feasible_pass_lane(self):
        o=observation();d=prepared(v3,o)
        d.traffic_plan(o,55.0,1/30)
        self.assertEqual(d.pass_seat,1)
        self.assertAlmostEqual(abs(d.pass_goal),3.2)
        self.assertGreater(abs(d.pass_goal),v3.TRAFFIC_CLEARANCE)

    def test_frozen_v2_reproduces_the_original_failure(self):
        o=observation();d=prepared(v2,o)
        d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat)

    def test_off_centre_leader_uses_relative_lane_targets(self):
        o=observation();o['rivals'][0]['dn']=1.0;d=prepared(v3,o)
        d.traffic_plan(o,55.0,1/30)
        self.assertGreater(abs(1.0-d.pass_goal),v3.TRAFFIC_CLEARANCE)
        self.assertAlmostEqual(d.pass_goal,-2.2)

    def test_occupied_side_selects_other_side(self):
        o=observation();o['rivals'].append(dict(seat=2,status='racing',gap_m=0.0,dn=-3.2,speed_mps=20.0))
        d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        self.assertGreater(d.pass_goal,0.0)

    def test_both_sides_occupied_do_not_initiate_pass(self):
        o=observation()
        for i,n in enumerate((-3.2,3.2),2):
            o['rivals'].append(dict(seat=i,status='racing',gap_m=0.0,dn=n,speed_mps=20.0))
        d=prepared(v3,o);target,_=d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat);self.assertLess(target,55.0)

    def test_narrow_road_does_not_clip_a_pass_into_the_leader(self):
        o=observation()
        for row in o['preview']['rows']:row[3]=row[4]=4.0
        d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat)

    def test_speed_guard_stays_until_actual_lateral_clearance(self):
        o=observation();d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        d.offset=d.pass_goal
        target,_=d.traffic_plan(o,55.0,1/30)
        self.assertLess(target,55.0)  # Intent alone does not remove protection.
        o['self']['n']=d.pass_goal;o['rivals'][0]['dn']=-d.pass_goal
        target,_=d.traffic_plan(o,55.0,1/30)
        self.assertEqual(target,55.0)

    def test_bend_does_not_force_merge_across_target(self):
        o=observation();d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        goal=d.pass_goal
        for row in o['preview']['rows']:row[1]=.01
        o['time']+=1/30
        d.act(o)
        self.assertEqual(d.pass_seat,1);self.assertEqual(d.pass_goal,goal)
        self.assertLessEqual(d.diagnostic['target'],v3.plan_speed(o,d.offset))

    def test_sharp_bend_does_not_start_new_pass(self):
        o=observation()
        for row in o['preview']['rows']:row[1]=.01
        d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat)

    def test_launch_zone_still_disables_pass(self):
        o=observation();o['self']['distance']=100.0
        d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat)

    def test_brief_missing_rival_does_not_trigger_merge(self):
        o=observation();d=prepared(v3,o);d.traffic_plan(o,55.0,1/30)
        d.offset=d.pass_goal;o['self']['n']=d.offset;o['rivals']=[];o['time']+=.1
        goal=d.pass_goal;d.traffic_plan(o,55.0,1/30)
        self.assertEqual(d.offset,goal);self.assertEqual(d.pass_seat,1)
        o['time']+=1.1;d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat);self.assertLess(abs(d.offset),abs(goal))

    def test_rear_fast_car_blocks_return_merge(self):
        o=observation();d=prepared(v3,o)
        d.pass_seat=1;d.pass_goal=d.offset=3.2
        o['self'].update(n=3.2,speed_mps=40.0)
        o['rivals']=[dict(seat=1,status='racing',gap_m=-15.0,dn=-3.2,speed_mps=60.0)]
        d.traffic_plan(o,55.0,1/30)
        self.assertEqual(d.offset,3.2)

    def test_target_pulling_away_can_end_pass_when_merge_is_clear(self):
        o=observation();d=prepared(v3,o)
        d.pass_seat=1;d.pass_goal=d.offset=3.2
        o['self'].update(n=3.2,speed_mps=20.0)
        o['rivals']=[dict(seat=1,status='racing',gap_m=80.0,dn=-3.2,speed_mps=45.0)]
        d.traffic_plan(o,55.0,1/30)
        self.assertIsNone(d.pass_seat);self.assertLess(d.offset,3.2)

    def test_sweep_covers_the_full_lane_change_duration(self):
        o=observation();d=prepared(v3,o)
        rival=dict(seat=2,status='racing',gap_m=-80.0,dn=3.2,speed_mps=40.0)
        self.assertFalse(d.clear_sweep(o,3.2,[rival]))

    def test_lateral_target_rate_is_unchanged(self):
        o=observation();d=prepared(v3,o)
        for i in range(30):
            before=d.offset;o['time']+=1/30;d.traffic_plan(o,55.0,1/30)
            self.assertLessEqual(abs(d.offset-before),.8/30+1e-12)

    def test_feedback_scenario_completes_pass_and_return(self):
        old=traffic_scenario(v2)
        new=traffic_scenario(v3)
        self.assertFalse(old['passed'])
        self.assertTrue(new['passed']);self.assertTrue(new['returned'])
        self.assertFalse(new['collision']);self.assertLess(new['max_n'],6.5)

    def test_gentle_curve_feedback_pass(self):
        new=traffic_scenario(v3,curve=.0008)
        self.assertTrue(new['passed']);self.assertTrue(new['returned'])
        self.assertFalse(new['collision']);self.assertLess(new['max_n'],6.5)

    def test_close_following_feedback_can_open_space_then_pass(self):
        new=traffic_scenario(v3,initial_gap=8.0,seconds=60.0)
        self.assertTrue(new['passed']);self.assertFalse(new['collision'])

    def test_minimum_public_map_width_feedback(self):
        new=traffic_scenario(v3,half_width=5.0)
        self.assertTrue(new['passed']);self.assertTrue(new['returned'])
        self.assertFalse(new['collision']);self.assertLess(new['max_n'],3.5)


class FrozenParameterTests(unittest.TestCase):
    def test_non_traffic_planners_and_actuator_law_are_unchanged(self):
        old=ast.parse((ROOT/'baselines/v2/agent.py').read_text())
        new=ast.parse((ROOT/'agent/agent.py').read_text())
        for tree in (old,new):
            for node in ast.walk(tree):
                for attr in ('lineno','col_offset','end_lineno','end_col_offset'):
                    if hasattr(node,attr):delattr(node,attr)
        funcs=lambda tree:{node.name:ast.dump(node) for node in tree.body if isinstance(node,ast.FunctionDef)}
        for name in ('clip','angle','grip_budget','plan_speed','pit_entry_distance'):
            self.assertEqual(funcs(old)[name],funcs(new)[name],name)
        methods=lambda tree:{n.name:ast.dump(n) for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Driver' for n in node.body if isinstance(n,ast.FunctionDef)}
        for name in ('choose_pit','road_bounds','act'):
            self.assertEqual(methods(old)[name],methods(new)[name],name)
        self.assertEqual(v2.TOP_SPEED,v3.TOP_SPEED)
        self.assertEqual(v2.TRACK_PITS,v3.TRACK_PITS)

    def test_design_bytes_are_unchanged(self):
        self.assertEqual((ROOT/'agent/design.json').read_bytes(),(ROOT/'baselines/v2/design.json').read_bytes())


if __name__=='__main__':unittest.main()
