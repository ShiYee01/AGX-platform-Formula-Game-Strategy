"""Frozen V5 (baselines/v5) safety invariants, longitudinal recovery, and lapping decisions."""
import ast,copy,hashlib,importlib.util,json,math
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

v4=load(ROOT/'baselines/v4/agent.py','v4_baseline_current_tests')
v5=load(ROOT/'baselines/v5/agent.py','v5_frozen_tests')
fixture=load(ROOT/'tests/test_control_v4.py','v4_fixture_current_tests')
feedback=load(ROOT/'tools/feedback_v5.py','feedback_v5_tests')
traffic=load(ROOT/'tests/test_traffic_v3.py','traffic_v5_tests')


def observation(station=400.,speed=25.,wet=1.):
    return fixture.observation(station,speed,wet,track=0)


def prepared(o):
    d=v5.Driver();d.reset_phase(o);d.start_distance=o['self']['distance']-1000
    d.start_time=o['time']-20;d.last_unstable_time=o['time']-5
    d.rival_motion={r['seat']:(o['time']-.4,o['self']['n']+r['dn'],0.,o['time']-.4) for r in o['rivals']}
    return d


def leader(gap=-300.,speed=70.,lane=0.,lap=2,seat=6):
    return dict(seat=seat,status='racing',gap_m=gap,dn=lane,speed_mps=speed,lap=lap)


def flat_state(o):
    st=v5.map_state(o)
    return st[0],(0.,)*len(st[1]),(0.,)*len(st[2]),st[3]


class SafetyInvariants(unittest.TestCase):
    def test_v4_geometry_steering_and_speed_safety_are_unchanged(self):
        trees=[ast.parse((ROOT/p).read_text()) for p in ('baselines/v4/agent.py','baselines/v5/agent.py')]
        for name in ('clip','angle','grip_budget','interpolate','lateral_budget','map_state','road_plan','plan_speed','steer_command','pit_entry_distance'):
            nodes=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
            self.assertEqual(ast.dump(nodes[0],include_attributes=False),ast.dump(nodes[1],include_attributes=False),name)
        for name in ('choose_pit','road_bounds','clear_sweep'):
            nodes=[next(n for c in t.body if isinstance(c,ast.ClassDef) and c.name=='Driver' for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
            self.assertEqual(ast.dump(nodes[0],include_attributes=False),ast.dump(nodes[1],include_attributes=False),name)
        self.assertEqual(v4.MAP_DATA,v5.MAP_DATA);self.assertEqual(v4.TOP_SPEED,v5.TOP_SPEED)
        self.assertEqual(v4.BRAKING_BUDGET,v5.BRAKING_BUDGET)

    def test_design_bytes_unchanged(self):
        self.assertEqual((ROOT/'baselines/v5/design.json').read_bytes(),(ROOT/'baselines/v4/design.json').read_bytes())

    def test_braking_command_unchanged_with_extra_enabled(self):
        for wet in (0.,.5,1.):
            for error in (-20.,-5.,-1.,0.):
                o=observation(wet=wet);s=v5.map_state(o);target=o['self']['speed_mps']+error
                self.assertEqual(v4.pedals(o,target,-2.,s,1),v5.pedals(o,target,-2.,s,1,allow_extra=True))

    def test_road_edge_and_spin_recovery_preserved(self):
        for n,h in [(8.6,.21),(0.,.8),(4.,.4)]:
            o=fixture.observation(258.,15.);o['self'].update(n=n,heading_error=h)
            a=v4.Driver().act(o);b=v5.Driver().act(o)
            self.assertEqual(a,b)

    def test_pit_and_low_fuel_safety_preserved(self):
        o=observation();o['self']['pit_state']=1
        self.assertEqual(v4.Driver().act(o),v5.Driver().act(o))
        o=observation();o['self']['fuel_kg']=3.
        self.assertEqual(v4.Driver().act(o),v5.Driver().act(o))

    def test_wet_grid_first_six_seconds_retains_v4_power(self):
        o=observation(speed=0.);o['self']['n']=3.5
        a=v4.Driver();b=v5.Driver()
        self.assertEqual(a.act(o),b.act(o))
        o['time']+=1.;o['self']['speed_mps']=5.
        self.assertEqual(a.act(o),b.act(o))

    def test_phases_reset_lapping_and_slip_memory(self):
        o=observation();d=prepared(o);o['rivals']=[leader()];d.act(o)
        self.assertIsNotNone(d.yield_seat)
        o['phase']='qualifying';o['rivals']=[];d.act(o)
        self.assertIsNone(d.yield_seat);self.assertEqual(d.rival_motion,{})

    def test_tight_curve_does_not_receive_new_acceleration(self):
        o=fixture.observation(234.,9.,1.);s=v5.map_state(o)
        self.assertEqual(v4.pedals(o,15.,1.,s,1),v5.pedals(o,15.,1.,s,1,allow_extra=True))


class AccelerationTests(unittest.TestCase):
    def test_stable_wet_straight_releases_old_limit(self):
        o=observation();s=flat_state(o)
        old=v4.pedals(o,50.,0.,s,2)[0];new=v5.pedals(o,50.,0.,s,2,allow_extra=True)[0]
        self.assertGreater(new,old*1.5)

    def test_slightly_under_target_has_no_extra(self):
        o=observation();s=flat_state(o)
        self.assertEqual(v4.pedals(o,25.3,0.,s,1),v5.pedals(o,25.3,0.,s,1,allow_extra=True))

    def test_sliding_heading_offset_low_fuel_and_missing_flags_use_v4(self):
        for values in ({'rear_sliding':True},{'front_sliding':True},{'heading_error':.11},{'n':1.},{'fuel_kg':3.}):
            o=observation();o['self'].update(values);s=flat_state(o)
            self.assertEqual(v4.pedals(o,50.,0.,s,1),v5.pedals(o,50.,0.,s,1,allow_extra=True),values)
        o=observation();del o['self']['rear_sliding'];s=flat_state(o)
        self.assertEqual(v4.pedals(o,50.,0.,s,1),v5.pedals(o,50.,0.,s,1,allow_extra=True))

    def test_unknown_map_or_missing_force_retains_v4(self):
        o=observation();self.assertEqual(v4.pedals(o,50.,0.,None,1),v5.pedals(o,50.,0.,None,1,allow_extra=True))
        for force in ([],[0.]*4):
            o['self']['drive_force_n']=force
            self.assertEqual(v4.pedals(o,50.,0.,None,1),v5.pedals(o,50.,0.,None,1,allow_extra=True))

    def test_growing_lateral_load_releases_less_propulsion(self):
        o=observation();s=flat_state(o);th=[]
        for util in (.1,.5,.7,.85):
            o['self']['g_lat']=12*v5.grip_budget(o)*util/9.81
            th.append(v5.pedals(o,50.,0.,s,1,allow_extra=True)[0])
        self.assertEqual(th,sorted(th,reverse=True))

    def test_uphill_force_compensation_does_not_exceed_new_wheel_budget(self):
        o=observation();s=flat_state(o);s=(s[0],s[1],(.15,)*len(s[2]),s[3])
        t,b=v5.pedals(o,50.,0.,s,1,allow_extra=True)
        wheel_acc=t*o['self']['drive_force_n'][1]/o['self']['effective_mass_kg']
        self.assertLessEqual(wheel_acc,11*v5.grip_budget(o)+1e-9)
        self.assertEqual(b,0.)

    def test_slip_requires_stable_recovery_time(self):
        o=observation();d=prepared(o);o['self']['rear_sliding']=True;d.act(o)
        o['self']['rear_sliding']=False;o['time']+=.5;d.act(o)
        self.assertFalse(d.diagnostic['extra_accel_enabled'])
        o['time']+=.3;d.act(o);self.assertTrue(d.diagnostic['extra_accel_enabled'])

    def test_traffic_guard_blocks_extra(self):
        o=observation(speed=35.);d=prepared(o);o['rivals']=[leader(-1.,20.,lap=1)]
        o['rivals'][0]['gap_m']=12.;a=d.act(o)
        self.assertFalse(d.diagnostic['extra_accel_enabled']);self.assertEqual(a['throttle'],0.)

    def test_straight_speed_step_improves_in_declared_force_model(self):
        for wet in (0.,.56,1.):
            old=feedback.acceleration_scenario(v4,wet);new=feedback.acceleration_scenario(v5,wet)
            self.assertTrue(new['reached']);self.assertLess(new['seconds'],old['seconds']*.8)
            self.assertEqual(new['sliding_seconds'],0.)


class LappingTests(unittest.TestCase):
    def test_early_lapping_car_selects_side_lane(self):
        o=observation(speed=20.);o['rivals']=[leader()];d=prepared(o);d.act(o)
        self.assertEqual(d.yield_seat,6);self.assertEqual(d.yield_state,'move')
        self.assertGreaterEqual(abs(d.yield_goal),3.2-1e-9)
        self.assertLessEqual(abs(d.offset),.8/30+1e-9)

    def test_newly_visible_opponent_is_observed_before_side_choice(self):
        o=observation(speed=20.);o['rivals']=[leader()];d=prepared(o);d.rival_motion={}
        d.act(o);self.assertEqual(d.offset,0.);self.assertEqual(d.yield_state,'watch')
        o['time']+=.3;d.act(o);self.assertEqual(d.yield_state,'move')

    def test_same_lap_fast_car_does_not_trigger_yield(self):
        o=observation();o['rivals']=[leader(lap=1)];d=prepared(o);d.act(o)
        self.assertIsNone(d.yield_seat)

    def test_missing_lap_does_not_invent_lapping(self):
        o=observation();r=leader();del r['lap'];o['rivals']=[r];d=prepared(o);d.act(o)
        self.assertIsNone(d.yield_seat)

    def test_start_line_wrap_still_identifies_lapping_leader(self):
        o=observation(station=100.,speed=20.);o['rivals']=[leader(lap=1)]
        d=prepared(o);d.act(o);self.assertEqual(d.yield_seat,6)

    def test_start_line_wrap_does_not_mistake_normal_follower(self):
        o=observation(station=100.,speed=20.);o['rivals']=[leader(lap=0)]
        d=prepared(o);d.act(o);self.assertIsNone(d.yield_seat)

    def test_non_closing_or_far_leader_does_not_trigger(self):
        for r in (leader(-300.,20.),leader(-800.,70.)):
            o=observation();o['rivals']=[r];d=prepared(o);d.act(o);self.assertIsNone(d.yield_seat)

    def test_late_rear_threat_keeps_lane_and_does_not_add_braking(self):
        o=observation(speed=20.);o['rivals']=[leader(-12.,60.)];d=prepared(o);a=d.act(o)
        self.assertEqual(d.yield_goal,0.);self.assertEqual(d.offset,0.);self.assertEqual(a['brake'],0.)
        self.assertEqual(d.yield_state,'hold')

    def test_curve_or_short_straight_defers_lateral_move(self):
        for station in (1050.,1100.):
            o=observation(station=station,speed=20.);o['rivals']=[leader()];d=prepared(o);d.act(o)
            self.assertEqual(d.offset,0.);self.assertEqual(d.yield_state,'watch')

    def test_existing_lateral_clearance_is_kept(self):
        o=observation();o['rivals']=[leader(lane=-4.)];d=prepared(o);d.act(o)
        self.assertEqual(d.offset,0.);self.assertEqual(d.yield_state,'hold')

    def test_blocked_sides_hold_instead_of_cutting_across(self):
        o=observation(speed=20.);o['rivals']=[leader(),leader(0.,20.,-3.2,lap=1,seat=8),leader(0.,20.,3.2,lap=1,seat=9)]
        d=prepared(o);d.act(o);self.assertEqual(d.offset,0.)

    def test_one_blocked_side_uses_other(self):
        o=observation(speed=20.);o['rivals']=[leader(),leader(0.,20.,-3.2,lap=1,seat=8)]
        d=prepared(o);d.act(o);self.assertGreater(d.yield_goal,0.)

    def test_insufficient_ttc_holds_lane(self):
        o=observation(speed=20.);o['rivals']=[leader(-100.,45.)];d=prepared(o);d.act(o)
        self.assertEqual(d.offset,0.)

    def test_narrow_road_rejects_clipped_lane(self):
        o=observation(speed=20.);o['rivals']=[leader()]
        for row in o['preview']['rows']:row[3]=row[4]=4.
        d=prepared(o);d.act(o);self.assertEqual(d.offset,0.)

    def test_drifting_rival_blocks_projected_sweep(self):
        o=observation();d=prepared(o);o['rivals']=[leader(0.,25.,3.4,lap=1)]
        d.rival_motion[6]=(o['time'],3.4,-1.)
        self.assertTrue(d.clear_sweep(o,0.,o['rivals']))
        self.assertFalse(d.clear_yield_sweep(o,0.,o['rivals']))

    def test_missing_leader_holds_lane_for_two_seconds(self):
        o=observation(speed=20.);o['rivals']=[leader()];d=prepared(o);d.act(o);goal=d.yield_goal
        o['rivals']=[];o['time']+=1.;d.act(o)
        self.assertEqual(d.yield_goal,goal);self.assertEqual(d.yield_state,'lost_hold')
        o['time']+=1.2;d.act(o);self.assertIsNone(d.yield_seat)

    def test_return_waits_until_leader_is_ahead(self):
        o=observation(speed=20.);o['rivals']=[leader()];d=prepared(o);d.act(o);goal=d.yield_goal
        o['time']+=.1;o['rivals'][0]['gap_m']=10.;d.act(o)
        self.assertEqual(d.yield_goal,goal);self.assertEqual(d.yield_seat,6)
        o['time']+=.1;o['rivals'][0]['gap_m']=25.;d.act(o);self.assertIsNone(d.yield_seat)

    def test_already_clear_close_pass_holds_current_speed(self):
        o=observation(speed=25.);o['rivals']=[leader(-20.,60.,-4.)];d=prepared(o);d.act(o)
        self.assertLessEqual(d.diagnostic['target'],25.)

    def test_launch_and_pit_never_initiate_yield(self):
        for pit in (0,1):
            o=observation(speed=0.);o['self']['pit_state']=pit;o['rivals']=[leader()]
            d=v5.Driver();d.act(o);self.assertIsNone(d.yield_seat)

    def test_lapping_feedback_has_clearance_and_safe_return(self):
        for wet,width in [(0.,8.),(1.,8.),(1.,5.)]:
            r=feedback.lapping_scenario(v5,wet=wet,half_width=width)
            details={k:v for k,v in r.items() if k!='samples'}
            self.assertFalse(r['collision'],details);self.assertTrue(r['passed'],details);self.assertTrue(r['returned'],details)
            self.assertLess(r['max_n'],width-1.5)

    def test_visible_drifting_leader_is_given_the_other_side(self):
        r=feedback.lapping_scenario(v5,rival_lane=lambda t:max(-1.4,-.15*t))
        details={k:v for k,v in r.items() if k!='samples'}
        self.assertFalse(r['collision'],details);self.assertTrue(r['returned'],details)
        moving=[s for s in r['samples'] if abs(s['offset'])>.1]
        self.assertGreater(moving[0]['offset'],0.)

    def test_original_slow_leader_pass_still_works(self):
        for kw in ({},{'half_width':5.},{'initial_gap':8.,'seconds':60.}):
            r=traffic.traffic_scenario(v5,**kw)
            self.assertFalse(r['collision']);self.assertTrue(r['passed']);self.assertTrue(r['returned'])

    def test_input_is_not_mutated_and_output_is_native(self):
        o=observation(speed=20.);o['rivals']=[leader()];old=copy.deepcopy(o)
        a=prepared(o).act(o);json.dumps(a,allow_nan=False);self.assertEqual(o,old)


if __name__=='__main__':unittest.main()
