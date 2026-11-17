"""V4 regressions for the actual Highland failure, telemetry, and traffic."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

v4=load(ROOT/'baselines/v4/agent.py','control_v4')
v3=load(ROOT/'baselines/v3/agent.py','control_v3')
feedback=load(ROOT/'tools/feedback_v4.py','feedback_v4_tests')
traffic=load(ROOT/'tests/test_traffic_v3.py','traffic_fixture_v4')


def observation(station=190.,speed=15.,wet=.56,track=1):
    length,curves,grades=v4.MAP_DATA[track]
    interp=lambda vals,s:v4.interpolate(vals,s,length)
    return dict(phase='race',time=100.,track_length_m=length,laps_total=4,
      self=dict(n=0.,heading_error=0.,speed_mps=speed,s=station,distance=length+station,
                fuel_kg=58.,tyre_wear=[.02]*4,tyre_temp_c=[70.]*4,compound='soft',ers_soc_mj=3.,
                pit_state=0,pit_stops=0,lap=1,front_sliding=False,rear_sliding=False,
                yaw_rate=0.,steer=0.,g_lat=0.,drag_n=.53*speed**2,rolling_resistance_n=140.,
                effective_mass_kg=950.,drive_force_n=[12000.,12000.,18000.,24000.],brake_force_max_n=56000.),
      weather=dict(condition='wet' if wet>.5 else 'dry',track_wetness=wet),rivals=[],
      preview=dict(rows=[[float(x),interp(curves,station+x),interp(grades,station+x),8.,8.,0] for x in range(15,361,15)]))


class ControlTests(unittest.TestCase):
    def test_map_matches_all_three_track_previews(self):
        for track in range(3):self.assertIsNotNone(v4.map_state(observation(track=track)))

    def test_unknown_map_uses_conservative_fallback(self):
        o=observation();o['track_length_m']=4000
        self.assertIsNone(v4.map_state(o));self.assertLessEqual(v4.plan_speed(o),45)

    def test_changed_geometry_is_rejected(self):
        o=observation();o['preview']['rows'][0][1]+=.01
        self.assertIsNone(v4.map_state(o))

    def test_curvature_data_is_periodic(self):
        for length,ks,gs in v4.MAP_DATA:
            self.assertEqual(v4.interpolate(ks,0,length),v4.interpolate(ks,length,length))
            self.assertAlmostEqual(v4.interpolate(gs,-1,length),v4.interpolate(gs,length-1,length))

    def test_no_fixed_15m_early_steering_at_hairpin_entry(self):
        o=observation(200.,12.)
        new=v4.steer_command(o,0.,v4.map_state(o))
        old=v3.Driver().act(o)['steer']
        self.assertLess(abs(new),.05);self.assertGreater(abs(old),.1)

    def test_current_bend_still_limits_speed_after_preview_opens(self):
        o=observation(241.,12.);state=v4.map_state(o)
        self.assertGreater(abs(v4.interpolate(state[1],241.,state[0])),.07)
        self.assertLess(v4.plan_speed(o),11.)
        self.assertGreater(v3.plan_speed(o),40.)

    def test_first_false_extrapolated_curve_no_longer_caps_the_straight(self):
        o=observation(190.,15.)
        self.assertGreater(v4.plan_speed(o),v3.plan_speed(o)+2.)

    def test_side_lane_reduces_tight_corner_speed(self):
        o=observation(234.,8.)
        self.assertLess(v4.plan_speed(o,3.2),v4.plan_speed(o,0.))

    def test_wet_and_worn_reduce_speed(self):
        o=observation(234.,8.,0.);dry=v4.plan_speed(o)
        o['weather']['track_wetness']=1.;wet=v4.plan_speed(o)
        o['self']['tyre_wear']=[.8]*4
        self.assertLess(wet,dry);self.assertLess(v4.plan_speed(o),wet)

    def test_road_edge_recovery_is_below_old_15mps_cap(self):
        o=observation(258.,15.);o['self'].update(n=8.6,heading_error=.21)
        self.assertLessEqual(v4.plan_speed(o),6.)
        a=v4.Driver().act(o);self.assertEqual(a['throttle'],0.);self.assertGreater(a['brake'],0.)

    def test_spin_and_low_fuel_keep_limits(self):
        o=observation();o['self']['heading_error']=.8
        self.assertLessEqual(v4.plan_speed(o),5.)
        o=observation(1000.,20.);o['self']['fuel_kg']=3.
        self.assertLessEqual(v4.plan_speed(o),27.)

    def test_resistance_is_compensated_at_target(self):
        o=observation(1000.,30.,0.);state=v4.map_state(o)
        # Isolate resistance from map grade for this actuator regression.
        state=(state[0],state[1],(0.,)*len(state[2]),state[3])
        throttle,brake=v4.pedals(o,30.,0.,state,1)
        self.assertGreater(throttle,0.);self.assertEqual(brake,0.)
        self.assertEqual(.25*(30.-o['self']['speed_mps']),0.)

    def test_uphill_requires_more_propulsion_than_downhill(self):
        o=observation(1000.,25.,0.);state=v4.map_state(o)
        uphill=(state[0],state[1],(.1,)*len(state[2]),state[3])
        downhill=(state[0],state[1],(-.1,)*len(state[2]),state[3])
        self.assertGreater(v4.pedals(o,25.,0.,uphill,1)[0],v4.pedals(o,25.,0.,downhill,1)[0])

    def test_sliding_reduces_power_request(self):
        o=observation(1000.,25.,0.);s=v4.map_state(o)
        good=v4.pedals(o,35.,3.,s,1)[0]
        o['self']['rear_sliding']=True
        bad=v4.pedals(o,35.,3.,s,1)[0]
        self.assertLessEqual(bad,.25);self.assertLess(bad,good)

    def test_missing_force_telemetry_uses_finite_fallback(self):
        for force in ([],[0.]*4):
            o=observation();o['self']['drive_force_n']=force
            a,b=v4.pedals(o,20.,1.,v4.map_state(o),1)
            self.assertTrue(math.isfinite(a+b));self.assertFalse(a>0 and b>0)

    def test_pits_and_dry_compound_rule_preserved(self):
        o=observation(6500.,20.,0.);self.assertTrue(v4.Driver().act(o)['pit_request'])
        o['self']['pit_state']=1;a=v4.Driver().act(o)
        self.assertEqual((a['throttle'],a['brake'],a['steer']),(0.,0.,0.))

    def test_traffic_blocks_acceleration_feedforward(self):
        o=observation(1000.,35.,0.);o['rivals']=[dict(seat=2,status='racing',gap_m=12.,dn=0.,speed_mps=15.)]
        d=v4.Driver();a=d.act(o)
        self.assertEqual(d.diagnostic['feed_accel'],0.);self.assertEqual(a['throttle'],0.);self.assertGreater(a['brake'],0.)

    def test_grid_lanes_and_adjacent_merge_guard_preserved(self):
        o=observation(0.,0.,1.);o['self']['n']=3.5
        d=v4.Driver();d.act(o);self.assertEqual(d.offset,3.5)
        o['rivals']=[dict(seat=2,status='racing',gap_m=0.,dn=-3.5,speed_mps=20.)]
        self.assertFalse(d.clear_sweep(o,0.,o['rivals']))

    def test_phase_change_resets_pass_and_wet_latch(self):
        o=observation();d=v4.Driver();d.act(o);d.pass_seat=3
        o['phase']='qualifying';d.act(o)
        self.assertIsNone(d.pass_seat);self.assertFalse(d.wet_race)

    def test_design_bytes_preserved(self):
        self.assertEqual((ROOT/'agent/design.json').read_bytes(),(ROOT/'baselines/v3/design.json').read_bytes())

    def test_full_output_is_native_json_and_input_unchanged(self):
        o=observation();original=copy.deepcopy(o);a=v4.Driver().act(o)
        json.dumps(a,allow_nan=False);self.assertEqual(o,original)
        self.assertEqual(set(a),{'throttle','brake','steer','ers_mode','aero_mode','pit_request','pit_compound'})

    def test_v4_can_pass_and_return_in_declared_traffic_model(self):
        for kw in ({}, {'half_width':5.}, {'initial_gap':8., 'seconds':60.}, {'curve':.0008}):
            r=traffic.traffic_scenario(v4,**kw)
            self.assertTrue(r['passed'],kw);self.assertTrue(r['returned'],kw);self.assertFalse(r['collision'],kw)

    def test_tight_bends_across_weather_have_geometric_margin_in_declared_model(self):
        for track in ('parkland','highland','crossover'):
            for wet in (0.,.56,1.):
                r=feedback.scenario(v4,track,wet,lateral=10)
                self.assertTrue(r['completed'],(track,wet));self.assertGreater(r['min_road_margin'],2.,(track,wet))

    def test_hairpin_controller_delay_sensitivity(self):
        for lag in (.06,.12,.25):
            r=feedback.scenario(v4,'highland',.56,lateral=10,lag=lag)
            self.assertTrue(r['completed']);self.assertGreater(r['min_road_margin'],2.)


if __name__=='__main__':unittest.main()
