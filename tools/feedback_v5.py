"""Explicit acceleration/lapping sensitivity models, not the AGX engine.
The opponent follows a declared constant-speed/constant-lane path; it does
not react to us. Tests cover our clearance/merging decisions, not real AI.
"""
import importlib.util
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def lapping_scenario(module, wet=1.0, gap=-300., rival_speed=70., rival_lane=0.,
                     half_width=8., same_lap=False, blockers=(), seconds=35.):
    length,curves,grades=module.MAP_DATA[0]
    at=lambda v,s:module.interpolate(v,s,length)
    station=100.;rival_station=station+gap;speed=20.;n=heading=steer=yaw=0.
    driver=module.Driver();grip=1-.45*wet;mass=950.;sl=rs=False
    samples=[];max_n=0.;collision=False;passed=returned=False;first_move=None
    for step in range(round(seconds*30)):
        t=step/30.;k=at(curves,station);grade=at(grades,station)
        drive=min(24000.,330000./max(5.,speed));force=[drive,drive,1.5*drive,2*drive]
        drag=.53*speed**2;roll=140.
        gap=rival_station-station
        lane_now = rival_lane(t) if callable(rival_lane) else rival_lane
        rival_lap=int(((length if same_lap else 2*length)+rival_station)//length)
        rivals=[dict(seat=6,status='racing',gap_m=gap,dn=lane_now-n,speed_mps=rival_speed,lap=rival_lap)]
        for seat,lane in enumerate(blockers,10):
            rivals.append(dict(seat=seat,status='racing',gap_m=0.,dn=lane-n,speed_mps=speed,lap=1))
        o=dict(phase='race',time=100.+t,track_length_m=length,laps_total=4,
          self=dict(n=n,heading_error=heading,speed_mps=speed,s=station%length,distance=length+station,
                    fuel_kg=58.,tyre_wear=[0.]*4,tyre_temp_c=[70.]*4,compound='soft',ers_soc_mj=3.,
                    pit_state=0,pit_stops=0,lap=1,front_sliding=sl,rear_sliding=rs,
                    yaw_rate=yaw,steer=steer,g_lat=speed*yaw/9.81,drag_n=drag,
                    rolling_resistance_n=roll,effective_mass_kg=mass,drive_force_n=force,brake_force_max_n=56000.),
          weather=dict(condition='wet' if wet>.5 else 'dry',track_wetness=wet),rivals=rivals,
          preview=dict(rows=[[float(x),at(curves,station+x),at(grades,station+x),half_width,half_width,0] for x in range(15,361,15)]))
        if step==0:
            driver.reset_phase(o);driver.start_time=0.;driver.last_unstable_time=0.;driver.start_distance=o['self']['distance']-500
        a=driver.act(o)
        if abs(gap)<4.5 and abs(n-lane_now)<2.:collision=True
        if gap>20.:passed=True
        if first_move is None and abs(driver.offset)>.05:first_move=t
        if step%3==0:
            samples.append(dict(t=t,s=station,gap=gap,n=n,speed=speed,offset=driver.offset,
                                target=driver.diagnostic['target'],rival_lane=lane_now,brake=a['brake'],throttle=a['throttle'],
                                yield_state=driver.diagnostic.get('yield_state'),yield_seat=driver.diagnostic.get('yield_seat')))
        for _ in range(2):
            dt=1/60.;steer+=(dt/.12)*(.3*a['steer']-steer)
            desired=speed/3.4*math.tan(steer);limit=12*grip/max(speed,1.)
            sl=abs(desired)>limit*1.02;yaw+=(dt/.06)*(max(-limit,min(limit,desired))-yaw)
            traction=math.sqrt(max(.25,(14*grip)**2-(speed*yaw)**2))
            request=(a['throttle']*force[a['ers_mode']]-a['brake']*56000.)/mass
            rs=request>traction*1.02
            accel=max(-traction,min(traction,request))-(drag+roll)/mass-9.81*grade
            speed=max(0.,speed+accel*dt)
            n+=speed*math.sin(heading)*dt
            station+=speed*math.cos(heading)/max(.35,1-k*n)*dt
            heading=module.angle(heading+(yaw-speed*k*math.cos(heading)/max(.35,1-k*n))*dt)
            rival_station+=rival_speed*dt
        max_n=max(max_n,abs(n))
        if passed and abs(n)<.2 and abs(driver.offset)<.05:
            returned=True;break
        if abs(n)>half_width or station>1500:break
    return dict(method='Chosen bicycle/force model, constant-path faster opponent; no official engine.',
                wet=wet,initial_gap=gap if first_move is None else samples[0]['gap'],rival_speed=rival_speed,
                half_width=half_width,collision=collision,passed=passed,returned=returned,
                first_move=first_move,max_n=max_n,seconds=t,samples=samples)


def acceleration_scenario(module, wet=1., grade=0., speed0=100/3.6,
                          target=160/3.6, traction_factor=14.):
    # Straight speed-step isolates longitudinal control from map braking.
    length,curves,grades=module.MAP_DATA[0]
    state=(length,(0.,)*len(curves),(grade,)*len(grades),[])
    speed=speed0;mass=950.;grip=1-.45*wet;slide=False;sliding=0.;peak_oil=0.
    for step in range(1800):
        t=step/30.;drive=min(24000.,330000./max(5.,speed));forces=[drive,drive,1.5*drive,2*drive]
        drag=.53*speed**2
        o=dict(self=dict(speed_mps=speed,n=0.,heading_error=0.,s=500.,fuel_kg=58.,
                        tyre_wear=[0.]*4,front_sliding=False,rear_sliding=slide,g_lat=0.,
                        drive_force_n=forces,effective_mass_kg=mass,drag_n=drag,
                        rolling_resistance_n=140.,brake_force_max_n=56000.),
               weather=dict(track_wetness=wet),preview=dict(rows=[[15.,0.,grade,8.,8.,0]]))
        if module.__name__.startswith('v5'):
            throttle,brake=module.pedals(o,target+1.,0.,state,2,allow_extra=True)
        else:throttle,brake=module.pedals(o,target+1.,0.,state,2)
        request=(throttle*forces[2]-brake*56000.)/mass
        cap=traction_factor*grip;slide=request>cap*1.02
        sliding+=slide/30.;peak_oil=max(peak_oil,throttle)
        speed=max(0.,speed+(max(-cap,min(cap,request))-(drag+140.)/mass-9.81*grade)/30.)
        if speed>=target:break
    return dict(wet=wet,grade=grade,traction_factor=traction_factor,seconds=(step+1)/30.,
                reached=speed>=target,sliding_seconds=sliding,peak_throttle=peak_oil)
