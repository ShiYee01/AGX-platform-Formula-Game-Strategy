"""Declared bicycle/traction sensitivity model, NOT the AGX engine.
Wheelbase 3.4 m agrees with stable seat-19 telemetry; lag, force curves,
traction limits and drag are chosen sensitivity assumptions, not an engine fit.
No model time is a predicted official lap time; no collisions/pits/rival AI.
"""
import json,math,importlib.util
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
v3=load(R/'baselines/v3/agent.py','v3');v4=load(R/'baselines/v4/agent.py','v4')

def scenario(mod,track='highland',wet=.56,start=None,end=None,v0=30,lateral=12,lag=.12,lane=0):
 d=json.loads((R/f'data/racing-{track}.mapdata.json').read_text());pts=d['centerline'];length=d['length_m'];N=len(pts)
 def at(key,s):
  q=(s%length)*N/length;i=int(q);return pts[i][key]+(pts[(i+1)%N][key]-pts[i][key])*(q-i)
 if start is None:
  peak=max(pts,key=lambda p:abs(p['curvature_1pm']))['s_m'];start=peak-150;end=peak+180
 if end is None:end=start+1000
 s=start;n=lane;h=0.;v=v0;steer=0.;yaw=0.;t=0.;mass=950.;grip=1-.45*wet
 driver=mod.Driver();mx=0;mh=0;rows=[];sl=False;rs=False;minmargin=99
 # Start with normal launched race state rather than inventing a grid at this bend.
 while s<end and t<600:
  k=at('curvature_1pm',s);grade=at('grade_dz_ds',s);drag=.53*v*v;roll=140
  drive=min(24000,330000/max(5,v));force=[drive,drive,1.5*drive,2*drive]
  o=dict(phase='race',time=100+t,track_length_m=length,laps_total=4,
   self=dict(n=n,heading_error=h,speed_mps=v,s=s%length,distance=s,tyre_wear=[0]*4,tyre_temp_c=[80]*4,fuel_kg=58,compound='soft',ers_soc_mj=3.,pit_state=0,pit_stops=0,lap=1,front_sliding=sl,rear_sliding=rs,yaw_rate=yaw,steer=steer,g_lat=v*yaw/9.81,drag_n=drag,rolling_resistance_n=roll,effective_mass_kg=mass,drive_force_n=force,brake_force_max_n=56000),
   weather=dict(condition='wet' if wet>.5 else 'dry',track_wetness=wet),rivals=[],preview=dict(rows=[[float(x),at('curvature_1pm',s+x),at('grade_dz_ds',s+x),at('half_width_left_m',s+x),at('half_width_right_m',s+x),0] for x in range(15,361,15)]))
  if t==0:
   driver.reset_phase(o);driver.start_distance=s-1000;driver.start_time=0;driver.grid_n=driver.offset=lane
  a=driver.act(o)
  rows.append(dict(t=t,s=s,n=n,h=h,v=v,target=driver.diagnostic['target'],throttle=a['throttle'],brake=a['brake'],steer=a['steer']))
  for sub in range(2):
   dt=1/60;steer+=min(1,dt/lag)*(.3*a['steer']-steer)
   desired_yaw=v/3.4*math.tan(steer);latcap=lateral*grip
   yawlimit=latcap/max(v,1)
   limited=max(-yawlimit,min(yawlimit,desired_yaw));sl=abs(desired_yaw)>yawlimit*1.02
   yaw+=min(1,dt/.06)*(limited-yaw)
   request=(a['throttle']*force[a['ers_mode']]-a['brake']*56000)/mass
   longcap=math.sqrt(max(.25,(14*grip)**2-(v*yaw)**2))
   rs=request>longcap*1.02
   acc=max(-longcap,min(longcap,request))-(drag+roll)/mass-9.81*grade
   v=max(0,v+acc*dt)
   n+=v*math.sin(h)*dt
   s+=v*math.cos(h)/max(.35,1-k*n)*dt
   h=math.atan2(math.sin(h+(yaw-v*k*math.cos(h)/max(.35,1-k*n))*dt),math.cos(h+(yaw-v*k*math.cos(h)/max(.35,1-k*n))*dt))
   mx=max(mx,abs(n));mh=max(mh,abs(h));minmargin=min(minmargin,at('half_width_left_m',s)-n,at('half_width_right_m',s)+n)
  t+=1/30
  if minmargin<0 or v<.05 and t>10:break
 return dict(track=track,wet=wet,lat=lateral,lag=lag,initial_lane=lane,completed=s>=end,seconds=t,max_n=mx,max_heading=mh,min_road_margin=minmargin,end_s=s,rows=rows)
