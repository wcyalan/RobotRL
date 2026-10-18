"""Shared predeclared Go2 behavior acceptance criteria."""
import numpy as np

def assess_behavior(data, profile):
    assert data["step_dt"] == .02 and len(data["episodes"]) == 32
    samples=data['behavior_samples']
    t=np.array([s['time_s'] for s in samples])
    active=np.array([s['active'] for s in samples])
    v=np.array([s['linear_b'] for s in samples])
    yaw=np.array([s['yaw_rate'] for s in samples])
    cmd=np.array([s['command'] for s in samples])
    tilt=np.degrees(np.arccos(np.clip(np.array([s['upright_cosine'] for s in samples]),-1,1)))
    xy=np.linalg.norm(v[:,:,:2]-cmd[:,None,:2],axis=-1)
    speed=np.linalg.norm(v[:,:,:2],axis=-1)
    yaw_error=np.abs(yaw-cmd[:,None,2])
    if profile=='stop_restart':
        assert np.allclose(cmd[:,0],np.where((t//5).astype(int)%2==0,.5,0))
        assert np.allclose(cmd[:,1:],0)
        windows=[(1,5,.15),(6,10,.10),(11,15,.15),(16,20,.10)]
    else:
        assert np.allclose(cmd,np.array([0,0,.5 if profile=='turn_left' else -.5]))
        windows=[(1,20,.10)]
    rows=[]
    for env,episode in enumerate(data['episodes']):
        passed=episode['time_out'] and not episode['terminated'] and episode['length_steps']==1000
        row={'env':env,'windows':[],'stop_response_seconds':[]}
        for start,end,threshold in windows:
            mask=(t>=start)&(t<end)&active[:,env]
            count=int(mask.sum());expected=round((end-start)/data['step_dt'])
            e=float(xy[mask,env].mean()) if count else None
            y=float(yaw_error[mask,env].mean()) if count else None
            a=float(tilt[mask,env].max()) if count else None
            ok=count==expected and e<=threshold and y<=.15 and a<=45
            passed=passed and ok
            row['windows'].append({'start':start,'end':end,'samples':count,'xy_error':e,'yaw_error':y,'max_tilt_degrees':a,'passed':bool(ok)})
        if profile=='stop_restart':
            for start in [5,15]:
                response=None
                for i in np.flatnonzero((t>=start)&(t<=start+2+1e-8)):
                    j=i+25
                    if j<=len(t) and t[j-1]<start+5 and np.all(active[i:j,env]) and np.all(speed[i:j,env]<=.1) and np.all(np.abs(yaw[i:j,env])<=.15):
                        response=round(float(t[i]-start),3);break
                row['stop_response_seconds'].append(response)
                passed=passed and response is not None
        row['passed']=bool(passed);rows.append(row)
    passed_count=sum(r['passed'] for r in rows)
    return {"passed_episodes":passed_count,"episodes":32,"passed":passed_count>=29,
            "failed_envs":[r["env"] for r in rows if not r["passed"]],"rows":rows}
