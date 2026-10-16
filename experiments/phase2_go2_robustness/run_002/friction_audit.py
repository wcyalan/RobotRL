"""Read active Newton and MuJoCo Warp contact parameters, without altering them."""
import numpy as np
import inspect
from pathlib import Path

def snapshot(base):
    from isaaclab_newton.physics.newton_manager import NewtonManager
    import isaaclab.envs.mdp.events as events
    Path('runtime_material_events.py').write_text(inspect.getsource(events), encoding='utf-8')
    model = NewtonManager.get_model()
    solver = NewtonManager._solver
    Path('runtime_solver.py').write_text(inspect.getsource(type(solver)), encoding='utf-8')
    def array(x):
        return x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
    result = {'solver_type':str(type(solver)), 'arrays':{}}
    for label, obj, attrs in [
        ('newton',model,['shape_material_mu','shape_body']),
        ('mjw_model',solver.mjw_model,['geom_friction','geom_priority','geom_solmix']),
        ('mjw_data',solver.mjw_data,['ncon']),
        ('contact',solver.mjw_data.contact,['friction','geom','dim'])]:
        for attr in attrs:
            if not hasattr(obj,attr): continue
            a=array(getattr(obj,attr))
            result['arrays'][label+'.'+attr]={'shape':list(a.shape),'unique':np.unique(a).tolist()[:40]}
            if label=='contact' and attr=='friction':
                result['arrays'][label+'.'+attr]['unique_rows']=np.unique(a.reshape(-1,5),axis=0).tolist()[:40]
    return result
