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
    contact = solver.mjw_data.contact
    count = int(array(solver.mjw_data.nacon).reshape(-1)[0])
    pairs = array(contact.geom)
    assert 0 <= count <= len(pairs), 'Contact buffer overflow'
    friction = array(contact.friction)
    dims = array(contact.dim)
    distances = array(contact.dist)
    margins = array(contact.includemargin)
    types = array(solver.mjw_model.geom_type).reshape(-1)
    worlds = array(contact.worldid)
    rows = []
    for i in range(count):
        a,b = map(int,pairs[i])
        if a < 0 or b < 0 or dims[i] <= 0 or distances[i] >= margins[i]:
            continue
        is_ground = int(types[a]) == 0 or int(types[b]) == 0  # MuJoCo plane
        rows.append({'world':int(worlds[i]),'geom_pair':[a,b],
                     'geom_names':[solver.mj_model.geom(a).name,solver.mj_model.geom(b).name],
                     'body_names':[solver.mj_model.body(int(solver.mj_model.geom_bodyid[g])).name for g in [a,b]],
                     'ground_contact':is_ground,'sliding_friction':float(friction[i,0]),
                     'distance':float(distances[i]),'include_margin':float(margins[i])})
    ground = [r for r in rows if r['ground_contact']]
    result['active_contacts'] = {'nacon':count,'constraint_eligible_count':len(rows),
                                 'ground_count':len(ground),
                                 'ground_sliding_friction':sorted({r['sliding_friction'] for r in ground}),
                                 'rows':rows}
    assert ground, 'No active ground contacts in audit snapshot'
    return result
