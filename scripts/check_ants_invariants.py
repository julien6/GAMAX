"""Check native identities, food conservation, FSM and spatial bounds for foraging."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def check(frames,identities):
    if not frames or len(frames)!=len(identities):raise ValueError('Missing frames/identities')
    original=frames[0];fixed=identities[0];checked=0
    for state,ids in zip(frames,identities):
        if ids!=fixed:raise ValueError('Fixed populations/identities changed')
        if len(ids['ant'])!=state['global.ants_number']:raise ValueError('Ant count differs')
        if len(ids['ant_grid'])!=state['global.gridsize']**2:raise ValueError('Grid size differs')
        for key in ('global.food_placed','global.grid_values','ant_grid.is_nest','ant_grid.location'):
            if state[key]!=original[key]:raise ValueError('Immutable field changed: '+key)
        food=np.asarray(state['ant_grid.food'])
        if np.any(food<0) or np.any(food>5):raise ValueError('Food bounds')
        if int(food.sum())+sum(state['ant.has_food'])+state['global.food_gathered']!=state['global.food_placed']:raise ValueError('Food conservation')
        road=np.asarray(state['ant_grid.road'])
        if not np.isfinite(road).all() or np.any(road<0) or np.any(road>240):raise ValueError('Pheromone bounds')
        points=np.asarray(state['ant.location']);heading=np.asarray(state['ant.heading'])
        if not np.isfinite(points).all() or np.any(points[:,:2]<0) or np.any(points[:,:2]>state['global.gridsize']) or np.any(points[:,2]!=0):raise ValueError('Spatial bounds')
        if not np.isfinite(heading).all() or np.any(heading<0) or np.any(heading>=360):raise ValueError('Heading bounds')
        if any(value not in ('wandering','carryingFood','followingRoad') for value in state['ant.state']):raise ValueError('Unknown FSM state')
        colors=np.asarray(state['ant_grid.color'])
        if np.any(colors<0) or np.any(colors>255):raise ValueError('Color bounds')
        checked+=len(ids['ant'])
    return dict(passed=True,frames=len(frames),agents_checked=checked)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('campaign',type=Path);a=parser.parse_args()
    summary=json.loads((a.campaign/'summary.json').read_text())
    if not summary.get('complete') or not summary.get('passed'):raise ValueError('Complete passing campaign required')
    reports=[]
    for record in summary['reports']:
        if record['id']!='ants':raise ValueError('Ants-only campaign required')
        directory=a.campaign/'ants'/str(record['seed'])
        for engine in ('gama','jax'):
            paths=[directory/(engine+'_trace.json'),directory/(engine+'_identities.json')]
            result=check(*[json.loads(path.read_text()) for path in paths])
            reports.append(dict(seed=record['seed'],engine=engine,hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},**result))
    result=dict(passed=all(record['passed'] for record in reports),reports=reports)
    (a.campaign/'invariants.json').write_text(json.dumps(result,indent=2)+'\n');print('Ants invariants',result['passed'],flush=True)

if __name__=='__main__':main()
