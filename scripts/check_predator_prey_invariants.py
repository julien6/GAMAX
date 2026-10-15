"""Check ecological bounds, native identities, references and lifecycle."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def check(frames,identities):
    if not frames or len(frames)!=len(identities):raise ValueError('Missing frames/identities')
    checked=0;seen={owner:set() for owner in ('prey','predator')};retired={owner:set() for owner in seen};largest={owner:-1 for owner in seen}
    grid=identities[0]['vegetation_cell']
    original=np.asarray(frames[0]['vegetation_cell.food_prod'])
    for step,state in enumerate(frames):
        if identities[step]['vegetation_cell']!=grid or identities[step]['generic_species']:raise ValueError('Grid/abstract population changed')
        food=np.asarray(state['vegetation_cell.food']);production=np.asarray(state['vegetation_cell.food_prod'])
        if not np.isfinite(food).all() or np.any(food<0) or np.any(food>np.asarray(state['vegetation_cell.max_food'])):raise ValueError('Ecological food bound')
        if not np.array_equal(production,original) or np.any(production<0):raise ValueError('Production changed')
        for owner,global_count in [('prey','nb_preys'),('predator','nb_predators')]:
            ids=identities[step][owner];current=set(ids)
            if ids!=sorted(current) or len(ids)!=state['global.'+global_count]:raise ValueError('Identity/count invariant')
            if current & retired[owner]:raise ValueError('Dead native identity was reused')
            births=current-seen[owner]
            if step and births and min(births)<=largest[owner]:raise ValueError('Birth identities are not monotonic')
            if births:largest[owner]=max(largest[owner],max(births))
            if step:retired[owner].update(set(identities[step-1][owner])-current)
            seen[owner].update(current)
            energy=np.asarray(state[owner+'.energy'])
            if not np.isfinite(energy).all() or np.any(energy<=0) or np.any(energy>np.asarray(state[owner+'.max_energy'])):raise ValueError('Alive energy bound')
            cells=state[owner+'.my_cell']
            if any(cell not in grid for cell in cells):raise ValueError('Dangling spatial reference')
            points=np.asarray(state[owner+'.location']).reshape((-1,3))
            expected=np.asarray(state['vegetation_cell.location'])[cells].reshape((-1,3))
            if not np.array_equal(points,expected):raise ValueError('Cell/location consistency')
            checked+=len(ids)
    return dict(passed=True,frames=len(frames),agents_checked=checked)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('campaign',type=Path);a=parser.parse_args()
    summary=json.loads((a.campaign/'summary.json').read_text())
    if not summary.get('complete') or not summary.get('passed'):raise ValueError('Complete passing campaign required')
    reports=[]
    for record in summary['reports']:
        if record['id']!='predator_prey':raise ValueError('Predator-prey-only campaign required')
        directory=a.campaign/'predator_prey'/str(record['seed'])
        for engine in ('gama','jax'):
            paths=[directory/(engine+'_trace.json'),directory/(engine+'_identities.json')]
            result=check(*[json.loads(path.read_text()) for path in paths])
            reports.append(dict(seed=record['seed'],engine=engine,hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},**result))
    result=dict(passed=all(record['passed'] for record in reports),reports=reports)
    (a.campaign/'invariants.json').write_text(json.dumps(result,indent=2)+'\n');print('Predator-prey invariants',result['passed'],flush=True)

if __name__=='__main__':main()
