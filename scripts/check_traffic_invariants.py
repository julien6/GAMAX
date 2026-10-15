"""Check the official one-road model without assuming unique cell occupancy."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def check(frames,identities):
    if not frames or len(frames)!=len(identities):raise ValueError('Missing states/identities')
    original=identities[0];checked=0
    for step,state in enumerate(frames):
        ids=identities[step]['car'];count=state['global.nb_cars']
        if identities[step]!=original or len(ids)!=count or len(set(ids))!=count:raise ValueError('Population/identity invariant')
        width=state['global.pavement_width']*2.;y=(state['global.y_road']+0.5)*2.
        positions=np.asarray(state['car.location']);speed=np.asarray(state['car.speed'])
        if not np.isfinite(positions).all() or np.any(positions[:,0]<0) or np.any(positions[:,0]>width) or np.any(positions[:,1]!=y) or np.any(positions[:,2]!=0):raise ValueError('Road/topology invariant')
        if np.any(speed<np.asarray(state['car.speed_min'])) or np.any(speed>np.asarray(state['car.speed_limit'])):raise ValueError('Speed bounds')
        if any(h!=0 for h in state['car.heading']):raise ValueError('Native toroidal heading preservation')
        sample=state['global.sample_car']
        if sample not in ids:raise ValueError('Sample reference')
        for index,color,icon,cell in zip(ids,state['car.color'],state['car.icon'],state['car.my_pavement']):
            red=index==sample
            if color!=([255,0,0,255] if red else [0,0,255,255]):raise ValueError('Selected car color')
            if icon!=state['global.'+('voit_red_image_file' if red else 'voit_image_file')]:raise ValueError('Selected image identity')
            if state['pavement.grid_y'][cell]!=state['global.y_road']:raise ValueError('Agent/cell road association')
        if step:
            previous=np.asarray(frames[step-1]['car.location'])
            delta=positions[:,0]-previous[:,0]
            corrected=np.where(delta<0,delta+width,delta)
            if not np.allclose(corrected,speed,atol=1e-12,rtol=1e-10):raise ValueError('Displacement/speed invariant')
            if not np.allclose(np.abs(delta),state['car.real_speed'],atol=1e-12,rtol=1e-10):raise ValueError('Native actual speed including wrapping')
        checked+=count
    return dict(passed=True,frames=len(frames),agents_checked=checked)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args()
    summary=json.loads((a.campaign/'summary.json').read_text())
    if not summary.get('complete') or not summary.get('passed'):raise ValueError('Complete passing coupled campaign required')
    reports=[]
    for record in summary['reports']:
        if record['id']!='traffic':raise ValueError('Traffic-only campaign required')
        directory=a.campaign/'traffic'/str(record['seed'])
        for engine in ('gama','jax'):
            paths=[directory/(engine+'_trace.json'),directory/(engine+'_identities.json')]
            result=check(*[json.loads(path.read_text()) for path in paths])
            reports.append(dict(seed=record['seed'],engine=engine,hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},**result))
    result=dict(passed=all(record['passed'] for record in reports),reports=reports)
    (a.campaign/'invariants.json').write_text(json.dumps(result,indent=2)+'\n');print('Traffic invariants',result['passed'],flush=True)

if __name__=='__main__':main()
