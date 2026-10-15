"""Check stable populations, heading/speed bounds and displacement in full Boids traces."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def check(frames,identities):
    if not frames or len(frames)!=len(identities):raise ValueError('Missing frames or identities')
    checked=0
    original=identities[0]
    for step,state in enumerate(frames):
        if identities[step]!=original:raise ValueError('Unexpected lifecycle or identity change')
        for species,count in [('boids',state['global.number_of_agents']),('obstacle',state['global.number_of_obstacles']),('boids_goal',1)]:
            ids=identities[step][species]
            if len(ids)!=count or len(set(ids))!=count:raise ValueError('Population/identity invariant')
            locations=np.asarray(state[species+'.location'],dtype=np.float64).reshape((-1,3))
            headings=np.asarray(state[species+'.heading'])
            speeds=np.asarray(state[species+'.real_speed'])
            if not np.isfinite(locations).all() or np.any(locations[:,2]!=0):raise ValueError('Invalid spatial state')
            if np.any((headings<0)|(headings>=360)) or not np.isfinite(headings).all():raise ValueError('Heading normalization')
            if np.any(speeds<0) or not np.isfinite(speeds).all():raise ValueError('Invalid actual speed')
            checked+=count
        if any(speed>state['global.maximal_speed'] for speed in state['boids.speed']):raise ValueError('Declared speed limit')
        if any(radius!=2*state['global.minimal_distance'] for radius in state['boids.range']):raise ValueError('Per-agent interaction range')
        if step:
            positions=np.asarray(state['boids.location']);previous=np.asarray(frames[step-1]['boids.location'])
            expected=positions-previous
            for axis,minimum,maximum in [(0,'xmin','xmax'),(1,'ymin','ymax')]:
                expected[:,axis]+=np.where(positions[:,axis]<state['global.'+minimum],state['global.bounds'],np.where(positions[:,axis]>state['global.'+maximum],-state['global.bounds'],0))
            if not np.allclose(expected,state['boids.velocity'],atol=1e-12,rtol=1e-10):raise ValueError('Displacement/bounding velocity')
    return dict(passed=True,frames=len(frames),agents_checked=checked)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path);a=p.parse_args()
    summary=json.loads((a.campaign/'summary.json').read_text())
    if not summary.get('complete') or not summary.get('passed'):raise ValueError('Complete passing coupled campaign required')
    reports=[]
    for record in summary['reports']:
        if record['id']!='boids':raise ValueError('Boids-only campaign required')
        directory=a.campaign/'boids'/str(record['seed'])
        for engine in ('gama','jax'):
            paths=[directory/(engine+'_trace.json'),directory/(engine+'_identities.json')]
            result=check(*[json.loads(path.read_text()) for path in paths])
            reports.append(dict(seed=record['seed'],engine=engine,hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},**result))
    result=dict(passed=all(record['passed'] for record in reports),reports=reports)
    (a.campaign/'invariants.json').write_text(json.dumps(result,indent=2)+'\n');print('Boids invariants',result['passed'],flush=True)

if __name__=='__main__':main()
