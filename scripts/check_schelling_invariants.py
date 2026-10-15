"""Check occupancy, shared lists, neighborhood membership and pre-agent counts."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def check(frames,identities):
    if not frames or len(frames)!=len(identities):raise ValueError('Missing states or identities')
    checked=0
    for step,state in enumerate(frames):
        count=state['global.number_of_people'];ids=identities[step]['people']
        cells=state['space.location'];places=state['people.my_place'];free=state['global.free_places']
        if len(ids)!=count or len(set(ids))!=count or len(places)!=count:raise ValueError('Population/identity invariant')
        if len(set(places))!=count or len(set(free))!=len(free):raise ValueError('Duplicate occupied/free cell')
        if set(places)&set(free) or set(places)|set(free)!=set(range(len(cells))):raise ValueError('Occupancy partition')
        if free!=state['global.all_places']:raise ValueError('Shared list aliases differ')
        if state['global.all_people']!=ids:raise ValueError('Ordered global population list')
        if any(cells[p]!=loc for p,loc in zip(places,state['people.location'])):raise ValueError('Agent/cell locations differ')
        palette=state['global.colors'][:state['global.number_of_groups']]
        if any(color not in palette for color in state['people.color']):raise ValueError('Invalid group color')
        positions=dict(zip(ids,state['people.location']));colors=dict(zip(ids,state['people.color']))
        radius=state['global.neighbours_distance'];wanted=state['global.percent_similar_wanted']
        coordinates=np.asarray(state['people.location'],dtype=np.float64)
        differences=coordinates[:,None,:]-coordinates[None,:,:]
        neighborhoods=np.sum(differences*differences,axis=2)<=radius*radius
        np.fill_diagonal(neighborhoods,False)
        expected_by_id={index:{ids[j] for j in np.flatnonzero(neighborhoods[i])} for i,index in enumerate(ids)}
        for index,nearby,similar,total,happy in zip(ids,state['people.my_neighbours'],state['people.similar_nearby'],state['people.total_nearby'],state['people.is_happy']):
            expected=expected_by_id[index]
            if len(nearby)!=len(set(nearby)) or set(nearby)!=expected:raise ValueError('Spatial neighborhood set')
            if total!=len(expected) or similar!=sum(colors[index]==colors[other] for other in expected):raise ValueError('Computed counts')
            if happy!=(similar>=wanted*total):raise ValueError('Computed happiness')
        if step:
            previous=frames[step-1]
            expected=(sum(previous['people.is_happy']),sum(previous['people.similar_nearby']),max(1,sum(previous['people.total_nearby'])))
            actual=tuple(state['global.'+name] for name in ('sum_happy_people','sum_similar_neighbours','sum_total_neighbours'))
            if actual!=expected:raise ValueError('Pre-agent global count scheduling')
        checked+=count
    return dict(passed=True,frames=len(frames),agents_checked=checked)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('campaign',type=Path)
    args=parser.parse_args();summary=json.loads((args.campaign/'summary.json').read_text())
    if not summary.get('complete') or not summary.get('passed'):raise ValueError('Complete passing coupled campaign required')
    reports=[]
    for record in summary['reports']:
        if record['id']!='schelling':raise ValueError('Schelling-only invariant campaign required')
        directory=args.campaign/record['id']/str(record['seed'])
        for engine in ('gama','jax'):
            paths=[directory/(engine+'_trace.json'),directory/(engine+'_identities.json')]
            result=check(*[json.loads(path.read_text()) for path in paths])
            reports.append(dict(seed=record['seed'],engine=engine,hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},**result))
    result=dict(passed=all(record['passed'] for record in reports),reports=reports)
    (args.campaign/'invariants.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Schelling invariants',result['passed'],flush=True)

if __name__=='__main__':main()
