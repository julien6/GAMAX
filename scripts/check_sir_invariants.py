"""Model-specific invariants for the frozen official SIR coupled traces."""
from pathlib import Path
import argparse
import hashlib
import json


def check(frames,identities):
    if len(frames)!=len(identities) or not frames:raise ValueError('Missing frames/identities')
    colors=([46,204,113,255],[231,76,60,255],[52,152,219,255])
    checked=0
    for step,f in enumerate(frames):
        pop=f['global.numberHosts'];ids=identities[step]['Host']
        if len(ids)!=pop or len(set(ids))!=pop:raise ValueError('Population/identity invariant at '+str(step))
        flags=list(zip(f['Host.is_susceptible'],f['Host.is_infected'],f['Host.is_immune']))
        if len(flags)!=pop or any(sum(row)!=1 for row in flags):raise ValueError('S/I/R partition at '+str(step))
        if any(c!=colors[row.index(True)] for c,row in zip(f['Host.color'],flags)):raise ValueError('State/color consistency at '+str(step))
        if any(type(p) is not int or not 0<=p<len(f['sir_grid.location']) or loc!=f['sir_grid.location'][p] for p,loc in zip(f['Host.myPlace'],f['Host.location'])):raise ValueError('Cell/location invariant at '+str(step))
        if step and f['global.number_I']!=sum(frames[step-1]['Host.is_infected']):raise ValueError('Pre-agent global count scheduling at '+str(step))
        checked+=pop
    return {'passed':True,'frames':len(frames),'agents_checked':checked}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('campaign',type=Path)
    a=p.parse_args();summary=json.loads((a.campaign/'summary.json').read_text());reports=[]
    if not summary.get('complete') or not summary.get('passed'):raise ValueError('Complete passing coupled campaign required')
    for record in summary['reports']:
        directory=a.campaign/record['id']/str(record['seed'])
        for engine in ('gama','jax'):
            paths=[directory/(engine+'_trace.json'),directory/(engine+'_identities.json')]
            row=dict(seed=record['seed'],engine=engine,**check(*[json.loads(p.read_text()) for p in paths]))
            row['hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};reports.append(row)
    out=dict(passed=all(r['passed'] for r in reports),reports=reports)
    (a.campaign/'invariants.json').write_text(json.dumps(out,indent=2)+'\n');print('SIR invariants',out['passed'],flush=True)

if __name__=='__main__':main()
