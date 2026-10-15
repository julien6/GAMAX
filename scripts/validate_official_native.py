"""Smoke-test unchanged development sources with the seeded native oracle.

No GAMAX compilation or cross-engine equivalence is claimed by this runner.
Reserved variants are not opened or compiled.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gamax.native_ws import observe


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,default=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13')
    parser.add_argument('-o','--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1,11)))
    parser.add_argument('--steps',type=int,default=100)
    args=parser.parse_args()
    registry=json.loads((args.corpus/'manifest.json').read_text())
    cases=[c for c in registry['cases'] if c['role']=='development']
    for case in cases:
        if hashlib.sha256((args.corpus/case['source']).read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Frozen development source changed: '+case['id'])
    for name,digest in registry['assets'].items():
        if hashlib.sha256((args.corpus/'official'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen asset changed: '+name)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__).resolve(),*[(ROOT/'gamax'/name) for name in ('native_ws.py','native_server.py','gama_client.py')]]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for p in files:
        target=out/'implementation'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    protocol=dict(classification='native smoke tests only; no GAMAX acceptance, coupled fidelity or performance result',
                  corpus_manifest_sha256=hashlib.sha256((args.corpus/'manifest.json').read_bytes()).hexdigest(),
                  seeds=args.seeds,steps=args.steps,implementation=hashes,reserved='uncompiled')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    reports=[]
    for case in cases:
        for seed in args.seeds:
            directory=out/case['id']/str(seed);directory.mkdir(parents=True)
            native_tree=directory/'oracle_sources'
            shutil.copytree(args.corpus/'official',native_tree)
            native_source=native_tree/(args.corpus/case['source']).relative_to(args.corpus/'official')
            result=observe(native_source,case['experiment'],
                           {'cycle':'simulation.cycle','seed':'simulation.seed','rng':'simulation.rng'},
                           seed=seed,steps=args.steps,output=directory/'native',home=ROOT/'.tools/gama')
            reports.append(dict(id=case['id'],seed=seed,status=result['status'],snapshots=len(result['trace']),
                                errors=result['errors'],error=result.get('error'),source_sha256=case['sha256']))
            summary=dict(complete=len(reports)==len(cases)*len(args.seeds),passed=all(r['status']=='passed' for r in reports),reports=reports)
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(case['id'],seed,result['status'],result.get('error'),flush=True)
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):
        raise RuntimeError('Native oracle changed during campaign')
    for name,digest in registry['assets'].items():
        if hashlib.sha256((args.corpus/'official'/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen asset changed during campaign: '+name)
    if not all(r['status']=='passed' for r in reports):raise SystemExit(1)

if __name__=='__main__':main()
