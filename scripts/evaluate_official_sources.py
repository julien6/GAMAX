#!/usr/bin/env python3
"""Run untouched official sources natively and record honest compiler coverage separately."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamax.compiler import compile_model, CompileError, implementation_identity
from gamax.native_server import native_server

EXPERIMENTS = dict(schelling='schelling', predator_prey='prey_predator', ants='With Inspector', boids='Basic', sir_spatial='Simulation', traffic='NetlogoTrafficmodel')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, default=ROOT/'benchmarks/corpus-v0.3')
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1: parser.error('Positive step count required')
    output = args.output.resolve()
    if output.exists(): parser.error('Choose a new evidence directory')
    output.mkdir(parents=True)
    # Retain source bytes, relative imports and assets in the native workload archive.
    shutil.copytree(args.corpus/'official', output/'official')
    records = json.loads((args.corpus/'coverage.json').read_text())['cases']
    sources = [item for item in records if 'source' in item]
    protocol = dict(classification='native smoke validation of complete official sources plus compiler coverage; no cross-engine equivalence or speed claim for rejected sources',
        steps=args.steps, experiments=EXPERIMENTS, source_hashes={item['id']: item['sha256'] for item in sources},
        compiler_sha256=hashlib.sha256((ROOT/'gamax/compiler.py').read_bytes()).hexdigest(), implementation_sha256=implementation_identity()[1],
        seed_policy='unchanged official default configuration; actual native seed/rng observed, not a paired-seed experiment',
        rendering='native server headless; no GUI timing')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    implementation = output/'implementation'; implementation.mkdir()
    for relative in ['scripts/evaluate_official_sources.py', *[str(p.relative_to(ROOT)) for p in sorted((ROOT/'gamax').glob('*.py'))]]:
        target=implementation/relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/relative,target)
    reports=[]
    with native_server(ROOT/'.tools/gama', output/'native') as client:
        for item in sources:
            source=output/item['source']; directory=output/item['id']; directory.mkdir()
            report=dict(id=item['id'], source=item['source'], sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            try:
                _,manifest=compile_model(source,directory/'generated')
                report.update(gamax='accepted_unvalidated',manifest=manifest)
            except CompileError as error:
                report.update(gamax='rejected',diagnostic=error.diagnostic)
            exp=None
            try:
                exp=client.command('load',model=str(source),experiment=EXPERIMENTS[item['id']],console=False,runtime=True)
                species=re.findall(r'^species\s+(\w+)',source.read_text(),re.M)
                entries=[f"'{name}' :: length({name})" for name in species]
                expr="['cycle' :: simulation.cycle, 'seed' :: string(simulation.seed), 'rng' :: simulation.rng, 'populations' :: ["+', '.join(entries)+']]'
                observations=[]
                for tick in range(args.steps+1):
                    if tick: client.command('step',exp_id=exp,nb_step=1,sync=True)
                    observations.append(client.expression(exp,expr))
                (directory/'native_observations.json').write_text(json.dumps(observations,indent=2)+'\n')
                report.update(native='passed',steps=args.steps,initial=observations[0],final=observations[-1])
            except Exception as error: report.update(native='failed',error=str(error))
            finally:
                if exp is not None:
                    try: client.command('stop',exp_id=exp)
                    except Exception as error: report['stop_error']=str(error)
            reports.append(report)
            summary=dict(complete=len(reports)==len(sources),native_passed=sum(r['native']=='passed' for r in reports),
                gamax_accepted=sum(r['gamax']!='rejected' for r in reports),paired_validated=0,reports=reports)
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(f"{item['id']}: native={report['native']} gamax={report['gamax']}",flush=True)

if __name__=='__main__':main()
