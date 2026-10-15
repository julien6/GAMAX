#!/usr/bin/env python3
"""Measure each already-validated family using isolated native/JAX workers."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    validation = args.validation.resolve()
    summary = json.loads((validation/'summary.json').read_text())
    if not summary['complete'] or not summary['passed']:
        raise SystemExit('Fidelity/invariants/controls must pass before benchmarking the suite')
    selected = [r for r in summary['reports'] if 'control' in r]
    if not selected: raise SystemExit('No validated control scenarios found')
    if args.steps > summary['steps']: raise SystemExit('Benchmark horizon cannot exceed the validated horizon')
    if args.output.exists(): raise SystemExit('Choose a new output directory')
    args.output.mkdir(parents=True)
    protocol = dict(validation_summary_sha256=hashlib.sha256((validation/'summary.json').read_bytes()).hexdigest(),
                    validation=str(validation), steps=args.steps, stride=args.stride, repeats=args.repeats,
                    scenarios=[dict(case=r['case'],seed=r['seed'],source_sha256=r['source_sha256']) for r in selected],
                    scope='exploratory fixed validated scenario per family; timing repetitions are not stochastic seeds',
                    comparison='fresh process per engine/mode; common offscreen renderer; no native GUI or energy claim')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results = []
    for report in selected:
        case = report['case']
        command = [sys.executable,str(Path(__file__).with_name('benchmark_rendering.py')),
                   '--source',str(validation/report['source']), '--steps',str(args.steps),
                   '--stride',str(args.stride), '--repeats',str(args.repeats),'-o',str((args.output/case).resolve())]
        with (args.output/(case+'.log')).open('w') as log:
            status = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        path = args.output/case/'summary.json'
        child = json.loads(path.read_text()) if path.exists() else {}
        results.append(dict(case=case,seed=report['seed'],returncode=status,passed=status==0 and child.get('passed',False),
                            ratios=child.get('gama_over_jax_ratios'),error=None if status==0 else 'See per-family log/failure'))
        final = dict(complete=len(results)==len(selected),passed=len(results)==len(selected) and all(r['passed'] for r in results),results=results)
        (args.output/'summary.json').write_text(json.dumps(final,indent=2)+'\n')
        print(json.dumps(results[-1]),flush=True)
    return 0 if final['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
