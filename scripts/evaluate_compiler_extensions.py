#!/usr/bin/env python3
"""Freeze extension workloads, then compare complete native/JAX trajectories and controls."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import numpy as np
from gamax.compiler import compile_model, load_model, Generator
from gamax.runtime import load_module, run, export_trace
from gamax.native_server import native_server
from gamax.validation import compare_traces, environment

CASES = {
    'imported_actions': 'examples/imported_actions/main.gaml',
    'moving_skill': 'examples/moving_skill.gaml',
    'stochastic_moving': 'examples/stochastic_moving/main.gaml',
    'control_flow': 'examples/control_flow/main.gaml',
    'random_initialization': 'examples/random_initialization/main.gaml',
    'mersenne_initialization': 'examples/mersenne_initialization/main.gaml',
    'dynamic_loops': 'examples/dynamic_loops/main.gaml',
    'declaration_facets': 'examples/declaration_facets/main.gaml',
    'expression_semantics': 'examples/expression_semantics/main.gaml',
    'creation_batches': 'examples/creation_batches/main.gaml',
    'creation_implicit_location': 'examples/creation_implicit_location/main.gaml',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.steps < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) <= 0:
        parser.error('Positive horizon and distinct positive seeds required')
    output = args.output.resolve()
    if output.exists(): parser.error('Choose a new evidence directory')
    output.mkdir(parents=True)
    workloads = []
    for case in args.cases:
        entry = CASES[case]
        original = ROOT/entry
        for seed in args.seeds:
            directory = output/case/str(seed); directory.mkdir(parents=True)
            if original.name == 'main.gaml':
                for dependency in original.parent.glob('*.gaml'):
                    (directory/dependency.name).write_bytes(dependency.read_bytes())
                source = directory/'main.gaml'
                source.write_text(source.read_text().replace('seed <- 17;', f'seed <- {seed};'))
            else:
                source = directory/original.name
                source.write_text(original.read_text().replace('heading <- index * 45.0;', f'heading <- index * 45.0 + {seed}.0;'))
            workloads.append(dict(case=case, seed=seed, source=str(source.relative_to(output)),
                sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.gaml')}))
    implementation = output/'implementation'
    hashes = {}
    for relative in ['scripts/evaluate_compiler_extensions.py', *[str(p.relative_to(ROOT)) for p in sorted((ROOT/'gamax').glob('*.py'))]]:
        path = ROOT/relative; target = implementation/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target); hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = dict(classification='development validation of authored feature probes; not full official ABM compilation or performance evidence',
        cases=args.cases,
        steps=args.steps, seeds=args.seeds, workloads=workloads,
        randomness='imported_actions and stochastic_moving: genuine runtime Java RNG, reseeded after init in identical GAML; moving_skill: deterministic initial heading configurations',
        tolerance=dict(atol=1e-12, rtol=1e-10), discrete='exact', observation='all declared and supported inherited attributes, after init and every step',
        controls='first seed: reverse agents for imported/stochastic moving/random initialization; mutate speed 2 to 3 for deterministic moving; remove one outer iteration for control_flow; require divergence', environment=environment(), implementation_sha256=hashes)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    reports = []
    def save():
        complete = len(reports) == len(workloads)
        summary = dict(complete=complete, passed=complete and all(r['passed'] for r in reports), planned=len(workloads), reports=reports,
                       compared_values=sum(r.get('comparison',{}).get('compared_values',0) for r in reports),
                       max_float_absolute_error=max((r.get('comparison',{}).get('max_float_absolute_error',0) for r in reports), default=0))
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
    with native_server(ROOT/'.tools/gama', output/'native') as client:
        for item in workloads:
            source = output/item['source']; directory = source.parent
            try:
                module, manifest = compile_model(source, directory/'generated')
                candidate = run(load_module(module), args.steps)
                reference = client.trace(source, manifest, args.steps, 'validation')
                export_trace(candidate, directory, 'gamax'); export_trace(reference, directory, 'gama')
                comparison = compare_traces(reference, candidate, manifest['fields'])
                invariants = []
                for tick, state in enumerate(candidate):
                    if any(not np.isfinite(np.asarray(v)).all() for v in state.values()): invariants.append(dict(tick=tick, rule='finite state'))
                    for key, value in state.items():
                        if key.endswith('.location') and np.asarray(value).size and not np.all((np.asarray(value)[:,:2] >= 0) & (np.asarray(value)[:,:2] <= 100)):
                            invariants.append(dict(tick=tick, rule='default world bounds'))
                control = None
                if item['seed'] == args.seeds[0]:
                    ir, _ = load_model(source)
                    mutant = directory/'reverse_agents.py';  
                    if item['case'] == 'moving_skill':
                        ir.scopes[1].init[1].value.args[0] = 3.0
                    elif item['case'] in ('creation_batches','creation_implicit_location'):
                        next(f for f in ir.scopes[1].fields if f.name == 'value').value.args[0].args[0] = 8
                    elif item['case'] == 'expression_semantics':
                        ir.scopes[0].fields[0].value.args[0] = 3
                    elif item['case'] == 'declaration_facets':
                        ir.scopes[0].fields[0].value.args[0] = 3
                    elif item['case'] == 'dynamic_loops':
                        ir.scopes[0].fields[0].value.args[0] = 4
                    elif item['case'] == 'control_flow':
                        ir.scopes[0].behaviors[0].body[0].value.args.pop()
                    mutant.write_text(Generator(ir, reverse_agents=item['case'] != 'moving_skill').generate())
                    control = compare_traces(reference, run(load_module(mutant), args.steps), manifest['fields'])
                report = dict(case=item['case'], seed=item['seed'], comparison=comparison, invariants=invariants, control=control,
                    passed=comparison['passed'] and not invariants and (control is None or not control['passed']))
            except Exception as error:
                report = dict(case=item['case'], seed=item['seed'], passed=False, error=str(error))
            reports.append(report); save(); print(f"{item['case']} seed={item['seed']} passed={report['passed']}", flush=True)
            jax.clear_caches()
    if not save()['passed']: raise SystemExit(1)

if __name__ == '__main__': main()
