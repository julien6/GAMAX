#!/usr/bin/env python3
"""Freeze sampled GAML inputs; compare every step of ten ABM families to native GAMA."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.scenarios import CASES, materialize
from gamax.compiler import compile_model, Parser, Generator
from gamax.native_server import native_server
from gamax.runtime import load_module, run, export_trace
from gamax.validation import compare_traces, environment

def invariant_failures(case, trace):
    failures = []
    def check(condition, step, rule):
        if not condition: failures.append(dict(step=step, rule=rule))
    for step, state in enumerate(trace):
        check(all(np.isfinite(np.asarray(value)).all() for value in state.values()), step, 'finite state')
        for key, value in state.items():
            values = np.asarray(value)
            if key.endswith(('.px', '.py')):
                bound = 7 if case in ('forest_fire','schelling','sir_grid','predator_prey','ants') else (21 if case == 'social_force' else 20)
                check(bool(np.all((values >= 0) & (values <= bound))), step, key + ' bounds')
            if key.endswith(('.status',)):
                check(set(values.tolist()) <= {0, 1, 2}, step, key + ' categories')
        if case == 'schelling':
            check(len(set(zip(state['resident.px'], state['resident.py']))) == 32, step, 'exclusive occupancy')
        if case == 'traffic':
            pos, vel = np.asarray(state['vehicle.position']), np.asarray(state['vehicle.velocity'])
            check(len(set(pos.tolist())) == 24 and bool(np.all((pos >= 0) & (pos < 128))), step, 'collision-free ring positions')
            check(bool(np.all((vel >= 0) & (vel <= 5))), step, 'bounded velocity')
        if case == 'ants': check(min(state['patch.trail']) >= 0, step, 'nonnegative pheromone')
        if step:
            previous = trace[step - 1]
            if case.startswith('sir_'):
                check(bool(np.all(np.asarray(state['person.status']) >= np.asarray(previous['person.status']))), step, 'irreversible SIR transitions')
            if case == 'granovetter':
                check(all(not old or new for old, new in zip(previous['person.adopted'], state['person.adopted'])), step, 'irreversible adoption')
            if case == 'ants':
                check(bool(np.all(np.asarray(state['ant.deliveries']) >= np.asarray(previous['ant.deliveries']))), step, 'monotone deliveries')
    return failures


def corrupt_neighborhoods(ir):
    """One generic IR mutation; no handwritten alternative model implementation."""
    def expression(expr):
        if expr.op == 'where':
            predicate = expr.args[1]
            predicate.op, predicate.args, predicate.type = 'literal', [True], 'bool'
        for arg in expr.args:
            if hasattr(arg, 'op'): expression(arg)
    def statements(items):
        for statement in items:
            if statement.value: expression(statement.value)
            statements(statement.body); statements(statement.alternative)
    for scope in ir.scopes:
        for behavior in scope.behaviors:
            expression(behavior.guard); statements(behavior.body)
    return ir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if args.steps < 1 or len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        parser.error('Positive horizon and distinct nonnegative scenario seeds required')
    output = args.output.resolve()
    if output.exists(): raise SystemExit('Choose a new output directory to preserve previous evidence')
    output.mkdir(parents=True)
    workloads = []
    for case in args.cases:
        for seed in args.seeds:
            directory = output / case / str(seed)
            directory.mkdir(parents=True)
            source = directory / f'{case}.gaml'
            source.write_text(materialize(case, seed))
            workloads.append(dict(case=case, seed=seed, source=str(source.relative_to(output)),
                                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    protocol = dict(classification='development multi-family differential validation; not held-out or performance evidence',
                    steps=args.steps, workloads=workloads, seeds=args.seeds,
                    randomness='NumPy PCG64 samples initialization/network only; frozen identical GAML in both engines; no runtime rnd/flip',
                    floating=dict(atol=1e-12, rtol=1e-10), discrete='exact', observer='every post-init/step state, all declared attributes',
                    controls='seed list first entry of each family: replace all where predicates by true; require divergence',
                    models='authored explicit-coordinate/contact variants, not compilation of the six official rejected sources',
                    environment=environment(), source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__).resolve(), ROOT/'benchmarks/scenarios.py', *sorted((ROOT/'gamax').glob('*.py')), *sorted((ROOT/'benchmarks/models').glob('*.gaml'))]})
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    archive = output / 'implementation'; archive.mkdir()
    for source in protocol['source_hashes']:
        target = archive / source; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    reports = []
    def save():
        complete = len(reports) == len(workloads)
        passed = complete and all(r.get('passed', False) for r in reports)
        summary = dict(complete=complete, passed=passed, planned=len(workloads), executed=len(reports), steps=args.steps,
                       reports=reports, purpose=protocol['classification'])
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return passed
    with native_server(ROOT/'.tools/gama', output/'native') as client:
        for item in workloads:
            case, seed = item['case'], item['seed']
            source = output/item['source']; directory = source.parent
            exp = None
            try:
                module_path, manifest = compile_model(source, directory/'generated')
                exp = client.command('load', model=str(source), experiment='validation', console=False, runtime=True)
                for species, count in manifest['populations'].items():
                    indices = client.expression(exp, f'({species} sort_by (each.index)) collect each.index')
                    if indices != list(range(count)): raise RuntimeError('Native identities do not match fixed-population contract')
                reference = [client.snapshot(exp, manifest)]
                for _ in range(args.steps):
                    client.command('step', exp_id=exp, nb_step=1, sync=True)
                    reference.append(client.snapshot(exp, manifest))
                client.command('stop', exp_id=exp); exp = None
                candidate = run(load_module(module_path), args.steps)
                export_trace(reference, directory, 'gama'); export_trace(candidate, directory, 'jax')
                comparison = compare_traces(reference, candidate, manifest['fields'])
                invariants = dict(gama=invariant_failures(case, reference), jax=invariant_failures(case, candidate))
                report = dict(**item, comparison=comparison, invariants=invariants,
                              passed=comparison['passed'] and not any(invariants.values()))
                if seed == args.seeds[0]:
                    mutant = directory/'generated/neighborhood_mutant.py'
                    mutant.write_text(Generator(corrupt_neighborhoods(Parser(source.read_text(), str(source)).parse())).generate())
                    control = compare_traces(reference, run(load_module(mutant), args.steps), manifest['fields'])
                    report['control'] = dict(mutation='all where predicates true', detected=not control['passed'], comparison=control)
                    report['passed'] &= not control['passed']
            except Exception as error:
                report = dict(**item, passed=False, error=str(error))
            finally:
                if exp is not None:
                    try: client.command('stop', exp_id=exp)
                    except RuntimeError: pass
            (directory/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            reports.append(report); save()
            print(json.dumps(dict(case=case, seed=seed, passed=report['passed'],
                                 comparison=report.get('comparison'), error=report.get('error'))), flush=True)
            jax.clear_caches()
    return 0 if save() else 1


if __name__ == '__main__': raise SystemExit(main())
