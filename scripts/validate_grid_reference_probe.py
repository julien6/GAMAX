"""Native probe of guarded overlap counts, duplicates, fallback and lifecycle."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from gamax.compiler import compile_model
from gamax.runtime import load_module, run
from gamax.validation import compare_traces, compare_identities
from benchmarks.official_native_checkpoints import observe
from validate_spatial_compilation import expression


def main(a):
    import jax.numpy as jnp
    a.output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'examples/grid_reference_counts/main.gaml'
    path, manifest = compile_model(source, a.output / 'compiled', experiment='validation',
                                   capacities={'Actor': 16}, optimization='fusion-avx-batch')
    module = load_module(path)
    keys = list(manifest['fields'])
    observations = {'field' + str(i): expression(k, manifest['fields'][k]) for i, k in enumerate(keys)}
    species = ['Actor', 'cell']
    observations.update({'identity' + str(i): f'string(({name} sort_by each.index) collect each.index)'
                         for i, name in enumerate(species)})
    reports = []
    for seed in a.seeds:
        native = observe(source, 'validation', observations, seed=seed, steps=12,
                         output=a.output / str(seed) / 'native', home=ROOT / '.tools/gama',
                         checkpoints=list(range(13)), preserve_rng=True)
        if native['status'] != 'passed':
            raise RuntimeError(native)
        reference = [{k: row['field' + str(i)] for i, k in enumerate(keys)} for row in native['trace']]
        identities = [{name: row['identity' + str(i)] for i, name in enumerate(species)} for row in native['trace']]
        candidate = run(module, 12, seed=seed)
        guards=[]
        for frame in candidate:
            shape=jnp.asarray(frame['global.shape'])
            half=(jnp.max(shape,axis=0)-jnp.min(shape,axis=0))[:2]/jnp.asarray([4,3])/2
            guards.append(bool(module.grid_reference_membership_valid(
                jnp.asarray(frame['Actor.location']),jnp.asarray(frame['Actor.place'],jnp.int32),
                jnp.ones(len(frame['Actor.place']),jnp.bool_),jnp.asarray(frame['cell.location']),half,4,3)))
        if set(guards)!={True,False}:raise RuntimeError('Probe did not exercise both guarded and fallback geometry')
        comparison = compare_traces(reference, candidate, manifest['fields'])
        identity = compare_identities(identities, candidate.identities)
        row = {'seed': seed, 'comparison': comparison, 'identities': identity,'guard_by_snapshot':guards,
               'passed': comparison['passed'] and identity['passed']}
        reports.append(row)
        (a.output / 'summary.json').write_text(json.dumps({'complete': len(reports) == len(a.seeds),
            'passed': all(r['passed'] for r in reports), 'optimization': manifest['optimization'],
            'reports': reports}, indent=2) + '\n')
        print(seed, row['passed'], comparison.get('first_mismatch'), flush=True)
    return 0 if all(r['passed'] for r in reports) else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', type=int, nargs='+', default=[2, 7, 10])
    p.add_argument('-o', '--output', type=Path, required=True)
    raise SystemExit(main(p.parse_args()))
