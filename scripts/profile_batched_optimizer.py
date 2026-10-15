"""Validate independent-world vectorization, then measure exploratory kernels.

No emitted code is edited. This is throughput evidence, not single-run latency
or an end-to-end GAMA benchmark. Native states are checked at every cycle.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from benchmark_amortized import checked_configuration
from gamax.runtime import canonical, check_status, load_module
from gamax.validation import compare_identities, compare_traces


def snapshot(module, state):
    value = canonical(getattr(module, 'observe_jit', lambda s: s)(state),
                      getattr(module, 'ASSET_FIELDS', None), getattr(module, 'ASSET_KEYS', None))
    for key in getattr(module, 'MATRIX_FIELDS', []):
        value[key] = [v for row in value[key] for v in row]
    for key, names in getattr(module, 'FSM_FIELDS', {}).items():
        value[key] = [names[i] for i in value[key]]
    return value


def main(a):
    import jax
    import jax.numpy as jnp
    cfg = checked_configuration(a.config, 100)
    evidence = json.loads((ROOT / 'benchmarks/official_acceptance_evidence.json').read_text())
    module = load_module(cfg['module'])
    a.output.mkdir(parents=True, exist_ok=False)
    reports = []

    def all_states(s):
        def advance(state, _):
            nxt = module.step(state)
            return nxt, nxt
        return jax.lax.scan(advance, s, None, length=100)

    proof = jax.jit(jax.vmap(all_states), compiler_options=module.JIT_OPTIONS)
    serial = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda _, x: module.step(x), s),
                     compiler_options=module.JIT_OPTIONS)
    vectorized = jax.jit(jax.vmap(lambda s: jax.lax.fori_loop(0, 100, lambda _, x: module.step(x), s)),
                         compiler_options=module.JIT_OPTIONS)
    for offset in range(0, len(a.seeds), a.batch_size):
        seeds = a.seeds[offset:offset + a.batch_size]
        initial = [module.reset(seed=seed) for seed in seeds]
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *initial)
        final, states = proof(stacked)
        jax.block_until_ready((final, states))
        checks = []
        for i, seed in enumerate(seeds):
            trace = [snapshot(module, initial[i])]
            identities = [module.identities(initial[i])]
            for cycle in range(100):
                state = jax.tree.map(lambda leaf: leaf[i, cycle], states)
                check_status(state)
                trace.append(snapshot(module, state))
                identities.append(module.identities(state))
            archive = ROOT / evidence[cfg['case_id']]['coupled'] / cfg['case_id'] / str(seed)
            comparison = compare_traces(json.loads((archive / 'gama_trace.json').read_text()),
                                        trace, cfg['manifest']['fields'])
            identity = compare_identities(json.loads((archive / 'gama_identities.json').read_text()), identities)
            checks.append({'seed': seed, 'comparison': comparison, 'identities': identity,
                           'passed': comparison['passed'] and identity['passed']})
            if not checks[-1]['passed']:
                (a.output / f'divergence-{seed}.json').write_text(json.dumps({
                    'trace': trace, 'identities': identities}, allow_nan=False) + '\n')
        del states, final, trace, identities
        row = {'seeds': seeds, 'checks': checks, 'eligible': all(c['passed'] for c in checks)}
        if row['eligible']:
            tick = time.perf_counter()
            scalar_executable = serial.lower(initial[0]).compile()
            row['serial_compile_seconds'] = time.perf_counter() - tick
            tick = time.perf_counter()
            batched_executable = vectorized.lower(stacked).compile()
            row['batch_compile_seconds'] = time.perf_counter() - tick
            for state in initial:
                jax.block_until_ready(scalar_executable(state))
            jax.block_until_ready(batched_executable(stacked))
            timings = []
            for repetition in range(3):
                times = {}
                for kind in (('serial', 'batch') if repetition % 2 == 0 else ('batch', 'serial')):
                    tick = time.perf_counter()
                    if kind == 'serial':
                        for state in initial:
                            jax.block_until_ready(scalar_executable(state))
                    else:
                        jax.block_until_ready(batched_executable(stacked))
                    times[kind] = time.perf_counter() - tick
                timings.append({**times, 'serial_over_batch': times['serial'] / times['batch']})
            row['kernel_timings'] = timings
        reports.append(row)
        result = {'complete': offset + a.batch_size >= len(a.seeds), 'passed': all(r['eligible'] for r in reports),
                  'case': cfg['case_id'], 'batch_size': a.batch_size,
                  'module_sha256': hashlib.sha256(Path(cfg['module']).read_bytes()).hexdigest(),
                  'scope': 'Independent worlds, every cycle checked against native oracle. Exploratory same-process synchronized CPU kernels. Excludes initialization, observations, rendering, exports, memory and GAMA costs; throughput only.',
                  'reports': reports}
        (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(cfg['case_id'], seeds, row['eligible'], row.get('kernel_timings'), flush=True)
        if not row['eligible']:
            return 1
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('-o', '--output', type=Path, required=True)
    a = p.parse_args()
    if a.batch_size not in (2, 4) or not a.seeds or len(set(a.seeds)) != len(a.seeds) or any(s not in range(1, 11) for s in a.seeds):
        p.error('Batch size 2 or 4; unique certified seeds 1–10 required')
    raise SystemExit(main(a))
