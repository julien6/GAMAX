#!/usr/bin/env python3
"""First matched compute/render experiment; frozen config before execution, no speedup assumed."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import zipfile
import psutil
from gamax.compiler import compile_model, CompileError, load_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--agents', type=int, default=128)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--source', type=Path, help='Unmodified fixed-population GAML; default shared-resource model')
    args = parser.parse_args()
    if args.steps <= 0 or args.stride <= 0 or args.steps % args.stride or args.agents < 3 or args.repeats < 1:
        parser.error('Require agents>=3, repeats>=1, and steps positive and divisible by stride')
    output = args.output.resolve()
    if output.exists(): raise SystemExit('Use a new output directory to preserve previous runs')
    output.mkdir(parents=True)
    # Preserve the actual official Schelling source and its truthful compilation diagnostic.
    library = next((ROOT/'.tools/gama/plugins').glob('gama.library*.jar'))
    with zipfile.ZipFile(library) as archive:
        name = 'models/Toy Models/Segregation (Schelling)/models/Segregation (Agents).gaml'
        source = output/'official_schelling.gaml'; source.write_bytes(archive.read(name))
    try: compile_model(source, output/'schelling_generated')
    except CompileError as error: (output/'schelling_diagnostic.json').write_text(json.dumps(error.diagnostic, indent=2)+'\n')
    source = output/'model.gaml'
    if args.source:
        original = args.source.resolve()
        _, dependencies = load_model(original)
        common = Path(os.path.commonpath([str(Path(item['path']).parent) for item in dependencies]))
        for item in dependencies:
            dependency = Path(item['path'])
            target = output/'input'/dependency.relative_to(common)
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(dependency.read_bytes())
        source = output/'input'/original.relative_to(common)
    else:
        source.write_text((ROOT/'examples/resource_competition.gaml').read_text().replace('number: 8;', f'number: {args.agents};'))
    start = time.perf_counter(); cpu_start = time.process_time(); _, manifest = compile_model(source, output/'generated')
    translation_seconds = time.perf_counter()-start
    translation_cpu_seconds = time.process_time()-cpu_start
    compiler_pss_bytes = getattr(psutil.Process().memory_full_info(), "pss", 0)
    conditions = [(r, engine, mode) for r in range(args.repeats) for engine in ['gama','jax'] for mode in ['compute','render']]
    random.Random(42).shuffle(conditions)
    cpu = min(os.sched_getaffinity(0))
    config = dict(status='exploratory; not confirmatory Cl3 evidence', model=manifest['model'], agents=sum(manifest['populations'].values()), populations=manifest['populations'],
                  steps=args.steps, stride=args.stride, repeats=args.repeats, cpu=cpu, order=conditions,
                  native_vm='Official product/flags, Xms128m Xmx512m, AlwaysPreTouch disabled, one active CPU',
                  source_sha256=manifest['source_sha256'], compiler_sha256=manifest['compiler_sha256'], implementation_sha256=manifest['implementation_sha256'],
                  translation_seconds=translation_seconds, translation_cpu_seconds=translation_cpu_seconds, compiler_observed_pss_bytes=compiler_pss_bytes, gama='2025.06.4', jax='0.6.2',
                  rendering='21 frames by default; same Pillow offscreen renderer on observed checkpoints; not GAMA native GUI',
                  timing='Compute: one complete-horizon RPC and reset/final observations. Render: one RPC per frame interval. Engine observations and cadence match within each mode.',
                  memory='Sum of process-tree RSS/PSS, sampled every 50ms across full lifecycle; approximate peaks',
                  primary='Wall time including startup/warmup/export/shutdown; JAX source translation reported separately',
                  prediction='No numerical speedup or resource reduction assumed; report all outcomes',
                  statistics='Paired log-ratios, geometric mean and percentile bootstrap CI95 (10000 resamples, RNG 20261012); repetition is process replicate, not stochastic seed; joint wall/CPU/PSS criteria reported separately')
    config['cpu_model'] = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    config['host_ram_bytes'] = psutil.virtual_memory().total
    sources = [Path(__file__).resolve(), ROOT/'benchmarks/worker.py', ROOT/'benchmarks/rendering.py', *sorted((ROOT/'gamax').glob('*.py'))]
    snapshot_dir = output/'benchmark_source'; snapshot_dir.mkdir()
    config['code_sha256'] = {}
    for file in sources:
        (snapshot_dir/file.name).write_bytes(file.read_bytes())
        config['code_sha256'][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    (output/'protocol.json').write_text(json.dumps(config, indent=2)+'\n')
    measurements = []
    for repetition, engine, mode in conditions:
        directory = output/f'{engine}_{mode}_{repetition}'
        command = [sys.executable, '-m', 'benchmarks.worker', '--engine', engine, '--mode', mode,
                   '--directory', str(directory), '--source', str(source), '--generated', str(output/'generated'),
                   '--steps', str(args.steps), '--stride', str(args.stride), '--cpu', str(cpu),
                   '--gama-home', str(ROOT/'.tools/gama')]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=1', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
        start = time.perf_counter(); peak_rss = peak_pss = 0
        with (output/f'{engine}_{mode}_{repetition}.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                rss = pss = 0
                try:
                    root = psutil.Process(process.pid)
                    for member in [root, *root.children(recursive=True)]:
                        try:
                            memory = member.memory_full_info(); rss += memory.rss; pss += getattr(memory, 'pss', 0)
                        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                except psutil.NoSuchProcess: pass
                peak_rss = max(peak_rss, rss); peak_pss = max(peak_pss, pss)
                time.sleep(0.05)
        elapsed = time.perf_counter()-start
        if process.returncode:
            (output/'failure.json').write_text(json.dumps(dict(condition=[repetition,engine,mode], returncode=process.returncode)))
            raise RuntimeError(f'Worker failed: {engine}/{mode}/{repetition}; see its log')
        result = json.loads((directory/'measurement.json').read_text())
        result.update(repetition=repetition, whole_process_wall_seconds=elapsed,
                      whole_campaign_wall_seconds=elapsed+(translation_seconds if engine=='jax' else 0),
                      whole_campaign_cpu_seconds=result['whole_process_cpu_seconds']+(translation_cpu_seconds if engine=='jax' else 0),
                      process_tree_peak_rss_bytes=peak_rss, process_tree_peak_pss_bytes=peak_pss,
                      whole_campaign_peak_pss_bytes=max(peak_pss, compiler_pss_bytes if engine=='jax' else 0))
        (directory/'measurement.json').write_text(json.dumps(result, indent=2)+'\n')
        measurements.append(result)
        print(json.dumps(dict(engine=engine, mode=mode, repetition=repetition, wall_seconds=elapsed, cpu_seconds=result['whole_process_cpu_seconds'], peak_pss_mib=peak_pss/2**20)), flush=True)
    # Validation and paired visuals run in the observer, excluded from worker resource measurements.
    from gamax.validation import compare_traces
    comparisons = []
    for mode in ['compute','render']:
        for repetition in range(args.repeats):
            reference = json.loads((output/f'gama_{mode}_{repetition}/trace.json').read_text())
            candidate = json.loads((output/f'jax_{mode}_{repetition}/trace.json').read_text())
            comparisons.append(dict(mode=mode, repetition=repetition, **compare_traces(reference, candidate, manifest['fields'])))
    metrics = ['whole_campaign_wall_seconds','whole_campaign_cpu_seconds','whole_campaign_peak_pss_bytes','whole_process_cpu_seconds','process_tree_peak_pss_bytes',
               'process_tree_peak_rss_bytes','warmed_compute_seconds','warmed_campaign_seconds','render_seconds']
    medians = {f'{engine}_{mode}': {metric: statistics.median(m[metric] for m in measurements if m['engine']==engine and m['mode']==mode)
                                   for metric in metrics} for engine in ['gama','jax'] for mode in ['compute','render']}
    ratios = {mode: {metric: medians[f'gama_{mode}'][metric]/medians[f'jax_{mode}'][metric] for metric in metrics if medians[f'jax_{mode}'][metric]>0}
              for mode in ['compute','render']}
    summary = dict(protocol=config, measurements=measurements, median=medians, gama_over_jax_ratios=ratios,
                   comparisons=comparisons, passed=all(c['passed'] for c in comparisons),
                   limitations=['Authored fixed-population source when --source is supplied; official Schelling rejection is a separate coverage artifact', f'{args.repeats} independent processes per condition; single model, no general speedup claim',
                                'Graphics are common offscreen frames; no GAMA native GUI measurement', 'Native transition timing includes WebSocket control; one full-horizon request without graphics, checkpoint requests with graphics',
                                'RAM peaks approximate; CPU seconds include startup, warmup, render and shutdown; no energy measurement'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    from PIL import Image
    left = Image.open(output/'gama_render_0/animation.gif')
    right = Image.open(output/'jax_render_0/animation.gif')
    frames = []
    for index in range(min(left.n_frames, right.n_frames)):
        left.seek(index);right.seek(index)
        frame = Image.new('RGB',(left.width+right.width,left.height))
        frame.paste(left.convert('RGB'), (0,0));frame.paste(right.convert('RGB'), (left.width,0));frames.append(frame)
    frames[0].save(output/'comparison.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
    frames[-1].save(output/'comparison.png')
    print(json.dumps(dict(passed=summary['passed'], ratios=ratios)), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__': sys.exit(main())
