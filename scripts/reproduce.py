#!/usr/bin/env python3
"""Plan by default; explicitly execute fresh fidelity and CPU experiments.

No archived results required. See docs/reproduction.md. No remote operations.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/reproduction/cpu-v0.json')
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--regime', choices=['cold', 'amortized', 'both'], default='amortized')
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Actually run GAMA and JAX; otherwise only print the plan')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cases = args.cases or list(cfg['capacities'])
    seeds = args.seeds or cfg['seeds']
    repeats = args.repeats if args.repeats is not None else cfg['repetitions']
    if len(set(cases)) != len(cases) or not set(cases) <= set(cfg['capacities']):
        parser.error('Choose distinct development cases from the configuration')
    if repeats < 1 or len(set(seeds)) != len(seeds) or not set(seeds) <= set(range(1, 11)):
        parser.error('Positive repetitions and distinct seeds 1–10 required')
    if args.regime != 'cold' and len(seeds) < 2:
        parser.error('Amortized native reuse validation requires at least two different seeds')
    out = args.output.resolve()
    commands = []
    for case in cases:
        directory = out/case
        validation = directory/'validation'
        commands.append([sys.executable, str(ROOT/'scripts/validate_official_compilation.py'), '-o', str(validation), '--cases', case, '--seeds', *map(str,seeds), '--steps',str(cfg['steps']), '--capacities',json.dumps(cfg['capacities'][case]), '--optimization',cfg['optimization']])
        if args.regime in ('cold','both'):
            commands.append([sys.executable,str(ROOT/'scripts/benchmark_official.py'),'-o',str(directory/'cold'),'--validation-root',str(validation),'--cases',case,'--seeds',*map(str,seeds),'--repeats',str(repeats),'--steps',str(cfg['steps']),'--optimization',cfg['optimization']])
            commands.append([sys.executable,str(ROOT/'scripts/analyze_official_performance.py'),str(directory/'cold')])
        if args.regime in ('amortized','both'):
            commands.append(['INTERNAL:create_configuration',case])
            commands.append([sys.executable,str(ROOT/'scripts/validate_reused_native.py'),'--config',str(directory/'configuration.json'),'--oracle',str(validation/case),'--seeds',str(seeds[0]),str(seeds[1]),str(seeds[0]),'--steps',str(cfg['steps']),'-o',str(directory/'reuse')])
            commands.append([sys.executable,str(ROOT/'scripts/benchmark_amortized.py'),'--configs',str(directory/'configuration.json'),'--steps',str(cfg['steps']),'--seeds',*map(str,seeds),'--repeats',str(repeats),'-o',str(directory/'amortized')])
            commands.append([sys.executable,str(ROOT/'scripts/analyze_amortized_performance.py'),str(directory/'amortized')])
    plan = {'classification':'new CPU reproduction, not historical reanalysis or v1 confirmatory study','execute':args.execute,'cases':cases,'seeds':seeds,'regime':args.regime,'steps':cfg['steps'],'repeats':repeats,'commands':commands}
    print(json.dumps(plan,indent=2),flush=True)
    if not args.execute:
        return
    if not (ROOT/'.tools/gama/plugins/gama.core_2025.6.4.jar').is_file():
        parser.error('Install GAMA first: python scripts/install_gama.py')
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=dict(os.environ,JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for command in commands:
        if command[0]=='INTERNAL:create_configuration':
            case=command[1];directory=out/case;validation=directory/'validation'
            folder=validation/case/'generated';manifest=json.loads((folder/'manifest.json').read_text())
            data={'case_id':case,'source':manifest['source'],'experiment':manifest['selected_experiment'],'module':str(folder/'model_jax.py'),'manifest':manifest,'validation_root':str(validation),'validation_seeds':seeds,'native_reuse_validation':str(directory/'reuse/summary.json'),**json.loads((validation/case/'translation.json').read_text())}
            (directory/'configuration.json').write_text(json.dumps(data,indent=2)+'\n')
            continue
        subprocess.run(command,cwd=ROOT,env=env,check=True)
    # benchmark_official's legacy CLI does not fail on ineligible pairs: check explicitly.
    for case in cases:
        for regime in ('cold','amortized'):
            summary=out/case/regime/'summary.json'
            if summary.exists():
                value=json.loads(summary.read_text())
                if not value['complete'] or not value['pairs'] or not all(p['eligible'] for p in value['pairs']):
                    raise SystemExit('Incomplete or rejected pairs: '+str(summary))
    (out/'reproduction_status.json').write_text(json.dumps({'complete':True,'passed':True,'cases':cases,'seeds':seeds,'scope':plan['classification']},indent=2)+'\n')


if __name__=='__main__':
    main()
