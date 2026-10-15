from pathlib import Path
import argparse
import json
import sys
from datetime import datetime, timezone
from .compiler import compile_model, CompileError, load_model, Generator
from .runtime import load_module, run, export_trace
from .validation import GamaClient, compare_traces, environment


def json_object(value):
    try:decoded=json.loads(value)
    except json.JSONDecodeError as error:raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(decoded,dict):raise argparse.ArgumentTypeError('A JSON object is required')
    return decoded


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compile restricted GAML and validate against native GAMA')
    sub = parser.add_subparsers(dest='command', required=True)
    compile_ = sub.add_parser('compile')
    compile_.add_argument('source'); compile_.add_argument('-o', '--output', required=True)
    compile_.add_argument('--experiment',help='Selected native experiment, including inherited literal parameter defaults')
    compile_.add_argument('--capacities',type=json_object,help='Reserved species capacities as a JSON object')
    compile_.add_argument('--optimization',choices=['none','safe','fusion','fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined'],default='none',help='Opt-in conservative optimizer; validate optimized trajectories before performance claims')
    run_ = sub.add_parser('run')
    run_.add_argument('module'); run_.add_argument('--steps', type=int, default=100)
    run_.add_argument('-o', '--output', required=True)
    run_.add_argument('--seed',type=float,help='Native-compatible external seed; required for implicit Mersenne initialization')
    run_.add_argument('--parameters',type=json_object,help='Typed external global initializer overrides as a JSON object')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('sources', nargs='+'); evaluate.add_argument('-o', '--output', required=True)
    evaluate.add_argument('--steps', type=int, default=100)
    evaluate.add_argument('--experiment', default='validation')
    evaluate.add_argument('--gama-url', default='ws://127.0.0.1:6868')
    evaluate.add_argument('--gama-version', required=True, help='Version of externally launched GAMA (reported by operator)')
    evaluate.add_argument('--atol', type=float, default=1e-12); evaluate.add_argument('--rtol', type=float, default=1e-10)
    evaluate.add_argument('--negative-control', action='store_true', help='Reverse species reflex order and require divergence')
    evaluate.add_argument('--negative-agent-order', action='store_true', help='Reverse agent iteration order and require divergence on a scheduling-sensitive model')
    args = parser.parse_args(argv)
    try:
        if args.command == 'compile':
            path, manifest = compile_model(args.source, args.output,experiment=args.experiment,capacities=args.capacities,optimization=args.optimization)
            print(json.dumps(dict(module=str(path), model=manifest['model'])))
            return 0
        if args.steps < 0: raise ValueError('steps must be nonnegative')
        if args.command == 'run':
            trace = run(load_module(args.module), args.steps,seed=args.seed,parameters=args.parameters)
            export_trace(trace, args.output, 'jax')
            print(json.dumps(dict(snapshots=len(trace), output=args.output)))
            return 0
        if len({Path(s).stem for s in args.sources}) != len(args.sources):
            raise ValueError('Source basenames must be distinct')
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        reports = []
        client = GamaClient(args.gama_url)
        try:
            for source in args.sources:
                directory = output / Path(source).stem
                module_path, manifest = compile_model(source, directory / 'generated')
                reference = client.trace(source, manifest, args.steps, args.experiment)
                candidate = run(load_module(module_path), args.steps)
                export_trace(reference, directory, 'gama'); export_trace(candidate, directory, 'jax')
                comparison = compare_traces(reference, candidate, manifest['fields'], args.atol, args.rtol)
                report = dict(model=manifest['model'], manifest=manifest, comparison=comparison)
                if args.negative_control:
                    ir, _ = load_model(source)
                    for scope in ir.scopes[1:]: scope.behaviors.reverse()
                    mutant_path = directory / 'generated' / 'mutant_jax.py'
                    mutant_path.write_text(Generator(ir).generate())
                    mutant = run(load_module(mutant_path), args.steps)
                    export_trace(mutant, directory, 'mutant')
                    control = compare_traces(reference, mutant, manifest['fields'], args.atol, args.rtol)
                    report['negative_control'] = dict(mutation='reverse species reflex order', detected=not control['passed'], comparison=control)
                if args.negative_agent_order:
                    ir, _ = load_model(source)
                    mutant_path = directory / 'generated' / 'agent_order_mutant_jax.py'
                    mutant_path.write_text(Generator(ir, reverse_agents=True).generate())
                    mutant = run(load_module(mutant_path), args.steps)
                    export_trace(mutant, directory, 'agent_order_mutant')
                    control = compare_traces(reference, mutant, manifest['fields'], args.atol, args.rtol)
                    report['agent_order_control'] = dict(mutation='reverse agent iteration order', detected=not control['passed'], comparison=control)
                (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                reports.append(report)
                print(json.dumps(dict(model=manifest['model'], **comparison)))
        finally:
            (output / 'gama_commands.json').write_text(json.dumps(client.commands, indent=2)+'\n')
            client.close()
        passed = all(r['comparison']['passed'] and (not args.negative_control or r['negative_control']['detected']) and (not args.negative_agent_order or r['agent_order_control']['detected']) for r in reports)
        summary = dict(schema_version=1, status='passed' if passed else 'failed',
                       timestamp_utc=datetime.now(timezone.utc).isoformat(),
                       purpose='functional differential validation; no performance claim',
                       oracle_float_encoding='GAML string(float): Java Double.toString, parsed back to float64; avoids JsonFloat eight-decimal rounding',
                       gama_version_operator_reported=args.gama_version, gama_greeting=client.greeting,
                       environment=environment(), steps=args.steps, reports=reports,
                       scope='development fixtures; not independent held-out corpus; declared attributes only, not full GAMA engine state')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return 0 if passed else 1
    except CompileError as error:
        if args.command == 'evaluate':
            directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
            (directory/'summary.json').write_text(json.dumps(dict(status='error', diagnostic=error.diagnostic, reports=locals().get('reports', [])), indent=2)+'\n')
        print(json.dumps(error.diagnostic), file=sys.stderr); return 2
    except (RuntimeError, ValueError, OSError) as error:
        if args.command == 'evaluate':
            failure = dict(schema_version=1, status='error', error=str(error), reports=locals().get('reports', []),
                           timestamp_utc=datetime.now(timezone.utc).isoformat())
            directory = Path(args.output); directory.mkdir(parents=True, exist_ok=True)
            (directory / 'summary.json').write_text(json.dumps(failure, indent=2)+'\n')
        print(str(error), file=sys.stderr); return 2


if __name__ == '__main__': sys.exit(main())
