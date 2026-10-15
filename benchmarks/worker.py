"""One isolated process per engine/condition; outputs measured states and timings."""
from pathlib import Path
import argparse
import json
import os
import resource
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['gama', 'jax'], required=True)
    parser.add_argument('--mode', choices=['compute', 'render'], required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--gama-home', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    directory = args.directory; directory.mkdir(parents=True, exist_ok=True)
    block_steps = args.steps if args.mode == 'compute' else args.stride
    setup_start = time.perf_counter()
    server = client = log = None
    frames, trace = [], []
    try:
        if args.engine == 'gama':
            # No JAX is imported into the native GAMA process.
            from gamax.gama_client import GamaClient
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            headless = args.gama_home/'headless'
            # Launch the exact official product directly so wait4 collects JVM CPU time.
            # Keep official VM flags, but use a bounded heap rather than Xms2G pre-touch.
            ini = (args.gama_home/'Gama.ini').read_text().splitlines()
            flags = ini[ini.index('-server'):]
            flags = [f for f in flags if not f.startswith(('-Xms', '-Xmx')) and f != '-XX:+AlwaysPreTouch']
            launcher = next((args.gama_home/'plugins').glob('org.eclipse.equinox.launcher_*.jar'))
            command = [str(args.gama_home/'jdk/bin/java'), '-cp', str(launcher), *flags,
                       '-Xms128m', '-Xmx512m', '-XX:-AlwaysPreTouch',
                       'org.eclipse.equinox.launcher.Main', '-configuration', str(headless/'configuration'),
                       '-application', 'gama.headless.product', '-data', str(directory/'workspace'),
                       '-hpc', '1', '-socket', str(port)]
            (directory/'native_command.json').write_text(json.dumps(command, indent=2)+'\n')
            log = (directory/'gama_server.log').open('w')
            server = subprocess.Popen(command, cwd=headless, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+120
            while True:
                if server.poll() is not None: raise RuntimeError('GAMA startup failure')
                try:
                    client = GamaClient(f'ws://127.0.0.1:{port}', timeout=120); break
                except (OSError, ConnectionError):
                    if time.monotonic() > deadline: raise RuntimeError('GAMA startup timeout')
                    time.sleep(0.1)
            def reset():
                return client.command('load', model=str(args.source), experiment='validation', console=False, runtime=True)
            exp_id = reset()
            warm_start = time.perf_counter()
            client.command('step', exp_id=exp_id, nb_step=block_steps, sync=True)
            warm_seconds = time.perf_counter()-warm_start
            client.command('stop', exp_id=exp_id); exp_id = reset()
            def advance():
                client.command('step', exp_id=exp_id, nb_step=block_steps, sync=True)
            def snapshot():
                manifest = json.loads((args.generated/'manifest.json').read_text())
                return client.snapshot(exp_id, manifest)
        else:
            import jax
            from gamax.runtime import load_module, canonical
            module = load_module(args.generated/'model_jax.py')
            transition = jax.jit(lambda s: jax.lax.fori_loop(0, block_steps, lambda _, state: module.step(state), s),
                                 compiler_options=getattr(module, 'JIT_OPTIONS', {}))
            state = module.reset()
            warm_start = time.perf_counter()
            jax.block_until_ready(transition(state))
            warm_seconds = time.perf_counter()-warm_start
            state = module.reset()
            def advance():
                nonlocal state
                state = transition(state)
                jax.block_until_ready(state)
            def snapshot(): return canonical(state)
        if args.mode == 'render':
            from benchmarks.rendering import draw_state
        setup_seconds = time.perf_counter()-setup_start
        compute_seconds = observation_seconds = render_seconds = 0.0
        campaign_start = time.perf_counter()
        for checkpoint in range(args.steps//block_steps+1):
            if checkpoint:
                start = time.perf_counter(); advance(); compute_seconds += time.perf_counter()-start
            start = time.perf_counter(); current = snapshot(); observation_seconds += time.perf_counter()-start
            trace.append(current)
            if args.mode == 'render':
                start = time.perf_counter()
                frames.append(draw_state(current, args.engine))
                render_seconds += time.perf_counter()-start
        campaign_seconds = time.perf_counter()-campaign_start
        export_start = time.perf_counter()
        (directory/'trace.json').write_text(json.dumps(trace, indent=2)+'\n')
        if frames:
            frames[-1].save(directory/'last_frame.png')
            frames[0].save(directory/'animation.gif', save_all=True, append_images=frames[1:], duration=120, loop=0)
        export_seconds = time.perf_counter()-export_start
        result = dict(engine=args.engine, mode=args.mode, steps=args.steps, stride=block_steps,
                      checkpoints=len(trace), setup_seconds=setup_seconds, warmup_or_xla_seconds=warm_seconds,
                      warmed_compute_seconds=compute_seconds, observation_seconds=observation_seconds,
                      render_seconds=render_seconds, warmed_campaign_seconds=campaign_seconds, export_seconds=export_seconds,
                      cpu_affinity=sorted(os.sched_getaffinity(0)),
                      gama_rpc_included=(args.engine=='gama'),
                      rendering='common Pillow offscreen rasterization, no GAMA GUI' if frames else 'none')
        if client:
            client.command('stop', exp_id=exp_id)
            (directory/'gama_commands.json').write_text(json.dumps(client.commands, indent=2)+'\n')
    finally:
        if client: client.close()
        if server and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try: server.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(server.pid, signal.SIGKILL); server.wait()
        if log: log.close()
    own, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    result['cpu_accounting'] = 'getrusage SELF + waited CHILDREN; JVM is a direct child'
    result['whole_process_cpu_seconds'] = own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
    (directory/'measurement.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__': main()
