#!/usr/bin/env python3
"""Launch owned native GAMA, evaluate unmodified fixtures, save provenance, shut it down."""
from pathlib import Path
import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
from gamax.cli import main as gamax_main

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gama-home', type=Path, default=ROOT/'.tools/gama')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('-o', '--output', type=Path, default=ROOT/'results/local/pilot')
    args = parser.parse_args()
    home = args.gama_home.resolve()
    wrapper = home/'headless/gama-headless.sh'
    if not wrapper.exists(): raise SystemExit('Install GAMA first: python scripts/install_gama.py')
    if not (home/'plugins/gama.core_2025.6.4.jar').exists():
        raise SystemExit('Runner requires GAMA 2025.06.4; use gamax evaluate for another declared version.')
    output = args.output.resolve()
    if (output/'summary.json').exists(): raise SystemExit('Output already contains a run: choose a new directory to preserve it.')
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    command = ['bash', str(wrapper), '-m', '2g', '-socket', str(port)]
    with (output/'gama_server.log').open('w') as log:
        process = subprocess.Popen(command, cwd=wrapper.parent, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+120
            while True:
                if process.poll() is not None: raise RuntimeError(f'GAMA exited: see {output}/gama_server.log')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1): break
                except OSError:
                    if time.monotonic() > deadline: raise RuntimeError('GAMA server startup timeout')
                    time.sleep(0.5)
            version = subprocess.run([str(home/'jdk/bin/java'), '-version'], capture_output=True, text=True, check=True)
            provenance = dict(gama_release='2025.06.4', release_url='https://github.com/gama-platform/gama/releases/tag/2025.06.4',
                              archive_sha256='043a81de03a79e0a71055c20fc1a1d8d6d290045e8c7ea7162aa5e2be622834f',
                              wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                              java=version.stderr, command=command, steps=args.steps,
                              version_note='Release pin applies to the bundled installation; verify external --gama-home separately.')
            (output/'gama_installation.json').write_text(json.dumps(provenance, indent=2)+'\n')
            sources = [ROOT/'examples'/f'{name}.gaml' for name in ['ordered_agents', 'initialization', 'int_overflow', 'float_decay', 'resource_competition']]
            status = gamax_main(['evaluate', *map(str, sources), '--steps', str(args.steps), '-o', str(output),
                               '--gama-url', f'ws://127.0.0.1:{port}', '--gama-version', '2025.06.4', '--negative-control'])
            if status: return status
            control_status = gamax_main(['evaluate', str(ROOT/'examples/resource_competition.gaml'), '--steps', str(args.steps),
                                        '-o', str(output/'agent-order-control'), '--gama-url', f'ws://127.0.0.1:{port}',
                                        '--gama-version', '2025.06.4', '--negative-agent-order'])
            summary_path = output/'summary.json'
            summary = json.loads(summary_path.read_text())
            summary['agent_order_control'] = json.loads((output/'agent-order-control/summary.json').read_text())
            if control_status: summary['status'] = 'failed'
            summary_path.write_text(json.dumps(summary, indent=2)+'\n')
            return control_status
        finally:
            # Kill the owned process group (wrapper and JVM); never an unrelated GAMA instance.
            import os, signal
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()


if __name__ == '__main__': sys.exit(main())
