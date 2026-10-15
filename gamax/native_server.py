"""Own a direct, bounded-heap JVM; never stop another user's GAMA instance."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import time

from .gama_client import GamaClient


@contextmanager
def native_server(home, directory, *, record_response_payloads=False):
    home, directory = Path(home).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not (home / 'plugins/gama.core_2025.6.4.jar').is_file():
        raise ValueError('This runner requires the installed GAMA 2025.06.4 product')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    headless = home / 'headless'
    ini = (home / 'Gama.ini').read_text().splitlines()
    flags = [f for f in ini[ini.index('-server'):]
             if not f.startswith(('-Xms', '-Xmx')) and f != '-XX:+AlwaysPreTouch']
    launcher = next((home / 'plugins').glob('org.eclipse.equinox.launcher_*.jar'))
    command = [str(home / 'jdk/bin/java'), '-cp', str(launcher), *flags,
               '-Xms128m', '-Xmx512m', '-XX:-AlwaysPreTouch',
               'org.eclipse.equinox.launcher.Main', '-configuration', str(headless / 'configuration'),
               '-application', 'gama.headless.product', '-data', str(directory / 'workspace'),
               '-hpc', '1', '-socket', str(port)]
    (directory / 'native_command.json').write_text(json.dumps(command, indent=2) + '\n')
    client = None
    with (directory / 'gama_server.log').open('w') as log:
        process = subprocess.Popen(command, cwd=headless, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while client is None:
                if process.poll() is not None:
                    raise RuntimeError('GAMA exited during startup; see gama_server.log')
                try:
                    client = GamaClient(f'ws://127.0.0.1:{port}',record_response_payloads=record_response_payloads)
                except (OSError, ConnectionError):
                    if time.monotonic() > deadline: raise RuntimeError('GAMA startup timeout')
                    time.sleep(0.1)
            yield client
        finally:
            if client:
                (directory / 'gama_commands.json').write_text(json.dumps(client.commands, indent=2) + '\n')
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if client: client.close()
