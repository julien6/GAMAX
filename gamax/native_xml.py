"""Fresh-process official XML oracle with external seed and observer-only outputs.

GAMA's WebSocket load parameters do not configure the simulation RNG seed.
The XML Simulation.seed attribute does. Observers inherit the selected experiment;
source files and their imports are not rewritten. Expressions must be pure reads.
This is a fidelity oracle, not a computation-only timing runner.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


def observe(source, experiment, observations, *, seed, steps, output, home,
            parameters=None, timeout=1800, cpu=None):
    source, output, home = (Path(p).resolve() for p in (source, output, home))
    if not source.is_file(): raise ValueError('Missing source')
    if not math.isfinite(float(seed)) or float(seed) == 0: raise ValueError('Finite nonzero seed required')
    if type(steps) is not int or steps < 1: raise ValueError('Positive integer horizon required')
    if timeout <= 0 or timeout > 1800: raise ValueError('Timeout must be in (0, 1800] seconds')
    if not observations or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) for name in observations):
        raise ValueError('Nonempty observations with identifier names required')
    if not (home/'plugins/gama.core_2025.6.4.jar').is_file(): raise ValueError('Pinned GAMA 2025.06.4 required')
    # Never overwrite existing evidence, including evidence of a failed run.
    output.mkdir(parents=True, exist_ok=False)
    wrapper = output/'observer.gaml'
    text = 'model gamax_observer\nimport ' + json.dumps(os.path.relpath(source, output)) + '\n'
    text += 'experiment gamax_observation type: gui parent: ' + json.dumps(experiment) + ' { output {\n'
    for name, expr in observations.items():
        guarded = 'simulation = nil ? nil : ('+expr+')'
        text += 'monitor '+json.dumps('initial_'+name)+' value: ('+guarded+') refresh: false;\n'
        text += 'monitor '+json.dumps('current_'+name)+' value: ('+guarded+');\n'
    wrapper.write_text(text+'}}\n')
    plan = ET.Element('Experiment_plan')
    sim = ET.SubElement(plan, 'Simulation', experiment='gamax_observation', finalStep=str(steps),
                        id='observation', seed=str(float(seed)), sourcePath=str(wrapper))
    params = ET.SubElement(sim, 'Parameters')
    for name, value in (parameters or {}).items():
        if type(value) not in (bool,int,float,str): raise ValueError('Unsupported XML parameter type: '+name)
        if type(value) is float and not math.isfinite(value): raise ValueError('Nonfinite XML parameter')
        kind = {bool:'BOOLEAN',int:'INT',float:'FLOAT',str:'STRING'}[type(value)]
        ET.SubElement(params, 'Parameter', name=name, var=name, type=kind,
                      value=str(value).lower() if type(value) is bool else str(value))
    outputs = ET.SubElement(sim, 'Outputs')
    for index, name in enumerate(observations):
        for prefix in ('initial_', 'current_'):
            ET.SubElement(outputs, 'Output', id=str(index*2+(prefix=='current_')),
                          name=prefix+name, framerate='1')
    ET.ElementTree(plan).write(output/'plan.xml', encoding='utf-8', xml_declaration=True)
    ini = (home/'Gama.ini').read_text().splitlines()
    flags = [f for f in ini[ini.index('-server'):] if not f.startswith(('-Xms','-Xmx')) and f != '-XX:+AlwaysPreTouch']
    launcher = next((home/'plugins').glob('org.eclipse.equinox.launcher_*.jar'))
    command = [str(home/'jdk/bin/java'), '-cp', str(launcher), *flags, '-Xms128m', '-Xmx512m',
               '-XX:-AlwaysPreTouch', '-XX:ActiveProcessorCount=1', 'org.eclipse.equinox.launcher.Main',
               '-configuration', str(home/'headless/configuration'), '-application', 'gama.headless.product',
               '-data', str(output/'workspace'), '-hpc', '1', str(output/'plan.xml'), str(output/'native_output')]
    if cpu is not None:
        if type(cpu) is not int or cpu not in os.sched_getaffinity(0): raise ValueError('CPU not available')
        command = ['taskset','-c',str(cpu),*command]
    protocol = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    seed=float(seed), experiment=experiment, parameters=parameters or {}, steps=steps,
                    observations=observations, command=command, timeout=timeout,
                    contract='observer-only inherited experiment; caller supplies pure expressions; XML step 0 follows the first transition; initial monitors capture post-init state',
                    timing='whole observed native process; includes startup and monitoring, not computation-only')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic(); timed_out=False
    with (output/'native.log').open('w') as log:
        process=subprocess.Popen(command,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: returncode=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGKILL);returncode=process.wait()
    errors = [line for line in (output/'native.log').read_text(errors='replace').splitlines()
              if re.search(r'(?:Exception|Division by zero|OutOfMemoryError|Runtime error|GamlCompilationError)',line)]
    result=dict(status='timeout' if timed_out else 'native_error' if returncode or errors else 'incomplete',
                returncode=returncode, wall_seconds=time.monotonic()-started, errors=errors, trace=[])
    try:
        files=list((output/'native_output').glob('*.xml'))
        if len(files)!=1: raise ValueError('Expected exactly one native trace XML')
        nodes=ET.parse(files[0]).findall('./Step')
        if len(nodes)!=steps: raise ValueError(f'Expected {steps} steps, got {len(nodes)}')
        def decode(text):
            try:return json.loads(text)
            except (ValueError,TypeError):return text
        required={prefix+name for prefix in ('initial_','current_') for name in observations}
        snapshots=[]
        for node in nodes:
            values={v.attrib['name']:decode(v.text) for v in node.findall('Variable')}
            if set(values)!=required: raise ValueError('Missing or unexpected monitored values')
            snapshots.append(values)
        initial={name:snapshots[0]['initial_'+name] for name in observations}
        if any({name:s['initial_'+name] for name in observations} != initial for s in snapshots):
            raise ValueError('Initial observers changed: refresh:false contract failed')
        result['trace']=[initial]+[{name:s['current_'+name] for name in observations} for s in snapshots]
        if not returncode and not errors and not timed_out:result['status']='passed'
    except (ValueError,ET.ParseError,OSError) as error:result['parse_error']=str(error)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
