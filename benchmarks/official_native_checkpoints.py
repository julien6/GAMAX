"""Checkpoint performance adapter derived from the validated seeded oracle.

Cold timings include the seed-zero preflight initialization. Report this cost
explicitly; native transition timings include synchronous WebSocket control.
The extra observer action preserves clock and RNG and does not replace rules.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import time
from gamax.native_server import native_server


def observe(source, experiment, observations, *, seed, steps, output, home, timeout=1800, parameters=None,preserve_rng=False,checkpoints=None):
    source,output = Path(source).resolve(),Path(output).resolve()
    if type(seed) not in (int,float) or not math.isfinite(seed) or seed==0:
        raise ValueError('Finite nonzero seed required')
    if type(steps) is not int or steps<1: raise ValueError('Positive horizon required')
    if not 0 < timeout <= 1800: raise ValueError('Timeout must be within (0,1800] seconds')
    parameters={} if parameters is None else dict(parameters)
    for name,value in parameters.items():
        if not re.fullmatch(r'[A-Za-z_]\w*',name) or name in ('seed','rng'):
            raise ValueError('Invalid or protected native parameter name')
        if type(value) not in (int,float,bool) or type(value) is float and not math.isfinite(value):
            raise ValueError('Native parameters currently require finite scalar values')
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),preserve_rng=preserve_rng,
                  experiment=experiment,seed=seed,steps=steps,observations=observations,timeout=timeout,parameters=parameters,
                  classification='exploratory checkpoint benchmark; native total includes preflight; transition timings include RPC',
                  seed_contract='load official experiment; set experiment seed; create simulation before retiring preflight simulation; verify native seed/RNG and one simulation; one checked activation step before observing transitions',
                  shutdown='terminate owned JVM after last synchronous snapshot, before closing socket; no stop/dispose command')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    expression='['+', '.join(json.dumps(name)+' :: ('+value+')' for name,value in observations.items())+']'
    oracle_source=source
    if preserve_rng:
        # ASK uses the target simulation's scope; expression RPCs use the experiment
        # scope, whose random generator is different. Keep the imported model intact.
        oracle_source=output/'observer.gaml'
        oracle_source.write_text('model gamax_rng_observer\nimport '+json.dumps(str(source))+'\n'
            'global { map gamax_observed <- []; action gamax_snapshot { '
            'int gamax_saved_usage <- rng_usage; float gamax_saved_seed <- seed; '
            'gamax_observed <- '+expression.replace('simulation.','')+'; '
            'seed <- gamax_saved_seed; rng_usage <- gamax_saved_usage; } }\n')
        protocol['observer_source_sha256']=hashlib.sha256(oracle_source.read_bytes()).hexdigest()
        protocol['observation_scope']='simulation-owned ASK action; restore seed and native 32-bit RNG usage after capture'
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic(); result=dict(status='incomplete',trace=[],compute_seconds=0.0,observation_seconds=0.0)
    checkpoints=list(range(steps+1)) if checkpoints is None else checkpoints
    if checkpoints[0]!=0 or checkpoints[-1]!=steps or sorted(set(checkpoints))!=checkpoints:raise ValueError('Invalid checkpoints')
    try:
        with native_server(home,output/'process') as client:
            exp=client.command('load',model=str(oracle_source),experiment=experiment,console=False,runtime=True)
            entries=', '.join(json.dumps(name)+' :: '+(str(value).lower() if type(value) is bool else repr(value)) for name,value in parameters.items())
            configuration=' with: ['+entries+']' if parameters else ''
            client.command('expression',exp_id=exp,expr=f'agent previous <- simulation; seed <- {float(seed)!r}; create simulation{configuration}; ask previous {{ do die; }} return seed;')
            actual=client.expression(exp,"['seed' :: simulation.seed, 'rng' :: simulation.rng, 'simulation_count' :: length(simulations)]")
            result['native_configuration']=actual
            if actual['seed']!=float(seed) or actual['rng']!='mersenne' or actual['simulation_count']!=1:
                raise RuntimeError('Native seed/RNG/simulation count mismatch: '+repr(actual))
            expression='['+', '.join(json.dumps(name)+' :: ('+value+')' for name,value in observations.items())+']'
            phase_expression="['cycle' :: simulation.cycle, 'rng_usage' :: simulation.rng_usage]"
            def snapshot():
                if not preserve_rng:return client.expression(exp,expression)
                phase_before=client.expression(exp,phase_expression)
                simulation_index=client.expression(exp,'simulation.index')
                client.command('ask',exp_id=exp,agent=f'simulation[{simulation_index}]',action='gamax_snapshot',args='{}')
                response=client.expression(exp,'simulation.gamax_observed')
                phase_after=client.expression(exp,phase_expression)
                if phase_before!=phase_after:raise RuntimeError('Observation changed native clock/RNG usage despite restoration')
                return response
            before=snapshot()
            phase=client.expression(exp,phase_expression)
            client.command('step',exp_id=exp,nb_step=1,sync=True)
            after=snapshot()
            activated=client.expression(exp,phase_expression)
            result['activation']=dict(before=phase,after=activated,unchanged=before==after and phase==activated)
            if not result['activation']['unchanged']:
                result['activation_observations']=dict(before=before,after=after)
                raise RuntimeError('Activation changed state/clock/RNG; unsupported native scheduling phase')
            result["setup_seconds"]=time.monotonic()-started
            previous_cycle=0
            for cycle in checkpoints:
                remaining=timeout-(time.monotonic()-started)
                if remaining<=0: raise TimeoutError('Native observation timeout')
                client.socket.settimeout(min(120,remaining))
                if cycle:
                    tick=time.perf_counter();client.command('step',exp_id=exp,nb_step=cycle-previous_cycle,sync=True);result['compute_seconds']+=time.perf_counter()-tick
                previous_cycle=cycle
                tick=time.perf_counter()
                native_cycle=client.expression(exp,'simulation.cycle')
                if native_cycle!=cycle: raise RuntimeError('Native cycle mismatch: '+repr((cycle,native_cycle)))
                observed=snapshot()
                if not isinstance(observed,dict) or set(observed)!=set(observations):
                    raise RuntimeError('Native observation schema mismatch')
                for name,value in observed.items():
                    if isinstance(value,str):
                        try: observed[name]=json.loads(value)
                        except ValueError: pass
                result['trace'].append(observed)
                result['observation_seconds']+=time.perf_counter()-tick
            result['status']='passed'
    except Exception as error:
        result.update(status='native_error',error=repr(error))
    log=output/'process/gama_server.log'
    errors=[line for line in log.read_text(errors='replace').splitlines()
            if re.search(r'(?:Exception|Division by zero|OutOfMemoryError|Runtime error|GamlCompilationError)',line)] if log.exists() else ['Missing native process log']
    result['errors']=errors
    if errors: result['status']='native_error'
    result['wall_seconds']=time.monotonic()-started
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
