"""Reuse one owned JVM and compiled experiment, never substitute model rules."""
from pathlib import Path
import hashlib
import json
import math
import re
import time
from gamax.native_server import native_server


class ReusedNativeExperiment:
    def __init__(self, source, experiment, observations, *, home, output, preserve_rng=False):
        self.source=Path(source).resolve();self.experiment=experiment
        self.observations=dict(observations);self.home=Path(home)
        self.output=Path(output).resolve();self.preserve_rng=preserve_rng
        self.expression='['+', '.join(json.dumps(k)+' :: ('+v+')' for k,v in self.observations.items())+']'

    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=False)
        self.context=native_server(self.home,self.output/'process',record_response_payloads=False)
        started=time.perf_counter();self.client=self.context.__enter__()
        try:
            source=self.source
            if self.preserve_rng:
                source=self.output/'observer.gaml'
                source.write_text('model gamax_reused_observer\nimport '+json.dumps(str(self.source))+'\n'
                    'global { map gamax_observed <- []; action gamax_snapshot { '
                    'int gamax_saved_usage <- rng_usage; float gamax_saved_seed <- seed; '
                    'gamax_observed <- '+self.expression.replace('simulation.','')+'; '
                    'seed <- gamax_saved_seed; rng_usage <- gamax_saved_usage; } }\n')
            self.exp=self.client.command('load',model=str(source),experiment=self.experiment,console=False,runtime=True)
            self.startup_seconds=time.perf_counter()-started
            (self.output/'protocol.json').write_text(json.dumps({
                'source':str(self.source),'source_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),
                'experiment':self.experiment,'observations':self.observations,'preserve_rng':self.preserve_rng,
                'scope':'Owned JVM and compiled experiment reused; each seed creates a new native simulation and retires the previous one. Startup includes native seed-zero preflight; transitions include RPC.',
                'startup_seconds':self.startup_seconds,'rpc_logging':'Requests and successful response length/SHA-256 only; full payloads preserved in run traces, failures retained in full.'},indent=2)+'\n')
            return self
        except BaseException:
            self.context.__exit__(*__import__('sys').exc_info());raise

    def __exit__(self,*args):
        result=self.context.__exit__(*args)
        if args[0] is None:
            log=self.output/'process/gama_server.log'
            errors=[line for line in log.read_text(errors='replace').splitlines() if re.search(r'(?:Exception|Division by zero|OutOfMemoryError|Runtime error|GamlCompilationError)',line)]
            if errors:raise RuntimeError('Errors in reused native process log: '+repr(errors))
        return result

    def phase(self):
        return self.client.expression(self.exp,"['cycle' :: simulation.cycle, 'rng_usage' :: simulation.rng_usage]")

    def snapshot(self):
        if self.preserve_rng:
            before=self.phase();index=self.client.expression(self.exp,'simulation.index')
            self.client.command('ask',exp_id=self.exp,agent=f'simulation[{index}]',action='gamax_snapshot',args='{}')
            observed=self.client.expression(self.exp,'simulation.gamax_observed')
            if before!=self.phase():raise RuntimeError('Reused native observer changed clock or RNG usage')
        else:observed=self.client.expression(self.exp,self.expression)
        if not isinstance(observed,dict) or set(observed)!=set(self.observations):
            raise RuntimeError('Reused native observation schema mismatch')
        for name,value in observed.items():
            if isinstance(value,str):
                try:observed[name]=json.loads(value)
                except ValueError:pass
        return observed

    def run(self, *, seed, steps, checkpoints=None, timeout=1800):
        if type(seed) not in (int,float) or not math.isfinite(seed) or seed==0:raise ValueError('Finite nonzero seed required')
        if type(steps) is not int or steps<1:raise ValueError('Positive horizon required')
        if not 0<timeout<=1800:raise ValueError('Timeout must be within (0,1800] seconds')
        checkpoints=list(range(steps+1)) if checkpoints is None else list(checkpoints)
        if not checkpoints or checkpoints[0]!=0 or checkpoints[-1]!=steps or sorted(set(checkpoints))!=checkpoints:raise ValueError('Invalid checkpoints')
        started=time.perf_counter()
        self.client.command('expression',exp_id=self.exp,expr=f'agent previous <- simulation; seed <- {float(seed)!r}; create simulation; ask previous {{ do die; }} return seed;')
        configuration=self.client.expression(self.exp,"['seed' :: simulation.seed, 'rng' :: simulation.rng, 'simulation_count' :: length(simulations)]")
        if configuration!={'seed':float(seed),'rng':'mersenne','simulation_count':1}:raise RuntimeError('Reused native seed/RNG/simulation-count mismatch: '+repr(configuration))
        before=self.snapshot();initial_phase=self.phase()
        if initial_phase['cycle']!=0:raise RuntimeError('Reused native simulation did not reset its clock')
        # A fresh experiment has an activation-only first step. A reused
        # experiment may immediately advance the newly created simulation.
        # Count that transition; never discard an actual model cycle.
        tick=time.perf_counter();self.client.command('step',exp_id=self.exp,nb_step=1,sync=True)
        first_step_seconds=time.perf_counter()-tick
        activated=self.phase();after=self.snapshot()
        cycle=activated['cycle'];compute=first_step_seconds if cycle==1 else 0.
        if cycle==0 and (before!=after or initial_phase!=activated):raise RuntimeError('Activation changed native state without advancing its clock')
        if cycle not in (0,1):raise RuntimeError('Unexpected native activation cycle')
        result={'status':'passed','trace':[before],'compute_seconds':compute,'observation_seconds':0.,
                'setup_seconds':time.perf_counter()-started-compute,
                'activation':{'before':initial_phase,'after':activated,'advanced':cycle==1},
                'native_configuration':configuration}
        for target in checkpoints[1:]:
            remaining=timeout-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('Reused native execution timeout')
            self.client.socket.settimeout(min(120,remaining))
            tick=time.perf_counter()
            if target>cycle:self.client.command('step',exp_id=self.exp,nb_step=target-cycle,sync=True)
            result['compute_seconds']+=time.perf_counter()-tick;cycle=target
            tick=time.perf_counter()
            if self.phase()['cycle']!=target:raise RuntimeError('Reused native cycle mismatch')
            result['trace'].append(after if target==1 and activated['cycle']==1 else self.snapshot())
            result['observation_seconds']+=time.perf_counter()-tick
        result['wall_seconds']=time.perf_counter()-started
        return result
