"""Load standalone generated code and export canonical declared-variable traces."""
from pathlib import Path
import csv
import importlib.util
import json
import jax
import numpy as np


def load_module(path):
    spec = importlib.util.spec_from_file_location('gamax_generated', Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(state, asset_fields=None, asset_keys=None):
    check_status(state)
    result = {}
    for key, value in state.items():
        if key.startswith('$'): continue
        array = np.asarray(jax.device_get(value))
        mask_key = '$listmask.'+key
        if mask_key in state:
            masks = np.asarray(jax.device_get(state[mask_key]))
            presence_key='$present.'+key.split('.')[0]
            if masks.ndim>1 and presence_key in state:
                present=np.asarray(jax.device_get(state[presence_key]));array=array[present];masks=masks[present]
            result[key] = array[masks].tolist() if masks.ndim == 1 else [row[mask].tolist() for row,mask in zip(array,masks)]
        else:
            presence_key='$present.'+key.split('.')[0]
            if presence_key in state:
                array=array[np.asarray(jax.device_get(state[presence_key]))]
            result[key] = array.tolist()
    if asset_fields:
        def decode(value):
            if isinstance(value,list):return [decode(item) for item in value]
            if type(value) is not int or value < -1 or value>=len(asset_keys):raise RuntimeError('Invalid image handle')
            return None if value==-1 else asset_keys[value]
        for key in asset_fields:result[key]=decode(result[key])
    return result

def check_status(state):
    if "$error" in state:
        error = int(jax.device_get(state["$error"]))
        if error:
            reasons = {1: "invalid dynamic-loop step/count or loop capacity exceeded", 2: "division by zero", 3: "invalid square world dimension", 4: "initial creation count differs from reserved fixed population", 5: "nil agent reference dereference", 6: "reserved population capacity exceeded or invalid initial count",7:'ordered agent-list capacity exceeded',8:'collection index outside its checked domain'}
            reasons[9]='invalid dynamic random bounds'
            reasons[10]='invalid spatial query radius'
            reasons[11]='invalid goto target or speed'
            reasons[12]='triangle side must be finite and strictly positive'
            reasons[13]='invalid image pixel index/handle'
            reasons[14]='matrix index outside its checked domain'
            reasons[15]='invalid diffusion proportion'
            raise RuntimeError(f"GAMAX runtime error {error}: {reasons.get(error, 'unknown error')}")


class Trace(list):
    """Backwards-compatible list of states, with separately preserved outputs."""
    def __init__(self, states, outputs, identities=None):
        super().__init__(states)
        self.outputs = outputs
        self.identities = [] if identities is None else identities


def run(module, steps, compiled=True, *, seed=None, parameters=None):
    if steps < 0: raise ValueError('steps must be nonnegative')
    state = module.reset() if seed is None and parameters is None else module.reset(seed=seed, parameters=parameters)
    check_status(state)
    observation=module.observe_jit if hasattr(module,'observe_jit') else lambda s:s
    def snapshot(state):
        result=canonical(observation(state),getattr(module,'ASSET_FIELDS',None),getattr(module,'ASSET_KEYS',None))
        for key in getattr(module,'MATRIX_FIELDS',[]):result[key]=[value for row in result[key] for value in row]
        for key,names in getattr(module,'FSM_FIELDS',{}).items():result[key]=[names[index] for index in result[key]]
        return result
    trace = Trace([snapshot(state)], module.outputs(state) if hasattr(module,'outputs') else [],
                  [module.identities(state)] if hasattr(module,'identities') else [])
    output_generations={event['id']:event['generation'] for event in trace.outputs if event['kind']=='save'}
    if any(generation>1 for generation in output_generations.values()):
        raise RuntimeError('Multiple save emissions per statement during initialization cannot be reconstructed')
    transition = module.step_jit if compiled else module.step
    for _ in range(steps):
        state = transition(state)
        check_status(state)
        trace.append(snapshot(state))
        if hasattr(module,'outputs'):
            for event in module.outputs(state):
                if event['kind']=='save':
                    previous=output_generations.get(event['id'],0)
                    if event['generation']>previous+1:raise RuntimeError('Multiple save emissions per statement in one cycle cannot be reconstructed')
                    if event['generation']>previous:
                        trace.outputs.append(event);output_generations[event['id']]=event['generation']
        if hasattr(module,'identities'):trace.identities.append(module.identities(state))
    return trace


def export_trace(trace, directory, engine):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f'{engine}_trace.json').write_text(json.dumps(trace, indent=2, allow_nan=False)+'\n')
    if getattr(trace,'outputs',None):
        (directory/f'{engine}_outputs.json').write_text(json.dumps(trace.outputs,indent=2,allow_nan=False)+'\n')
    for event in getattr(trace,'outputs',[]):
        if event['kind']=='save':
            relative=Path(event['path'])
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe output path')
            target=directory/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('w' if event['rewrite'] else 'a') as stream:stream.write(event['text'])
    if getattr(trace,'identities',None):
        (directory/f'{engine}_identities.json').write_text(json.dumps(trace.identities,indent=2,allow_nan=False)+'\n')
    with (directory / f'{engine}_variables.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['engine', 'step', 'field', 'agent_index', 'value'])
        for step, state in enumerate(trace):
            for field, value in state.items():
                values = enumerate(value) if isinstance(value, list) else [('', value)]
                for index, scalar in values:
                    writer.writerow([engine, step, field, index, scalar])
