"""Confirmatory campaign schedule; planning does not imply executed evidence."""
import math
import random

SEEDS=tuple(range(1,11))
SCALES=(0.5,1.0,2.0)
HORIZONS=(100,1000,10000)
REPETITIONS=10
TIMEOUT_SECONDS=1800
FAILURES=('timeout','oom','capacity_exceeded','divergence','native_error','compiler_error')


def demographic_configuration(native, names, scale, bounds=None):
    """Keep ecological settings unchanged; round halves up then clip constraints.

    Names and bounds come from frozen experiment metadata, never inferred from
    the model filename. A missing demographic field blocks configuration.
    """
    if scale not in SCALES:raise ValueError('Unsupported scale')
    result=dict(native);bounds={} if bounds is None else bounds
    if not names:raise ValueError('Demographic metadata required')
    for name in names:
        if name not in native or type(native[name]) is not int or native[name]<0:
            raise ValueError('Invalid native demographic parameter: '+name)
        low,high=bounds.get(name,(0,None))
        if type(low) is not int or low<0 or (high is not None and (type(high) is not int or high<low)):
            raise ValueError('Invalid demographic bounds: '+name)
        value=max(low,math.floor(native[name]*scale+0.5))
        result[name]=min(high,value) if high is not None else value
    return result


def frame_cycles(horizon):
    if horizon not in HORIZONS:raise ValueError('Unsupported horizon')
    return [i*horizon//10 for i in range(11)]


def campaign_schedule(case_ids, *, order_seed=20261013):
    """Small cells first; randomize independent process conditions within cells.

    Each job needs frozen source/dependency/compiler/configuration identities and
    a passed fidelity certificate before execution. Returned rows are plans only.
    Native reusable-simulation amortization is an additional experiment, not one
    of these fresh-process jobs.
    """
    case_ids=list(case_ids)
    if not case_ids or len(set(case_ids))!=len(case_ids):raise ValueError('Distinct case IDs required')
    rng=random.Random(order_seed)
    rows=[]
    for scale in SCALES:
        for horizon in HORIZONS:
            for case in case_ids:
                cell=[]
                for seed in SEEDS:
                    for repetition in range(REPETITIONS):
                        for mode in ('compute','render'):
                            pair=f'{case}/scale-{scale}/cycles-{horizon}/seed-{seed}/rep-{repetition}/{mode}'
                            for engine in ('gama','jax'):
                                cell.append(dict(id=pair+'/'+engine,pair_id=pair,case=case,scale=scale,horizon=horizon,
                                                 seed=seed,repetition=repetition,mode=mode,engine=engine,
                                                 frames=frame_cycles(horizon) if mode=='render' else [],
                                                 timeout_seconds=TIMEOUT_SECONDS,status='planned_fidelity_gate_required'))
                rng.shuffle(cell);rows.extend(cell)
    return rows
