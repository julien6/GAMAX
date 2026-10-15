"""GAMA-native oracle via its documented headless WebSocket protocol.

GAMA must execute the unmodified source. The adapter contains no source dynamics.
Steps are synchronous; step 0 is the post-init state, not GAMA's cycle variable.
"""
import json
import math
import platform
import sys
import jax
import jaxlib
import numpy as np


from .gama_client import GamaClient


def compare_identities(reference,candidate):
    """Compare native ordinal IDs, including empty and noncontiguous populations."""
    report=dict(passed=True,snapshots=len(reference),first_mismatch=None)
    def fail(detail):
        report.update(passed=False,first_mismatch=detail)
        return report
    if len(reference)!=len(candidate):return fail(dict(reason='identity_trace_length'))
    for step,(left,right) in enumerate(zip(reference,candidate)):
        if set(left)!=set(right):return fail(dict(step=step,reason='identity_schema'))
        for species in left:
            for values in (left[species],right[species]):
                if not isinstance(values,list) or any(type(i) is not int or i<0 for i in values):
                    return fail(dict(step=step,species=species,reason='identity_type'))
                if values!=sorted(set(values)):
                    return fail(dict(step=step,species=species,reason='identity_order_or_duplicate'))
            if left[species]!=right[species]:
                return fail(dict(step=step,species=species,reason='identity_value',reference=left[species],candidate=right[species]))
    return report


def compare_traces(reference, candidate, fields, atol=1e-12, rtol=1e-10):
    if not (math.isfinite(atol) and math.isfinite(rtol)) or atol < 0 or rtol < 0:
        raise ValueError('Tolerances must be finite and nonnegative')
    report = dict(passed=True, snapshots=len(reference), compared_values=0,
                  mismatched_values=0, max_float_absolute_error=0.0,
                  max_float_relative_error=0.0, float_atol=atol, float_rtol=rtol,
                  discrete_comparison='exact', first_mismatch=None)
    def mismatch(detail, count=1):
        report['passed'] = False
        report['mismatched_values'] += count
        if report['first_mismatch'] is None: report['first_mismatch'] = detail
    if len(reference) != len(candidate):
        mismatch(dict(reason='trace_length', reference=len(reference), candidate=len(candidate)))
        return report
    for step, (expected, actual) in enumerate(zip(reference, candidate)):
        if set(expected) != set(fields) or set(actual) != set(fields):
            mismatch(dict(step=step, reason='field_schema')); continue
        for field, type_ in fields.items():
            if type_ in ('image_file','values:image_file','fsm_state'):
                left,right=expected[field],actual[field]
                def shape(value):return [len(value),*[shape(item) for item in value]] if isinstance(value,list) else None
                def flatten_assets(value):
                    if isinstance(value,list):return [item for child in value for item in flatten_assets(child)]
                    return [value]
                if shape(left)!=shape(right):mismatch(dict(step=step,field=field,reason='asset_shape'));continue
                left,right=flatten_assets(left),flatten_assets(right);report['compared_values']+=len(left)
                for position,(a,b) in enumerate(zip(left,right)):
                    if any(value is not None and type(value) is not str for value in (a,b)) or a!=b:mismatch(dict(step=step,field=field,offset=position,reference=a,candidate=b,reason='asset_value'))
                continue
            if type_.startswith('agents:'):
                left,right = expected[field],actual[field]
                def valid(value):
                    return isinstance(value,list) and all(valid(v) if isinstance(v,list) else type(v) is int for v in value)
                def lengths(value):
                    return [len(value),*[lengths(v) for v in value if isinstance(v,list)]]
                def flatten(value):
                    return [x for v in value for x in (flatten(v) if isinstance(v,list) else [v])]
                if not valid(left) or not valid(right):
                    mismatch(dict(step=step,field=field,reason='type'));continue
                if lengths(left)!=lengths(right):
                    mismatch(dict(step=step,field=field,reason='ordered_collection_shape'));continue
                a,b=np.asarray(flatten(left),dtype=np.int64),np.asarray(flatten(right),dtype=np.int64)
                report['compared_values']+=a.size
                if not np.array_equal(a,b):
                    offset=int(np.flatnonzero(a!=b)[0])
                    mismatch(dict(step=step,field=field,reason='ordered_collection_value',offset=offset,reference=int(a[offset]),candidate=int(b[offset])),int(np.count_nonzero(a!=b)))
                continue
            left,right=expected[field],actual[field]
            if type_=='polygon':
                def polygon_structure(value):
                    return [len(value),*[polygon_structure(v) for v in value if isinstance(v,list)]] if isinstance(value,list) else None
                def polygon_flatten(value):
                    return [x for v in value for x in (polygon_flatten(v) if isinstance(v,list) else [v])]
                if polygon_structure(left)!=polygon_structure(right):
                    mismatch(dict(step=step,field=field,reason='geometry_shape'));continue
                left,right=polygon_flatten(left),polygon_flatten(right)
            a, b = np.asarray(left), np.asarray(right)
            if a.shape != b.shape:
                mismatch(dict(step=step, field=field, reason='shape', reference=list(a.shape), candidate=list(b.shape))); continue
            report['compared_values'] += a.size
            if a.size == 0: continue  # JSON empty arrays have no recoverable scalar dtype.
            scalar_type=type_.split(':')[1] if type_.startswith(('values:','matrix:')) else type_
            expected_kind = 'iu' if type_.startswith('agent:') else {'int': 'iu', 'bool': 'b', 'float': 'fiu', 'point': 'fiu', 'geometry': 'fiu', 'polygon':'fiu', 'rgb': 'iu'}[scalar_type]
            if a.dtype.kind not in expected_kind or b.dtype.kind not in expected_kind:
                mismatch(dict(step=step, field=field, reason='type'), a.size); continue
            if scalar_type in ('float', 'point', 'geometry','polygon'):
                finite = np.isfinite(a) & np.isfinite(b)
                delta = np.abs(a.astype(float)-b.astype(float))
                good = finite & (delta <= atol+rtol*np.abs(a))
                if np.any(finite):
                    report['max_float_absolute_error'] = max(report['max_float_absolute_error'], float(np.max(delta[finite])))
                    denominator = np.maximum(np.abs(a.astype(float)), np.finfo(float).tiny)
                    with np.errstate(over='ignore', invalid='ignore'):
                        relative = np.minimum(delta / denominator, np.finfo(float).max)
                    report['max_float_relative_error'] = max(report['max_float_relative_error'], float(np.max(relative[finite])))
            else: good = a == b
            if not np.all(good):
                flat_index = int(np.flatnonzero(~good)[0])
                mismatch(dict(step=step, field=field, agent_index=flat_index if a.ndim else None,
                              reference=a.reshape(-1)[flat_index].item(), candidate=b.reshape(-1)[flat_index].item(),
                              reason='value'), int(np.count_nonzero(~good)))
    return report


def environment():
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
                jax=jax.__version__, jaxlib=jaxlib.__version__, numpy=np.__version__,
                devices=[str(device) for device in jax.devices()], x64=bool(jax.config.jax_enable_x64))


def normalize_asset_observations(frames,manifest,path_map):
    """Map exact private-oracle file paths to verified portable resource keys."""
    from pathlib import Path
    mapping={str(Path(path).resolve()):key for path,key in path_map.items()}
    def normalize(value):
        if isinstance(value,list):return [normalize(item) for item in value]
        if value=='':return None
        if type(value) is not str:raise ValueError('Native image path is not a string')
        path=str(Path(value).resolve())
        if path not in mapping:raise ValueError('Native image path is outside the frozen declared resources: '+path)
        return mapping[path]
    return [{key:normalize(value) if key in manifest.get('asset_fields',{}) else value for key,value in frame.items()} for frame in frames]
