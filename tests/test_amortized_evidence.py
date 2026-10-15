"""Reject reuse evidence that does not cover the exact source and adapter."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('amortized_evidence',ROOT/'scripts/benchmark_amortized.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('change',(None,'seed','snapshot','tolerance','source','adapter','incomplete'))
def test_reuse_proof_requires_repeat_fidelity_and_exact_binding(tmp_path,change):
    comparison={'passed':True,'snapshots':101,'float_atol':1e-12,'float_rtol':1e-10}
    rows=[{'seed':s,'comparison':dict(comparison),'identities':{'passed':True,'snapshots':101}} for s in (2,7,2)]
    binding={'source_sha256':'original','fields':{'global.value':'int'},'experiment':'validation',
             'adapter_sha256':hashlib.sha256((ROOT/'benchmarks/reused_native_checkpoints.py').read_bytes()).hexdigest(),
             'preparation_sha256':hashlib.sha256((ROOT/'scripts/benchmark_official.py').read_bytes()).hexdigest()}
    evidence={'complete':True,'passed':True,'steps':100,'configuration':binding,'reports':rows}
    if change=='seed':rows[-1]['seed']=10
    if change=='snapshot':rows[0]['comparison']['snapshots']=100
    if change=='tolerance':rows[0]['comparison']['float_atol']=1e-6
    if change=='source':binding['source_sha256']='different'
    if change=='adapter':binding['adapter_sha256']='different'
    if change=='incomplete':evidence['complete']=False
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(evidence))
    cfg={'native_reuse_validation':str(proof),'experiment':'validation','manifest':{'source_sha256':'original','fields':{'global.value':'int'}}}
    if change is None:module.checked_native_reuse(cfg,100)
    else:
        with pytest.raises(RuntimeError):module.checked_native_reuse(cfg,100)
