from pathlib import Path
import json
import numpy as np
import pytest
from gamax.compiler import compile_model, implementation_identity, CompileError
from gamax.runtime import load_module, run, canonical

ROOT=Path(__file__).resolve().parents[1]


def test_modular_compiler_identity_covers_semantics(tmp_path):
    path,manifest=compile_model(ROOT/'examples/resource_competition.gaml',tmp_path/'generated')
    files,digest=implementation_identity()
    assert manifest['implementation_sha256']==digest
    assert manifest['implementation_files']==files
    assert {'frontend.py','analysis.py','lowering.py','ir.py','randomness.py'} <= set(files)
    assert path.is_file()


def test_mersenne_native_initialization_and_twist_boundaries(tmp_path):
    path,manifest=compile_model(ROOT/'examples/mersenne_initialization/main.gaml',tmp_path/'generated')
    module=load_module(path)
    trace=run(module,100)
    # Independent native traces saved by the ten-seed differential campaign.
    native=json.loads((ROOT/'tests/fixtures/native/mersenne-initialization-corrected-2026-10-13/mersenne_initialization/1/gama_trace.json').read_text())
    # Seed 17 fixture is checked against a fresh seed-1 compilation to match archive.
    source=tmp_path/'seed1.gaml';source.write_text((ROOT/'examples/mersenne_initialization/main.gaml').read_text().replace('seed <- 17;', 'seed <- 1;'))
    p,_=compile_model(source,tmp_path/'seed1')
    assert run(load_module(p),100)==native
    assert trace==run(module,100)
    assert manifest['random_profile']['generator']=='mersenne'
    np.testing.assert_equal(module.reset()['$rng'][0].dtype,np.dtype('uint32'))


def test_dynamic_loop_evaluation_and_scope(tmp_path):
    path,_=compile_model(ROOT/'examples/dynamic_loops/main.gaml',tmp_path/'generated')
    module=load_module(path);compiled=run(module,2)
    assert compiled==run(module,2,compiled=False)
    assert compiled[0]['walker.count']==[2]*4
    assert compiled[1]['global.total']==13
    assert compiled[2]['global.total']==26
    assert compiled[1]['walker.count']==[5]*4
    assert not any(k.startswith('$') for k in compiled[1])


@pytest.mark.parametrize('statement', [
    'int stride <- 0; loop k from: 0 to: n step: stride {}',
    'loop times: n + 1000000 {}',
])
def test_invalid_dynamic_trip_count_is_explicit(tmp_path,statement):
    source=tmp_path/'invalid.gaml';source.write_text('model invalid global { int n <- 2; reflex r {'+statement+'} }')
    path,_=compile_model(source,tmp_path/'generated');module=load_module(path)
    with pytest.raises(RuntimeError,match='dynamic-loop'):
        run(module,1)
    with pytest.raises(RuntimeError,match='dynamic-loop'):
        canonical(module.step_jit(module.reset()))


def test_dynamic_loop_body_local_does_not_escape(tmp_path):
    source=tmp_path/'bad.gaml';source.write_text('model bad global { int n <- 2; reflex r { loop k from: 0 to: n { int inside <- k; } n <- inside; } }')
    with pytest.raises(CompileError):compile_model(source,tmp_path/'generated')
    assert not (tmp_path/'generated').exists()


def test_external_reset_configuration_and_validation(tmp_path):
    source=tmp_path/'config.gaml'
    source.write_text('model config global { int n <- 2; int dependent <- n + 1; init { rng <- "mersenne"; seed <- 17; } reflex r { n <- n + rnd(9); } } experiment validation type: gui {}')
    path,manifest=compile_model(source,tmp_path/'out',experiment='validation',capacities={})
    module=load_module(path)
    initial=canonical(module.reset(seed=2,parameters={'n':7}))
    assert initial=={'global.n':7,'global.dependent':8}
    assert run(module,1,seed=2,parameters={'n':7})[1]['global.n']==11
    for parameters in ({'unknown':1},{'n':True},{'n':2.5},{'n':2**31}):
        with pytest.raises(ValueError):module.reset(parameters=parameters)
    with pytest.raises(ValueError):module.reset(seed=0)
    assert manifest['selected_experiment']=='validation'
    with pytest.raises(CompileError,match='Unknown experiment'):
        compile_model(source,tmp_path/'invalid',experiment='missing')
    assert not (tmp_path/'invalid').exists()


def test_capacity_below_default_population_is_rejected_before_emission(tmp_path):
    with pytest.raises(CompileError,match='GAMAX_CAPACITY'):
        compile_model(ROOT/'examples/dynamic_loops/main.gaml',tmp_path/'generated',capacities={'walker':3})
    assert not (tmp_path/'generated').exists()


def test_generated_module_is_standalone_and_name_independent(tmp_path):
    import subprocess
    import sys
    original=(ROOT/'examples/mersenne_initialization/main.gaml').read_text()
    source=tmp_path/'renamed'/'different.gaml';source.parent.mkdir()
    source.write_text(original.replace('model mersenne_initialization','model unrelated_name'))
    first,_=compile_model(ROOT/'examples/mersenne_initialization/main.gaml',tmp_path/'first')
    second,_=compile_model(source,tmp_path/'elsewhere')
    assert first.read_bytes()==second.read_bytes()
    # -I removes the repository from sys.path; the generated module imports JAX
    # and standard-library code only, without GAMAX or a GAMA process.
    script='import importlib.util, sys; assert "gamax" not in sys.modules; s=importlib.util.spec_from_file_location("standalone",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); state=m.step_jit(m.reset(seed=2)); import jax; jax.block_until_ready(state); assert "gamax" not in sys.modules; print("standalone-ok")'
    result=subprocess.run([sys.executable,'-I','-c',script,str(second)],cwd=tmp_path,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='standalone-ok'
