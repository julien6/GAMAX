from pathlib import Path
import json
import pytest
from gamax.compiler import compile_model,CompileError
from gamax.runtime import load_module,run,canonical
from gamax.validation import compare_traces
ROOT=Path(__file__).resolve().parents[1]


def test_facets_defaults_and_external_coercion(tmp_path):
    path,manifest=compile_model(ROOT/'examples/declaration_facets/main.gaml',tmp_path/'out')
    model=load_module(path)
    native=json.loads((ROOT/'tests/fixtures/native/declaration-facets-probe-2026-10-13/declaration_facets/1/gama_trace.json').read_text())
    assert compare_traces(native,run(model,10),manifest['fields'])['passed']
    state=canonical(model.reset(parameters={'bounded':99,'continuous':-20.0}))
    assert state['global.bounded']==7 and state['global.continuous']==-1.5
    assert state['global.inconsistent']==9
    assert 'global.lower' in manifest['effects']['global.constraints']['reads']
    source=tmp_path/'before.gaml'
    source.write_text('model facets global { int n min: 1 max: 3 <- 7; }')
    p,_=compile_model(source,tmp_path/'before')
    assert canonical(load_module(p).reset())['global.n']==3


@pytest.mark.parametrize('fragment',['int n <- 1 min: true;','int n min: 1 min: 2;','bool n min: 0;','int n category: "display";'])
def test_unknown_or_ill_typed_facets_fail_closed(tmp_path,fragment):
    source=tmp_path/'bad.gaml';source.write_text('model bad global {'+fragment+'}')
    with pytest.raises(CompileError):compile_model(source,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('operator',['/','div','mod'])
def test_dynamic_zero_divisor_errors_are_masked_and_explicit(tmp_path,operator):
    type_='float' if operator=='/' else 'int'
    source=tmp_path/'checked.gaml'
    source.write_text(f'model checked global {{ int divisor; bool enabled <- false; {type_} result <- 0; reflex r {{ result <- enabled ? 9 {operator} divisor : 3; }} }}')
    p,_=compile_model(source,tmp_path/'out');model=load_module(p)
    assert run(model,1)[1]['global.result']==3
    with pytest.raises(RuntimeError,match='division by zero'):
        run(model,1,parameters={'enabled':True})
    assert run(model,1,parameters={'enabled':True,'divisor':2})[1]['global.result']==(4.5 if operator=='/' else 4 if operator=='div' else 1)
