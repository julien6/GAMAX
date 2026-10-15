"""Recheck emitted action semantics against independently archived native states."""
from pathlib import Path
import json
import pytest
from gamax.compiler import compile_model, CompileError
from gamax.runtime import load_module,run
from gamax.validation import compare_traces,compare_identities

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('seed',range(1,11))
def test_returns_nested_calls_and_conditional_draws_match_native(tmp_path,seed):
    path,manifest=compile_model(ROOT/'examples/returning_actions/main.gaml',tmp_path/'generated')
    trace=run(load_module(path),100,seed=seed)
    reference=ROOT/'tests/fixtures/native/returning-actions-ten-seeds-native-coupled-2026-10-13/returning_actions'/str(seed)
    assert compare_traces(json.loads((reference/'gama_trace.json').read_text()),trace,manifest['fields'])['passed']
    assert compare_identities(json.loads((reference/'gama_identities.json').read_text()),trace.identities)['passed']

@pytest.mark.parametrize('body',['return unknown();','if true { return 1; }','return recurse();'])
def test_unknown_recursive_or_partial_return_rejected(tmp_path,body):
    source=tmp_path/'bad.gaml'
    source.write_text('model bad global { int value <- 0; int recurse { '+body+' } reflex r { value <- recurse(); } }')
    with pytest.raises(CompileError):compile_model(source,tmp_path/'generated')
    assert not (tmp_path/'generated').exists()

def test_rgba_pixels_match_native(tmp_path):
    path,manifest=compile_model(ROOT/'examples/raster_pixels/main.gaml',tmp_path/'generated')
    trace=run(load_module(path),3,seed=2)
    root=ROOT/'tests/fixtures/native/raster-pixels-native-contract-2026-10-13/raster_pixels/2'
    assert compare_traces(json.loads((root/'gama_trace.json').read_text()),trace,manifest['fields'])['passed']
