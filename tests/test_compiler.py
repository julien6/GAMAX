from pathlib import Path
import json
import pytest
import numpy as np
from gamax.compiler import compile_model, CompileError
from gamax.runtime import load_module, run
from gamax.validation import compare_traces

ROOT = Path(__file__).resolve().parents[1]


def module(tmp_path, source):
    path = tmp_path / 'unseen_model.gaml'
    path.write_text(source)
    generated, manifest = compile_model(path, tmp_path / 'generated')
    return load_module(generated), manifest


def test_effect_order_and_frozen_guards(tmp_path):
    # If must evaluate its condition once, even when the body changes that condition.
    model, _ = module(tmp_path, '''model unseen
    global { init { create actor number: 3; } }
    species actor {
        int x <- index; int snapshot <- 0; bool enabled <- true;
        reflex ordered when: enabled {
            if x < 2 { x <- 10; snapshot <- x; } else { snapshot <- -1; }
            enabled <- false;
            x <- x + 1;
        }
        reflex next when: not enabled { snapshot <- snapshot + 1; }
    }
    experiment validation type: gui {}''')
    assert run(model, 1)[1] == {'actor.x': [11, 11, 3], 'actor.snapshot': [11, 11, 0], 'actor.enabled': [False]*3}
    assert run(model, 2)[2]['actor.x'] == [11, 11, 3]


def test_initialization_at_creation_point(tmp_path):
    model, _ = module(tmp_path, (ROOT/'examples/initialization.gaml').read_text())
    assert run(model, 0)[0] == {'global.base': 7, 'cohort_a.value': [10, 12, 14],
                             'cohort_a.previous': [0]*3, 'cohort_b.value': [7, 8], 'cohort_b.previous': [0]*2}


def test_int32_overflow(tmp_path):
    model, _ = module(tmp_path, (ROOT/'examples/int_overflow.gaml').read_text())
    assert run(model, 1)[1]['counter.value'] == [-2147483639, -2147483638, -2147483637, -2147483636]


@pytest.mark.parametrize('name', ['ordered_agents', 'initialization', 'int_overflow', 'float_decay'])
def test_eager_and_jit_agree(tmp_path, name):
    path, manifest = compile_model(ROOT/'examples'/f'{name}.gaml', tmp_path)
    generated = load_module(path)
    report = compare_traces(run(generated, 20, False), run(generated, 20, True), manifest['fields'])
    assert report['passed'], report


@pytest.mark.parametrize('source,code', [
    ('model m global { int x <- rnd(10); }', 'GAMAX_RANDOM'),
    ('model m global { int x <- true; }', 'GAMAX_TYPE'),
    ('model m global { int x <- missing; }', 'GAMAX_NAME'),
    ('model m global { int x <- 0; reflex r { x <- 1 / 2; } }', 'GAMAX_TYPE'),
    ('model m global { init { if true { create a number: 1; } } } species a {}', 'GAMAX_EFFECT'),
    ('model m global {} species a {}', 'GAMAX_UNSUPPORTED'),
    ('model m global {} experiment e type: gui { init { write 1; } }', 'GAMAX_EXPERIMENT'),
    ('model m global { int cycle <- 0; }', 'GAMAX_UNSUPPORTED'),
    ('model m global { int x <- 2147483648; }', 'GAMAX_UNSUPPORTED'),
])
def test_unsupported_rejected_before_codegen(tmp_path, source, code):
    path = tmp_path/'bad.gaml'; path.write_text(source)
    target = tmp_path/'generated'
    with pytest.raises(CompileError) as error: compile_model(path, target)
    assert error.value.diagnostic['code'] == code
    assert error.value.diagnostic['line'] >= 1 and error.value.diagnostic['column'] >= 1
    assert not target.exists()


def test_codegen_deterministic_and_source_map_present(tmp_path):
    path, manifest = compile_model(ROOT/'examples/ordered_agents.gaml', tmp_path/'a')
    other, second = compile_model(ROOT/'examples/ordered_agents.gaml', tmp_path/'b')
    assert path.read_bytes() == other.read_bytes()
    assert manifest['generated_sha256'] == second['generated_sha256']
    mapping = json.loads((path.parent/'source_map.json').read_text())
    assert any(item['source_line'] == 15 for item in mapping)


def test_comparator_rejects_missing_or_misaligned_data():
    fields = {'a.x': 'int'}
    assert not compare_traces([{'a.x': [1, 2]}], [{'a.x': [1]}], fields)['passed']
    assert not compare_traces([{'a.x': [1]}], [], fields)['passed']
    assert not compare_traces([{'a.x': [1]}], [{}], fields)['passed']
    assert not compare_traces([{'a.x': [1]}], [{'a.x': [1.0]}], fields)['passed']


def test_float_tolerance_and_discrete_exactness():
    assert compare_traces([{'x': 1.0}], [{'x': 1.0+1e-11}], {'x': 'float'})['passed']
    assert not compare_traces([{'x': 1.0}], [{'x': 1.01}], {'x': 'float'})['passed']
    assert not compare_traces([{'x': 1}], [{'x': 2}], {'x': 'int'}, atol=1000)['passed']
    assert not compare_traces([{'x': float('nan')}], [{'x': float('nan')}], {'x': 'float'})['passed']
    with pytest.raises(ValueError): compare_traces([], [], {}, atol=float('nan'))


def test_shared_resource_requires_sequential_agent_schedule(tmp_path):
    model, _ = module(tmp_path, (ROOT/'examples/resource_competition.gaml').read_text())
    trace = run(model, 2)
    assert trace[1]['forager.collected'] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert trace[2]['forager.collected'] == [2, 2, 2, 0, 0, 0, 0, 0]
    assert trace[2]['forager.previous'] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert trace[2]['global.available'] == 0


def test_float_equality_rejected_until_gama_rounding_is_lowered(tmp_path):
    path = tmp_path/'rounding.gaml'
    path.write_text('model m global { bool x <- 0.1 + 0.2 = 0.3; }')
    with pytest.raises(CompileError, match='floating equality'): compile_model(path, tmp_path/'output')
