from pathlib import Path

import numpy as np
import pytest

from gamax.compiler import compile_model, CompileError, Parser, Generator
from gamax.runtime import load_module, run
from benchmarks.scenarios import CASES, materialize, small_world

ROOT = Path(__file__).resolve().parents[1]


def compile_text(tmp_path, text):
    source = tmp_path / 'source.gaml'
    source.write_text(text)
    module, manifest = compile_model(source, tmp_path / 'generated')
    return load_module(module), manifest


def test_double_precision_reducers_and_synchronous_neighbor_reads(tmp_path):
    module, _ = compile_text(tmp_path, (ROOT / 'examples/neighborhood.gaml').read_text())
    trace = run(module, 2)
    assert trace[1]['patch.average'][1] == pytest.approx(4 / 3, abs=1e-15)
    assert trace[1]['patch.previous'] == trace[0]['patch.value']
    assert trace[2]['patch.previous'] == trace[1]['patch.value']
    assert trace[1]['patch.neighbors'] == [2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2]


def test_ask_reads_latest_shared_state_in_agent_order(tmp_path):
    source = '''model ordered_ask
    global { int budget <- 2; init { create worker number: 4; }
      reflex allocate { ask worker { if budget > 0 { reward <- reward + 1; budget <- budget - 1; } } }
    }
    species worker { int reward <- 0; }
    experiment validation type: gui {}'''
    module, manifest = compile_text(tmp_path, source)
    assert run(module, 1)[1]['worker.reward'] == [1, 1, 0, 0]
    assert manifest['effects']['global.allocate']['writes'] == ['global.budget', 'worker.reward']
    mutant = tmp_path / 'mutant.py'
    mutant.write_text(Generator(Parser(source, 'test').parse(), reverse_agents=True).generate())
    assert run(load_module(mutant), 1)[1]['worker.reward'] == [0, 0, 1, 1]


def test_double_division_degrees_and_truncating_integer_arithmetic(tmp_path):
    module, _ = compile_text(tmp_path, '''model numeric
      global { float ratio <- 1 / 3; float angle <- sin(30); float magnitude <- sqrt(2);
        int quotient <- -7 div 3; int remainder <- -7 mod 3; int saturated <- int(3000000000.0); }
      experiment validation type: gui {}''')
    state = run(module, 0)[0]
    assert state['global.ratio'] == pytest.approx(1 / 3, abs=1e-15)
    assert state['global.angle'] == pytest.approx(0.5, abs=1e-15)
    assert state['global.magnitude'] == pytest.approx(np.sqrt(2), abs=1e-15)
    assert state['global.quotient'] == -2 and state['global.remainder'] == -1
    assert state['global.saturated'] == 2147483647


def test_jit_keeps_intermediate_double_rounding(tmp_path):
    module, manifest = compile_text(tmp_path, '''model rounding
      global { float a <- 0.17554443101104242; float b <- 12.567229590660790;
        float c <- 13.574042765875694; float result <- 0.0;
        reflex compute { result <- a + (b - c) * 0.01; } }
      experiment validation type: gui {}''')
    expected = 0.17554443101104242 + (12.567229590660790 - 13.574042765875694) * 0.01
    assert run(module, 1)[1]['global.result'] == expected
    assert manifest['jit_contract']['floating_models_disable_hlo_fusion']


def test_reserved_native_species_rejected(tmp_path):
    with pytest.raises(CompileError, match='built-in agent species'):
        compile_text(tmp_path, 'model invalid global { init { create agent number: 1; } } species agent {}')


@pytest.mark.parametrize('fragment', [
    'float x <- 1 / 0;',
    'int x <- length(actor);',
    'float x <- myself.x;',
    'float x <- ' + '9' * 400 + '.0;',
])
def test_new_unsafe_expressions_rejected_before_writes(tmp_path, fragment):
    source = tmp_path / 'bad.gaml'
    source.write_text('model bad global { ' + fragment + ' init { create actor number: 2; } } species actor {}')
    with pytest.raises(CompileError): compile_model(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('body', [
    'ask worker { value <- length(worker where (length(worker where each.index = index) > 0)); }',
])
def test_nested_collection_queries_fail_closed(tmp_path, body):
    source = tmp_path / 'nested.gaml'
    source.write_text('model nested global { init { create worker number: 2; } reflex r { ' + body + ' } } species worker { int value <- 0; }')
    with pytest.raises(CompileError): compile_model(source, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('case', CASES)
def test_all_family_sources_compile_without_python_dynamics(tmp_path, case):
    text = materialize(case, 4)
    assert text == materialize(case, 4) and text != materialize(case, 5)
    module, manifest = compile_text(tmp_path, text)
    assert manifest['model'] == case
    assert set(run(module, 1)[1]) == set(manifest['fields'])


def test_frozen_small_world_has_no_duplicate_edges_or_self_loops():
    for seed in range(10):
        graph = small_world(np.random.default_rng(seed))
        edges = [tuple(sorted((i, j))) for i, links in enumerate(graph) for j in links]
        assert len(edges) == len(set(edges)) == 64
        assert all(i != j for i, j in edges)
