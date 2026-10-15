from pathlib import Path

import numpy as np
import pytest

from gamax.compiler import compile_model, CompileError
from gamax.runtime import load_module, run

ROOT = Path(__file__).resolve().parents[1]


def compile_text(tmp_path, text):
    path = tmp_path / 'model.gaml'; path.write_text(text)
    module, manifest = compile_model(path, tmp_path / 'generated')
    return load_module(module), manifest


def test_imported_actions_native_random_stream_and_bundle(tmp_path):
    source = ROOT / 'examples/imported_actions/main.gaml'
    module, manifest = compile_model(source, tmp_path / 'generated')
    trace = run(load_module(module), 2)
    # Independent native GAMA probe, retained under results/local/compiler-extension-probe.
    assert trace[1]['walker.ticks'] == [1, 2, 1, 2, 3, 3, 1, 4]
    assert trace[1]['walker.value'][0] == 0.14185948788509606
    assert trace[2]['global.budget'] == 97
    assert len(manifest['dependencies']) == 2
    assert not any(key.startswith('$') for key in trace[0])
    rebuilt, _ = compile_model(tmp_path / 'generated' / manifest['source_bundle_entry'], tmp_path / 'rebuilt')
    assert run(load_module(rebuilt), 2) == trace
    assert manifest['effects']['walker.evolve']['writes'] == ['global.budget', 'walker.ticks', 'walker.value']


def test_action_parameter_evaluated_before_action_writes(tmp_path):
    module, _ = compile_text(tmp_path, '''model parameters
      global { int value <- 4;
        action change(int amount) { value <- 10; value <- value + amount; }
        reflex invoke { do change(value); }
      } experiment validation type: gui {}''')
    assert run(module, 1)[1]['global.value'] == 14


def test_block_local_does_not_escape_and_branch_mask_frozen(tmp_path):
    module, _ = compile_text(tmp_path, '''model locals
      global { int value <- 1; int result <- 0;
        reflex r { if value = 1 { int before <- value; value <- 2; before <- before + 3; result <- before; } }
      } experiment validation type: gui {}''')
    assert run(module, 1)[1] == {'global.value': 2, 'global.result': 4}


def test_random_false_guards_branches_and_short_circuit_do_not_consume(tmp_path):
    def source(skip):
        return '''model random_branches global { float value <- 0.0;
          init { rng <- "java"; seed <- 17; }
          reflex disabled when: false { value <- rnd(1.0); }
          reflex run { ''' + skip + ''' value <- rnd(1.0); }
        } experiment validation type: gui {}'''
    reference, _ = compile_text(tmp_path, source(''))
    expected = run(reference, 5)
    candidate, _ = compile_text(tmp_path, source('if false { value <- rnd(1.0); } if false and flip(1.0) { value <- 4.0; } if true or flip(1.0) {}'))
    assert run(candidate, 5) == expected
    assert run(candidate, 5, compiled=False) == expected


def test_moving_boundary_reverses_without_clipping(tmp_path):
    module, manifest = compile_text(tmp_path, (ROOT / 'examples/moving_skill.gaml').read_text())
    trace = run(module, 4)
    assert trace[2]['walker.location'][0] == [99.0, 5.0, 0.0]
    assert trace[3]['walker.location'][0] == [99.0, 5.0, 0.0]
    assert trace[3]['walker.heading'][0] == 180.0
    assert trace[3]['walker.real_speed'][0] == 0.0
    assert trace[4]['walker.location'][0][0] == 97.0
    assert manifest['spatial_profile']['step_seconds'] == 1
    assert set(manifest['effects']['walker.walk']['writes']) == {'walker.location','walker.heading','walker.real_speed','walker.displacement'}


def test_point_components_and_distance(tmp_path):
    module, _ = compile_text(tmp_path, '''model points global {
      point origin <- {0, 0}; point target <- {3, 4, 0};
      float distance <- origin distance_to target; float coordinate <- target.x;
      point doubled <- target * 2;
    } experiment validation type: gui {}''')
    state = run(module, 0)[0]
    assert state['global.distance'] == 5.0 and state['global.coordinate'] == 3.0
    assert state['global.doubled'] == [6.0, 8.0, 0.0]


@pytest.mark.parametrize('member', [
    'action recurse { do recurse; }',
    'action unused { int temporary <- missing; }',
    'action a(int x) { } reflex r { do a(true); }',
    'reflex r { int temporary <- 1; } reflex s { value <- temporary; }',
    'init { rng <- "java"; seed <- 0; }',
    'init { rng <- "unknown"; seed <- 17; } reflex r { value <- rnd(10); }',
    'init { value <- 1; rng <- "java"; seed <- 17; value <- 1; }',
    'init { rng <- "java"; seed <- 17; } reflex r { value <- rnd(-1); }',
])
def test_invalid_extensions_fail_before_output(tmp_path, member):
    source = tmp_path / 'bad.gaml'; source.write_text('model bad global { int value <- 0; ' + member + ' }')
    with pytest.raises(CompileError): compile_model(source, tmp_path / 'generated')
    assert not (tmp_path / 'generated').exists()


def test_import_cycles_missing_and_source_locations(tmp_path):
    entry = tmp_path / 'entry.gaml'; dependency = tmp_path / 'common.gaml'
    entry.write_text('model entry import "common.gaml" global {}')
    dependency.write_text('model common import "entry.gaml" global {}')
    with pytest.raises(CompileError, match='Cyclic import'): compile_model(entry, tmp_path / 'generated')
    dependency.unlink()
    with pytest.raises(CompileError, match='does not exist'): compile_model(entry, tmp_path / 'generated')
    dependency.write_text('model common global { int value <- missing; }')
    with pytest.raises(CompileError) as error: compile_model(entry, tmp_path / 'generated')
    assert error.value.diagnostic['file'] == str(dependency)
    assert not (tmp_path / 'generated').exists()


def test_location_rhs_evaluated_before_native_heading_setter(tmp_path):
    module, _ = compile_text(tmp_path, (ROOT / 'examples/moving_assignment.gaml').read_text())
    state = run(module, 1)[1]
    # Native GAMA probe also checks the sub-0.01 move retains the setter's cap.
    assert state['walker.location'][0] == pytest.approx([22.004472135954998, 21.0022360679775, 0.0], abs=1e-13)
    assert state['walker.heading'][0] == pytest.approx(26.56505117707799, abs=1e-13)


def test_emitted_locals_stable_across_hash_seeds(tmp_path):
    import os
    import subprocess
    import sys
    source = tmp_path / 'model.gaml'
    source.write_text('model stable global { int value <- 0; reflex r { int aa <- 1; int bb <- 2; value <- aa + bb; } }')
    for seed in ('1', '2'):
        subprocess.run([sys.executable, '-c', 'from gamax.compiler import compile_model; import sys; compile_model(sys.argv[1],sys.argv[2])', str(source), str(tmp_path/seed)],
                       env=dict(os.environ, PYTHONHASHSEED=seed), check=True)
    assert (tmp_path/'1/model_jax.py').read_bytes() == (tmp_path/'2/model_jax.py').read_bytes()


def test_static_loops_branches_and_named_arguments(tmp_path):
    source = (ROOT / 'examples/control_flow/main.gaml').read_text().replace('seed <- 17;', 'seed <- 1;')
    module, _ = compile_text(tmp_path, source)
    compiled = run(module, 2)
    assert compiled == run(module, 2, compiled=False)
    assert compiled[1]['global.total'] == 27
    # Native seed-1 probe: declaration-order argument draws, despite reversed facets.
    assert compiled[1]['global.code'] == 25
    assert compiled[1]['global.sampled'] == 1.6767703211753724
    assert compiled[1]['walker.count'] == [3] * 4
    assert compiled[2]['global.total'] == 54
    assert compiled[2]['walker.count'] == [6] * 4


@pytest.mark.parametrize('statement', [
    'loop times: 0 { total <- missing; }',
    'loop times: 1025 { total <- k; }',
    'loop k from: 0 to: 2 step: 0 {}',
    'loop times: true {}',
    'loop total from: 0 to: 1 {}',
    'loop k from: 0 to: 0 {} total <- k;',
    'loop k from: 0 to: 0 { loop k times: 2 {} }',
    'do a wrong: 2;',
    'do a amount: 2 amount: 3;',
])
def test_invalid_control_flow_rejected_without_output(tmp_path, statement):
    source = tmp_path / 'bad.gaml'
    source.write_text('model invalid global { int total <- 0; action a(int amount) {} reflex r {' + statement + '} }')
    with pytest.raises(CompileError): compile_model(source, tmp_path / 'generated')
    assert not (tmp_path / 'generated').exists()


def test_nested_static_loops_expansion_limit(tmp_path):
    with pytest.raises(CompileError, match='4096'):
        compile_text(tmp_path, '''model expansion global { int value <- 0;
          reflex r { loop times: 100 { loop times: 100 { value <- value + 1; } } }
        }''')
    assert not (tmp_path / 'generated').exists()


def test_random_species_initialization_native_order(tmp_path):
    module, manifest = compile_text(tmp_path, (ROOT / 'examples/random_initialization/main.gaml').read_text())
    state = run(module, 0)[0]
    assert state['walker.choice'] == [5, 3, 8, 9, 3, 7, 10, 9]
    assert state['walker.value'][0] == 0.14185948788509606
    assert state['walker.location'][0] == [39.39897884411283, 99.3550817543521, 0.0]
    assert run(module, 2) == run(module, 2, compiled=False)
    assert manifest['random_profile']['reseed'].startswith('prefix')


def test_random_species_initialization_requires_precreation_seed(tmp_path):
    source = (ROOT / 'examples/random_initialization/main.gaml').read_text()
    source = source.replace('rng <- "java"; seed <- 17; create walker number: 8;',
                            'create walker number: 8; rng <- "java"; seed <- 17;')
    with pytest.raises(CompileError, match='Random initialization'):
        compile_text(tmp_path, source)
    assert not (tmp_path / 'generated').exists()


def test_precreation_seed_with_moving_fails_explicitly(tmp_path):
    source = (ROOT / 'examples/moving_skill.gaml').read_text()
    source = source.replace('init {', 'init { rng <- "java"; seed <- 17;', 1)
    with pytest.raises(CompileError, match='implicit random initialization'):
        compile_text(tmp_path, source)
    assert not (tmp_path / 'generated').exists()


def test_monomorphic_parent_references_and_experiment_inheritance(tmp_path):
    source=tmp_path/'renamed.gaml'
    source.write_text("""model arbitrary_name
    global {int size <- 2; list<Ancestor> pool; int total update:sum(pool collect each.value); init {create Descendant number:size; pool <- Descendant as list;}}
    species Ancestor {int value <- 1;}
    species Descendant parent:Ancestor {int value <- 2;}
    experiment base virtual:true {parameter 'Size' category:'Population' var:size;}
    experiment 'Derived' type:gui parent:base {output {display scene {species Descendant;}}}
    """)
    path,manifest=compile_model(source,tmp_path/'out',experiment='Derived')
    assert manifest['reference_aliases']=={'Ancestor':'Descendant'}
    assert manifest['fields']['global.pool']=='agents:Descendant'
    assert manifest['presentation']['experiments']['Derived']['parameters']==[{'label':'Size','category':'Population','variable':'size'}]
    assert run(load_module(path),1)[1]['global.total']==4
    with pytest.raises(CompileError,match='virtual'):compile_model(source,tmp_path/'virtual',experiment='base')
    with pytest.raises(CompileError,match='monomorphic'):compile_model(source,tmp_path/'ambiguous',experiment='Derived',capacities={'Ancestor':1})
