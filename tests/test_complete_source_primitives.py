"""Regressions against native GAMA states for full-source compiler primitives."""
from pathlib import Path
import json
import pytest
from gamax.compiler import compile_model,CompileError
from gamax.runtime import load_module,run
from gamax.validation import compare_traces,compare_identities

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('case,campaign,steps',[
    ('grid_cast_sort_nil','grid-cast-sort-nil-fixed-native-coupled-2026-10-13',10),
    ('grid_geometry_distance','grid-geometry-distance-context-native-coupled-2026-10-13',3),
    ('deferred_diffusion','deferred-grid-primitives-native-coupled-2026-10-13',10),
    ('grid_species_order','grid-species-order-native-coupled-2026-10-13',10),
    ('return_argument_capture','return-argument-capture-native-coupled-2026-10-13',10),
    ('raster_matrix','raster-matrix-native-contract-2026-10-13',3),
])
def test_current_translation_matches_native_primitives(tmp_path,case,campaign,steps):
    module,manifest=compile_model(ROOT/'examples'/case/'main.gaml',tmp_path/'generated',experiment='validation')
    trace=run(load_module(module),steps,seed=2 if manifest["random_profile"] is not None else None)
    reference=ROOT/'results'/campaign/case/'2'
    assert compare_traces(json.loads((reference/'gama_trace.json').read_text()),trace,manifest['fields'])['passed']
    assert compare_identities(json.loads((reference/'gama_identities.json').read_text()),trace.identities)['passed']

@pytest.mark.parametrize('body',[
    'point value <- nil;',
    'init { unknown_effect; }',
    'init { create Actor number:1 with:(missing:1); }',
])
def test_unsupported_effects_and_nullable_persistent_points_fail_before_emission(tmp_path,body):
    source=tmp_path/'bad.gaml';source.write_text('model test global { '+body+' } species Actor {}')
    with pytest.raises(CompileError):compile_model(source,tmp_path/'generated')
    assert not (tmp_path/'generated').exists()


def test_experiment_initializers_apply_before_count_and_external_override(tmp_path):
    source=tmp_path/'selected.gaml'
    source.write_text('model arbitrary global { int count <- 7; init {create Actor number:count;} } species Actor {} experiment base type:gui virtual:true {parameter "Count" var:count init:3;} experiment chosen type:gui parent:base {}')
    module,manifest=compile_model(source,tmp_path/'generated',experiment='chosen',capacities={'Actor':10})
    loaded=load_module(module)
    assert manifest['populations']['Actor']==3
    assert len(run(loaded,0).identities[0]['Actor'])==3
    assert len(run(loaded,0,parameters={'count':5}).identities[0]['Actor'])==5
