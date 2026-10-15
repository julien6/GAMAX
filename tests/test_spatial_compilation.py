from pathlib import Path
import json
import subprocess
import sys
import pytest

from gamax.compiler import CompileError, compile_model
from gamax.runtime import canonical, load_module, run, export_trace
from gamax.validation import compare_traces

ROOT = Path(__file__).resolve().parents[1]


def test_mutable_list_shuffle_matches_all_native_seeds(tmp_path):
    path,manifest=compile_model(ROOT/'examples/mutable_lists_shuffle/main.gaml',tmp_path/'out',capacities={'Walker':6})
    for seed in range(1,11):
        native=json.loads((ROOT/f'tests/fixtures/native/mutable-lists-shuffle-ten-seeds-2026-10-13/mutable_lists_shuffle/{seed}/gama_trace.json').read_text())
        assert compare_traces(native,run(load_module(path),100,seed=seed),manifest['fields'])['passed']
    assert manifest['collection_fields']['global.cache']=={'capacity':6,'mutability':'ordered append/remove/replace'}


def test_typed_value_lists_preserve_boolean_and_fractional_point_values(tmp_path):
    source=tmp_path/'values.gaml'
    source.write_text('model values global {list<bool> flags <- [true,false]; list<point> points <- [{1.25,2.75},{-3.5,4.125}]; int rank; bool flag; point destination <- {0,0}; reflex select {flag <- flags at rank; destination <- points at rank;}}')
    path,manifest=compile_model(source,tmp_path/'out')
    model=load_module(path)
    trace=run(model,1,parameters={'rank':1})
    assert trace[1]['global.flags']==[True,False]
    assert all(type(v) is bool for v in trace[1]['global.flags'])
    assert trace[1]['global.destination']==[-3.5,4.125,0.0]
    assert trace[1]['global.flag'] is False
    assert manifest['runtime_errors']['8']=='collection index outside its checked domain'
    for rank in (-1,2):
        with pytest.raises(RuntimeError,match='collection index'):
            run(model,1,parameters={'rank':rank})


def test_value_lists_match_native_full_trace(tmp_path):
    path,manifest=compile_model(ROOT/'examples/value_lists/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/value-lists-native-probe-c-2026-10-13/value_lists/2/gama_trace.json').read_text())
    assert compare_traces(native,run(load_module(path),100,seed=2),manifest['fields'])['passed']


def test_dynamic_random_bounds_fail_explicitly(tmp_path):
    path,manifest=compile_model(ROOT/'examples/dynamic_random_bounds/main.gaml',tmp_path/'out')
    assert manifest['runtime_errors']['9']=='invalid dynamic random bounds'
    for parameters in ({'upper':-5},{'lower':8},{'start':4.0}):
        with pytest.raises(RuntimeError,match='random bounds'):
            run(load_module(path),100,seed=2,parameters=parameters)


def test_dynamic_random_bounds_match_native_all_seeds(tmp_path):
    path,manifest=compile_model(ROOT/'examples/dynamic_random_bounds/main.gaml',tmp_path/'out')
    model=load_module(path)
    for seed in range(1,11):
        native=json.loads((ROOT/f'tests/fixtures/native/dynamic-random-bounds-native-2026-10-13/dynamic_random_bounds/{seed}/gama_trace.json').read_text())
        assert compare_traces(native,run(model,100,seed=seed),manifest['fields'])['passed']


def test_import_overrides_and_global_init_order_match_native(tmp_path):
    path,manifest=compile_model(ROOT/'examples/import_overrides/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/import-overrides-coupled-native-2026-10-13/import_overrides/2/gama_trace.json').read_text())
    trace=run(load_module(path),100)
    assert trace[0]['global.signature']==2331
    assert compare_traces(native,trace,manifest['fields'])['passed']


def test_unresolved_virtual_action_is_rejected(tmp_path):
    source=tmp_path/'virtual.gaml'
    source.write_text('model virtual global {action hook virtual:true; init {do hook;}}')
    with pytest.raises(CompileError,match='virtual action'):
        compile_model(source,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_parameter_grid_matches_native_and_requires_recompile_for_dimensions(tmp_path):
    path,manifest=compile_model(ROOT/'examples/parameter_grid/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/parameter-grid-coupled-native-2026-10-13/parameter_grid/2/gama_trace.json').read_text())
    assert compare_traces(native,run(load_module(path),100,seed=2),manifest['fields'])['passed']
    assert manifest['fixed_grid_parameters']=={'columns':4,'rows':3}
    with pytest.raises(ValueError,match='requires recompilation'):
        run(load_module(path),0,seed=2,parameters={'columns':5})


def test_update_attributes_match_native_before_reflexes(tmp_path):
    path,manifest=compile_model(ROOT/'examples/update_attributes/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/update-attributes-coupled-native-2026-10-13/update_attributes/2/gama_trace.json').read_text())
    model=load_module(path)
    trace=run(model,100,seed=2)
    assert trace[1]['global.signature']==212
    assert trace[1]['Walker.marker']==[3,3]
    assert compare_traces(native,trace,manifest['fields'])['passed']
    assert compare_traces(trace,run(model,100,seed=2,compiled=False),manifest['fields'])['passed']


def test_computed_attributes_are_lazy_and_match_native(tmp_path):
    path,manifest=compile_model(ROOT/'examples/computed_attributes/main.gaml',tmp_path/'out')
    model=load_module(path)
    native=json.loads((ROOT/'tests/fixtures/native/computed-attributes-coupled-native-b-2026-10-13/computed_attributes/2/gama_trace.json').read_text())
    trace=run(model,100)
    assert trace[0]['global.tripled']==3 and trace[1]['global.tripled']==6
    assert trace[1]['Walker.derived']==[9,9]
    assert compare_traces(native,trace,manifest['fields'])['passed']
    state=model.reset(); before=canonical(state)
    assert canonical(model.observe_jit(state))['global.tripled']==3
    assert canonical(state)==before


@pytest.mark.parametrize('members',[
    'int a -> b; int b -> a;',
    'int a -> 1 update:2;',
    'int a -> 1; reflex bad {a <- 2;}',
])
def test_invalid_or_mutating_getters_are_rejected_before_emission(tmp_path,members):
    source=tmp_path/'getter.gaml';source.write_text('model getter global {'+members+'}')
    with pytest.raises(CompileError):compile_model(source,tmp_path/'out')
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('case,archive',[
    ('moving_defaults','moving-defaults-ten-seeds-coupled-2026-10-13'),
    ('regular_grid','regular-grid-ten-seeds-coupled-2026-10-13'),
])
def test_native_implicit_skills_and_regular_grid_all_seeds(tmp_path,case,archive):
    path,manifest=compile_model(ROOT/'examples'/case/'main.gaml',tmp_path/'out')
    model=load_module(path)
    for seed in range(1,11):
        native=json.loads((ROOT/'results'/archive/case/str(seed)/'gama_trace.json').read_text())
        assert compare_traces(native,run(model,100,seed=seed),manifest['fields'])['passed']


def test_forward_update_dependencies_match_native(tmp_path):
    path,manifest=compile_model(ROOT/'examples/update_forward_dependencies/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/update-forward-dependencies-native-b-2026-10-13/update_forward_dependencies/2/gama_trace.json').read_text())
    assert compare_traces(native,run(load_module(path),100),manifest['fields'])['passed']
    assert manifest['variable_order']['global'][:2]==['a','b']


def test_wander_matches_native_and_preserves_failed_move_speed(tmp_path):
    path,manifest=compile_model(ROOT/'examples/wander_rules/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/wander-rules-native-probe-2026-10-13/wander_rules/2/gama_trace.json').read_text())
    assert compare_traces(native,run(load_module(path),100,seed=2),manifest['fields'])['passed']


def test_persistent_neighbors_keep_order_and_external_radius(tmp_path):
    path,manifest = compile_model(ROOT/'examples/persistent_neighbors/main.gaml',tmp_path/'out')
    model = load_module(path)
    initial = canonical(model.reset(seed=2))
    assert initial['cell.nearby'][0] == [8,5,1,2,4]
    assert initial['WalkerHost.place'] == [2,2,7]
    trace = run(model,3,seed=2)
    assert trace == run(model,3,seed=2,compiled=False)
    assert canonical(model.reset(seed=2,parameters={'radius':1}))['cell.nearby'][0] != initial['cell.nearby'][0]
    assert canonical(model.reset(seed=2,parameters={'radius':999})) == canonical(model.reset(seed=2,parameters={'radius':3}))
    assert manifest['fields']['cell.nearby']=='agents:cell'


def test_spatial_generated_module_is_standalone_and_name_independent(tmp_path):
    source=ROOT/'examples/persistent_neighbors/main.gaml'
    first,_=compile_model(source,tmp_path/'first')
    renamed=tmp_path/'unrelated.gaml'
    renamed.write_text(source.read_text().replace('model persistent_neighbors','model arbitrary_name'))
    second,_=compile_model(renamed,tmp_path/'second')
    assert first.read_bytes()==second.read_bytes()
    code = ('import runpy,sys; m=runpy.run_path(sys.argv[1]); '
            's=m["step_jit"](m["reset"](seed=2)); '
            'assert int(s["$error"])==0; assert s["cell.nearby"].shape==(12,24); '
            'assert "gamax" not in sys.modules; print("standalone-ok")')
    result=subprocess.run([sys.executable,'-I','-c',code,str(first.resolve())],cwd=tmp_path,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='standalone-ok'


def test_ordered_collection_comparator_detects_order_shape_and_type():
    fields={'grid.nearby':'agents:grid'}
    reference=[{'grid.nearby':[[2,1],[],[0]]}]
    assert compare_traces(reference,reference,fields)['passed']
    for corrupted in ([[1,2],[],[0]],[[2],[1],[0]],[[True,1],[],[0]]):
        assert not compare_traces(reference,[{'grid.nearby':corrupted}],fields)['passed']


def test_init_outputs_are_preserved(tmp_path):
    path,_=compile_model(ROOT/'examples/colors_outputs/main.gaml',tmp_path/'out')
    trace=run(load_module(path),2)
    assert trace[1]['global.matches']==3
    assert trace[2]['global.matches']==0
    assert trace.outputs==[{'kind':'write','phase':'init','id':0,'text':'Initial value: 0.25'}]
    export_trace(trace,tmp_path/'trace','jax')
    assert json.loads((tmp_path/'trace/jax_outputs.json').read_text())==trace.outputs


def test_count_resolution_tracks_initial_assignments_and_create_action_scope(tmp_path):
    source=tmp_path/'creation.gaml'
    source.write_text('model creation global {int n <- 2; init {n <- 3; create Walker number:n {do change;}}} '
                      'species Walker {int value; action change {value <- 7;}}')
    path,manifest=compile_model(source,tmp_path/'out')
    assert manifest['populations']['Walker']==3
    assert run(load_module(path),0)[0]['Walker.value']==[7,7,7]


def test_empty_collection_loops_and_nil_equality(tmp_path):
    source=tmp_path/'empty.gaml'
    source.write_text('model empty_collection global {int count; bool missing; Walker reference <- nil; init {create Walker number:0;} '
                      'reflex count_agents {missing <- reference = nil; count <- 0; loop person over:(Walker as list) {count <- count+1;}}} '
                      'species Walker {Walker reference <- nil; reflex check {reference <- nil;}}')
    path,_=compile_model(source,tmp_path/'out')
    result=run(load_module(path),1)[1]
    assert result['global.count']==0 and result['global.missing'] is True


def test_accumulation_preserves_order_sensitive_reduction(tmp_path):
    path,manifest=compile_model(ROOT/'examples/accumulated_neighbors/main.gaml',tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/accumulation-native-probe-2026-10-13/accumulated_neighbors/2/gama_trace.json').read_text())
    assert compare_traces(native,run(load_module(path),10,seed=2),manifest['fields'])['passed']


def test_reserved_capacity_masks_inactive_agents_and_signals_overflow(tmp_path):
    source=ROOT/'examples/persistent_neighbors/main.gaml'
    first,manifest=compile_model(source,tmp_path/'reserved',capacities={'WalkerHost':6})
    ordinary,_=compile_model(source,tmp_path/'ordinary')
    model=load_module(first)
    assert run(model,10,seed=2)==run(load_module(ordinary),10,seed=2)
    assert len(run(model,1,seed=2,parameters={'walkers':5})[1]['WalkerHost.place'])==5
    assert run(model,1,seed=2,parameters={'walkers':0})[1]['WalkerHost.place']==[]
    assert manifest['capacities']=={'WalkerHost':6}
    with pytest.raises(RuntimeError,match='capacity exceeded'):
        run(model,0,seed=2,parameters={'walkers':7})


def test_parameter_override_skips_effectful_or_invalid_initializer(tmp_path):
    source=tmp_path/'override.gaml'
    source.write_text('model override global {int divisor; float value <- 1/divisor;}')
    path,_=compile_model(source,tmp_path/'out');model=load_module(path)
    with pytest.raises(RuntimeError,match='division by zero'):run(model,0)
    assert run(model,0,parameters={'value':4.0})[0]['global.value']==4.0


def test_nil_reference_fails_only_when_evaluated(tmp_path):
    source=tmp_path/'nil.gaml'
    source.write_text('model nil_reference global { bool enabled <- false; init {create Walker number: 1;} } '
                      'grid cell width: 2 height: 2 use_regular_agents: false frequency: 0 {} '
                      'species Walker {cell place; reflex update {location <- enabled ? place.location : {0,0};}}')
    path,_=compile_model(source,tmp_path/'out'); model=load_module(path)
    assert run(model,1,seed=2)[1]['Walker.location']==[[0.0,0.0,0.0]]
    with pytest.raises(RuntimeError,match='nil agent reference'):
        run(model,1,seed=2,parameters={'enabled':True})


@pytest.mark.parametrize('declaration',['int radius <- 2;','int radius min: 0 max: 3 <- 2;','int radius min: 1 max: 6 <- 2;'])
def test_unverified_radius_bounds_are_rejected(tmp_path,declaration):
    source=tmp_path/'bad.gaml'
    source.write_text('model bad global {'+declaration+'} grid cell width: 4 height: 3 use_regular_agents: false frequency: 0 {list<cell> nearby <- self neighbors_at radius;}')
    with pytest.raises(CompileError,match='verified min/max'):
        compile_model(source,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_lifecycle_matches_native_states_and_identities(tmp_path):
    from gamax.validation import compare_identities
    source=ROOT/'examples/agent_lifecycle/main.gaml'
    with pytest.raises(CompileError,match='explicit reserved capacity'):
        compile_model(source,tmp_path/'missing')
    path,manifest=compile_model(source,tmp_path/'out',capacities={'WalkerHost':64})
    model=load_module(path)
    trace=run(model,10,seed=2)
    native=ROOT/'tests/fixtures/native/lifecycle-native-probe-b-2026-10-13/agent_lifecycle/2'
    reference=json.loads((native/'gama_trace.json').read_text())
    identities=json.loads((native/'gama_identities.json').read_text())
    assert compare_traces(reference,trace,manifest['fields'])['passed']
    assert compare_identities(identities,trace.identities)['passed']
    assert trace==run(model,10,seed=2,compiled=False)
    assert trace.identities[10]['WalkerHost']==[6,8,11]
    with pytest.raises(RuntimeError,match='capacity exceeded'):
        small,_=compile_model(source,tmp_path/'small',capacities={'WalkerHost':3})
        run(load_module(small),10,seed=2)


def test_death_stops_remaining_statements_and_keeps_native_ids(tmp_path):
    source=tmp_path/'death.gaml'
    source.write_text('model death global {int count; init {create Walker number:3;}} '
                      'species Walker {reflex remove when:index=1 {do die; count <- count+100;} '
                      'reflex update {count <- count+1;}}')
    path,_=compile_model(source,tmp_path/'out')
    trace=run(load_module(path),2)
    assert trace[1]['global.count']==2 and trace[2]['global.count']==4
    assert trace.identities[1]['Walker']==[0,2]


def test_identity_comparator_rejects_renumbering_and_missing_agent():
    from gamax.validation import compare_identities
    reference=[{'Agent':[0,2,7]}]
    for corrupted in ([0,1,2],[0,2],[0,7,2],[0,2,2],[0,2,True]):
        assert not compare_identities(reference,[{'Agent':corrupted}])['passed']


def test_overlap_reduction_matches_native_and_rejects_observable_order(tmp_path):
    source=ROOT/'examples/spatial_overlap/main.gaml'
    path,manifest=compile_model(source,tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/spatial-overlap-native-probe-2026-10-13/spatial_overlap/2/gama_trace.json').read_text())
    assert compare_traces(native,run(load_module(path),10,seed=2),manifest['fields'])['passed']
    corrupted=tmp_path/'ordered.gaml'
    corrupted.write_text(source.read_text().replace('occupied <- occupied + 1','occupied <- occupied * 31 + candidate.index'))
    with pytest.raises(CompileError,match='order-independent'):
        compile_model(corrupted,tmp_path/'rejected')
    floating=tmp_path/'floating.gaml'
    floating.write_text(source.read_text().replace('int occupied <- 0','float occupied <- 0.0'))
    with pytest.raises(CompileError,match='order-independent'):
        compile_model(floating,tmp_path/'floating_rejected')


def test_gui_declarations_retained_and_effects_rejected(tmp_path):
    source=tmp_path/'graphics.gaml'
    source.write_text('model graphics global {int n <- 2; init {create Walker number:n;}} '
                      'species Walker {aspect basic {draw circle(1) color:#red;}} '
                      'experiment view type:gui {parameter "Population" var:n; output {display scene {species Walker aspect:basic;}}}')
    _,manifest=compile_model(source,tmp_path/'out',experiment='view')
    assert manifest['presentation']['experiments']['view']['parameters']==[{'label':'Population','variable':'n'}]
    assert 'circle' in manifest['presentation']['aspects']['Walker']['basic']['tokens']
    source.write_text(source.read_text().replace('draw circle(1)','draw circle(rnd(1))'))
    _,random_manifest=compile_model(source,tmp_path/'random_render')
    assert random_manifest['presentation']['aspects']['Walker']['basic']['render_random']
    source.write_text(source.read_text().replace('draw circle(rnd(1))','n <- n+1; draw circle(1)'))
    with pytest.raises(CompileError):
        compile_model(source,tmp_path/'rejected')


def test_inheritance_composes_init_and_overrides_slots_like_native(tmp_path):
    source=ROOT/'examples/inheritance_rules/main.gaml'
    path,manifest=compile_model(source,tmp_path/'out')
    native=json.loads((ROOT/'tests/fixtures/native/inheritance-coupled-native-probe-2026-10-13/inheritance_rules/2/gama_trace.json').read_text())
    trace=run(load_module(path),100,seed=2)
    assert compare_traces(native,trace,manifest['fields'])['passed']
    assert trace[0]['global.init_calls']==22
    assert trace[0]['Child.b']==[172,192]
    assert trace[1]['Child.a']==[44,40]
    assert manifest['populations']['Parent']==0


def test_inherited_self_creation_targets_concrete_species(tmp_path):
    from gamax.compiler import load_model
    source=tmp_path/'inherited_birth.gaml'
    source.write_text('model inherited_birth global {init {create Child number:1;}} '
                      'species Parent {int n <- rnd(1); reflex replace {create species(self); do die;}} '
                      'species Child parent:Parent {}')
    ir,_=load_model(source)
    child=next(s for s in ir.scopes if s.name=='Child')
    assert child.behaviors[0].body[0].target=='Child'
    path,_=compile_model(source,tmp_path/'out',capacities={'Child':3})
    trace=run(load_module(path),2,seed=2)
    assert trace.identities[-1]['Child']==[2] and trace.identities[-1]['Parent']==[]


def test_random_getters_match_native_without_advancing_simulation_rng(tmp_path):
    path,manifest=compile_model(ROOT/'examples/random_getters/main.gaml',tmp_path/'out')
    model=load_module(path)
    assert manifest['rng_observation_required']
    for seed in range(1,11):
        native=json.loads((ROOT/'tests/fixtures/native/random-getters-ten-seeds-coupled-2026-10-13/random_getters'/str(seed)/'gama_trace.json').read_text())
        assert compare_traces(native,run(model,100,seed=seed),manifest['fields'])['passed']
    state=model.reset(seed=2)
    before=canonical(model.step_jit(state))
    observed=model.observe_jit(state)
    assert canonical(model.step_jit(observed))['Walker.value']==before['Walker.value']
    assert canonical(model.observe_jit(state))==canonical(model.observe_jit(state))


def test_ordered_spatial_queries_match_dense_native_index(tmp_path):
    for case,archive,steps in [('spatial_query_order','spatial-query-order-ten-seeds-native-2026-10-13',100)]:
        # The source and native trace are frozen by the coupled campaign.
        directory=ROOT/'results'/archive/case
        path,manifest=compile_model(ROOT/'examples'/case/'main.gaml',tmp_path/case)
        model=load_module(path)
        for seed in range(1,11):
            native=json.loads((directory/str(seed)/'gama_trace.json').read_text())
            assert compare_traces(native,run(model,steps,seed=seed),manifest['fields'])['passed']


def test_list_aliases_keep_mutations_and_replacement_separate(tmp_path):
    path,manifest=compile_model(ROOT/'examples/list_aliases/main.gaml',tmp_path/'out')
    model=load_module(path)
    for seed in range(1,11):
        native=json.loads((ROOT/'tests/fixtures/native/list-aliases-ten-seeds-coupled-2026-10-13/list_aliases'/str(seed)/'gama_trace.json').read_text())
        trace=run(model,100,seed=seed)
        assert compare_traces(native,trace,manifest['fields'])['passed']
        assert trace[1]['global.first_list']==trace[1]['global.second_list']==[1,2]
        assert trace[3]['global.first_list']==trace[3]['global.third_list']==[1,2,0]
        assert sorted(trace[3]['global.second_list'])==[0,1,2]


def test_spatial_getters_match_native_order_and_later_random_stream(tmp_path):
    path,manifest=compile_model(ROOT/'examples/spatial_getters/main.gaml',tmp_path/'out')
    model=load_module(path)
    assert manifest['rng_observation_required']
    for seed in range(1,11):
        native=json.loads((ROOT/'tests/fixtures/native/spatial-getters-ten-seeds-coupled-2026-10-13/spatial_getters'/str(seed)/'gama_trace.json').read_text())
        assert compare_traces(native,run(model,100,seed=seed),manifest['fields'])['passed']


@pytest.mark.parametrize('case,archive',[
    ('nested_asks','nested-asks-ten-seeds-coupled-2026-10-13'),
    ('native_collection_selection','native-collection-selection-ten-seeds-coupled-2026-10-13'),
    ('goto_points','goto-point-ten-seeds-coupled-2026-10-13'),
    ('circle_query_order','circle-query-order-ten-seeds-coupled-2026-10-13'),
    ('image_handles','image-handles-ten-seeds-coupled-2026-10-13'),
])
def test_new_primitives_replay_native_states_and_identities(tmp_path,case,archive):
    from gamax.validation import compare_identities
    path,manifest=compile_model(ROOT/'examples'/case/'main.gaml',tmp_path/case)
    model=load_module(path)
    for seed in range(1,11):
        directory=ROOT/'results'/archive/case/str(seed)
        reference=json.loads((directory/'gama_trace.json').read_text())
        identities=json.loads((directory/'gama_identities.json').read_text())
        trace=run(model,100,seed=seed)
        assert compare_traces(reference,trace,manifest['fields'])['passed']
        assert compare_identities(identities,trace.identities)['passed']


def test_image_handles_require_unchanged_bundled_assets(tmp_path):
    path,manifest=compile_model(ROOT/'examples/image_handles/main.gaml',tmp_path/'images')
    model=load_module(path)
    model.reset(seed=2)
    asset=path.parent/manifest['assets'][0]['asset_bundle_path']
    original=asset.read_bytes()
    asset.write_bytes(original+b'corrupted')
    with pytest.raises((RuntimeError,ValueError),match='asset|Asset|SHA|hash'):
        model.reset(seed=2)
    asset.unlink()
    with pytest.raises((RuntimeError,ValueError,FileNotFoundError)):
        model.reset(seed=2)
