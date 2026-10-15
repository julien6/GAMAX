"""Read-only getter maps preserve native traces; effectful getters stay ordered."""
import json
from pathlib import Path
import jax
import numpy as np
from gamax.compiler import compile_model
from gamax.runtime import load_module,run,check_status
from gamax.validation import compare_traces,compare_identities


def test_batched_getters_match_native_lifecycle_order_and_rng(tmp_path):
    root=Path(__file__).resolve().parents[1];archive=root/'tests/fixtures/native/optimizer-contact-cache-native-probe-v3'
    native=json.loads((archive/'native/result.json').read_text());keys=list(json.loads((archive/'compiled/manifest.json').read_text())['fields'])
    expected=[{key:row['field'+str(i)] for i,key in enumerate(keys)} for row in native['trace']]
    path,m=compile_model(root/'examples/contact_cache_lifecycle/main.gaml',tmp_path/'compiled',experiment='validation',capacities={'Walker':640},optimization='fusion-avx-batch')
    trace=run(load_module(path),10,seed=2)
    assert compare_traces(expected,trace,m['fields'])['passed']
    assert compare_identities([{'Walker':row['ids']} for row in native['trace']],trace.identities)['passed']


def test_random_getters_are_not_batched(tmp_path):
    source=tmp_path/'unrelated.gaml'
    source.write_text('''model unrelated global {int total; init {create Actor number:12;} reflex measure {total <- sum(Actor collect each.sample);}} species Actor {int sample -> rnd(1000);} experiment validation type:gui {}''')
    a,am=compile_model(source,tmp_path/'host',experiment='validation',optimization='fusion-avx-host')
    b,bm=compile_model(source,tmp_path/'batch',experiment='validation',optimization='fusion-avx-batch')
    assert next(p for p in bm['optimization']['passes'] if p['name']=='vectorized-discard-only-integer-getters')['applied_sites']==0
    reference=run(load_module(a),10,seed=2);candidate=run(load_module(b),10,seed=2)
    assert compare_traces(reference,candidate,bm['fields'])['passed']


def test_mersenne_discard_keeps_int32_index_with_int64_budget():
    from gamax.randomness import MERSENNE_CACHED_SOURCE,mersenne_initial_state
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True);ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE,ns)
    rng=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
    discard=jax.jit(ns['mersenne_discard_uniforms'])
    for count in (0,1,310,1300):
        a=discard(rng,jnp.asarray(count,jnp.int32));b=discard(rng,jnp.asarray(count,jnp.int64))
        assert b[1].dtype==jnp.int32
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        jax.jit(ns['mersenne_uniform'])(b)


def test_reused_geometry_preserves_native_draws_radius_changes_and_lifecycle(tmp_path):
    root=Path(__file__).resolve().parents[1];archive=root/'tests/fixtures/native/optimizer-pure-getter-native-probe-v2'
    native=json.loads((archive/'native/result.json').read_text())
    keys=list(json.loads((archive/'compiled/manifest.json').read_text())['fields'])
    expected=[{key:row['field'+str(i)] for i,key in enumerate(keys)} for row in native['trace']]
    path,manifest=compile_model(root/'examples/pure_getter_maps.gaml',tmp_path/'compiled',experiment='validation',capacities={'Walker':640},optimization='fusion-avx-batch')
    passes={p['name']:p['applied_sites'] for p in manifest['optimization']['passes']}
    assert passes['vectorized-discard-only-integer-getters']==3
    assert passes['reuse-pure-getter-query-geometry']>0
    trace=run(load_module(path),10,seed=2)
    assert compare_traces(expected,trace,manifest['fields'])['passed']
    assert compare_identities([{'Walker':row['ids']} for row in native['trace']],trace.identities)['passed']
