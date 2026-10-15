"""Incremental membership must match uncached queries through every mutation."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from gamax.spatial_index import QUADTREE_SOURCE,UNORDERED_DISTANCE_SOURCE,CONTACT_CACHE_SOURCE
from gamax.randomness import MERSENNE_CACHED_SOURCE,mersenne_initial_state


def functions():
    namespace={'jax':jax,'jnp':jnp}
    exec(MERSENNE_CACHED_SOURCE+QUADTREE_SOURCE+UNORDERED_DISTANCE_SOURCE+CONTACT_CACHE_SOURCE,namespace)
    return namespace


def test_contact_cache_moves_births_deaths_radius_changes_and_rng():
    f=functions();build=jax.jit(f['build_contact_cache']);update=jax.jit(f['update_contact_cache'])
    cached=jax.jit(f['cached_point_neighbors']);uncached=jax.jit(f['point_neighbors_unordered'])
    random=np.random.default_rng(821);size=41
    positions=jnp.asarray(random.uniform(-10,10,(size,3)),jnp.float64)
    alive=jnp.asarray(random.random(size)>.2);leaf=jnp.where(alive,0,-1).astype(jnp.int32);radius=jnp.asarray(7.,jnp.float64)
    cache=build(positions,alive,leaf,radius)
    rng=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
    for step in range(60):
        index=int(random.integers(size))
        positions=positions.at[index].set(jnp.asarray(random.uniform(-12,12,3)))
        if step%7==0:
            alive=alive.at[index].set(~alive[index]);leaf=leaf.at[index].set(jnp.where(alive[index],0,-1))
        cache=update(cache,positions,alive,leaf,jnp.asarray(index),radius)
        for expected,actual in zip(build(positions,alive,leaf,radius),cache):np.testing.assert_array_equal(expected,actual)
        query=int(random.integers(size));new_radius=jnp.asarray(3. if step%5==0 else float(radius),jnp.float64)
        a,(ia,ma)=uncached(rng,positions,leaf,alive,positions[query],jnp.asarray(query),new_radius)
        b,(ib,mb),valid,key,new_cache=cached(rng,jnp.asarray(True),radius,cache,positions,alive,leaf,jnp.asarray(query),new_radius)
        np.testing.assert_array_equal(np.asarray(ia)[np.asarray(ma)],np.asarray(ib)[np.asarray(mb)])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        if bool(alive[query]):
            cache=new_cache;radius=key
        rng=a


def test_generated_cache_matches_native_lifecycle_and_order(tmp_path):
    import json
    from pathlib import Path
    from gamax.compiler import compile_model
    from gamax.runtime import load_module,run
    from gamax.validation import compare_traces,compare_identities
    root=Path(__file__).resolve().parents[1];archive=root/'tests/fixtures/native/optimizer-contact-cache-native-probe-v3'
    native=json.loads((archive/'native/result.json').read_text());keys=list(json.loads((archive/'compiled/manifest.json').read_text())['fields'])
    reference=[{key:row['field'+str(i)] for i,key in enumerate(keys)} for row in native['trace']]
    module,manifest=compile_model(root/'examples/contact_cache_lifecycle/main.gaml',tmp_path/'compiled',experiment='validation',capacities={'Walker':640},optimization='fusion-avx')
    trace=run(load_module(module),10,seed=2)
    assert compare_traces(reference,trace,manifest['fields'])['passed']
    assert compare_identities([{'Walker':row['ids']} for row in native['trace']],trace.identities)['passed']


def test_row_memoization_invalidates_identity_radius_and_epoch_and_keeps_draws():
    from gamax.spatial_index import CONTACT_ROW_CACHE_SOURCE
    f=functions();exec(CONTACT_ROW_CACHE_SOURCE,f)
    memo=jax.jit(f['memoized_point_neighbors']);plain=jax.jit(f['point_neighbors_unordered'])
    random=np.random.default_rng(22);size=41
    positions=jnp.asarray(random.uniform(-10,10,(size,3)),jnp.float64);alive=jnp.ones(size,jnp.bool_);leaf=jnp.zeros(size,jnp.int32)
    cache=(jnp.zeros(size,jnp.bool_),jnp.zeros(size,jnp.bool_),jnp.asarray(0,jnp.int32),jnp.asarray(0,jnp.int32))
    valid=jnp.asarray(False);key=jnp.asarray(jnp.nan,jnp.float64);source=jnp.asarray(-1,jnp.int32);stamp=jnp.asarray(-1,jnp.int64)
    rng=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
    for epoch in range(15):
        index=int(random.integers(size));positions=positions.at[index].set(jnp.asarray(random.uniform(-10,10,3)))
        if epoch%3==0:alive=alive.at[index].set(~alive[index]);leaf=leaf.at[index].set(jnp.where(alive[index],0,-1))
        query=jnp.asarray(int(random.integers(size)),jnp.int64 if epoch%2 else jnp.int32)
        for radius in (7.,7.,3.,3.):
            a,(ia,ma)=plain(rng,positions,leaf,alive,positions[query],query,jnp.asarray(radius))
            b,(ib,mb),valid,key,cache,source,stamp=memo(rng,valid,key,cache,source,stamp,jnp.asarray(epoch,jnp.int64),positions,alive,leaf,query,jnp.asarray(radius))
            np.testing.assert_array_equal(np.asarray(ia)[np.asarray(ma)],np.asarray(ib)[np.asarray(mb)])
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            rng=a
