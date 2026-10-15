"""Integer ordering proof by differential comparison, including tree splits."""
import numpy as np
import jax
import jax.numpy as jnp
from gamax.spatial_index import QUADTREE_SOURCE, INCREMENTAL_ORDER_SOURCE


def test_incremental_order_matches_stable_sort_with_holes_ties_and_splits():
    namespace={'jax':jax,'jnp':jnp}
    exec(QUADTREE_SOURCE+INCREMENTAL_ORDER_SOURCE,namespace)
    update=jax.jit(namespace['quadtree_update_order'])
    keys=np.asarray(namespace['QUADTREE_DFS_KEYS'])
    rng=np.random.default_rng(20261014)
    for size in (1,2,17,128):
        for trial in range(40):
            previous=rng.integers(-1,1 if size<=100 else 100,size,dtype=np.int32)
            order=rng.integers(0,8,size,dtype=np.int64)
            indices=np.lexsort((order,keys[np.maximum(previous,0)])).astype(np.int32)
            index=int(rng.integers(size));leaf=previous.copy()
            leaf[index]=rng.integers(-1,1 if size<=100 else 100);order[index]=rng.integers(0,16)
            if trial%3==0 and size>1:leaf[(index+1)%size]=rng.integers(0,1 if size<=100 else 100)
            expected=np.lexsort((order,keys[np.maximum(leaf,0)])).astype(np.int32)
            actual=update(jnp.asarray(indices),jnp.asarray(previous),jnp.asarray(leaf),jnp.asarray(order),jnp.asarray(index))
            np.testing.assert_array_equal(actual,expected)


def test_integer_sum_preserves_overflow_masks_and_float_order(tmp_path):
    from gamax.compiler import compile_model
    from gamax.runtime import load_module
    source=tmp_path/'arbitrary.gaml'
    source.write_text('model arbitrary global {int count <- 1;}')
    reference=load_module(compile_model(source,tmp_path/'reference',optimization='none')[0])
    optimized=load_module(compile_model(source,tmp_path/'optimized',optimization='safe')[0])
    rng=np.random.default_rng(42)
    for shape in ((0,),(1,),(127,),(127,3)):
        values=jnp.asarray(rng.integers(-(2**31),2**31,size=shape,dtype=np.int32))
        mask=jnp.asarray(rng.random(shape[0])>.3)
        np.testing.assert_array_equal(reference.ordered_sum(values,mask,jnp.int32),optimized.ordered_sum(values,mask,jnp.int32))
    floats=jnp.asarray([1e16,1.,-1e16,2.],jnp.float64)
    mask=jnp.ones(4,dtype=jnp.bool_)
    assert float(optimized.ordered_sum(floats,mask,jnp.float64))==float(reference.ordered_sum(floats,mask,jnp.float64))==2.


def test_unobserved_permutations_preserve_membership_and_rng():
    from gamax.spatial_index import UNORDERED_DISTANCE_SOURCE
    from gamax.randomness import MERSENNE_CACHED_SOURCE,mersenne_initial_state
    namespace={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+QUADTREE_SOURCE+UNORDERED_DISTANCE_SOURCE,namespace)
    original=namespace['point_neighbors_at_distance'];optimized=namespace['point_neighbors_unordered'];uniform=namespace['mersenne_uniform']
    reference=jax.jit(lambda r,p,l,i,a,s,e,d:original(r,p,l,i,a,s,e,d,uniform))
    candidate=jax.jit(optimized)
    rng=np.random.default_rng(123)
    positions=jnp.asarray(rng.uniform(-10,10,(128,3)),jnp.float64)
    leaf=jnp.asarray(rng.integers(-1,5,128),jnp.int32)
    indices=jnp.asarray(rng.permutation(128),jnp.int32)
    allowed=jnp.asarray(rng.random(128)>.25)
    state=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
    for radius in (0.,.1,2.,8.,100.):
        a,(ia,ma)=reference(state,positions,leaf,indices,allowed,positions[0],jnp.asarray(0),jnp.asarray(radius))
        b,(ib,mb)=candidate(state,positions,leaf,allowed,positions[0],jnp.asarray(0),jnp.asarray(radius))
        np.testing.assert_array_equal(np.sort(np.asarray(ia)[np.asarray(ma)]),np.asarray(ib)[np.asarray(mb)])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_guarded_reflexes_match_native_scheduling_and_draws(tmp_path):
    import json
    from gamax.compiler import compile_model
    from gamax.runtime import load_module,run
    from gamax.validation import compare_traces,compare_identities
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    archive=root/'tests/fixtures/native/optimizer-guarded-reflex-native-probe'
    native=json.loads((archive/'native/result.json').read_text())
    fields=json.loads((archive/'compiled/manifest.json').read_text())['fields']
    reference=[{key:row['field'+str(i)] for i,key in enumerate(fields)} for row in native['trace']]
    identities=[{'Actor':row['ids']} for row in native['trace']]
    for mode in ('safe','fusion'):
        path,manifest=compile_model(archive/'probe.gaml',tmp_path/mode,experiment='validation',optimization=mode)
        candidate=run(load_module(path),10,seed=2)
        assert compare_traces(reference,candidate,manifest['fields'])['passed']
        assert compare_identities(identities,candidate.identities)['passed']


def test_compact_float_sum_is_bit_exact_with_sparse_masks(tmp_path):
    from gamax.compiler import compile_model
    from gamax.runtime import load_module
    source=tmp_path/'reduction.gaml';source.write_text('model reduction global {int n <- 1;}')
    baseline=load_module(compile_model(source,tmp_path/'baseline',optimization='none')[0])
    optimized=load_module(compile_model(source,tmp_path/'optimized',optimization='safe')[0])
    rng=np.random.default_rng(20261014)
    for shape in ((0,),(1,),(128,),(128,3)):
        values=rng.normal(size=shape)*rng.choice([1e-100,1.,1e100],size=shape)
        for density in (0.,.1,.8,1.):
            mask=jnp.asarray(rng.random(shape[0])<density);array=jnp.asarray(values,jnp.float64)
            a=np.asarray(baseline.ordered_sum(array,mask,jnp.float64))
            b=np.asarray(optimized.ordered_sum(array,mask,jnp.float64))
            np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))


def test_lazy_quadtree_matches_reference_births_moves_and_splits():
    from gamax.spatial_index import QUADTREE_OPTIMIZED_SOURCE
    original={'jax':jax,'jnp':jnp};optimized={'jax':jax,'jnp':jnp}
    exec(QUADTREE_SOURCE,original);exec(QUADTREE_OPTIMIZED_SOURCE,optimized)
    baseline=jax.jit(original['quadtree_insert']);candidate=jax.jit(optimized['quadtree_insert'])
    rng=np.random.default_rng(41)
    for size in (1,50,100,101,160,500):
        positions=rng.uniform(0,150,(size,3));positions[:min(size,130),:2]=.3
        positions=jnp.asarray(positions,jnp.float64)
        a=b=(jnp.zeros((4**8-1)//3,jnp.bool_),jnp.full(size,-1,jnp.int32),jnp.zeros(size,jnp.int64),jnp.asarray(0,jnp.int64))
        for index in [*range(size),*rng.integers(0,size,20)]:
            index=jnp.asarray(index,jnp.int32)
            a=baseline(*a,positions,index,jnp.asarray(100.),jnp.asarray(150.))
            b=candidate(*b,positions,index,jnp.asarray(100.),jnp.asarray(150.))
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_compiled_reset_reuses_executable_across_native_seeds(tmp_path):
    from gamax.compiler import compile_model
    from gamax.runtime import load_module,run
    from gamax.validation import compare_traces
    source=tmp_path/'seeded.gaml';source.write_text('model seeded global {float scale <- 1.0 / 3.0; int count <- 12; init {create Actor number:count;}} species Actor {int value <- rnd(1000);}')
    baseline=load_module(compile_model(source,tmp_path/'baseline',optimization='none')[0])
    optimized=load_module(compile_model(source,tmp_path/'optimized',optimization='safe')[0])
    manifest=__import__('json').loads((tmp_path/'optimized/manifest.json').read_text())
    for seed in (1,2,10,-2,.125):
        a=run(baseline,0,seed=seed);b=run(optimized,0,seed=seed)
        assert compare_traces(a,b,manifest['fields'])['passed']
    assert optimized._reset_native_jit._cache_size()==1
    import pytest
    for seed in (0,float('inf'),float('nan')):
        with pytest.raises(ValueError):optimized.reset(seed=seed)


def test_counted_quadtree_matches_native_leaf_counts_through_lifecycle():
    from gamax.spatial_index import QUADTREE_COUNTED_SOURCE
    reference={'jax':jax,'jnp':jnp};optimized={'jax':jax,'jnp':jnp}
    exec(QUADTREE_SOURCE,reference);exec(QUADTREE_COUNTED_SOURCE,optimized)
    baseline=jax.jit(reference['quadtree_insert']);candidate=jax.jit(optimized['quadtree_insert'])
    rng=np.random.default_rng(631)
    for size in (50,100,101,240):
        node_capacity=(4**8-1)//3
        positions=jnp.asarray(rng.uniform(-5,105,(size,3)),jnp.float64)
        positions=positions.at[:min(size,130),:2].set(.3)
        a=(jnp.zeros(node_capacity,jnp.bool_),jnp.full(size,-1,jnp.int32),jnp.zeros(size,jnp.int64),jnp.asarray(0,jnp.int64))
        b=(*a,jnp.zeros(node_capacity,jnp.int32))
        for turn,index in enumerate([*range(size),*rng.integers(0,size,40)]):
            index=jnp.asarray(index,jnp.int32)
            if turn>=size and turn%3==0:
                leaf=a[1];previous=leaf[index]
                a=(a[0],leaf.at[index].set(-1),a[2],a[3])
                counts=b[4].at[jnp.maximum(previous,0)].add(jnp.where(previous>=0,-1,0))
                b=(*a,counts)
            else:
                if turn>=size:positions=positions.at[index].set(jnp.asarray(rng.uniform(-5,105,3)))
                a=baseline(*a,positions,index,jnp.asarray(100.),jnp.asarray(100.))
                b=candidate(*b,positions,index,jnp.asarray(100.),jnp.asarray(100.))
            for x,y in zip(a,b[:4]):np.testing.assert_array_equal(x,y)
            leaves=np.asarray(a[1]);expected=np.bincount(leaves[leaves>=0],minlength=node_capacity)
            np.testing.assert_array_equal(b[4],expected)


def test_incremental_order_repairs_death_before_next_move():
    namespace={'jax':jax,'jnp':jnp};exec(QUADTREE_SOURCE+INCREMENTAL_ORDER_SOURCE,namespace)
    update=jax.jit(namespace['quadtree_update_order']);dfs=namespace['QUADTREE_DFS_KEYS']
    # A subdivided tree requires a reserved capacity above 100, even when
    # this fixture exercises only six active identities.
    leaf=jnp.asarray([10,3,7,10,2,7]+[-1]*122,jnp.int32)
    order=jnp.asarray([40,20,30,50,10,60]+[0]*122,jnp.int64)
    indices=jnp.lexsort((order,dfs[jnp.maximum(leaf,0)])).astype(jnp.int32)
    for index,new_leaf,new_order in ((2,-1,30),(5,2,70),(1,10,80)):
        previous=leaf;leaf=leaf.at[index].set(new_leaf);order=order.at[index].set(new_order)
        indices=update(indices,previous,leaf,order,jnp.asarray(index,jnp.int32))
        np.testing.assert_array_equal(indices,jnp.lexsort((order,dfs[jnp.maximum(leaf,0)])))
