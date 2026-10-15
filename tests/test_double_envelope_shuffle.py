"""Shared shuffle body must match both ordered native phases, even padding."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_CACHED_SOURCE,MERSENNE_BATCH_SOURCE,mersenne_initial_state
from gamax.spatial_index import QUADTREE_SOURCE,DOUBLE_SHUFFLE_SOURCE,DOUBLE_SHUFFLE_BATCH_SOURCE
import pytest


@pytest.mark.parametrize('source',(DOUBLE_SHUFFLE_SOURCE,DOUBLE_SHUFFLE_BATCH_SOURCE))
def test_two_forward_shuffles_match_shared_loop_bitwise(source):
    ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+MERSENNE_BATCH_SOURCE+QUADTREE_SOURCE+source,ns)
    def twice(rng,values,mask):
        rng,values,mask=ns['collection_shuffle_forward'](rng,values,mask,ns['mersenne_uniform'])
        return ns['collection_shuffle_forward'](rng,values,mask,ns['mersenne_uniform'])
    reference=jax.jit(twice);candidate=jax.jit(lambda r,v,m:ns['collection_shuffle_twice_forward'](r,v,m,ns['mersenne_uniform']))
    random=np.random.default_rng(58)
    for size in (0,1,2,32,128,350):
        values=jnp.asarray(random.permutation(size)*3,jnp.int32)
        for count in sorted({0,min(size,1),min(size,2),size//2,size}):
            mask=np.zeros(size,dtype=bool);mask[random.permutation(size)[:count]]=True
            rng=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
            a,av,am=reference(rng,values,jnp.asarray(mask));b,bv,bm=candidate(rng,values,jnp.asarray(mask))
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(av,bv);np.testing.assert_array_equal(am,bm)
