"""Bulk native doubles preserve samples and every raw/cached state bit."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_CACHED_SOURCE,MERSENNE_BATCH_SOURCE,mersenne_initial_state


def test_bulk_uniforms_match_scalar_draws_at_even_odd_and_twist_boundaries():
    ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+MERSENNE_BATCH_SOURCE,ns)
    for size in (0,1,2,20,312,313,1300,2500):
        def slow(state):
            if size==0:return state,jnp.zeros((0,),jnp.float64)
            def sample(i,carry):
                rng,values=carry;rng,v=ns['mersenne_uniform'](rng);return rng,values.at[i].set(v)
            return jax.lax.fori_loop(0,size,sample,(state,jnp.zeros(size,jnp.float64)))
        reference=jax.jit(slow);candidate=jax.jit(lambda state:ns['mersenne_uniforms'](state,size))
        for seed in (1,2,10):
            raw=jnp.asarray(mersenne_initial_state(seed),jnp.uint32);cache=ns['mersenne_cache'](raw)
            for offset in (0,1,2,623,624):
                state=(raw,jnp.asarray(offset,jnp.int32),cache)
                a,av=reference(state);b,bv=candidate(state)
                for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
                np.testing.assert_array_equal(np.asarray(av).view(np.uint64),np.asarray(bv).view(np.uint64))


def test_uniform_pair_preserves_masked_draws_and_twist_boundaries():
    ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+MERSENNE_BATCH_SOURCE,ns)
    def reference(rng,active):
        values=[]
        for _ in range(2):
            rng,value=jax.lax.cond(active,ns['mersenne_uniform'],lambda r:(r,jnp.asarray(0.,jnp.float64)),rng)
            values.append(value)
        return rng,jnp.stack(values)
    expected=jax.jit(reference)
    candidate=jax.jit(lambda rng,active:jax.lax.cond(active,ns['mersenne_uniform_pair'],lambda r:(r,jnp.zeros(2,jnp.float64)),rng))
    for seed in (1,2,10):
        raw=jnp.asarray(mersenne_initial_state(seed),jnp.uint32)
        for index in (0,1,620,621,622,623,624):
            state=(raw,jnp.asarray(index,jnp.int32),ns['mersenne_cache'](raw))
            for active in (False,True):
                for a,b in zip(jax.tree.leaves(expected(state,active)),jax.tree.leaves(candidate(state,active))):
                    np.testing.assert_array_equal(a,b)
