"""Discarded grid permutations preserve exact MT state and the next draw."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_CACHED_SOURCE,mersenne_initial_state
from gamax.spatial import GRID_SOURCE,implicit_grid_shuffle_draws


def test_static_grid_discard_matches_each_native_neighborhood():
    ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+GRID_SOURCE,ns)
    discard=jax.jit(ns['mersenne_discard_uniforms'])
    for width,height in ((1,1),(1,5),(5,1),(3,5),(5,3),(8,8)):
        for moore in (False,True):
            for torus in (False,True):
                start=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
                def slow(state):
                    return jax.lax.fori_loop(0,width*height,lambda i,s:ns['grid_neighbors'](s,i,width,height,1,moore=moore,torus=torus)[0],state)
                reference=jax.jit(slow)(start)
                optimized=discard(start,jnp.asarray(implicit_grid_shuffle_draws(width,height,moore=moore,torus=torus),jnp.int32))
                for a,b in zip(reference,optimized):np.testing.assert_array_equal(a,b)
                a,va=ns['mersenne_uniform'](reference);b,vb=ns['mersenne_uniform'](optimized)
                np.testing.assert_array_equal(va,vb)
