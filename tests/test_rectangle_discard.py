"""Unobserved rectangle permutations preserve membership and native MT words."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_CACHED_SOURCE,mersenne_initial_state
from gamax.spatial_index import QUADTREE_SOURCE,RECT_QUERY_SOURCE,RECT_UNORDERED_SOURCE


def test_unordered_rectangle_membership_and_rng():
    ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+QUADTREE_SOURCE+RECT_QUERY_SOURCE+RECT_UNORDERED_SOURCE,ns)
    reference=jax.jit(lambda r,p,l,i,a,s,h,e:ns['point_agents_inside_rectangle'](r,p,l,i,a,s,h,e,ns['mersenne_uniform']))
    candidate=jax.jit(ns['point_agents_inside_rectangle_unordered'])
    random=np.random.default_rng(791);size=128
    points=jnp.asarray(random.uniform(-10,10,(size,3)));points=points.at[:64,2].set(0.)
    leaf=jnp.asarray(random.integers(-1,5,size),jnp.int32);allowed=jnp.asarray(random.random(size)>.2);indices=jnp.asarray(random.permutation(size),jnp.int32)
    rng=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
    for half in ([0.,0.],[1.,2.],[5.,7.],[20.,20.]):
        source=jnp.zeros(3);half=jnp.asarray(half);exclude=jnp.asarray(-1,jnp.int32)
        a,(av,am)=reference(rng,points,leaf,indices,allowed,source,half,exclude)
        b,(bv,bm)=candidate(rng,points,leaf,allowed,source,half,exclude)
        np.testing.assert_array_equal(np.sort(np.asarray(av)[np.asarray(am)]),np.asarray(bv)[np.asarray(bm)])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
