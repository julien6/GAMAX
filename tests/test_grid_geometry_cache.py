"""Static grid geometry preserves ordering, masks and all shuffle draws."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_CACHED_SOURCE,mersenne_initial_state
from gamax.spatial import GRID_SOURCE,GRID_CACHED_SOURCE,grid_neighbor_geometry


def test_cached_geometry_matches_native_grid_algorithm():
    ns={'jax':jax,'jnp':jnp};exec(MERSENNE_CACHED_SOURCE+GRID_SOURCE+GRID_CACHED_SOURCE,ns)
    cached=jax.jit(lambda state,v,d,i,r:ns['grid_neighbors_from_geometry'](state,v,d,i,r,uniform=ns['mersenne_uniform']))
    for width,height,bound in ((1,1,3),(1,5,3),(5,1,3),(3,5,3),(5,3,3),(8,8,5)):
        for moore in (False,True):
            for torus in (False,True):
                values,radii=grid_neighbor_geometry(width,height,bound,moore=moore,torus=torus)
                values=jnp.asarray(values,jnp.int32);radii=jnp.asarray(radii,jnp.int32)
                reference=jax.jit(lambda state,i,r:ns['grid_neighbors'](state,i,width,height,r,max_radius=bound,moore=moore,torus=torus))
                state=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(623,jnp.int32),jnp.zeros(312,jnp.float64))
                for index in sorted({0,width*height//2,width*height-1}):
                    for radius in (-1,0,1,bound-1,bound):
                        a,(av,am)=reference(state,jnp.asarray(index),jnp.asarray(radius))
                        b,(bv,bm)=cached(state,values,radii,jnp.asarray(index),jnp.asarray(radius))
                        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
                        np.testing.assert_array_equal(av,bv);np.testing.assert_array_equal(am,bm)
                        state=a
