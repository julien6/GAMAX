"""Batch initialization retains ordered neighborhoods and the complete RNG state."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_CACHED_SOURCE, MERSENNE_BATCH_SOURCE, mersenne_initial_state
from gamax.spatial import GRID_CACHED_SOURCE, GRID_BATCH_INITIAL_SOURCE, grid_neighbor_geometry


def test_batch_neighborhoods_match_sequential_initialization():
    ns = {'jax': jax, 'jnp': jnp}
    exec(MERSENNE_CACHED_SOURCE + MERSENNE_BATCH_SOURCE + GRID_CACHED_SOURCE + GRID_BATCH_INITIAL_SOURCE, ns)
    for width, height, bound, moore, torus in ((1, 1, 0, False, False), (1, 5, 2, True, True), (5, 3, 3, False, True), (8, 8, 2, True, False)):
        geometry, radii = map(lambda a: jnp.asarray(a, jnp.int32), grid_neighbor_geometry(width, height, bound, moore=moore, torus=torus))
        def sequential(rng, radius):
            values = jnp.full(geometry.shape, -1, jnp.int32)
            masks = jnp.zeros(geometry.shape, jnp.bool_)
            if geometry.shape[1] == 0:
                return rng, (values, masks)
            def row(i, carry):
                rng, values, masks = carry
                rng, (v, m) = ns['grid_neighbors_from_geometry'](rng, geometry, radii, i, radius, uniform=ns['mersenne_uniform'])
                return rng, values.at[i].set(v), masks.at[i].set(m)
            rng, values, masks = jax.lax.fori_loop(0, width * height, row, (rng, values, masks))
            return rng, (values, masks)
        reference = jax.jit(sequential)
        candidate = jax.jit(lambda rng, radius: ns['initialized_grid_neighborhoods'](rng, geometry, radii, radius))
        raw = jnp.asarray(mersenne_initial_state(2), jnp.uint32)
        for offset in (0, 1, 623, 624):
            state = (raw, jnp.asarray(offset, jnp.int32), ns['mersenne_cache'](raw))
            for radius in (-1, 0, 1, bound):
                a = reference(state, jnp.asarray(radius))
                b = candidate(state, jnp.asarray(radius))
                for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
                    np.testing.assert_array_equal(x, y)
