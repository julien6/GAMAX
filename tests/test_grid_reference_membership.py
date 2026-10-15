"""The fast membership guard must never admit a different closed-box result."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gamax.spatial import GRID_REFERENCE_SOURCE

jax.config.update('jax_enable_x64', True)
namespace = {'jax': jax, 'jnp': jnp}
exec(GRID_REFERENCE_SOURCE, namespace)
guard = jax.jit(namespace['grid_reference_membership_valid'], static_argnames=('width', 'height'))


@pytest.mark.parametrize('width,height', [(1, 1), (1, 4), (5, 1), (5, 4)])
def test_admitted_reference_membership_matches_geometry(width, height):
    rng = np.random.default_rng(17)
    for scale in (1e-200, .3, 1., 1e100):
        centers = np.array([[(x + .5) * scale, (y + .5) * scale, 0.]
                            for y in range(height) for x in range(width)])
        references = rng.integers(len(centers), size=23, dtype=np.int32)
        locations = centers[references].copy()
        present = rng.random(23) > .2
        references[~present] = -1
        half = np.array([scale / 2, scale / 2])
        assert bool(guard(locations, references, present, centers, half, width, height))
        for cell in range(len(centers)):
            geometric = present & np.all(np.abs(locations[:, :2] - centers[cell, :2]) <= half, axis=1)
            np.testing.assert_array_equal(geometric, present & (references == cell))


@pytest.mark.parametrize('fault', ['off_center', 'wrong_reference', 'invalid_reference',
                                  'wide_box', 'nonrectilinear', 'duplicate_centers', 'infinite_centers'])
def test_unsafe_geometry_falls_back(fault):
    width, height = 4, 3
    centers = np.array([[x + .5, y + .5, 0.] for y in range(height) for x in range(width)])
    references = np.array([0, 5, 11], np.int32)
    locations = centers[references].copy()
    half = np.array([.5, .5])
    if fault == 'off_center': locations[0, 0] += .1
    if fault == 'wrong_reference': references[0] = 1
    if fault == 'invalid_reference': references[0] = -1
    if fault == 'wide_box': half[0] = 1.
    if fault == 'nonrectilinear': centers[5, 0] += .1
    if fault == 'duplicate_centers': centers[1] = centers[0]
    if fault == 'infinite_centers':
        centers[:,0]=np.inf
        locations[:,0]=np.inf
    assert not bool(guard(locations, references, np.ones(3, bool), centers, half, width, height))
