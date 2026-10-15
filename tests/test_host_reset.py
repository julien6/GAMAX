"""Functional host initialization must preserve the full emitted JAX state."""
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
from gamax.compiler import compile_model
from gamax.runtime import load_module


def test_host_initializer_matches_jax_and_keeps_private_namespace(tmp_path):
    source=tmp_path/'renamed.gaml'
    source.write_text('''model independent_initializer
    global {int count <- 8; float fraction <- 1.0 / 3.0; int total <- 2147483646;
      init {total <- total + 10; create Actor number:count;}}
    grid cell width:5 height:3 neighbors:8 {int nearby <- length(self neighbors_at 1); int sample <- rnd(1000);}
    species Actor skills:[moving] {int value <- rnd(1000); reflex evolve {value <- rnd(1000);}}
    experiment validation type:gui {}''')
    baseline=load_module(compile_model(source,tmp_path/'jax',experiment='validation',capacities={'Actor':16},optimization='fusion-avx-math')[0])
    host=load_module(compile_model(source,tmp_path/'host',experiment='validation',capacities={'Actor':16},optimization='fusion-avx-host')[0])
    def compare(seed):
        a=baseline.reset(seed=seed,parameters={'count':4});b=host.reset(seed=seed,parameters={'count':4})
        assert a.keys()==b.keys()
        for key in a:
            for x,y in zip(jax.tree.leaves(a[key]),jax.tree.leaves(b[key])):
                assert isinstance(y,jax.Array)
                assert x.dtype==y.dtype
                np.testing.assert_array_equal(np.asarray(x).view(np.uint8),np.asarray(y).view(np.uint8)) if x.ndim else np.testing.assert_array_equal(x,y)
        assert host.jnp is jnp
        assert host.jax is jax
        return b
    for seed in (1,2,-2,.125):compare(seed)
    # Simultaneous calls must not replace JAX bindings shared with transitions.
    with ThreadPoolExecutor(max_workers=4) as executor:states=list(executor.map(lambda seed:host.reset(seed=seed),[1,2,3,4]))
    for state in states:assert isinstance(host.step_jit(state)['Actor.value'],jax.Array)
    import pytest
    for seed in (0,float('nan'),float('inf')):
        with pytest.raises(ValueError):host.reset(seed=seed)
