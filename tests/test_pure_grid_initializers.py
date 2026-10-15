"""Pure initializer maps retain full state, dtype, draw stream and cache reuse."""
from pathlib import Path
import jax
import numpy as np
from gamax.compiler import compile_model
from gamax.runtime import load_module,check_status


def test_pure_grid_maps_match_ordered_initializer_with_parameters_and_rng(tmp_path):
    source=tmp_path/'unrelated.gaml'
    source.write_text('''model unrelated
global { float side <- 123.4; geometry shape <- square(side); point center <- {33.0,33.0}; float cap <- 0.75;
 int after; init { after <- rnd(1000); } }
grid tile width:33 height:33 neighbors:8 {
 bool central <- (topology(tile) distance_between [self,center]) < 4;
 float raw <- rnd(1.0);
 float scaled <- raw / 7.0 max:cap update:scaled+1.0;
 rgb color <- central ? #red : ((scaled>0.05) ? #blue : #green);
}
experiment validation type:gui {}
''')
    reference=load_module(compile_model(source,tmp_path/'host',experiment='validation',optimization='fusion-avx-host')[0])
    path,manifest=compile_model(source,tmp_path/'batch',experiment='validation',optimization='fusion-avx-batch')
    candidate=load_module(path)
    assert next(p for p in manifest['optimization']['passes'] if p['name']=='compiled-pure-cell-grid-initializers')['applied_sites']>=3
    cache_sizes=None
    for seed in (2,7,2):
        a=reference.reset(seed=seed,parameters={'cap':0.04});b=candidate.reset(seed=seed,parameters={'cap':0.04})
        check_status(a);check_status(b)
        assert set(a)==set(b)
        for key in a:
            for left,right in zip(jax.tree.leaves(a[key]),jax.tree.leaves(b[key])):
                assert left.dtype==right.dtype
                if np.issubdtype(left.dtype,np.floating):np.testing.assert_allclose(left,right,atol=1e-12,rtol=1e-10)
                else:np.testing.assert_array_equal(left,right)
        sizes=[function._cache_size() for function in candidate.PURE_GRID_INITIALIZERS.values()]
        if cache_sizes is None:cache_sizes=sizes
        assert sizes==cache_sizes
        assert candidate.jax is jax
