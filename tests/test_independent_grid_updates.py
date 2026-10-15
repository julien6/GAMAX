"""Only independent, nonstochastic grid updates may change scheduling."""
from gamax.compiler import compile_model
from gamax.runtime import load_module,run
from gamax.validation import compare_traces


def test_vector_grid_preserves_field_coercion_order_and_frequency(tmp_path):
    source=tmp_path/'different.gaml';source.write_text('''model different global {float increment <- 0.17;} grid cell width:7 height:5 neighbors:4 frequency:2 {float ceiling <- 1.0; float food <- rnd(1.0) update:food+increment max:ceiling; rgb color <- #black update:rgb(int(255*food),255,int(255*(1-food)));} experiment validation type:gui {}''')
    a,am=compile_model(source,tmp_path/'host',experiment='validation',optimization='fusion-avx-host');b,bm=compile_model(source,tmp_path/'batch',experiment='validation',optimization='fusion-avx-batch')
    assert next(p for p in bm['optimization']['passes'] if p['name']=='vectorized-independent-grid-updates')['applied_sites']==1
    assert compare_traces(run(load_module(a),10,seed=2),run(load_module(b),10,seed=2),bm['fields'])['passed']


def test_stochastic_grid_updates_stay_sequential(tmp_path):
    source=tmp_path/'random.gaml';source.write_text('''model random grid cell width:3 height:2 {float value <- 0.0 update:rnd(1.0);} experiment validation type:gui {}''')
    _,m=compile_model(source,tmp_path/'compiled',experiment='validation',optimization='fusion-avx-batch')
    assert next(p for p in m['optimization']['passes'] if p['name']=='vectorized-independent-grid-updates')['applied_sites']==0
