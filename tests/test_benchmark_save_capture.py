"""Compiled scans must not duplicate saves when a model's save guard stops."""
import importlib.util
from pathlib import Path
import jax
import pytest
from gamax.compiler import compile_model
from gamax.runtime import load_module,run


def test_scan_capture_preserves_conditional_save_events(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/benchmark_official.py'
    spec=importlib.util.spec_from_file_location('benchmark_save_capture',script)
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    source=tmp_path/'conditional.gaml'
    source.write_text('''model conditional
global {
 int count <- 0;
 reflex tick { count <- count + 1;
  if count <= 2 { save ("count="+count+"; value="+0.00028753457722419107) to: "result.txt" rewrite: count=1; }
 }
}
experiment validation type: gui {}
''')
    module=load_module(compile_model(source,tmp_path/'compiled',optimization='fusion')[0])
    trace=run(module,5)
    state=module.reset();keys=[k for k in state if k.startswith('$save.') or k.startswith('$output.')]
    def step(s,_):
        nxt=module.step(s)
        return nxt,{k:nxt[k] for k in keys}
    _,emissions=jax.jit(lambda s:jax.lax.scan(step,s,None,length=5),compiler_options=module.JIT_OPTIONS)(state)
    host=jax.device_get(emissions);generations={};events=[]
    for i in range(5):events.extend(benchmark.collect_new_saves(module,{k:v[i] for k,v in host.items()},generations))
    assert events==trace.outputs
    assert len(events)==2
    assert events[0]['text']=='count=1; value=2.8753457722419107E-4'
    assert [e['rewrite'] for e in events]==[True,False]


def test_multiple_save_payloads_are_rejected_instead_of_silently_lost(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/benchmark_official.py'
    spec=importlib.util.spec_from_file_location('benchmark_save_capture',script)
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    source=tmp_path/'repeated.gaml'
    source.write_text('''model repeated global { reflex tick { loop i from:0 to:1 { save ("i="+i) to:"result.txt" rewrite:false; } } } experiment validation type:gui {}''')
    module=load_module(compile_model(source,tmp_path/'compiled')[0])
    with pytest.raises(RuntimeError,match='Multiple save emissions'):run(module,1)
    state=module.step_jit(module.reset())
    with pytest.raises(RuntimeError,match='Multiple save emissions'):benchmark.collect_new_saves(module,state,{})
