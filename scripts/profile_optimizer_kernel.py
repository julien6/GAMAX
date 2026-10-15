"""Compare synchronized JAX kernels; exploratory, excludes render/startup costs."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gamax.runtime import load_module,canonical,check_status
from gamax.validation import compare_traces


def main(a):
    import jax
    rows=[]
    oracle=json.loads(a.oracle.read_text()) if a.oracle else None
    for path in a.modules:
        print('reset-start',str(path),flush=True);module=load_module(path);manifest=json.loads((path.parent/'manifest.json').read_text())
        tick=time.perf_counter();initial=module.reset(seed=a.seed);jax.block_until_ready(initial);reset=time.perf_counter()-tick
        print('reset-complete',reset,flush=True);kernel=jax.jit(lambda state:jax.lax.fori_loop(0,a.steps,lambda _,s:module.step(s),state),compiler_options=module.JIT_OPTIONS)
        tick=time.perf_counter();compiled=kernel.lower(initial).compile();compilation=time.perf_counter()-tick
        print('compile-complete',compilation,flush=True);tick=time.perf_counter();jax.block_until_ready(compiled(initial));warm=time.perf_counter()-tick
        print('warmup-complete',warm,flush=True);durations=[]
        for _ in range(a.repeats):
            tick=time.perf_counter();final=compiled(initial);jax.block_until_ready(final);durations.append(time.perf_counter()-tick);print('repeat-complete',durations[-1],flush=True)
        check_status(final)
        observed=getattr(module,'observe_jit',lambda state:state)(final)
        fields=canonical(observed,getattr(module,'ASSET_FIELDS',None),getattr(module,'ASSET_KEYS',None))
        for key in getattr(module,'MATRIX_FIELDS',[]):fields[key]=[v for row in fields[key] for v in row]
        for key,names in getattr(module,'FSM_FIELDS',{}).items():fields[key]=[names[i] for i in fields[key]]
        comparison=compare_traces([oracle[a.steps]],[fields],manifest['fields']) if oracle else None
        row={'module':str(path.resolve()),'generated_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'reset_seconds':reset,'compilation_seconds':compilation,'warmup_compute_seconds':warm,'warmed_compute_seconds':durations,'final_comparison':comparison}
        rows.append(row)
        a.output.write_text(json.dumps({'complete':len(rows)==len(a.modules),'scope':'Exploratory same-process warmed JAX kernels; host shared; no GAMA ratio or fresh-process cost conclusion.','seed':a.seed,'steps':a.steps,'reports':rows},indent=2)+'\n')
        print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--modules',type=Path,nargs='+',required=True);p.add_argument('--seed',type=int,default=2);p.add_argument('--steps',type=int,default=100);p.add_argument('--repeats',type=int,default=3);p.add_argument('--oracle',type=Path);p.add_argument('-o','--output',type=Path,required=True);main(p.parse_args())
