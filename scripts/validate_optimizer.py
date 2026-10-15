"""Compare opt-in optimized official sources to archived native trajectories.

Recompiles unchanged sources; checks baseline byte identity, every declared
state/identity, determinism, and times the synchronized warmed scan separately.
Only development sources from the acceptance registry; no held-out compilation.
"""
from pathlib import Path
import argparse,json,time,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model
from gamax.runtime import load_module,run
from gamax.validation import compare_traces,compare_identities

def archive_divergence(directory, reference, candidate, comparison, fields):
    """Keep replayable states and the first bit difference per floating field."""
    import struct
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'candidate_trace.json').write_text(json.dumps(candidate,allow_nan=False)+'\n')
    (directory/'candidate_identities.json').write_text(json.dumps(candidate.identities,allow_nan=False)+'\n')
    def flatten(value):
        if isinstance(value,list):
            for item in value:yield from flatten(item)
        else:yield value
    first={}
    for step,(expected,actual) in enumerate(zip(reference,candidate)):
        for field,kind in fields.items():
            if field in first or kind not in ('float','point','geometry','polygon','values:float','matrix:float'):continue
            for offset,(a,b) in enumerate(zip(flatten(expected[field]),flatten(actual[field]))):
                if struct.pack('>d',a)!=struct.pack('>d',b):
                    first[field]={'step':step,'flat_offset':offset,'reference':a,'candidate':b,'reference_hex':float(a).hex(),'candidate_hex':float(b).hex()};break
    (directory/'diagnostic.json').write_text(json.dumps({'comparison':comparison,'first_bit_difference_by_field':first,'note':'Bit differences are diagnostic; acceptance still uses the fixed numerical tolerances.'},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--cases',nargs='+',default=['sir_spatial']);p.add_argument('--seeds',nargs='+',type=int,default=[2]);p.add_argument('--fidelity-only',action='store_true');p.add_argument('--optimization',choices=['none','safe','fusion','fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined'],default='safe');p.add_argument('--steps',type=int,default=100);p.add_argument('--oracle-root',type=Path);p.add_argument('--capacities-file',type=Path);a=p.parse_args();
    if a.steps<1 or a.steps!=100 and a.oracle_root is None:raise ValueError('An explicit full-horizon native oracle is required outside 100 cycles')
    a.output.mkdir(parents=True,exist_ok=False)
    capacity_overrides=json.loads(a.capacities_file.read_text()) if a.capacities_file else {}
    corpus=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13';cases={c['id']:c for c in json.loads((corpus/'manifest.json').read_text())['cases'] if c['role']=='development'};evidence=json.loads((ROOT/'benchmarks/official_acceptance_evidence.json').read_text());accepted={r['id']:r for r in json.loads((ROOT/'results/official-six-final-verified-acceptance-2026-10-13/summary.json').read_text())['reports']};reports=[]
    import jax
    for name in a.cases:
        c=cases[name];caps={'Host':1024} if name=='sir_spatial' else {'prey':1000,'predator':300} if name=='predator_prey' else {};modules={}
        for mode in dict.fromkeys(['none',a.optimization]):
            candidate_caps=caps if mode=='none' else capacity_overrides.get(name,caps)
            path,m=compile_model(corpus/c['source'],a.output/name/mode,experiment=c['experiment'],capacities=candidate_caps,optimization=mode)
            if mode=='none' and hashlib.sha256(path.read_bytes()).hexdigest()!=accepted[name]['generated_sha256']:raise RuntimeError('Baseline code changed')
            modules[mode]=(load_module(path),m)
        for seed in a.seeds:
            archive=(a.oracle_root/name/str(seed)) if a.oracle_root else ROOT/evidence[name]['coupled']/name/str(seed);reference=json.loads((archive/'gama_trace.json').read_text());ids=json.loads((archive/'gama_identities.json').read_text());
            if len(reference)!=a.steps+1 or len(ids)!=a.steps+1:raise ValueError('Native oracle horizon mismatch')
            row={'case':name,'seed':seed,'modes':{}}
            for mode,(module,m) in modules.items():
                if a.fidelity_only and mode=='none' and a.optimization!='none':continue
                try:
                    candidate=run(module,a.steps,seed=seed)
                except (RuntimeError,ValueError,MemoryError) as exc:
                    status='capacity_overflow' if isinstance(exc,RuntimeError) and 'runtime error 6:' in str(exc) else 'oom' if isinstance(exc,MemoryError) else 'execution_error'
                    row['modes'][mode]={'comparison':{'passed':False,'snapshots':None},'identities':{'passed':False},'kernel_comparison':None,'failure':{'status':status,'exception':repr(exc)},'optimization':m['optimization'],'capacities':m['capacities']}
                    print(name,seed,mode,status,repr(exc),flush=True)
                    continue
                comparison=compare_traces(reference,candidate,m['fields']);identity=compare_identities(ids,candidate.identities)
                if any(event.get('kind')=='save' for event in candidate.outputs):
                    output_directory=a.output/name/mode/str(seed)
                    output_directory.mkdir(parents=True,exist_ok=True)
                    (output_directory/'output_events.json').write_text(json.dumps(candidate.outputs,indent=2,allow_nan=False)+'\n')
                if not comparison['passed'] or not identity['passed']:
                    archive_divergence(a.output/name/mode/str(seed)/'divergence',reference,candidate,comparison,m['fields'])
                if a.fidelity_only:
                    row['modes'][mode]={'comparison':comparison,'identities':identity,'kernel_comparison':None,'optimization':m['optimization']}
                    print(name,seed,mode,comparison['passed'],identity['passed'],flush=True)
                    continue
                def transition(s):return jax.lax.fori_loop(0,a.steps,lambda _,v:module.step(v),s)
                kernel=jax.jit(transition,compiler_options=module.JIT_OPTIONS);initial=module.reset(seed=seed);tick=time.perf_counter();jax.block_until_ready(kernel(initial));warm=time.perf_counter()-tick
                durations=[]
                for _ in range(3):
                    tick=time.perf_counter();final=kernel(initial);jax.block_until_ready(final);durations.append(time.perf_counter()-tick)
                # Compiled whole-horizon kernel must agree with checked per-cycle execution.
                from gamax.runtime import canonical
                observed=getattr(module,'observe_jit',lambda s:s)(final);f=canonical(observed,getattr(module,'ASSET_FIELDS',None),getattr(module,'ASSET_KEYS',None))
                for key in getattr(module,'MATRIX_FIELDS',[]):f[key]=[v for r in f[key] for v in r]
                for key,names in getattr(module,'FSM_FIELDS',{}).items():f[key]=[names[i] for i in f[key]]
                kernel_comparison=compare_traces([reference[-1]],[f],m['fields'])
                row['modes'][mode]={'comparison':comparison,'identities':identity,'kernel_comparison':kernel_comparison,'warmup_seconds':warm,'warmed_compute_seconds':durations,'optimization':m['optimization']}
                print(name,seed,mode,comparison['passed'],identity['passed'],kernel_comparison['passed'],durations,flush=True)
            row['passed']=all(v['comparison']['passed'] and v['identities']['passed'] and (v['kernel_comparison'] is None or v['kernel_comparison']['passed']) for v in row['modes'].values());reports.append(row);(a.output/'summary.json').write_text(json.dumps({'complete':False,'steps':a.steps,'reports':reports},indent=2)+'\n')
    passed=all(r['passed'] for r in reports);(a.output/'summary.json').write_text(json.dumps({'complete':True,'steps':a.steps,'passed':passed,'classification':'optimizer development, archived native oracle; kernel timings exclude observations/render/saves; three warm repetitions in same process, not fresh paired engine campaign','reports':reports},indent=2)+'\n');return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
