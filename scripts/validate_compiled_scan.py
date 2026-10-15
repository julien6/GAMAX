"""Compare every emitted scan state to the native oracle, including identities."""
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from gamax.runtime import load_module,canonical,check_status
from gamax.validation import compare_traces,compare_identities


def main(a):
    import jax
    a.output.mkdir(parents=True,exist_ok=False);reports=[]
    evidence=json.loads((ROOT/'benchmarks/official_acceptance_evidence.json').read_text())
    for config in a.configs:
        cfg=json.loads(config.read_text());case=cfg['case_id'];manifest=cfg['manifest'];module=load_module(cfg['module'])
        if hashlib.sha256(Path(cfg['module']).read_bytes()).hexdigest()!=manifest['generated_sha256']:raise RuntimeError('Changed module during scan validation')
        def transition(state):
            def advance(s,_):
                nxt=module.step(s);return nxt,nxt
            return jax.lax.scan(advance,state,None,length=a.steps)
        kernel=jax.jit(transition,compiler_options=module.JIT_OPTIONS)
        for seed in a.seeds:
            initial=module.reset(seed=seed);final,states=kernel(initial);jax.block_until_ready((final,states));check_status(final)
            def snapshot(state):
                fields=canonical(getattr(module,'observe_jit',lambda s:s)(state),getattr(module,'ASSET_FIELDS',None),getattr(module,'ASSET_KEYS',None))
                for key in getattr(module,'MATRIX_FIELDS',[]):fields[key]=[v for row in fields[key] for v in row]
                for key,names in getattr(module,'FSM_FIELDS',{}).items():fields[key]=[names[i] for i in fields[key]]
                return fields
            trace=[snapshot(initial)];identities=[module.identities(initial)]
            for cycle in range(a.steps):
                state=jax.tree.map(lambda value:value[cycle],states)
                trace.append(snapshot(state));identities.append(module.identities(state))
            archive=ROOT/evidence[case]['coupled']/case/str(seed)
            comparison=compare_traces(json.loads((archive/'gama_trace.json').read_text()),trace,manifest['fields'])
            identity=compare_identities(json.loads((archive/'gama_identities.json').read_text()),identities)
            row={'case':case,'seed':seed,'module_sha256':manifest['generated_sha256'],'comparison':comparison,'identities':identity,'passed':comparison['passed'] and identity['passed']}
            reports.append(row);print(case,seed,row['passed'],flush=True)
            if not row['passed']:
                failed=a.output/case/str(seed);failed.mkdir(parents=True,exist_ok=True)
                (failed/'candidate_trace.json').write_text(json.dumps(trace,allow_nan=False)+'\n');(failed/'candidate_identities.json').write_text(json.dumps(identities)+'\n')
            (a.output/'summary.json').write_text(json.dumps({'complete':False,'steps':a.steps,'reports':reports},indent=2)+'\n')
            del states,trace,identities,initial,final
    passed=all(r['passed'] for r in reports)
    (a.output/'summary.json').write_text(json.dumps({'complete':True,'passed':passed,'steps':a.steps,'scope':'All cycle states materialized from a compiled scan; native oracle, exact discrete/identity checks and fixed float tolerances. No speed claim; output materialization differs from the timing kernel.','reports':reports},indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--configs',type=Path,nargs='+',required=True);parser.add_argument('--seeds',type=int,nargs='+',default=list(range(1,11)));parser.add_argument('--steps',type=int,default=100);parser.add_argument('-o','--output',type=Path,required=True);arguments=parser.parse_args()
    if arguments.steps!=100:parser.error('This runner uses the archived 100-cycle oracle')
    raise SystemExit(main(arguments))
