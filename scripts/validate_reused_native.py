"""Full per-cycle check of native session reuse against the frozen oracle."""
from pathlib import Path
import argparse,json,sys
import hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from benchmark_official import prepare_native_inputs
from benchmarks.reused_native_checkpoints import ReusedNativeExperiment
from gamax.validation import compare_traces,compare_identities,normalize_asset_observations


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());manifest=cfg['manifest']
    native,execution,observations,keys,species,assets=prepare_native_inputs(a.output,cfg)
    rows=[]
    metadata={'source_sha256':manifest['source_sha256'],'generated_sha256':manifest['generated_sha256'],'experiment':cfg['experiment'],'fields':manifest['fields'],
              'adapter_sha256':hashlib.sha256((ROOT/'benchmarks/reused_native_checkpoints.py').read_bytes()).hexdigest(),
              'preparation_sha256':hashlib.sha256((ROOT/'scripts/benchmark_official.py').read_bytes()).hexdigest()}
    with ReusedNativeExperiment(execution,'gamax_benchmark',observations,home=ROOT/'.tools/gama',output=a.output/'session',preserve_rng=manifest.get('rng_observation_required',False)) as session:
        for ordinal,seed in enumerate(a.seeds):
            result=session.run(seed=seed,steps=a.steps)
            trace=[{k:row['field'+str(i)] for i,k in enumerate(keys)} for row in result['trace']]
            trace=normalize_asset_observations(trace,manifest,assets)
            ids=[{n:row['identity'+str(i)] for i,n in enumerate(species)} for row in result['trace']]
            reference=a.oracle/str(seed)
            comparison=compare_traces(json.loads((reference/'gama_trace.json').read_text()),trace,manifest['fields'])
            identities=compare_identities(json.loads((reference/'gama_identities.json').read_text()),ids)
            row={'ordinal':ordinal,'seed':seed,'comparison':comparison,'identities':identities,'activation':result['activation'],'setup_seconds':result['setup_seconds'],'compute_seconds':result['compute_seconds']}
            rows.append(row);print(seed,comparison['passed'],identities['passed'],result['activation'],flush=True)
            (a.output/f'run-{ordinal}.json').write_text(json.dumps(result,allow_nan=False)+'\n')
            (a.output/'summary.json').write_text(json.dumps({'complete':False,'steps':a.steps,'configuration':metadata,'reports':rows},indent=2)+'\n')
    passed=all(r['comparison']['passed'] and r['identities']['passed'] for r in rows)
    (a.output/'summary.json').write_text(json.dumps({'complete':True,'passed':passed,'steps':a.steps,'configuration':metadata,'reports':rows},indent=2)+'\n')
    return 0 if passed else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('-o','--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
