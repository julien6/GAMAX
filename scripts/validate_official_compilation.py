"""Strict coupled validation of frozen development ABMs; reserved sources untouched."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model,implementation_identity,CompileError
from gamax.native_ws import observe
from gamax.runtime import run,load_module
from gamax.validation import compare_traces,compare_identities,normalize_asset_observations
from validate_spatial_compilation import expression


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-o','--output',type=Path,required=True)
    p.add_argument('--corpus',type=Path,default=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13')
    p.add_argument('--cases',nargs='+',default=['sir_spatial'])
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(1,11)))
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--capacities',type=json.loads,default={})
    p.add_argument('--parameters',type=json.loads,default={})
    p.add_argument('--optimization',choices=['none','safe','fusion','fusion-avx-host','fusion-avx-batch'],default='none')
    a=p.parse_args();a.corpus=a.corpus.resolve();registry=json.loads((a.corpus/'manifest.json').read_text())
    cases={r['id']:r for r in registry['cases'] if r['role']=='development'}
    if any(c not in cases for c in a.cases):raise ValueError('Only development sources may be compiled')
    def check_corpus():
        for relative,digest in registry['assets'].items():
            if hashlib.sha256((a.corpus/'official'/relative).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen asset changed: '+relative)
    check_corpus()
    a.output.mkdir(parents=True,exist_ok=False)
    sources=[*(ROOT/'gamax').glob('*.py'),Path(__file__).resolve(),ROOT/'scripts/validate_spatial_compilation.py']
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    for f in sources:
        target=a.output/'implementation'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    protocol=dict(classification='official unchanged development source coupled validation; no performance evidence',
                  implementation_sha256=implementation_identity()[1],files=hashes,seeds=a.seeds,steps=a.steps,
                  optimization=a.optimization,capacities=a.capacities,parameters=a.parameters,atol=1e-12,rtol=1e-10,
                  corpus_sha256=hashlib.sha256((a.corpus/'manifest.json').read_bytes()).hexdigest(),
                  acceptance_gate='all requested coupled trajectories and separate negative controls; compilation alone insufficient')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    reports=[]
    for name in a.cases:
        case=cases[name];source=a.corpus/case['source']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=case['sha256']:raise RuntimeError('Changed source')
        compile_start=time.perf_counter();compile_cpu=time.process_time()
        try:module_path,manifest=compile_model(source,a.output/name/'generated',experiment=case['experiment'],capacities=a.capacities,optimization=a.optimization)
        except CompileError as e:
            reports.append(dict(id=name,status='compile_rejected',diagnostic=e.diagnostic));continue
        except Exception as e:
            reports.append(dict(id=name,status='compiler_error',error=repr(e)));continue
        (a.output/name/'translation.json').write_text(json.dumps({'translation_seconds':time.perf_counter()-compile_start,'translation_cpu_seconds':time.process_time()-compile_cpu})+'\n')
        model=load_module(module_path);keys=list(manifest['fields']);species=list(manifest['populations'])
        observations={'field'+str(i):expression(k,manifest['fields'][k]) for i,k in enumerate(keys)}
        observations.update({'identity'+str(i):'string(('+n+' sort_by each.index) collect each.index)' for i,n in enumerate(species)})
        for seed in a.seeds:
            directory=a.output/name/str(seed);directory.mkdir()
            record=dict(id=name,seed=seed,status='pending',source_sha256=case['sha256'])
            # Each oracle receives a private complete asset tree, including imports.
            native_tree=directory/'oracle_sources';shutil.copytree(a.corpus/'official',native_tree)
            native_source=native_tree/source.relative_to(a.corpus/'official')
            try:
                candidate=run(model,a.steps,seed=seed if manifest['implicit_rng'] else None,parameters=a.parameters)
                native=observe(native_source,case['experiment'],observations,seed=seed,steps=a.steps,parameters=a.parameters,output=directory/'native',home=ROOT/'.tools/gama',preserve_rng=manifest.get('rng_observation_required',False))
                reference=[{k:s['field'+str(i)] for i,k in enumerate(keys)} for s in native['trace']]
                ids=[{n:s['identity'+str(i)] for i,n in enumerate(species)} for s in native['trace']]
                reference=normalize_asset_observations(reference,manifest,{str(native_tree/Path(item['path']).relative_to(a.corpus/'official')):item['key'] for item in manifest.get('assets',[])})
                comparison=compare_traces(reference,candidate,manifest['fields']);identities=compare_identities(ids,candidate.identities)
                passed=native['status']=='passed' and comparison['passed'] and identities['passed']
                record.update(status='coupled_passed' if passed else 'failed',native_status=native['status'],comparison=comparison,identities=identities)
                for label,data in [('gama_trace',reference),('jax_trace',list(candidate)),('gama_identities',ids),('jax_identities',candidate.identities)]:
                    (directory/(label+'.json')).write_text(json.dumps(data,allow_nan=False)+'\n')
            except Exception as e:record.update(status='execution_error',error=repr(e))
            (directory/'comparison.json').write_text(json.dumps(record,indent=2)+'\n');reports.append(record)
            print(name,seed,record['status'],record.get('comparison',{}).get('first_mismatch'),record.get('error'),flush=True)
            (a.output/'summary.json').write_text(json.dumps(dict(complete=False,reports=reports),indent=2)+'\n')
    check_corpus()
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):raise RuntimeError('Implementation changed during campaign')
    passed=all(r['status']=='coupled_passed' for r in reports)
    (a.output/'summary.json').write_text(json.dumps(dict(complete=True,passed=passed,reports=reports),indent=2)+'\n')
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
