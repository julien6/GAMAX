"""Capture every native cycle from unchanged development models at a horizon.

An external empty-output experiment preserves inherited parameter defaults;
rendering and pacing are excluded. Held-out sources are never selected.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from gamax.compiler import compile_model
from gamax.validation import normalize_asset_observations
from benchmarks.official_native_checkpoints import observe
from validate_spatial_compilation import expression


def main(a):
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    corpus=(ROOT/'benchmarks/official-v0.4-frozen-2026-10-13').resolve()
    cases={c['id']:c for c in json.loads((corpus/'manifest.json').read_text())['cases'] if c['role']=='development'}
    rows=[]
    for name in a.cases:
        case=cases[name];source=corpus/case['source']
        _,manifest=compile_model(source,a.output/name/'configuration',experiment=case['experiment'])
        metadata=manifest['presentation']['experiments'][case['experiment']]
        if metadata.get('unsupported_effects'):raise ValueError('Effectful experiment cannot be replaced')
        declarations=[]
        for p in metadata.get('parameters',[]):
            value=p.get('initial')
            if value is not None and type(value) not in (int,float,bool):raise ValueError('Non-scalar experiment initializer')
            suffix=' init: '+(str(value).lower() if isinstance(value,bool) else repr(value)) if value is not None else ''
            declarations.append('parameter '+json.dumps(p['label'])+' var: '+p['variable']+suffix+';')
        keys=list(manifest['fields']);species=list(manifest['populations'])
        fields={'field'+str(i):expression(k,manifest['fields'][k]) for i,k in enumerate(keys)}
        fields.update({'identity'+str(i):'string(('+n+' sort_by each.index) collect each.index)' for i,n in enumerate(species)})
        for seed in a.seeds:
            folder=a.output/name/str(seed);folder.mkdir(parents=True)
            tree=folder/'sources';shutil.copytree(corpus/'official',tree)
            copied=tree/source.relative_to(corpus/'official');execution=copied.parent/'gamax_oracle.gaml'
            execution.write_text('model gamax_oracle\nimport '+json.dumps(str(copied))+'\nexperiment validation type:gui {float minimum_cycle_duration <- 0.0;\n'+'\n'.join(declarations)+'\noutput {}\n}\n')
            result=observe(execution,'validation',fields,seed=seed,steps=a.steps,output=folder/'native',home=ROOT/'.tools/gama',preserve_rng=manifest.get('rng_observation_required',False),checkpoints=list(range(a.steps+1)))
            row={'case':name,'seed':seed,'steps':a.steps,'status':result['status'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
            if result['status']=='passed' and len(result['trace'])==a.steps+1:
                trace=[{k:s['field'+str(i)] for i,k in enumerate(keys)} for s in result['trace']]
                mapping={str(tree/Path(item['path']).resolve().relative_to(corpus/'official')):item['key'] for item in manifest.get('assets',[])}
                trace=normalize_asset_observations(trace,manifest,mapping)
                identities=[{n:s['identity'+str(i)] for i,n in enumerate(species)} for s in result['trace']]
                for filename,value in [('gama_trace.json',trace),('gama_identities.json',identities)]:
                    path=folder/filename;path.write_text(json.dumps(value)+'\n');row[filename+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            else:row['status']='rejected_native_execution_or_horizon'
            rows.append(row);(a.output/'summary.json').write_text(json.dumps({'complete':len(rows)==len(a.cases)*len(a.seeds),'steps':a.steps,'reports':rows},indent=2)+'\n');print(row,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True);p.add_argument('--seeds',type=int,nargs='+',default=list(range(1,11)));p.add_argument('--steps',type=int,default=1000);main(p.parse_args())
