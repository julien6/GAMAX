"""Coupled native/JAX tests of authored compiler-development sources.

These are compiler primitive tests, never acceptance of an official full ABM.
The original GAML source is imported unchanged by the native observer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamax.compiler import compile_model, implementation_identity,CompileError
from gamax.native_xml import observe
from gamax.runtime import load_module, run
from gamax.validation import compare_traces,compare_identities,normalize_asset_observations


def expression(key, kind):
    scope, field = key.split('.')
    value = 'simulation.'+field if scope == 'global' else 'each.'+field
    if kind.startswith('matrix:'):
        value='('+value+' as list)'
    if kind=='image_file':
        value='('+value+' = nil ? "" : '+value+'.path)'
    elif kind=='values:image_file':
        value='('+value+' collect (each=nil ? "" : each.path))'
    elif kind == 'point':
        value = '['+', '.join(value+'.'+axis for axis in ('x','y','z'))+']'
    elif kind in ('geometry','polygon'):
        value += '.points collect [each.x, each.y, each.z]'
    elif kind == 'rgb':
        value = '('+value+' = nil ? [-1,-1,-1,-1] : ['+', '.join(value+'.'+channel for channel in ('red','green','blue','alpha'))+'])'
    elif kind.startswith('agent:'):
        value = '('+value+' = nil ? -1 : '+value+'.index)'
    elif kind.startswith('agents:'):
        value = '('+value+' collect each.index)'
    elif kind=='values:rgb':
        value='('+value+' collect [each.red,each.green,each.blue,each.alpha])'
    elif kind=='values:point':
        value='('+value+' collect [each.x,each.y,each.z])'
    if scope != 'global':
        value = '('+scope+' sort_by each.index) collect ('+value+')'
    if kind in ('image_file','values:image_file','fsm_state'):return value
    # The server's numeric JSON writer rounds scalar doubles to eight decimals.
    # GAML string(Double) retains Java's round-trip representation instead.
    return 'string('+value+')' if scope != 'global' or kind in ('float','point','geometry','polygon','rgb') or kind.startswith(('agents:','values:','matrix:')) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o','--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=['grid_references','persistent_neighbors','colors_outputs'])
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(1,11)))
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--oracle',choices=['xml','ws'],default='xml')
    parser.add_argument('--parameters',type=json.loads,default={})
    parser.add_argument('--capacities',type=json.loads,default={})
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    components, identity = implementation_identity()
    snapshot = out/'implementation'; snapshot.mkdir()
    for name in components:
        (snapshot/name).write_bytes((ROOT/'gamax'/name).read_bytes())
    validation_files = [Path(__file__).resolve(), *[(ROOT/'gamax'/name) for name in ('native_ws.py','native_server.py','native_xml.py','gama_client.py','runtime.py','validation.py')]]
    validation_hashes = {}
    for path in validation_files:
        relative = str(path.relative_to(ROOT))
        validation_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = snapshot/relative; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    protocol = dict(classification='authored compiler-development validation; not official ABM acceptance',
                    seeds=args.seeds,steps=args.steps,cases=args.cases,oracle=args.oracle,implementation_sha256=identity,
                    components=components,atol=1e-12,rtol=1e-10)
    protocol['validation_files'] = validation_hashes
    protocol['parameters']=args.parameters
    protocol['capacities']=args.capacities
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    reports=[]
    def save_summary(complete=False):
        (out/'summary.json').write_text(json.dumps(dict(complete=complete,passed=complete and all(r['passed'] for r in reports),reports=reports),indent=2)+'\n')
    if args.oracle=='ws':
        from gamax.native_ws import observe as native_observe
    else: native_observe=observe
    for case in args.cases:
        source = ROOT/'examples'/case/'main.gaml'
        try:compiled,manifest = compile_model(source,out/case/'generated',experiment='validation',capacities=args.capacities)
        except Exception as error:
            directory=out/case;directory.mkdir(parents=True,exist_ok=True)
            (directory/'error.txt').write_text(traceback.format_exc())
            for seed in args.seeds:
                report=dict(case=case,seed=seed,passed=False,status='compile_rejected' if isinstance(error,CompileError) else 'compiler_error',error=repr(error))
                if isinstance(error,CompileError):report['diagnostic']=error.diagnostic
                reports.append(report)
            save_summary();continue
        native_source=compiled.parent/manifest['source_bundle_entry']
        if hashlib.sha256(native_source.read_bytes()).hexdigest()!=manifest['source_sha256']:
            raise RuntimeError('Bundled source differs from compiled source')
        module = load_module(compiled)
        keys = list(manifest['fields'])
        observations = {'field'+str(i):expression(key,manifest['fields'][key]) for i,key in enumerate(keys)}
        species=list(manifest['populations'])
        observations.update({'identity'+str(i):'string(('+name+' sort_by each.index) collect each.index)' for i,name in enumerate(species)})
        for seed in args.seeds:
            folder=out/case/str(seed);folder.mkdir()
            try:candidate=run(module,args.steps,seed=seed if manifest.get('implicit_rng') else None,parameters=args.parameters)
            except Exception as error:
                report=dict(case=case,seed=seed,passed=False,status='candidate_error',error=repr(error),source_sha256=manifest['source_sha256'])
                (folder/'error.txt').write_text(traceback.format_exc());(folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
                reports.append(report);save_summary();print(case,seed,'candidate_error',repr(error),flush=True);continue
            try:
                observer_options={'preserve_rng':True} if manifest.get('rng_observation_required') else {}
                if observer_options and args.oracle!='ws':raise RuntimeError('Random getters require the simulation-scope WS observer')
                native=native_observe(native_source,'validation',observations,seed=seed,steps=args.steps,parameters=args.parameters,output=folder/'native',home=ROOT/'.tools/gama',**observer_options)
                reference=[{key:state['field'+str(i)] for i,key in enumerate(keys)} for state in native['trace']]
                reference=normalize_asset_observations(reference,manifest,{str(compiled.parent/item['source_bundle_path']):item['key'] for item in manifest.get('assets',[])})
                comparison=compare_traces(reference,candidate,manifest['fields'])
                reference_ids=[{name:state['identity'+str(i)] for i,name in enumerate(species)} for state in native['trace']]
                identities=compare_identities(reference_ids,candidate.identities)
                passed=native['status']=='passed' and comparison['passed'] and identities['passed']
                report=dict(case=case,seed=seed,passed=passed,native_status=native['status'],comparison=comparison,identities=identities,
                            source_sha256=manifest['source_sha256'])
                for name,data in [('comparison',report),('gama_trace',reference),('jax_trace',list(candidate)),('outputs',candidate.outputs),('gama_identities',reference_ids),('jax_identities',candidate.identities)]:
                    (folder/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
            except Exception as error:
                report=dict(case=case,seed=seed,passed=False,status='validation_error',error=repr(error),source_sha256=manifest['source_sha256'])
                (folder/'error.txt').write_text(traceback.format_exc())
                (folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
            reports.append(report)
            save_summary()
            print(case,seed,report.get('native_status',report.get('status')),report['passed'],report.get('comparison',{}).get('first_mismatch'),flush=True)
    if implementation_identity()[1]!=identity: raise RuntimeError('Compiler changed during validation')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in validation_hashes.items()):
        raise RuntimeError('Validation implementation changed during campaign')
    save_summary(complete=True)
    if not all(r['passed'] for r in reports): raise SystemExit(1)

if __name__=='__main__': main()
