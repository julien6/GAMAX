"""Corrupt generic compiler behavior or archived source copies to test fidelity gates."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model,load_model,implementation_identity
from gamax.lowering import Generator
from gamax.runtime import load_module,run
from gamax.validation import compare_traces,compare_identities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--capacities',type=json.loads,required=True)
    p.add_argument('--seed',type=int,default=2)
    p.add_argument('--steps',type=int,help='Compare a checked prefix of the complete native reference')
    p.add_argument('--mutations',type=json.loads,help='Map control name to [unique source fragment, replacement]')
    p.add_argument('-o','--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    # Mutation sites are experiment inputs, never branches in the compiler.
    text=a.source.read_text()
    mutations=a.mutations or {
        'rng':('reflex basic_move {','reflex basic_move { if (flip(1.0)) {}'),
        'selection':('one_of (myPlace.neighbours)','one_of (myPlace.neighbours where (each.index mod 2 = 0))'),
        'lifecycle':('do die;',''),
    }
    if set(mutations)!={'rng','selection','lifecycle'}:raise ValueError('Exactly RNG, selection and lifecycle fault sites required')
    _,bundle_manifest=compile_model(a.source,a.output/'source_bundle',experiment=a.experiment,capacities=a.capacities)
    reference=json.loads((a.reference/'gama_trace.json').read_text())
    identities=json.loads((a.reference/'gama_identities.json').read_text())
    steps=len(reference)-1
    if a.steps is not None:
        if not 1<=a.steps<=steps:raise ValueError('Invalid native reference prefix horizon')
        steps=a.steps;reference=reference[:steps+1];identities=identities[:steps+1]
    reports=[]
    for name in ('baseline','scheduling',*mutations):
        directory=a.output/name;directory.mkdir()
        shutil.copytree(a.output/'source_bundle'/'sources',directory/'sources')
        source=directory/bundle_manifest['source_bundle_entry'];candidate_text=text
        if name in mutations:
            before,after=mutations[name]
            if text.count(before)!=1:raise ValueError('Fault injection site must be unique: '+name)
            candidate_text=text.replace(before,after)
        source.write_text(candidate_text)
        module,manifest=compile_model(source,directory/'generated',experiment=a.experiment,capacities=a.capacities)
        if name=='scheduling':
            ir,_=load_model(source,experiment=a.experiment);ir.capacities=dict(a.capacities)
            module.write_text(Generator(ir,reverse_agents=True).generate())
        try:trace=run(load_module(module),steps,seed=a.seed)
        except Exception as error:
            record=dict(control=name,passed=name!='baseline',detected=name!='baseline',error=repr(error),
                        mutated_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
            (directory/'comparison.json').write_text(json.dumps(record,indent=2)+'\n');reports.append(record)
            print(name,record['passed'],repr(error),flush=True);continue
        comparison=compare_traces(reference,trace,manifest['fields']);ids=compare_identities(identities,trace.identities)
        detected=not comparison['passed'] or not ids['passed']
        passed=not detected if name=='baseline' else detected
        record=dict(control=name,passed=passed,detected=detected,comparison=comparison,identities=ids,
                    mutated_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(module.read_bytes()).hexdigest())
        (directory/'jax_trace.json').write_text(json.dumps(list(trace))+'\n')
        (directory/'comparison.json').write_text(json.dumps(record,indent=2)+'\n');reports.append(record)
        first=comparison['first_mismatch']
        print(name,passed,{'step':first['step'],'field':first['field']} if first else None, bool(ids['first_mismatch']),flush=True)
    summary=dict(passed=all(r['passed'] for r in reports),reports=reports,classification='deliberately corrupted copies; not source acceptance',seed=a.seed,steps=steps,reference_hashes={name:hashlib.sha256((a.reference/name).read_bytes()).hexdigest() for name in ('gama_trace.json','gama_identities.json')},implementation_sha256=implementation_identity()[1],validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['passed']:raise SystemExit(1)

if __name__=='__main__':main()
