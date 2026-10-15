#!/usr/bin/env python3
"""Verify archived functional gates and byte identity of current emitted modules.

The registry distinguishes compilation from validated coupled execution. It does
not infer any performance result or compile reserved independent variants.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model,CompileError,implementation_identity


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)


def verify(case,entry,current,corpus):
    coupled=ROOT/entry['coupled'];controls=ROOT/entry['negative_controls'];standalone=ROOT/entry['standalone']
    protocol=read(coupled/'protocol.json');summary=read(coupled/'summary.json')
    require(summary.get('complete') and summary.get('passed'),'Incomplete/failed coupled campaign')
    require(protocol['seeds']==list(range(1,11)) and protocol['steps']==100,'Required ten-seed, 100-cycle gate missing')
    require(protocol['atol']==1e-12 and protocol['rtol']==1e-10,'Tolerance contract changed')
    require(protocol['corpus_sha256']==digest(corpus/'manifest.json'),'Different frozen corpus')
    frozen=coupled/case['id']/'generated'/'model_jax.py'
    require(digest(current)==digest(frozen),'Current emitted module differs from validated frozen module; replay required')
    frozen_manifest=read(frozen.parent/'manifest.json')
    # Older manifests hashed read_text(), which normalizes CRLF. Verify the
    # untouched bundled entry itself, rather than trusting that metadata field.
    require(digest(frozen.parent/frozen_manifest['source_bundle_entry'])==case['sha256'],'Validated bundled source differs from official source')
    reports=[r for r in summary['reports'] if r['id']==case['id']]
    require(sorted(r['seed'] for r in reports)==list(range(1,11)),'Missing or duplicate seed evidence')
    for report in reports:
        comparison=report['comparison']
        require(report['status']=='coupled_passed' and report['native_status']=='passed','Native/coupled failure')
        require(report['source_sha256']==case['sha256'],'Report source differs')
        require(comparison['passed'] and comparison['mismatched_values']==0 and comparison['snapshots']==101,'State mismatch/missing frames')
        require(comparison['float_atol']==1e-12 and comparison['float_rtol']==1e-10 and comparison['discrete_comparison']=='exact','Comparison contract differs')
        require(report['identities']['passed'] and report['identities']['snapshots']==101,'Identity comparison failed')
    invariants=read(coupled/'invariants.json')
    require(invariants['passed'],'Invariant failure')
    require(sorted((r['seed'],r['engine']) for r in invariants['reports'])==[(seed,engine) for seed in range(1,11) for engine in ('gama','jax')],'Incomplete invariants')
    trace_hashes={}
    for report in invariants['reports']:
        require(report['passed'] and report['frames']==101,'Invariant frames missing')
        for name,expected in report['hashes'].items():
            relative=case['id']+'/'+str(report['seed'])+'/'+name
            require(digest(coupled/relative)==expected,'Archived trace changed: '+relative)
            trace_hashes[relative]=expected
    negative=read(controls/'summary.json')
    require(negative['passed'],'Negative controls failed')
    require(sorted(r['control'] for r in negative['reports'])==sorted(['baseline','scheduling','rng','selection','lifecycle']),'Required negative controls missing')
    for report in negative['reports']:
        require(report['passed'],'Negative control failed')
        if report['control']=='baseline':
            require(report['generated_sha256']==digest(frozen) and report['comparison']['passed'],'Control baseline differs')
        else:require(report['detected'],'Injected defect was not detected')
    independent=read(standalone/'result.json')
    require(independent['passed'] and independent['returncode']==0 and independent['require_no_gamax'],'Standalone gate failed')
    require(independent['module_sha256']==digest(frozen),'Standalone module differs')
    return dict(status='coupled_validated',seeds=list(range(1,11)),cycles=100,
                compared_values=sum(r['comparison']['compared_values'] for r in reports),
                generated_sha256=digest(current),current_emission_byte_identical=True,
                evidence=entry,trace_hashes=trace_hashes,
                evidence_hashes={str(path.relative_to(ROOT)):digest(path) for path in (coupled/'protocol.json',coupled/'summary.json',coupled/'invariants.json',controls/'summary.json',standalone/'result.json')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13')
    parser.add_argument('--evidence',type=Path,default=ROOT/'benchmarks/official_acceptance_evidence.json')
    parser.add_argument('-o','--output',type=Path,required=True)
    args=parser.parse_args();corpus=args.corpus.resolve();registry=read(corpus/'manifest.json');evidence=read(args.evidence)
    for relative,expected in registry['assets'].items():require(digest(corpus/'official'/relative)==expected,'Frozen asset changed: '+relative)
    for case in registry['cases']:require(digest(corpus/case['source'])==case['sha256'],'Frozen source changed: '+case['id'])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);reports=[]
    for case in registry['cases']:
        if case['role']!='development':continue
        record=dict(id=case['id'],source_sha256=case['sha256'],performance_eligible=False)
        try:
            entry=evidence.get(case['id'])
            capacities=read(ROOT/entry['coupled']/'protocol.json')['capacities'] if entry else {}
            module,manifest=compile_model(corpus/case['source'],out/'generated'/case['id'],experiment=case['experiment'],capacities=capacities)
            record.update(verify(case,entry,module,corpus) if entry else dict(status='compiled_without_fidelity_evidence'))
        except CompileError as error:record.update(status='rejected',diagnostic=error.diagnostic)
        except Exception as error:record.update(status='acceptance_error',error=repr(error))
        reports.append(record);print(case['id']+': '+record['status'],flush=True)
    accepted=sum(r['status']=='coupled_validated' for r in reports)
    _,identity=implementation_identity()
    (out/'summary.json').write_text(json.dumps(dict(complete=True,all_six_accepted=accepted==6,accepted=accepted,development_count=len(reports),implementation_sha256=identity,performance_eligible=0,reserved_policy='not compiled',reports=reports),indent=2)+'\n')


if __name__=='__main__':main()
