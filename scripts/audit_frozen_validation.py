#!/usr/bin/env python3
"""Verify frozen official bytes, snapshot the compiler, audit development coverage.

Reserved cases are never parsed or compiled here. Compilation acceptance alone
cannot unlock performance: strict native/JAX fidelity is a separate milestone.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model, CompileError, implementation_identity


def audit(corpus,output,capacities=None):
    corpus,output=Path(corpus).resolve(),Path(output).resolve()
    registry=json.loads((corpus/'manifest.json').read_text())
    capacities={} if capacities is None else capacities
    for relative,digest in registry['assets'].items():
        path=corpus/'official'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen asset changed: '+relative)
    for case in registry['cases']:
        if hashlib.sha256((corpus/case['source']).read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Frozen source changed: '+case['id'])
    output.mkdir(parents=True,exist_ok=False)
    files,identity=implementation_identity()
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),
                  corpus=str(corpus),corpus_manifest_sha256=hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest(),
                  implementation_sha256=identity,implementation_files=files,
                  capacities=capacities,
                  purpose='development coverage only; no fidelity or performance result',
                  fidelity=dict(discrete='exact',atol=1e-12,rtol=1e-10),
                  reserved_policy='no parsing or compilation until a separate final compiler freeze',
                  performance_gate='blocked until full coupled init/every-cycle trajectories and negative controls pass')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for source in sorted((ROOT/'gamax').glob('*.py'))+[Path(__file__).resolve()]:
        target=output/'implementation'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    shutil.copyfile(corpus/'manifest.json',output/'corpus_manifest.json')
    records=[]
    for case in registry['cases']:
        record=dict(id=case['id'],role=case['role'],source_sha256=case['sha256'],performance_eligible=False)
        if case['role']=='reserved':record.update(status='reserved_uncompiled',first_compilation=None)
        else:
            try:
                _,manifest=compile_model(corpus/case['source'],output/'generated'/case['id'],experiment=case['experiment'],capacities=capacities.get(case['id'],{}))
                record.update(status='accepted_unvalidated',generated_sha256=manifest['generated_sha256'])
            except CompileError as error:record.update(status='rejected',diagnostic=error.diagnostic)
            except Exception as error:record.update(status='compiler_error',error=repr(error))
            print(case['id']+': '+record['status'],flush=True)
        records.append(record)
    summary=dict(complete=True,development_count=sum(r['role']=='development' for r in records),
                 accepted=sum(r['status']=='accepted_unvalidated' for r in records),coupled_validated=0,
                 reserved_count=sum(r['role']=='reserved' for r in records),performance_eligible=0,reports=records)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,default=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13')
    parser.add_argument('-o','--output',type=Path,required=True)
    parser.add_argument('--capacities',type=json.loads,default={})
    args=parser.parse_args();audit(args.corpus,args.output,args.capacities)

if __name__=='__main__':main()
