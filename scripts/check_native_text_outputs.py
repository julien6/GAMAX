"""Compare emitted relative text saves against a private unchanged native run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gamax.runtime import load_module,run,export_trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('module',type=Path);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--original-source',type=Path,required=True);p.add_argument('--seed',type=int,default=2);p.add_argument('--steps',type=int,default=100);p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    trace=run(load_module(a.module),a.steps,seed=a.seed)
    paths=sorted({event['path'] for event in trace.outputs if event['kind']=='save'})
    if not paths:raise ValueError('No text saves exercised')
    for name in paths:
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Nonportable save path')
        initial=a.original_source.parent/relative
        if initial.exists():
            destination=a.output/'exports'/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(initial.read_bytes())
    export_trace(trace,a.output/'exports','jax')
    records=[]
    for name in paths:
        native=a.native_source.parent/name;candidate=a.output/'exports'/name
        records.append(dict(path=name,passed=native.read_bytes()==candidate.read_bytes(),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),jax_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),save_events=sum(event['kind']=='save' and event['path']==name for event in trace.outputs)))
    result=dict(passed=all(record['passed'] for record in records),reports=records,module_sha256=hashlib.sha256(a.module.read_bytes()).hexdigest(),seed=a.seed,steps=a.steps,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('native text outputs',result['passed'],flush=True)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
