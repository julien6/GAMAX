#!/usr/bin/env python3
"""Fresh, serial performance cells for already validated compiler feature workloads."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_robustness import analyze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('-o','--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    validation=args.validation.resolve(); summary=json.loads((validation/'summary.json').read_text())
    if not summary['complete'] or not summary['passed']: parser.error('Require complete passed differential validation')
    protocol=json.loads((validation/'protocol.json').read_text())
    if protocol['steps'] != 100 or args.repeats < 2: parser.error('Require a validated 100-step horizon and at least two repetitions')
    output=args.output.resolve()
    if output.exists(): parser.error('Choose a new evidence directory')
    output.mkdir(parents=True)
    seed=protocol['seeds'][0]
    workloads=[r for r in protocol['workloads'] if r['seed']==seed]
    frozen=dict(classification='exploratory startup-inclusive costs on authored feature workloads; not full official-model speed or ten-seed cost robustness',
        validation=str(validation),steps=100,stride=10,repeats=args.repeats,seed=seed,workloads=workloads,
        validation_summary_sha256=hashlib.sha256((validation/'summary.json').read_bytes()).hexdigest(),
        concurrency='serial benchmark cells, no competing validation processes',rendering='same Pillow offscreen renderer, not native GUI')
    (output/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n')
    shutil.copyfile(Path(__file__),output/'benchmark_compiler_extensions.py')
    reports=[]
    for item in workloads:
        case=item['case']; source=validation/item['source']; destination=output/case
        command=[sys.executable,str(ROOT/'scripts/benchmark_rendering.py'),'--source',str(source),'--steps','100','--stride','10','--repeats',str(args.repeats),'-o',str(destination)]
        with (output/(case+'.log')).open('w') as log:
            status=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        report=dict(case=case,returncode=status,passed=False)
        if status==0:
            result=json.loads((destination/'summary.json').read_text())
            robustness=analyze(result)
            robustness['replication_unit']='independent fresh process repetitions of one fixed, explicitly reseeded source; not different random seeds'
            (destination/'robustness.json').write_text(json.dumps(robustness,indent=2)+'\n')
            report.update(passed=result['passed'],summary=result,robustness=robustness)
        reports.append(report)
        (output/'summary.json').write_text(json.dumps(dict(complete=len(reports)==len(workloads),passed=len(reports)==len(workloads) and all(r['passed'] for r in reports),reports=reports),indent=2)+'\n')
        print(case,report['passed'],flush=True)
    if not all(r['passed'] for r in reports): raise SystemExit(1)

if __name__=='__main__':main()
