"""Bind completed ten-seed native comparisons to exactly re-emitted modules.

The certificate covers only the tested configuration and 100-cycle horizon.
A selected subset keeps the complete parent report hash, including rejections.
"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model,implementation_identity


def checked_rows(summary,case,mode,steps=100):
    rows=[r for r in summary['reports'] if r['case']==case]
    if len(rows)!=10 or sorted(r['seed'] for r in rows)!=list(range(1,11)):
        raise ValueError(f'{case}: exactly seeds 1–10 required')
    for row in rows:
        value=row['modes'][mode]
        comparison=value['comparison'];identities=value['identities']
        if not (row['passed'] and comparison['passed'] and identities['passed']):raise ValueError(f'{case}: rejected trajectory')
        if comparison['snapshots']!=steps+1 or identities['snapshots']!=steps+1:raise ValueError(f'{case}: incomplete horizon')
        if comparison['float_atol']!=1e-12 or comparison['float_rtol']!=1e-10:raise ValueError(f'{case}: changed tolerances')
        kernel=value.get('kernel_comparison')
        if kernel is not None and not kernel['passed']:raise ValueError(f'{case}: rejected whole-horizon kernel')
    return rows


def certify(directory,output,mode,cases=None,steps=100):
    parent=directory/'summary.json';summary=json.loads(parent.read_text())
    if not summary.get('complete'):raise ValueError('Validation still incomplete')
    cases=cases or sorted({r['case'] for r in summary['reports']})
    rows=[r for case in cases for r in checked_rows(summary,case,mode,steps)]
    output.mkdir(parents=True,exist_ok=False);modules={};configurations={}
    for case in cases:
        original=directory/case/mode
        manifest=json.loads((original/'manifest.json').read_text())
        expected=manifest['generated_sha256']
        if hashlib.sha256((original/'model_jax.py').read_bytes()).hexdigest()!=expected:raise ValueError('Changed validated module')
        emitted,current=compile_model(manifest['source'],output/'reemitted'/case,experiment=manifest['selected_experiment'],capacities=manifest.get('capacities'),optimization=mode)
        if hashlib.sha256(emitted.read_bytes()).hexdigest()!=expected:raise ValueError(f'{case}: current compiler differs from validated code')
        modules[case]=expected
        configurations[case]={'capacities':manifest.get('capacities',{}),'experiment':manifest['selected_experiment'],'source_sha256':manifest['source_sha256']}
    proof=output/'proof.json';proof.write_text(json.dumps({'complete':True,'passed':True,'steps':steps,'scope':f'Selected tested configurations, seeds 1–10, {steps} cycles; no extrapolation to other parameters or horizons.','parent_report':str(parent.resolve()),'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'reports':rows},indent=2)+'\n')
    result={'steps':steps,'implementation_sha256':implementation_identity()[1],'modules':modules,'configurations':configurations,'proof':str(proof.resolve()),'proof_sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}
    (output/'certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('validation',type=Path);p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--optimization',choices=['safe','fusion','fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined'],required=True);p.add_argument('--cases',nargs='+');p.add_argument('--steps',type=int,default=100);a=p.parse_args()
    certify(a.validation,a.output,a.optimization,a.cases,a.steps)
