"""Check deterministic emitted code after renaming and relocating development sources.

Reserved independent variants are not opened or compiled. This finite check
supports independence from source/model names; it is not a language proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gamax.compiler import compile_model,implementation_identity


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('-o','--output',type=Path,required=True);a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    corpus=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13'
    registry=json.loads((ROOT/'benchmarks/official_acceptance_evidence.json').read_text());reports=[]
    for case in json.loads((corpus/'manifest.json').read_text())['cases']:
        if case['role']!='development':continue
        directory=a.output/case['id'];directory.mkdir()
        entry=registry.get(case['id']);capacities=json.loads((ROOT/entry['coupled']/'protocol.json').read_text())['capacities'] if entry else {}
        baseline,manifest=compile_model(corpus/case['source'],directory/'baseline',experiment=case['experiment'],capacities=capacities)
        repeated,_=compile_model(corpus/case['source'],directory/'repeat',experiment=case['experiment'],capacities=capacities)
        tree=directory/'arbitrary_location'/'sources';shutil.copytree(baseline.parent/'sources',tree)
        source=directory/'arbitrary_location'/manifest['source_bundle_entry']
        text=source.read_bytes().decode()
        renamed,count=re.subn(r'(?m)^(\s*model\s+)[A-Za-z_][A-Za-z_0-9]*',r'\1unrelated_model_identifier',text,count=1)
        if count!=1:raise ValueError('Expected one model declaration')
        moved=source.with_name('unrelated_entry_filename.gaml');moved.write_bytes(renamed.encode());source.unlink()
        candidate,_=compile_model(moved,directory/'renamed_relocated',experiment=case['experiment'],capacities=capacities)
        record=dict(id=case['id'],passed=digest(baseline)==digest(repeated)==digest(candidate),generated_sha256=digest(baseline),repeated_sha256=digest(repeated),renamed_relocated_sha256=digest(candidate),source_sha256=case['sha256'])
        reports.append(record);print(case['id'],record['passed'],flush=True)
    result=dict(passed=len(reports)==6 and all(r['passed'] for r in reports),reports=reports,implementation_sha256=implementation_identity()[1],reserved_policy='not opened or compiled')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()
