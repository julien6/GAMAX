"""Native differential probe for read-only maps and grouped discarded draws."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from gamax.compiler import compile_model
from gamax.runtime import load_module,run
from gamax.validation import compare_traces,compare_identities
from benchmarks.official_native_checkpoints import observe
from validate_spatial_compilation import expression


def main(out):
 out=out.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'examples/pure_getter_maps.gaml'
 path,m=compile_model(source,out/'compiled',experiment='validation',capacities={'Walker':640},optimization='fusion-avx-batch')
 keys=list(m['fields']);observations={'field'+str(i):expression(k,m['fields'][k]) for i,k in enumerate(keys)};observations['ids']='string((Walker sort_by each.index) collect each.index)'
 native=observe(source,'validation',observations,seed=2,steps=10,output=out/'native',home=ROOT/'.tools/gama',checkpoints=list(range(11)),preserve_rng=True)
 if native['status']!='passed':raise RuntimeError(native)
 reference=[{k:row['field'+str(i)] for i,k in enumerate(keys)} for row in native['trace']]
 candidate=run(load_module(path),10,seed=2);comparison=compare_traces(reference,candidate,m['fields']);identity=compare_identities([{'Walker':r['ids']} for r in native['trace']],candidate.identities)
 result={'passed':comparison['passed'] and identity['passed'],'comparison':comparison,'identities':identity,'optimization':m['optimization']};(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True);return 0 if result['passed'] else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.output))
