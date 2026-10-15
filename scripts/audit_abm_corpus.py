#!/usr/bin/env python3
"""Freeze actual official sources and report compiler coverage, including unavailable requested cases."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from gamax.compiler import compile_model, CompileError, implementation_identity
import gamax.compiler as compiler

ROOT = Path(__file__).resolve().parents[1]
CASES = [
 ('schelling', 'grid', 'models/Toy Models/Segregation (Schelling)/models/Segregation (Agents).gaml', ['imports/inheritance','grids','agent references/collections','random draws','native graphics']),
 ('predator_prey', 'grid/lifecycle', 'models/Tutorials/Predator Prey/models/Model 12.gaml', ['actions/inheritance','grids','birth/death','random draws','field updates']),
 ('ants', 'grid/continuous', 'models/Toy Models/Ants (Foraging and Sorting)/models/Ant Foraging.gaml', ['FSM','diffusion','grid/matrix','moving skill','random draws']),
 ('boids', 'continuous', 'models/Toy Models/Boids/models/Boids.gaml', ['point/vector arithmetic','neighborhood queries','actions/ask','moving skill','random draws']),
 ('sir_spatial', 'grid', 'models/Toy Models/Epidemiology/models/Susceptible Infected Recovered (SIR).gaml', ['grids','collections/loops','random draws','agent lifecycle']),
 ('traffic', 'grid/continuous', 'models/Toy Models/Traffic/models/Netlogo - Traffic model - 1 road.gaml', ['toroidal topology','spatial queries','moving skill','actions','random draws']),
 ('forest_fire', 'grid', None, ['synchronous grid fields','neighbor propagation','random draws']),
 ('sir_small_world', 'network', None, ['graph construction','bounded adjacency','random draws','infection/recovery schedule']),
 ('granovetter', 'network', None, ['bounded adjacency','individual thresholds','adoption schedule']),
 ('social_force', 'continuous', None, ['force equations','spatial queries','integration','obstacles']),
]


def main():
 parser=argparse.ArgumentParser();parser.add_argument('-o','--output',type=Path,required=True);args=parser.parse_args()
 if args.output.exists(): raise SystemExit('Choose a new corpus directory')
 args.output.mkdir(parents=True)
 library=next((ROOT/'.tools/gama/plugins').glob('gama.library*.jar'))
 records=[]
 with zipfile.ZipFile(library) as archive:
  for name, topology, member, extensions in CASES:
   record=dict(id=name,topology=topology,origin='GAMA 2025.06.4 library' if member else 'requested family; source not yet selected',required_extensions=extensions)
   if member:
    # Copy the model subtree to retain relative imports/assets, without rewriting source.
    subtree=Path(member).parent.parent.as_posix()+'/'
    for entry in archive.namelist():
     if entry.startswith(subtree) and not entry.endswith('/'):
      target=args.output/'official'/entry
      target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(entry))
    source=args.output/'official'/member
    record.update(source=str(source.relative_to(args.output)),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    try:
     _,manifest=compile_model(source,args.output/'generated'/name)
     record.update(status='accepted_unvalidated',manifest=manifest)
    except CompileError as error:
     record.update(status='rejected',first_diagnostic=error.diagnostic)
   else: record['status']='source_pending'
   records.append(record)
 summary=dict(schema_version=1,classification='target corpus/coverage audit; not held-out evaluation or performance evidence',
              compiler_sha256=hashlib.sha256(Path(compiler.__file__).read_bytes()).hexdigest(), implementation_sha256=implementation_identity()[1],
              library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),cases=records,
              denominator=len(records),counts={status:sum(r['status']==status for r in records) for status in ['rejected','source_pending','accepted_unvalidated']},
              native_validity='Official provenance only; full compilation/runs with GAMA still to validate independently',
              notes=['SIR spatial is not substituted for requested small-world SIR','No ODE-only model substituted for predator-prey ABM',
                     'Required extensions are design annotations, not a complete automatic feature inventory', 'First compiler diagnostic is not an exhaustive list of unsupported features'])
 (args.output/'coverage.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(summary['counts']))


if __name__=='__main__':main()
