#!/usr/bin/env python3
"""Freeze source bytes and reserve independent variants without compiling them."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = [
 ('sir_spatial', 'Toy Models/Epidemiology', 'Susceptible Infected Recovered (SIR).gaml', 'Simulation'),
 ('schelling', 'Toy Models/Segregation (Schelling)', 'Segregation (Agents).gaml', 'schelling'),
 ('boids', 'Toy Models/Boids', 'Boids.gaml', 'Basic'),
 ('traffic', 'Toy Models/Traffic', 'Netlogo - Traffic model - 1 road.gaml', 'NetlogoTrafficmodel'),
 ('predator_prey', 'Tutorials/Predator Prey', 'Model 12.gaml', 'prey_predator'),
 ('ants', 'Toy Models/Ants (Foraging and Sorting)', 'Ant Foraging.gaml', 'With Inspector'),
]
RESERVED = [
 ('schelling_cellular', 'Toy Models/Segregation (Schelling)', 'Segregation (Cellular Automata).gaml'),
 ('boids_flocks', 'Toy Models/Boids', 'Boids With Flocks.gaml'),
 ('predator_prey_13', 'Tutorials/Predator Prey', 'Model 13.gaml'),
 ('ant_sorting', 'Toy Models/Ants (Foraging and Sorting)', 'Ant Sorting.gaml'),
 ('si', 'Toy Models/Epidemiology', 'Susceptible Infected (SI).gaml'),
 ('traffic_two_roads', 'Toy Models/Traffic', 'Netlogo - Traffic model - 2 roads.gaml'),
]

def main():
 p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);args=p.parse_args()
 output=args.output.resolve()
 if output.exists(): p.error('Choose a new corpus directory')
 library=next((ROOT/'.tools/gama/plugins').glob('gama.library*.jar'))
 records=[];files={}
 with zipfile.ZipFile(library) as z:
  specifications=[('development',*item) for item in PRIMARY]+[('reserved',*item,None) for item in RESERVED]
  for role,identifier,subtree,filename,experiment in specifications:
   prefix=f'models/{subtree}/'; entry=prefix+'models/'+filename
   if entry not in z.namelist(): raise ValueError(f'Missing official source: {entry}')
   records.append(dict(id=identifier,role=role,source='official/'+entry,experiment=experiment,
                       sha256=hashlib.sha256(z.read(entry)).hexdigest(),first_compilation=None))
   for name in z.namelist():
    if name.startswith(prefix) and not name.endswith('/'):
     files[name]=z.read(name)
  output.mkdir(parents=True)
  for name,data in sorted(files.items()):
   path=output/'official'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 manifest=dict(schema_version=1,frozen_at=datetime.now(timezone.utc).isoformat(),
  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),cases=records,
  assets={name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())},
  reservation='Never compile reserved entries during feature development. Prior bundling/path listing is not a claim of blinded source access.',
  fidelity=dict(atol=1e-12,rtol=1e-10,discrete='exact',seed_policy='coupled native streams, seeds 1 through 10'),
  performance=dict(seeds=list(range(1,11)),size_factors=[0.5,1.0,2.0],horizons=[100,1000,10000],repetitions=10,
                   modes=['compute','render'],timeout_seconds=1800,render_frames=11,requires_fidelity=True))
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(development=6,reserved=6,files=len(files),manifest=str(output/'manifest.json'))))

if __name__=='__main__':main()
