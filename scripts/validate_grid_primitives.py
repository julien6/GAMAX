#!/usr/bin/env python3
"""Native/JAX development probes of ordered and shuffled grid neighborhoods.

This validates spatial primitives, not full compilation of an official model.
"""
from pathlib import Path
import argparse,json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gamax.native_xml import observe
from gamax.randomness import MERSENNE_SOURCE,mersenne_initial_state
from gamax.spatial import GRID_SOURCE
from gamax.runtime import load_module


def main():
    parser=argparse.ArgumentParser();parser.add_argument('-o','--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sources={name:(ROOT/'gamax'/name).read_bytes() for name in ('randomness.py','spatial.py','native_xml.py')}
    (out/'implementation').mkdir()
    for name,data in sources.items():(out/'implementation'/name).write_bytes(data)
    (out/'protocol.json').write_text(json.dumps(dict(classification='spatial primitive development validation; not official ABM compilation',seeds=args.seeds,implementation={n:hashlib.sha256(d).hexdigest() for n,d in sources.items()}),indent=2)+'\n')
    module=out/'grid_jax.py'
    import inspect
    module.write_text('import math, struct\nimport jax\njax.config.update("jax_enable_x64",True)\nimport jax.numpy as jnp\n'+inspect.getsource(mersenne_initial_state)+'\n'+MERSENNE_SOURCE+'\n'+GRID_SOURCE+'''\ndef initialize(seed, width, height, radius, moore, torus):
    rng=(jnp.asarray(mersenne_initial_state(seed),jnp.uint32),jnp.asarray(624,jnp.int32))
    capacity=(2*radius+1)**2-1 if moore else 2*radius*(radius+1)
    arrays=jnp.full((width*height,capacity),-1,jnp.int32)
    masks=jnp.zeros((width*height,capacity),jnp.bool_)
    def create(i,state):
        rng,arrays,masks=state
        rng,(neighbors,present)=grid_neighbors(rng,i,width,height,radius,moore=moore,torus=torus)
        return rng,arrays.at[i].set(neighbors),masks.at[i].set(present)
    def implicit_neighbors(i,rng):
        rng,_=grid_neighbors(rng,i,width,height,1,moore=moore,torus=torus)
        return rng
    rng=jax.lax.fori_loop(0,width*height,implicit_neighbors,rng)
    rng,arrays,masks=jax.lax.fori_loop(0,width*height,create,(rng,arrays,masks))
    rng,sample=mersenne_uniform(rng)
    return arrays,masks,jnp.trunc(sample*1000001).astype(jnp.int32)
initialize_jit=jax.jit(initialize,static_argnums=(0,1,2,3,4,5))
''')
    m=load_module(module);reports=[]
    import jax,numpy as np
    for width,height,radius in ((4,3,2),(3,2,3)):
        for moore in (False,True):
            for torus in (False,True):
                for seed in args.seeds:
                    case=f'{width}x{height}-r{radius}-moore{int(moore)}-torus{int(torus)}-seed{seed}'
                    folder=out/case;folder.mkdir()
                    source=folder/'probe.gaml'
                    source.write_text(f'model spatial_probe\nglobal torus: {str(torus).lower()} {{ geometry shape <- square(12); int next_draw; init {{ next_draw <- rnd(1000000); }} }}\ngrid cell width: {width} height: {height} neighbors: {8 if moore else 4} use_regular_agents: false frequency: 0 {{ list<cell> nearby <- self neighbors_at {radius}; }}\nexperiment validation type: gui {{}}\n')
                    native=observe(source,'validation',{'neighbors':'string(cell collect (each.nearby collect each.index))','next_draw':'next_draw'},seed=seed,steps=1,output=folder/'native',home=ROOT/'.tools/gama')
                    arrays,masks,next_draw=jax.device_get(m.initialize_jit(seed,width,height,radius,moore,torus))
                    candidate=dict(neighbors=[row[mask].tolist() for row,mask in zip(arrays,masks)],next_draw=int(next_draw))
                    passed=native['status']=='passed' and native['trace'][0]==candidate
                    report=dict(case=case,passed=passed,candidate=candidate,native=native)
                    (folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
                    (out/'summary.json').write_text(json.dumps(dict(complete=len(reports)==8*len(args.seeds),passed=all(r['passed'] for r in reports),reports=reports),indent=2)+'\n')
                    print(case,passed,flush=True)
    if not all(r['passed'] for r in reports):raise SystemExit(1)
if __name__=='__main__':main()
