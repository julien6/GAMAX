"""Six official sources, fresh paired processes, common rendering and checkpoint fidelity.

Exploratory native-size campaign. Native cold cost includes seeded preflight;
compute includes WebSocket step control. Neither is native GUI performance.
"""
from pathlib import Path
import argparse, hashlib, json, os, random, resource, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def collect_new_saves(module,snapshot,generations):
    """A retained save payload is not a new emission after its guard turns false."""
    result=[]
    for event in module.outputs(snapshot):
        if event['kind']!='save':continue
        generation=event['generation']
        if generation>generations.get(event['id'],0)+1:
            raise RuntimeError('Multiple save emissions per statement in one cycle cannot be reconstructed')
        if generation>generations.get(event['id'],0):
            result.append(event);generations[event['id']]=generation
    return result

def draw(frame):
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(640,640),'white');d=ImageDraw.Draw(image)
    boundary=frame.get('global.shape',[])
    if isinstance(boundary,list) and boundary and isinstance(boundary[0],list):
        width=max(float(v[0]) for v in boundary);height=max(float(v[1]) for v in boundary)
    else:width=height=100.0
    width=max(width,1.0);height=max(height,1.0)
    for name in sorted(frame,key=lambda k:(-len(frame[k]) if k.endswith('.location') and isinstance(frame[k],list) else 0,k)):
        positions=frame[name]
        if not name.endswith('.location') or not isinstance(positions,list):continue
        species=name.rsplit('.',1)[0];colors=frame.get(species+'.color',[])
        for i,p in enumerate(positions):
            if not isinstance(p,list) or len(p)<2:continue
            # Same projection, clipping and rasterizer for both engines.
            x,y=float(p[0])*639/width,float(p[1])*639/height
            c=colors[i] if isinstance(colors,list) and i<len(colors) else [30,80,160]
            if not isinstance(c,list) or len(c)<3:c=[30,80,160]
            c=tuple(max(0,min(255,round(v))) for v in c[:3])
            d.ellipse((x-2,y-2,x+2,y+2),fill=c)
    return image

def prepare_native_inputs(directory,cfg):
    """Import the unchanged source through the same zero-pacing experiment."""
    directory=Path(directory).resolve()
    from validate_spatial_compilation import expression
    manifest=cfg['manifest'];source=Path(cfg['source']);tree=directory/'sources'
    official=(ROOT/'benchmarks/official-v0.4-frozen-2026-10-13/official').resolve()
    shutil.copytree(official,tree)
    native_source=tree/source.resolve().relative_to(official)
    keys=list(manifest['fields']);species=list(manifest['populations'])
    observations={'field'+str(i):expression(k,manifest['fields'][k]) for i,k in enumerate(keys)}
    observations.update({'identity'+str(i):f'simulation.{n} collect each.index' for i,n in enumerate(species)})
    metadata=manifest['presentation']['experiments'][cfg['experiment']];declarations=[]
    if metadata.get('unsupported_effects'):raise RuntimeError('Effectful experiment cannot be replaced for benchmarking')
    for parameter in metadata.get('parameters',[]):
        initial=parameter.get('initial')
        if initial is not None and type(initial) not in (bool,int,float):raise RuntimeError('Unsupported experiment default')
        suffix=' init: '+(str(initial).lower() if isinstance(initial,bool) else repr(initial)) if initial is not None else ''
        declarations.append('parameter '+json.dumps(parameter['label'])+' var: '+parameter['variable']+suffix+';')
    execution_source=native_source.parent/'gamax_execution.gaml'
    execution_source.write_text('model gamax_execution\nimport '+json.dumps(str(native_source))+'\nexperiment gamax_benchmark type: gui {\nfloat minimum_cycle_duration <- 0.0;\n'+'\n'.join(declarations)+'\noutput {}\n}\n')
    assets={str(tree/Path(item['path']).resolve().relative_to(official)):item['key'] for item in manifest.get('assets',[])}
    return native_source,execution_source,observations,keys,species,assets

def worker(a):
    os.sched_setaffinity(0,{a.cpu});directory=a.output;directory.mkdir(parents=True)
    cfg=json.loads(a.config.read_text());manifest=cfg['manifest'];source=Path(cfg['source'])
    checkpoints=list(range(0,a.steps+1,a.steps//10)) if a.mode=='render' else [0,a.steps]
    setup=time.perf_counter();render_seconds=0.0
    if a.engine=='gama':
        from benchmarks.official_native_checkpoints import observe
        from validate_spatial_compilation import expression
        from gamax.validation import normalize_asset_observations
        tree=directory/'sources';shutil.copytree(ROOT/'benchmarks/official-v0.4-frozen-2026-10-13/official',tree)
        native_source=tree/source.resolve().relative_to((ROOT/'benchmarks/official-v0.4-frozen-2026-10-13/official').resolve())
        keys=list(manifest['fields']);species=list(manifest['populations'])
        observations={'field'+str(i):expression(k,manifest['fields'][k]) for i,k in enumerate(keys)}
        observations.update({'identity'+str(i):'string(('+n+' sort_by each.index) collect each.index)' for i,n in enumerate(species)})
        # Select an external execution experiment; original imports and
        # simulation rules remain intact. Copy all parameter declarations and
        # their inherited literal defaults, omit GUI outputs and real-time pacing.
        metadata=manifest['presentation']['experiments'][cfg['experiment']]
        if metadata.get('unsupported_effects'):raise RuntimeError('Effectful experiment cannot be replaced for benchmarking')
        declarations=[]
        for parameter in metadata.get('parameters',[]):
            variable=parameter['variable']
            initial=parameter.get('initial')
            if initial is not None and type(initial) not in (bool,int,float):raise RuntimeError('Unsupported experiment default')
            suffix=' init: '+(str(initial).lower() if isinstance(initial,bool) else repr(initial)) if initial is not None else ''
            declarations.append('parameter '+json.dumps(parameter['label'])+' var: '+variable+suffix+';')
        execution_source=native_source.parent/'gamax_execution.gaml'
        execution_source.write_text('model gamax_execution\nimport '+json.dumps(str(native_source))+'\nexperiment gamax_benchmark type: gui {\nfloat minimum_cycle_duration <- 0.0;\n'+'\n'.join(declarations)+'\noutput {}\n}\n')
        result=observe(execution_source,'gamax_benchmark',observations,seed=a.seed,steps=a.steps,output=directory/'native',home=ROOT/'.tools/gama',preserve_rng=manifest.get('rng_observation_required',False),checkpoints=checkpoints)
        if result['status']!='passed':raise RuntimeError(result)
        trace=[{k:s['field'+str(i)] for i,k in enumerate(keys)} for s in result['trace']]
        trace=normalize_asset_observations(trace,manifest,{str(tree/Path(item['path']).resolve().relative_to((ROOT/'benchmarks/official-v0.4-frozen-2026-10-13/official').resolve())):item['key'] for item in manifest.get('assets',[])})
        saved_files={}
        for event in manifest.get('outputs',[]):
            if event.get('kind')=='save':
                target=native_source.parent/event['path']
                saved_files[event['path']]=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        ids=[{n:s['identity'+str(i)] for i,n in enumerate(species)} for s in result['trace']]
        metrics={k:result[k] for k in ('compute_seconds','observation_seconds','setup_seconds')};metrics['warmup_seconds']=None
    else:
        import jax
        from gamax.runtime import load_module,canonical,check_status
        module=load_module(cfg['module']);state=module.reset(seed=a.seed)
        jax.block_until_ready(state);metrics={'setup_seconds':time.perf_counter()-setup,'compute_seconds':0.,'observation_seconds':0.}
        block_steps=a.steps//10 if a.mode=='render' else a.steps
        output_keys=[k for k in state if k.startswith(('$output.','$save.')) or k=='$error']
        has_saves=any(e.get('kind')=='save' for e in manifest.get('outputs',[]))
        def body(s,_):
            nxt=module.step(s)
            return nxt,{k:nxt[k] for k in output_keys}
        transition=jax.jit(lambda s:jax.lax.scan(body,s,None,length=block_steps),compiler_options=getattr(module,'JIT_OPTIONS',{}))
        tick=time.perf_counter();jax.block_until_ready(transition(state));metrics['warmup_seconds']=time.perf_counter()-tick
        # Functional transition leaves the initial arrays intact: reuse them
        # after warmup rather than initializing the identical simulation twice.
        jax.block_until_ready(state)
        trace=[];ids=[];previous=0;events=[];saved_files={};save_generations={}
        def collect_saves(snapshot):
            events.extend(collect_new_saves(module,snapshot,save_generations))
        if has_saves:collect_saves(state)
        for cycle in checkpoints:
            tick=time.perf_counter()
            if cycle>previous:
                state,emissions=transition(state);jax.block_until_ready((state,emissions));check_status(state)
                # Scan records status and every save emission without exporting full states.
                import numpy as np
                host={k:np.asarray(v) for k,v in jax.device_get(emissions).items()}
                if '$error' in host and np.any(host['$error']):raise RuntimeError('Runtime error during compiled block')
                if has_saves:
                    for index in range(block_steps):
                        collect_saves({k:v[index] for k,v in host.items()})
            jax.block_until_ready(state);metrics['compute_seconds']+=time.perf_counter()-tick;previous=cycle
            tick=time.perf_counter();f=canonical(getattr(module,'observe_jit',lambda value:value)(state),getattr(module,'ASSET_FIELDS',None),getattr(module,'ASSET_KEYS',None))
            for k in getattr(module,'MATRIX_FIELDS',[]):f[k]=[v for row in f[k] for v in row]
            for k,names in getattr(module,'FSM_FIELDS',{}).items():f[k]=[names[i] for i in f[k]]
            trace.append(f);ids.append(module.identities(state));metrics['observation_seconds']+=time.perf_counter()-tick
        write(directory/'save_events.json',events)
        for e in events:
            target=directory/'simulation_outputs'/e['path'];target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('w' if e['rewrite'] else 'a') as out:out.write(e['text'])
        for e in events:saved_files[e['path']]=hashlib.sha256((directory/'simulation_outputs'/e['path']).read_bytes()).hexdigest()
    tick=time.perf_counter();hashes=[]
    if a.mode=='render':
        for cycle,frame in zip(checkpoints,trace):
            image=draw(frame);image.save(directory/f'frame-{cycle}.png');hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
    render_seconds=time.perf_counter()-tick
    tick=time.perf_counter();write(directory/'trace.json',trace);write(directory/'identities.json',ids)
    metrics.update(engine=a.engine,mode=a.mode,seed=a.seed,render_seconds=render_seconds,export_seconds=time.perf_counter()-tick,frame_hashes=hashes,saved_files=saved_files)
    own,children=resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)
    metrics['cpu_seconds']=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
    write(directory/'measurement.json',metrics)

def main(a):
    import psutil
    from gamax.compiler import compile_model,implementation_identity
    from gamax.validation import compare_traces,compare_identities
    a.output.mkdir(parents=True,exist_ok=False)
    corpus=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13';cases=[c for c in json.loads((corpus/'manifest.json').read_text())['cases'] if c['role']=='development']
    fresh = a.validation_root.resolve() if a.validation_root else None
    acceptance_path = fresh/'summary.json' if fresh else ROOT/'results/official-six-final-verified-acceptance-2026-10-13/summary.json'
    acceptance=json.loads(acceptance_path.read_text())
    if fresh:
        from reproduction_validation import verified_modules
        validated = verified_modules(fresh, a.cases or [c['id'] for c in cases], a.seeds, a.steps, a.optimization, corpus)
        accepted = {}
    else:
        if not acceptance['all_six_accepted']:raise RuntimeError('Missing reference functional acceptance')
        accepted={r['id']:r for r in acceptance['reports']}
    if a.cases:cases=[c for c in cases if c['id'] in a.cases]
    if not cases or a.cases and set(a.cases)!={c['id'] for c in cases}:raise ValueError('Unknown development case')
    certificate=json.loads(a.fidelity_certificate.read_text()) if a.fidelity_certificate else None
    if any(s not in range(1,11) for s in a.seeds):raise ValueError('Only seeds 1–10 are covered by the certification protocol')
    if not fresh and a.steps!=100 and (a.optimization=='none' or not certificate or certificate.get('steps',100)!=a.steps):raise ValueError('A transformed module certificate covering this exact horizon is required')
    files=[Path(__file__).resolve(),ROOT/'benchmarks/official_native_checkpoints.py',ROOT/'scripts/validate_spatial_compilation.py',*sorted((ROOT/'gamax').glob('*.py'))]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for f in files:
        dest=a.output/'implementation'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
    cfgs={};cpu=min(os.sched_getaffinity(0))
    for case in cases:
        caps={'Host':1024} if case['id']=='sir_spatial' else {'prey':1000,'predator':300} if case['id']=='predator_prey' else {}
        configured_caps=validated[case['id']]['capacities'] if fresh else (certificate.get('configurations',{}).get(case['id'],{}).get('capacities',caps) if certificate and a.optimization!='none' else caps)
        tick=time.perf_counter();module,manifest=compile_model(corpus/case['source'],a.output/case['id']/'generated',experiment=case['experiment'],capacities=configured_caps,optimization=a.optimization)
        translation_seconds=time.perf_counter()-tick
        verification_start=time.perf_counter()
        baseline_sha=manifest['generated_sha256']
        if fresh:
            if manifest['generated_sha256'] != validated[case['id']]['generated_sha256']:
                raise RuntimeError('Generated module differs from fresh coupled validation')
        elif a.optimization!='none':
            _,baseline=compile_model(corpus/case['source'],a.output/case['id']/'reference',experiment=case['experiment'],capacities=caps)
            baseline_sha=baseline['generated_sha256']
            if manifest['generated_sha256']!=baseline_sha:
                if not certificate or certificate['implementation_sha256']!=implementation_identity()[1] or certificate['modules'].get(case['id'])!=manifest['generated_sha256']:raise RuntimeError('Missing optimized fidelity certificate')
                proof_path=ROOT/certificate['proof'];proof=json.loads(proof_path.read_text())
                if hashlib.sha256(proof_path.read_bytes()).hexdigest()!=certificate['proof_sha256'] or not proof.get('complete') or not proof.get('passed'):raise RuntimeError('Invalid optimized fidelity proof')
                rows=[r for r in proof['reports'] if r['case']==case['id']]
                if sorted(r['seed'] for r in rows)!=list(range(1,11)) or any(not r['passed'] or r['modes'][a.optimization]['comparison']['snapshots']!=a.steps+1 for r in rows):raise RuntimeError('Incomplete optimized seed coverage')
        if not fresh and baseline_sha!=accepted[case['id']]['generated_sha256']:raise RuntimeError('Reference code differs from accepted code')
        if not fresh and a.optimization=='none' and hashlib.sha256(module.read_bytes()).hexdigest()!=accepted[case['id']]['generated_sha256']:raise RuntimeError('Generated code differs from accepted code')
        cfg=dict(source=str(corpus/case['source']),experiment=case['experiment'],module=str(module),manifest=manifest,translation_seconds=translation_seconds,verification_seconds=time.perf_counter()-verification_start)
        path=a.output/case['id']/'configuration.json';write(path,cfg);cfgs[case['id']]=(path,cfg)
    rng=random.Random(20261013);jobs=[]
    for rep in range(a.repeats):
        conditions=[(c['id'],seed,rep,mode) for c in cases for seed in a.seeds for mode in ('compute','render')];rng.shuffle(conditions)
        for condition in conditions:
            engines=['gama','jax'];rng.shuffle(engines);jobs.extend([(*condition,e) for e in engines])
    write(a.output/'protocol.json',dict(classification='exploratory official native-size performance; not full confirmatory campaign',optimization=a.optimization,code_sha256=hashes,implementation_sha256=implementation_identity()[1],acceptance_sha256=hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),steps=a.steps,seeds=a.seeds,repeats=a.repeats,cpu=cpu,jobs=jobs,rendering='common 640x640 Pillow points, eleven frames; not native GAMA GUI',limitations=['Native cold total includes seeded preflight simulation','Native compute includes WebSocket RPC; JAX compute uses compiled scan with native JIT options, checked status and text-save capture','Warmup explicit for JAX; native JVM JIT evolves during run','Rendering uses declared global world bounds and point/color attributes; not full GAML display interpretation','Other host applications may contend for resources','Native external experiment copies parameter defaults, omits native outputs and forces zero real-time pacing; original source is imported unchanged'],versions=dict(gama='2025.06.4',jax=__import__('importlib.metadata',fromlist=['version']).version('jax'),pillow=__import__('importlib.metadata',fromlist=['version']).version('Pillow'),python=sys.version),hardware=dict(cpu=psutil.cpu_count(),ram_bytes=psutil.virtual_memory().total,cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'))))
    rows=[];pair_results=[]
    for case,seed,rep,mode,engine in jobs:
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):raise RuntimeError('Benchmark implementation changed; stop campaign')
        directory=a.output/case/f'{seed}-{rep}-{mode}-{engine}'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(directory),'--config',str(cfgs[case][0]),'--engine',engine,'--mode',mode,'--seed',str(seed),'--steps',str(a.steps),'--cpu',str(cpu)]
        env=dict(os.environ,JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
        tick=time.perf_counter();rss=pss=0;status='measured'
        with (a.output/f'{case}-{seed}-{rep}-{mode}-{engine}.log').open('w') as log:
            p=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                try:
                    members=[psutil.Process(p.pid),*psutil.Process(p.pid).children(recursive=True)];mem=[x.memory_full_info() for x in members];rss=max(rss,sum(x.rss for x in mem));pss=max(pss,sum(getattr(x,'pss',0) for x in mem))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                if time.perf_counter()-tick>1800:
                    import signal
                    for child in members[1:]:
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    os.killpg(p.pid,signal.SIGKILL);p.wait();status='timeout';break
                time.sleep(.05)
        if p.returncode:status='execution_error' if status!='timeout' else status
        row=dict(case=case,seed=seed,repetition=rep,mode=mode,engine=engine,status=status,total_seconds=time.perf_counter()-tick,peak_rss_bytes=rss,peak_pss_bytes=pss,directory=str(directory))
        if status=='measured':row.update(json.loads((directory/'measurement.json').read_text()))
        rows.append(row);write(a.output/'measurements.json',rows);print(case,seed,rep,mode,engine,status,round(row['total_seconds'],3),flush=True)
        for c,s,r,m in [(case,seed,rep,mode)]:
            key=(c,s,r,m)
            pair=[x for x in rows if (x['case'],x['seed'],x['repetition'],x['mode'])==key]
            if len(pair)!=2:continue
            record=dict(case=c,seed=s,repetition=r,mode=m,eligible=False)
            if all(x['status']=='measured' for x in pair):
                g=next(x for x in pair if x['engine']=='gama');j=next(x for x in pair if x['engine']=='jax')
                comparison=compare_traces(json.loads((Path(g['directory'])/'trace.json').read_text()),json.loads((Path(j['directory'])/'trace.json').read_text()),cfgs[c][1]['manifest']['fields'])
                identity=compare_identities(json.loads((Path(g['directory'])/'identities.json').read_text()),json.loads((Path(j['directory'])/'identities.json').read_text()))
                same_pixels=g['frame_hashes']==j['frame_hashes'];same_saves=g.get('saved_files')==j.get('saved_files');record.update(same_saved_files=same_saves,comparison=comparison,identities=identity,same_pixels=same_pixels,eligible=comparison['passed'] and identity['passed'] and same_pixels and same_saves)
                if record['eligible']:record['gama_over_jax']={k:g[k]/j[k] for k in ('total_seconds','cpu_seconds','peak_pss_bytes','compute_seconds') if j[k]>0}
            pair_results.append(record)
        write(a.output/'summary.json',dict(complete=len(rows)==len(jobs),completed=len(rows),planned=len(jobs),pairs=pair_results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--steps',type=int,default=100);p.add_argument('--seeds',type=int,nargs='+',default=list(range(1,11)));p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--optimization',choices=['none','safe','fusion','fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined'],default='none');p.add_argument('--cases',nargs='+');p.add_argument('--fidelity-certificate',type=Path);p.add_argument('--validation-root',type=Path,help='Fresh strict validation from validate_official_compilation.py; no historical archives required');p.add_argument('--worker',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--engine');p.add_argument('--mode');p.add_argument('--seed',type=int);p.add_argument('--cpu',type=int)
    a=p.parse_args();a.output=a.output.resolve()
    if a.steps<10 or a.steps%10:p.error('Horizon must be positive and divisible by ten')
    worker(a) if a.worker else main(a)
