"""Measure repeated seeded simulations with startup amortized for both engines.

One fresh process per engine/model/rendering condition; one native JVM/compiled
experiment or JAX executable reused. Keep startup and complete batch costs.
"""
from pathlib import Path
import argparse,hashlib,json,os,random,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from benchmark_official import write,draw,prepare_native_inputs,collect_new_saves

def checked_configuration(path,steps):
    from gamax.compiler import implementation_identity
    from certify_optimizer import checked_rows
    cfg=json.loads(path.read_text());manifest=cfg['manifest'];case=cfg['case_id']
    if cfg.get('validation_root'):
        from reproduction_validation import verified_modules
        corpus=ROOT/'benchmarks/official-v0.4-frozen-2026-10-13'
        manifests=verified_modules(Path(cfg['validation_root']),[case],cfg['validation_seeds'],steps,manifest['optimization']['mode'],corpus)
        if manifests[case]['generated_sha256']!=manifest['generated_sha256'] or hashlib.sha256(Path(cfg['module']).read_bytes()).hexdigest()!=manifest['generated_sha256']:
            raise RuntimeError('Configuration differs from fresh validation')
        if manifests[case]['capacities']!=manifest['capacities'] or cfg['experiment']!=manifests[case]['selected_experiment'] or cfg['source']!=manifests[case]['source']:
            raise RuntimeError('Changed fresh configuration')
        return cfg
    certificate=json.loads(Path(cfg['fidelity_certificate']).read_text())
    if certificate['steps']!=steps or certificate['implementation_sha256']!=implementation_identity()[1]:raise RuntimeError('Certificate does not cover the current compiler/horizon')
    if certificate['modules'].get(case)!=manifest['generated_sha256']:raise RuntimeError('Module outside certificate')
    if hashlib.sha256(Path(cfg['module']).read_bytes()).hexdigest()!=manifest['generated_sha256']:raise RuntimeError('Changed certified module')
    if hashlib.sha256(Path(cfg['source']).read_bytes()).hexdigest()!=manifest['source_sha256']:raise RuntimeError('Changed official source')
    proof=Path(certificate['proof']);summary=json.loads(proof.read_text())
    if hashlib.sha256(proof.read_bytes()).hexdigest()!=certificate['proof_sha256'] or not summary.get('complete') or not summary.get('passed'):raise RuntimeError('Invalid fidelity proof')
    checked_rows(summary,case,manifest['optimization']['mode'],steps)
    configuration=certificate['configurations'][case]
    if configuration['capacities']!=manifest['capacities'] or configuration['source_sha256']!=manifest['source_sha256'] or configuration['experiment']!=cfg['experiment']:raise RuntimeError('Configuration outside certificate')
    return cfg

def checked_native_reuse(cfg,steps):
    proof=Path(cfg['native_reuse_validation']);data=json.loads(proof.read_text());manifest=cfg['manifest']
    if cfg.get('native_reuse_sha256') and hashlib.sha256(proof.read_bytes()).hexdigest()!=cfg['native_reuse_sha256']:raise RuntimeError('Changed native reuse proof')
    if not data.get('complete') or not data.get('passed') or data['steps']!=steps:raise RuntimeError('Native reuse is not validated')
    binding=data['configuration']
    if binding['source_sha256']!=manifest['source_sha256'] or binding['fields']!=manifest['fields'] or binding['experiment']!=cfg['experiment']:raise RuntimeError('Native reuse proof covers another configuration')
    if binding['adapter_sha256']!=hashlib.sha256((ROOT/'benchmarks/reused_native_checkpoints.py').read_bytes()).hexdigest() or binding['preparation_sha256']!=hashlib.sha256((ROOT/'scripts/benchmark_official.py').read_bytes()).hexdigest():raise RuntimeError('Native reuse adapter changed')
    rows=data['reports'];seeds=[r['seed'] for r in rows]
    if len(rows)<3 or len(set(seeds))<2 or len(set(seeds))==len(seeds):raise RuntimeError('Native reuse needs distinct seeds and a repeated seed')
    if any(not r['comparison']['passed'] or not r['identities']['passed'] or r['comparison']['snapshots']!=steps+1 or r['identities']['snapshots']!=steps+1 or r['comparison']['float_atol']!=1e-12 or r['comparison']['float_rtol']!=1e-10 for r in rows):raise RuntimeError('Native reuse fidelity rejected')


def cpu_seconds():
    import psutil
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    total=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
    for child in psutil.Process().children(recursive=True):
        try:
            value=child.cpu_times();total+=value.user+value.system
        except psutil.NoSuchProcess:pass
    return total


def worker(a):
    os.sched_setaffinity(0,{a.cpu});a.output.mkdir(parents=True,exist_ok=False)
    cfg=checked_configuration(a.config,a.steps);checked_native_reuse(cfg,a.steps);manifest=cfg['manifest'];sequence=json.loads(a.sequence.read_text())
    if hashlib.sha256(Path(cfg['module']).read_bytes()).hexdigest()!=manifest['generated_sha256']:raise RuntimeError('Changed certified module')
    checkpoints=list(range(0,a.steps+1,a.steps//10)) if a.mode=='render' else [0,a.steps]
    started=time.perf_counter();startup_cpu=cpu_seconds();rows=[]
    write(a.output/'current_run.json',{'phase':'startup','started_monotonic':time.monotonic()})
    if a.engine=='gama':
        from benchmarks.reused_native_checkpoints import ReusedNativeExperiment
        from gamax.validation import normalize_asset_observations
        native,execution,observations,keys,species,assets=prepare_native_inputs(a.output,cfg)
        session=ReusedNativeExperiment(execution,'gamax_benchmark',observations,home=ROOT/'.tools/gama',output=a.output/'session',preserve_rng=manifest.get('rng_observation_required',False))
        session.__enter__()
        try:session.run(seed=sequence[0]['seed'],steps=a.steps,checkpoints=checkpoints)
        except BaseException:session.__exit__(*sys.exc_info());raise
        def execute(seed,directory):
            for event in manifest.get('outputs',[]):
                if event.get('kind')=='save':(native.parent/event['path']).unlink(missing_ok=True)
            result=session.run(seed=seed,steps=a.steps,checkpoints=checkpoints)
            trace=normalize_asset_observations([{k:r['field'+str(i)] for i,k in enumerate(keys)} for r in result['trace']],manifest,assets)
            identities=[{n:r['identity'+str(i)] for i,n in enumerate(species)} for r in result['trace']]
            saved={}
            for event in manifest.get('outputs',[]):
                if event.get('kind')!='save':continue
                path=native.parent/event['path']
                saved[event['path']]=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
                if path.exists():
                    target=directory/'simulation_outputs'/event['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
            return trace,identities,saved,{k:result[k] for k in ('setup_seconds','compute_seconds','observation_seconds','activation')}
    else:
        import jax
        import numpy as np
        from gamax.runtime import load_module,canonical,check_status
        module=load_module(cfg['module']);initial=module.reset(seed=sequence[0]['seed']);jax.block_until_ready(initial);check_status(initial)
        output_keys=[k for k in initial if k.startswith(('$output.','$save.')) or k=='$error']
        block_steps=a.steps//10 if a.mode=='render' else a.steps
        def body(s,_):
            nxt=module.step(s);return nxt,{k:nxt[k] for k in output_keys}
        transition=jax.jit(lambda s:jax.lax.scan(body,s,None,length=block_steps),compiler_options=module.JIT_OPTIONS)
        tick=time.perf_counter();compiled=transition.lower(initial).compile();compilation=time.perf_counter()-tick
        warm=initial
        for _ in range(a.steps//block_steps):warm,_=compiled(warm)
        jax.block_until_ready(warm);check_status(warm)
        def snapshot(state):
            observed=getattr(module,'observe_jit',lambda s:s)(state)
            fields=canonical(observed,getattr(module,'ASSET_FIELDS',None),getattr(module,'ASSET_KEYS',None))
            for key in getattr(module,'MATRIX_FIELDS',[]):fields[key]=[v for row in fields[key] for v in row]
            for key,names in getattr(module,'FSM_FIELDS',{}).items():fields[key]=[names[i] for i in fields[key]]
            return fields
        snapshot(initial);snapshot(warm)
        def execute(seed,directory):
            tick=time.perf_counter();state=module.reset(seed=seed);jax.block_until_ready(state);check_status(state)
            metrics={'setup_seconds':time.perf_counter()-tick,'compute_seconds':0.,'observation_seconds':0.}
            trace=[];identities=[];events=[];generations={};previous=0
            if any(e.get('kind')=='save' for e in manifest.get('outputs',[])):events.extend(collect_new_saves(module,state,generations))
            for cycle in checkpoints:
                tick=time.perf_counter()
                if cycle>previous:
                    state,emissions=compiled(state);jax.block_until_ready((state,emissions));check_status(state)
                    host={k:np.asarray(v) for k,v in jax.device_get(emissions).items()}
                    if np.any(host['$error']):raise RuntimeError('Runtime error during amortized compiled block')
                    if any(e.get('kind')=='save' for e in manifest.get('outputs',[])):
                        for index in range(block_steps):events.extend(collect_new_saves(module,{k:v[index] for k,v in host.items()},generations))
                metrics['compute_seconds']+=time.perf_counter()-tick;previous=cycle
                tick=time.perf_counter();trace.append(snapshot(state));identities.append(module.identities(state));metrics['observation_seconds']+=time.perf_counter()-tick
            saved={}
            for event in events:
                target=directory/'simulation_outputs'/event['path'];target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('w' if event['rewrite'] else 'a') as out:out.write(event['text'])
                saved[event['path']]=hashlib.sha256(target.read_bytes()).hexdigest()
            # Missing conditional saves have the same explicit representation.
            for event in manifest.get('outputs',[]):
                if event.get('kind')=='save':saved.setdefault(event['path'],None)
            return trace,identities,saved,metrics
    startup=time.perf_counter()-started;startup_cpu=cpu_seconds()-startup_cpu
    try:
        for item in sequence:
            write(a.output/'current_run.json',{'phase':'simulation','started_monotonic':time.monotonic(),**item})
            directory=a.output/f"{item['seed']}-{item['repetition']}";directory.mkdir()
            tick=time.perf_counter();cpu=cpu_seconds()
            trace,identities,saved,metrics=execute(item['seed'],directory)
            render=time.perf_counter();hashes=[]
            if a.mode=='render':
                for cycle,frame in zip(checkpoints,trace):
                    image=draw(frame);image.save(directory/f'frame-{cycle}.png');hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
            metrics['render_seconds']=time.perf_counter()-render
            export=time.perf_counter();write(directory/'trace.json',trace);write(directory/'identities.json',identities)
            metrics.update(export_seconds=time.perf_counter()-export,total_seconds=time.perf_counter()-tick,cpu_seconds=cpu_seconds()-cpu,frame_hashes=hashes,saved_files=saved,directory=str(directory),**item)
            rows.append(metrics);write(a.output/'summary.json',{'complete':False,'startup_seconds':startup,'startup_cpu_seconds':startup_cpu,'reports':rows});print(a.engine,a.mode,item['seed'],item['repetition'],metrics['total_seconds'],flush=True)
    finally:
        if a.engine=='gama':session.__exit__(*sys.exc_info())
    write(a.output/'summary.json',{'complete':True,'startup_seconds':startup,'startup_cpu_seconds':startup_cpu,'cpu_seconds':cpu_seconds(),'compilation_seconds':compilation if a.engine=='jax' else None,'reports':rows})


def main(a):
    import psutil
    from gamax.validation import compare_traces,compare_identities
    a.output.mkdir(parents=True,exist_ok=False);cpu=min(os.sched_getaffinity(0));rng=random.Random(20261014)
    sequence=[{'seed':s,'repetition':r} for r in range(a.repeats) for s in a.seeds];rng.shuffle(sequence)
    sequence_path=a.output/'sequence.json';write(sequence_path,sequence)
    jobs=[]
    for config in a.configs:
        cfg=checked_configuration(config,a.steps);checked_native_reuse(cfg,a.steps);case=cfg['case_id']
        for mode in ('compute','render'):
            engines=['gama','jax'];rng.shuffle(engines)
            jobs.extend((case,config,mode,engine) for engine in engines)
    write(a.output/'protocol.json',{'scope':'Amortized repeated seeded simulations; fresh process per engine/model/render condition, owned JVM/compiled experiment or JAX executable reused. One full-horizon warmup for each engine; complete batch totals include startup/warmup. Native transitions include RPC; common renderer, not native GUI. Shared host.','steps':a.steps,'seeds':a.seeds,'repeats':a.repeats,'cpu':cpu,'jobs':[(c,str(p),m,e) for c,p,m,e in jobs]})
    rows=[];pairs=[]
    env=dict(os.environ,JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    for case,config,mode,engine in jobs:
        directory=a.output/case/mode/engine;directory.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(config),'--sequence',str(sequence_path),'--engine',engine,'--mode',mode,'--steps',str(a.steps),'--cpu',str(cpu),'-o',str(directory)]
        started=time.perf_counter();rss=pss=0;status='measured';members=[]
        with (directory.parent/(engine+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                try:
                    members=[psutil.Process(process.pid),*psutil.Process(process.pid).children(recursive=True)];memory=[p.memory_full_info() for p in members]
                    rss=max(rss,sum(m.rss for m in memory));pss=max(pss,sum(getattr(m,'pss',0) for m in memory))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                try:progress=json.loads((directory/'current_run.json').read_text());expired=time.monotonic()-progress['started_monotonic']>1800
                except (FileNotFoundError,json.JSONDecodeError):expired=time.perf_counter()-started>1800
                if expired:
                    import signal
                    for child in reversed(members[1:]):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    os.killpg(process.pid,signal.SIGKILL);process.wait();status='timeout';break
                time.sleep(.05)
        if process.returncode and status!='timeout':status='execution_error'
        row={'case':case,'mode':mode,'engine':engine,'status':status,'total_seconds':time.perf_counter()-started,'peak_rss_bytes':rss,'peak_pss_bytes':pss,'directory':str(directory),'configuration':str(config)}
        row['translation_seconds']=json.loads(config.read_text()).get('translation_seconds',0.) if engine=='jax' else 0.
        row['translation_cpu_seconds']=json.loads(config.read_text()).get('translation_cpu_seconds',0.) if engine=='jax' else 0.
        row['total_including_translation_seconds']=row['total_seconds']+row['translation_seconds']
        if status=='measured':row['batch']=json.loads((directory/'summary.json').read_text())
        rows.append(row);write(a.output/'measurements.json',rows)
        pair=[r for r in rows if r['case']==case and r['mode']==mode]
        if len(pair)==2:
            record={'case':case,'mode':mode,'eligible':False,'runs':[]}
            if all(r['status']=='measured' and r['batch']['complete'] for r in pair):
                g=next(r for r in pair if r['engine']=='gama');j=next(r for r in pair if r['engine']=='jax');cfg=json.loads(config.read_text());gr=g['batch']['reports'];jr=j['batch']['reports']
                if [(r['seed'],r['repetition']) for r in gr]!=[(r['seed'],r['repetition']) for r in jr]:raise RuntimeError('Amortized pairing order mismatch')
                for left,right in zip(gr,jr):
                    gd,jd=Path(left['directory']),Path(right['directory'])
                    comparison=compare_traces(json.loads((gd/'trace.json').read_text()),json.loads((jd/'trace.json').read_text()),cfg['manifest']['fields'])
                    identities=compare_identities(json.loads((gd/'identities.json').read_text()),json.loads((jd/'identities.json').read_text()))
                    same_pixels=left['frame_hashes']==right['frame_hashes'];same_saves=left['saved_files']==right['saved_files']
                    record['runs'].append({'seed':left['seed'],'repetition':left['repetition'],'comparison':comparison,'identities':identities,'same_pixels':same_pixels,'same_saved_files':same_saves,'eligible':comparison['passed'] and identities['passed'] and same_pixels and same_saves,'gama_over_jax':{k:left[k]/right[k] for k in ('total_seconds','cpu_seconds','compute_seconds') if right[k]>0}})
                record['eligible']=len(record['runs'])==len(sequence) and all(r['eligible'] for r in record['runs'])
                if record['eligible']:record['batch_gama_over_jax']={'total_seconds':g['total_seconds']/j['total_seconds'],'total_including_translation_seconds':g['total_including_translation_seconds']/j['total_including_translation_seconds'],'peak_pss_bytes':g['peak_pss_bytes']/j['peak_pss_bytes'],'cpu_seconds':g['batch']['cpu_seconds']/j['batch']['cpu_seconds'],'cpu_including_translation_seconds':(g['batch']['cpu_seconds']+g['translation_cpu_seconds'])/(j['batch']['cpu_seconds']+j['translation_cpu_seconds'])}
            pairs.append(record)
        write(a.output/'summary.json',{'complete':len(rows)==len(jobs),'completed':len(rows),'planned':len(jobs),'pairs':pairs})
        print(case,mode,engine,status,row['total_seconds'],flush=True)
    return 0 if pairs and all(p['eligible'] for p in pairs) else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--configs',type=Path,nargs='+');p.add_argument('--config',type=Path);p.add_argument('--sequence',type=Path);p.add_argument('--steps',type=int,default=100);p.add_argument('--seeds',type=int,nargs='+',default=list(range(1,11)));p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');p.add_argument('--engine');p.add_argument('--mode');p.add_argument('--cpu',type=int)
    a=p.parse_args();a.output=a.output.resolve()
    if a.steps<10 or a.steps%10:p.error('Horizon must be divisible by ten')
    if a.repeats<1 or not a.seeds or len(set(a.seeds))!=len(a.seeds) or any(s not in range(1,11) for s in a.seeds):p.error('Unique certified seeds 1–10 and positive repetitions required')
    raise SystemExit(worker(a) if a.worker else main(a))
