"""Report full batch ratios separately from seed variation; never pool models."""
from pathlib import Path
import argparse,json,math,random,statistics


def geometric(values):return math.exp(statistics.mean(math.log(v) for v in values))


def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text());measurements=json.loads((directory/'measurements.json').read_text());reports=[]
    if summary.get('performance_valid') is False:raise ValueError('Campaign rejected: '+summary.get('reason','invalid performance instrumentation'))
    for pair in summary['pairs']:
        rows=[r for r in measurements if r['case']==pair['case'] and r['mode']==pair['mode']]
        result={k:pair[k] for k in ('case','mode','eligible')};result['completed_pairs']=len(pair['runs']);result['eligible_pairs']=sum(r['eligible'] for r in pair['runs'])
        result['batch_gama_over_jax']=pair.get('batch_gama_over_jax');result['seed_statistics']={}
        if pair['eligible']:
            g=next(r for r in rows if r['engine']=='gama');j=next(r for r in rows if r['engine']=='jax');count=len(pair['runs'])
            # Uniformly attribute measured non-simulation overhead to each run.
            # The allocated durations sum back to the complete measured batch.
            overhead={r['engine']:r['total_including_translation_seconds']-sum(x['total_seconds'] for x in r['batch']['reports']) for r in (g,j)}
            allocated=[]
            for left,right in zip(g['batch']['reports'],j['batch']['reports']):
                allocated.append({'seed':left['seed'],'ratio':(left['total_seconds']+overhead['gama']/count)/(right['total_seconds']+overhead['jax']/count)})
            for metric in ('total_seconds','cpu_seconds','compute_seconds','total_including_allocated_startup_translation_seconds'):
                data=allocated if metric=='total_including_allocated_startup_translation_seconds' else [{'seed':r['seed'],'ratio':r['gama_over_jax'][metric]} for r in pair['runs']]
                seeds=sorted({r['seed'] for r in data});values=[geometric([r['ratio'] for r in data if r['seed']==seed]) for seed in seeds]
                rng=random.Random(20261014);bootstrap=sorted(geometric(rng.choices(values,k=len(values))) for _ in range(10000)) if len(values)>1 else []
                result['seed_statistics'][metric]={'gama_over_jax':geometric(values),'seed_min':min(values),'seed_max':max(values),'seeds':seeds,'ci95':None if not bootstrap else [bootstrap[249],bootstrap[9749]]}
            result['batch_joint_gain_observed']=all(pair['batch_gama_over_jax'][key]>1 for key in ('total_including_translation_seconds','cpu_including_translation_seconds','peak_pss_bytes'))
        reports.append(result)
    output={'complete':summary['complete'],'completed':summary['completed'],'planned':summary['planned'],
            'scope':'Complete batch ratios include startup/warmup and separately measured translation wall time. Seed bootstrap is conditional on the observed batch and uniformly allocated startup; it does not quantify between-process startup uncertainty. One batch per engine/model/mode; no memory confidence interval. CPU includes cumulative worker and child usage through shutdown; translation CPU is reported separately and included in its named ratio. Peak memory covers execution batches, excluding translation; joint gain refers to these measured scopes. The orchestration parent is excluded. Shared host; no model pooling.',
            'reports':reports}
    (directory/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    for row in reports:print(row['case'],row['mode'],row['eligible'],row['batch_gama_over_jax'])
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);analyze(parser.parse_args().directory)
