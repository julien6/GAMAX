"""Summarize only fidelity- and pixel-validated official benchmark pairs."""
from pathlib import Path
import argparse,json,math,random,statistics

def geometric(values):return math.exp(statistics.mean(math.log(v) for v in values))

def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text());rows=json.loads((directory/'measurements.json').read_text());report=[]
    for case in sorted({r['case'] for r in rows}):
        for mode in ('compute','render'):
            pairs=[p for p in summary['pairs'] if p['case']==case and p['mode']==mode];accepted=[p for p in pairs if p['eligible']]
            r=dict(case=case,mode=mode,completed_pairs=len(pairs),eligible_pairs=len(accepted),failed_pairs=len(pairs)-len(accepted),seeds=sorted({p['seed'] for p in accepted}),metrics={})
            for metric in ('total_seconds','cpu_seconds','peak_pss_bytes','compute_seconds'):
                by_seed={seed:[p['gama_over_jax'][metric] for p in accepted if p['seed']==seed] for seed in r['seeds']}
                values=[geometric(v) for v in by_seed.values()]
                if not values:continue
                rng=random.Random(20261013);boot=sorted(geometric(rng.choices(values,k=len(values))) for _ in range(10000)) if len(values)>1 else []
                r['metrics'][metric]=dict(gama_over_jax=geometric(values),seed_min=min(values),seed_max=max(values),ci95=[boot[249],boot[9749]] if boot else None)
            r['joint_gain_observed']=bool(r['metrics']) and all(r['metrics'][k]['gama_over_jax']>1 for k in ('total_seconds','cpu_seconds','peak_pss_bytes'))
            report.append(r)
    result=dict(complete=summary['complete'],completed=summary['completed'],planned=summary['planned'],statistics='Geometric mean paired ratios; repetitions aggregated within seed; bootstrap over seeds, 10000 resamples. Ratios >1 favor GAMAX. Partial results exploratory; native cold includes seeded preflight, translation additional.',reports=report)
    (directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('| ABM | mode | paires validées | GAMA/JAX temps total | GAMA/JAX calcul | GAMA/JAX CPU | GAMA/JAX PSS |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for r in report:
        m=r['metrics'];fmt=lambda k:f"{m[k]['gama_over_jax']:.3f}" if k in m else '—'
        print(f"| {r['case']} | {r['mode']} | {r['eligible_pairs']}/{r['completed_pairs']} | {fmt('total_seconds')} | {fmt('compute_seconds')} | {fmt('cpu_seconds')} | {fmt('peak_pss_bytes')} |")
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);analyze(p.parse_args().directory)
