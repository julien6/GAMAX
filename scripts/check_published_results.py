"""Recompute published scalar batch ratios without historical local archives."""
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'benchmarks/published_results/v0-costs.json').read_text())
    count=0
    for campaign in data['campaigns']:
        for report in campaign['reported_ratios']:
            rows=[r for r in campaign['measurements'] if r['case']==report['case'] and r['mode']==report['mode']]
            g=next(r for r in rows if r['engine']=='gama');j=next(r for r in rows if r['engine']=='jax')
            assert len(rows)==2 and all(r['complete'] and r['status']=='measured' for r in rows)
            recomputed={
                'total_seconds':g['total_seconds']/j['total_seconds'],
                'total_including_translation_seconds':g['total_including_translation_seconds']/j['total_including_translation_seconds'],
                'cpu_seconds':g['cpu_seconds']/j['cpu_seconds'],
                'cpu_including_translation_seconds':(g['cpu_seconds']+g['translation_cpu_seconds'])/(j['cpu_seconds']+j['translation_cpu_seconds']),
                'peak_pss_bytes':g['peak_pss_bytes']/j['peak_pss_bytes']}
            for metric,value in recomputed.items():
                assert math.isclose(value,report['ratios'][metric],rel_tol=1e-12,abs_tol=1e-12),(campaign['campaign'],metric)
            count+=1
    print(f'{count} published case/mode batch ratios recomputed. This does not revalidate raw trajectories.')


if __name__=='__main__':
    main()
