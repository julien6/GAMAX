"""Choose an observed winning certified profile; retain failures and regressions.

This reports measured choices, not a universal compiler correctness certificate
or a claim that no faster implementation exists.
"""
from pathlib import Path
import argparse,json,statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--functional',type=Path,nargs='+',required=True);p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args()
    metrics=json.loads((a.campaign/'measurements.json').read_text());pairs=json.loads((a.campaign/'summary.json').read_text());functional={}
    for path in a.functional:
        for r in json.loads(path.read_text())['reports']:functional.setdefault(r['case'],[]).append(r)
    rows=[]
    for case,proof in functional.items():
        valid=sorted(r['seed'] for r in proof)==list(range(1,11)) and all(r['passed'] for r in proof)
        row={'case':case,'selected':'none','candidate':'fusion','functional_passed':valid,'reason':'candidate fidelity failed or incomplete','modes':{}}
        if valid:
            row['reason']='no complete favorable cost evidence'
            passed=True
            for mode in ('compute','render'):
                entries=[r for r in pairs['pairs'] if r['case']==case and r['mode']==mode]
                if not entries or any(not r['eligible'] for r in entries):passed=False;continue
                g=[r for r in metrics if r['case']==case and r['mode']==mode and r['engine']=='gama' and r['status']=='measured'];j=[r for r in metrics if r['case']==case and r['mode']==mode and r['engine']=='jax' and r['status']=='measured']
                ratios={key:statistics.median(r[key] for r in g)/statistics.median(r[key] for r in j) for key in ('total_seconds','cpu_seconds','peak_pss_bytes','compute_seconds')}
                row['modes'][mode]=ratios
                if any(ratios[key]<=1 for key in ('total_seconds','cpu_seconds','peak_pss_bytes')):passed=False
            if passed:row.update(selected='fusion',reason='full coupled seed coverage and observed joint time/CPU/PSS gain in both modes')
            else:row['reason']='cost regression, incomplete pairs, or absent measured gain; keep reference'
        rows.append(row)
    result={'classification':'observed profile recommendation for measured configuration; source/module-bound certificates required; repetitions on one seed are exploratory','campaign_complete':pairs['complete'],'recommendations':rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print(r['case'],r['selected'],r['reason'])
if __name__=='__main__':main()
