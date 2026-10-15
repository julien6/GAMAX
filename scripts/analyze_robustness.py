#!/usr/bin/env python3
"""Predeclared paired bootstrap of process-repetition ratios, not a seed analysis."""
from pathlib import Path
import argparse
import json
import numpy as np

METRICS=['whole_campaign_wall_seconds','whole_campaign_cpu_seconds','whole_campaign_peak_pss_bytes','warmed_compute_seconds']


def analyze(summary, draws=10000, seed=20261012):
 if not isinstance(draws,int) or draws<1:raise ValueError('Positive integer bootstrap draws required')
 rng=np.random.default_rng(seed);results={}
 for mode in ['compute','render']:
  repetitions=sorted({m['repetition'] for m in summary['measurements'] if m['mode']==mode})
  if len(repetitions)<2:raise ValueError('At least two process repetitions per mode required')
  paired={}
  for r in repetitions:
   matches={engine:[m for m in summary['measurements'] if m['mode']==mode and m['repetition']==r and m['engine']==engine] for engine in ['gama','jax']}
   if any(len(v)!=1 for v in matches.values()):raise ValueError('Missing or duplicated paired run')
   paired[r]={engine:values[0] for engine,values in matches.items()}
  mode_results={}
  for metric in METRICS:
   a=np.asarray([paired[r]['gama'][metric] for r in repetitions],dtype=float)
   b=np.asarray([paired[r]['jax'][metric] for r in repetitions],dtype=float)
   if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a<=0) or np.any(b<=0):raise ValueError('Ratios require finite positive measurements')
   ratios=a/b;logs=np.log(ratios)
   bootstrap=np.exp(logs[rng.integers(0,len(logs),size=(draws,len(logs)))].mean(axis=1))
   low,high=np.quantile(bootstrap,[.025,.975])
   mode_results[metric]=dict(n=len(logs),geometric_mean_ratio=float(np.exp(logs.mean())),ci95=[float(low),float(high)],
                           individual_ratios=ratios.tolist(),min_ratio=float(ratios.min()),max_ratio=float(ratios.max()),
                           lower_bound_above_one=bool(low>1),all_pairs_favor_jax=bool(np.all(ratios>1)))
  joint=[all(paired[r]['gama'][metric]>paired[r]['jax'][metric] for metric in METRICS[:3]) for r in repetitions]
  results[mode]=dict(metrics=mode_results,joint_improvement_pairs=sum(joint),total_pairs=len(joint),
                     all_three_marginal_lower_bounds_above_one=all(mode_results[m]['lower_bound_above_one'] for m in METRICS[:3]))
 return dict(method='Paired log ratios / geometric mean / percentile bootstrap',draws=draws,bootstrap_seed=seed,
             replication_unit='independent process repetition on fixed deterministic inputs; not random scenario seed',
             passed_fidelity=summary['passed'],results=results,
             limitations=['Marginal CI95, not simultaneous family-wise intervals','Ratios do not establish speedups on other models, hardware or workloads',
                          'CPU time is not electrical energy','PSS sampling and native control costs retain benchmark limitations'])


def main():
 parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
 summary=json.loads((args.directory/'summary.json').read_text());result=analyze(summary)
 (args.directory/'robustness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
