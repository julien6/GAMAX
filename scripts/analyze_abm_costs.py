#!/usr/bin/env python3
"""Reproduce per-family cost intervals without rerunning the engine measurements."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_robustness import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory/'summary.json').read_text())
    if not summary['complete'] or not summary['passed']:
        raise SystemExit('Require a complete suite with successful trace comparisons')
    results = []
    for entry in summary['results']:
        directory = args.directory/entry['case']
        child = json.loads((directory/'summary.json').read_text())
        analysis = analyze(child)
        (directory/'robustness.json').write_text(json.dumps(analysis,indent=2)+'\n')
        results.append(dict(case=entry['case'],analysis=analysis))
    aggregate = dict(scope='per-family marginal intervals from timing pairs on one scenario; not seed robustness or simultaneous intervals',analyses=results)
    (args.directory/'robustness.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    sources = [Path(__file__).resolve(), ROOT/'scripts/analyze_robustness.py']
    archive = args.directory/'analysis_source'; archive.mkdir(exist_ok=True)
    for source in sources: (archive/source.name).write_bytes(source.read_bytes())
    (args.directory/'analysis_provenance.json').write_text(json.dumps(dict(source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        summary_sha256=hashlib.sha256((args.directory/'summary.json').read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(families=len(results),passed=True)))


if __name__=='__main__':main()
