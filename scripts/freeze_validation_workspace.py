"""Copy the compiler and validators so a long campaign cannot observe live edits.

The official corpus and installed native oracle are read-only inputs linked from
this workspace. Validators still hash those inputs and copy native assets per run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-o','--output',type=Path,required=True)
    a=p.parse_args();target=a.output.resolve();target.mkdir(parents=True,exist_ok=False)
    hashes={}
    for name in ('gamax','scripts'):
        destination=target/name;destination.mkdir()
        for source in sorted((ROOT/name).glob('*.py')):
            relative=source.relative_to(ROOT);shutil.copyfile(source,target/relative)
            hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ('benchmarks','examples','.tools'):(target/name).symlink_to(ROOT/name,target_is_directory=True)
    (target/'freeze.json').write_text(json.dumps(dict(files=hashes,inputs='linked immutable corpus, authored fixtures and installed native oracle; private asset copies made per native run'),indent=2)+'\n')
    print(target)

if __name__=='__main__':main()
