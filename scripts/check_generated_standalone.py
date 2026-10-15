"""Execute an emitted module in isolated Python without GAMAX/GAMA imports."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('module',type=Path)
    p.add_argument('--seed',type=float,required=True)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--require-no-gamax',action='store_true')
    p.add_argument('-o','--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    program='''import json,runpy,sys,jax,importlib.util
if sys.argv[4]=='true':assert importlib.util.find_spec('gamax') is None
model=runpy.run_path(sys.argv[1])
state=model['reset'](seed=float(sys.argv[2]))
for _ in range(int(sys.argv[3])):state=model['step_jit'](state)
jax.block_until_ready(state)
assert '$error' not in state or int(state['$error'])==0
assert not any(name=='gamax' or name.startswith('gamax.') or name=='gama' for name in sys.modules)
print(json.dumps({'identities':model['identities'](state),'standalone':True}))
'''
    (a.output/'isolated_check.py').write_text(program)
    with tempfile.TemporaryDirectory(prefix='gamax-standalone-') as directory:
        # Resolving a venv's interpreter symlink would bypass its pyvenv.cfg.
        command=[str(a.python.absolute()),'-I','-c',program,str(a.module.resolve()),str(a.seed),str(a.steps),str(a.require_no_gamax).lower()]
        result=subprocess.run(command,cwd=directory,capture_output=True,text=True,timeout=1800)
    (a.output/'stdout.log').write_text(result.stdout);(a.output/'stderr.log').write_text(result.stderr)
    report=dict(passed=result.returncode==0,returncode=result.returncode,seed=a.seed,steps=a.steps,require_no_gamax=a.require_no_gamax,
                module_sha256=hashlib.sha256(a.module.read_bytes()).hexdigest(),command=command)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('standalone',report['passed'],flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
