"""Run the six-model smoke gate then the accepted native-size 10-seed campaign."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    def status(**data):
        temporary=a.output/'status.tmp';temporary.write_text(json.dumps(dict(pid=os.getpid(),**data),indent=2)+'\n');temporary.replace(a.output/'status.json')
    status(stage='waiting_for_runner_debug',waiting_pid=a.wait_pid)
    if a.wait_pid:
        import psutil
        while psutil.pid_exists(a.wait_pid):
            try:
                if psutil.Process(a.wait_pid).status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(10)
    for stage,seeds,repeats in [('smoke',[2],1),('campaign',list(range(1,11)),10)]:
        directory=a.output/stage;command=[sys.executable,str(ROOT/'scripts/benchmark_official.py'),'-o',str(directory),'--steps','100','--seeds',*map(str,seeds),'--repeats',str(repeats)]
        status(stage=stage,state='running',directory=str(directory),command=command)
        with (a.output/(stage+'.log')).open('w') as log:code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if code:status(stage=stage,state='runner_failed',returncode=code);return 1
        summary=json.loads((directory/'summary.json').read_text())
        subprocess.call([sys.executable,str(ROOT/'scripts/analyze_official_performance.py'),str(directory)],stdout=(a.output/(stage+'-table.md')).open('w'))
        if stage=='smoke' and (not summary['complete'] or len(summary['pairs'])!=12 or not all(x['eligible'] for x in summary['pairs'])):
            status(stage=stage,state='fidelity_or_render_gate_failed',directory=str(directory));return 1
    status(stage='campaign',state='complete',directory=str(a.output/'campaign'));return 0

if __name__=='__main__':raise SystemExit(main())
