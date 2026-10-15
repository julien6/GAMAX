"""Finish frozen validation, issue byte-bound certificates, then measure costs.

Runs are serialized behind an existing performance campaign. Rejected cases
remain in the parent proof and status; they are never measured as faithful.
"""
from pathlib import Path
import argparse,json,subprocess,sys,time


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    status={'state':'waiting','frozen_workspace':str(a.workspace.resolve()),'jobs':[]}
    def save():
        temporary=a.output/'status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(a.output/'status.json')
    def complete(path):
        try:return json.loads(path.read_text()).get('complete',False)
        except (FileNotFoundError,json.JSONDecodeError):return False
    save()
    # Wait for the currently running cost campaign to release its CPU.
    while not complete(a.wait_for):time.sleep(10)
    for text in a.validation:
        directory=Path(text).resolve()
        status.update(state='waiting_for_validation',validation=str(directory));save()
        while not complete(directory/'summary.json'):time.sleep(10)
        summary=json.loads((directory/'summary.json').read_text())
        cases=sorted({r['case'] for r in summary['reports']})
        for case in cases:
            rows=[r for r in summary['reports'] if r['case']==case]
            modes=set.intersection(*(set(r['modes']) for r in rows))
            for mode in sorted(modes-{'none'}):
                job={'validation':str(directory),'case':case,'optimization':mode,'state':'certifying'}
                status['jobs'].append(job);status['state']='running';save()
                prefix=a.output/(directory.name+'-'+case+'-'+mode)
                certificate=prefix.with_name(prefix.name+'-certificate')
                command=[sys.executable,str(a.workspace/'scripts/certify_optimizer.py'),str(directory),'-o',str(certificate),'--optimization',mode,'--cases',case]
                with prefix.with_suffix('.log').open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        job.update(state='certificate_rejected',returncode=result.returncode);save();continue
                    job['state']='measuring';job['results']=str(prefix);save()
                    command=[sys.executable,str(a.workspace/'scripts/benchmark_official.py'),'-o',str(prefix),'--cases',case,'--optimization',mode,'--fidelity-certificate',str(certificate/'certificate.json'),'--seeds',*[str(i) for i in range(1,11)],'--repeats',str(a.repeats)]
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                    job.update(state='complete' if result.returncode==0 else 'runner_failed',returncode=result.returncode);save()
                    if (prefix/'summary.json').exists():subprocess.run([sys.executable,str(a.workspace/'scripts/analyze_official_performance.py'),str(prefix)],stdout=log,stderr=subprocess.STDOUT)
    status['state']='complete';save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--validation',nargs='+',required=True);p.add_argument('--wait-for',type=Path,required=True);p.add_argument('-o','--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);main(p.parse_args())
