"""Re-emit a certified configuration and account for translation explicitly."""
from pathlib import Path
import argparse,json,sys,time,hashlib,resource
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from gamax.compiler import compile_model
from benchmark_official import write
from benchmark_amortized import checked_configuration


def prepare(certificate,case,output):
    output.mkdir(parents=True,exist_ok=False)
    cert=json.loads(certificate.read_text())
    original=certificate.parent/'reemitted'/case/'manifest.json';manifest=json.loads(original.read_text())
    tick=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);cpu_before=usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime
    module,current=compile_model(manifest['source'],output/'generated',experiment=manifest['selected_experiment'],capacities=manifest['capacities'],optimization=manifest['optimization']['mode'])
    elapsed=time.perf_counter()-tick
    usage=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN);translation_cpu=usage.ru_utime+usage.ru_stime+children.ru_utime+children.ru_stime-cpu_before
    if hashlib.sha256(module.read_bytes()).hexdigest()!=cert['modules'][case]:raise RuntimeError('Re-emission differs from certificate')
    cfg={'case_id':case,'source':current['source'],'experiment':current['selected_experiment'],'module':str(module.resolve()),'manifest':current,'fidelity_certificate':str(certificate.resolve()),'translation_seconds':elapsed,'translation_cpu_seconds':translation_cpu}
    path=output/'configuration.json';write(path,cfg);checked_configuration(path,cert['steps']);return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--certificate',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('-o','--output',type=Path,required=True);a=p.parse_args();print(prepare(a.certificate.resolve(),a.case,a.output.resolve()))
