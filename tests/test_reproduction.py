"""Publication-path tests using synthetic metadata, never native simulations."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import reproduction_validation as validation


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(validation,'implementation_identity',lambda: ({},'compiler'))
    corpus=tmp_path/'corpus';write(corpus/'manifest.json',{'assets':{}})
    root=tmp_path/'evidence';source=tmp_path/'model.gaml';source.write_text('model example')
    module=root/'traffic/generated/model_jax.py';module.parent.mkdir(parents=True);module.write_text('pass')
    write(module.with_name('manifest.json'),{'capacities':{},'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(module.read_bytes()).hexdigest()})
    write(root/'protocol.json',{'implementation_sha256':'compiler','steps':100,'optimization':'safe','parameters':{},'atol':1e-12,'rtol':1e-10,'files':{},'corpus_sha256':hashlib.sha256((corpus/'manifest.json').read_bytes()).hexdigest()})
    write(root/'summary.json',{'complete':True,'passed':True,'reports':[{'id':'traffic','seed':1,'status':'coupled_passed','comparison':{'passed':True,'snapshots':101,'float_atol':1e-12,'float_rtol':1e-10},'identities':{'passed':True,'snapshots':101}}]})
    return root,corpus,module


def test_matching_fresh_evidence_is_accepted(evidence):
    root,corpus,_=evidence
    assert 'traffic' in validation.verified_modules(root,['traffic'],[1],100,'safe',corpus)


@pytest.mark.parametrize('change',['seed','horizon','optimization','module','source','tolerance','partial','corpus'])
def test_changed_or_incomplete_evidence_is_rejected(evidence,change):
    root,corpus,module=evidence
    seeds,steps,mode=[1],100,'safe'
    if change=='seed':seeds=[2]
    elif change=='horizon':steps=1000
    elif change=='optimization':mode='none'
    elif change=='module':module.write_text('changed')
    elif change=='source':Path(json.loads(module.with_name('manifest.json').read_text())['source']).write_text('changed')
    elif change=='corpus':write(corpus/'manifest.json',{'assets':{},'changed':True})
    else:
        data=json.loads((root/'summary.json').read_text())
        if change=='partial':data['complete']=False
        else:data['reports'][0]['comparison']['float_atol']=1
        write(root/'summary.json',data)
    with pytest.raises(ValueError):
        validation.verified_modules(root,['traffic'],seeds,steps,mode,corpus)


def test_plan_does_not_create_output(tmp_path):
    out=tmp_path/'absent'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--cases','traffic','--seeds','1','--regime','cold','-o',str(out)],check=True,capture_output=True,text=True)
    data=json.loads(result.stdout)
    assert not data['execute'] and not out.exists()
    assert any('--validation-root' in command for command in data['commands'])


def test_amortized_plan_includes_reuse_gate(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--cases','traffic','--seeds','1','2','-o',str(tmp_path/'absent')],check=True,capture_output=True,text=True)
    commands=json.loads(result.stdout)['commands']
    reuse=next(i for i,c in enumerate(commands) if any('validate_reused_native.py' in x for x in c))
    measure=next(i for i,c in enumerate(commands) if any('benchmark_amortized.py' in x for x in c))
    assert reuse<measure
