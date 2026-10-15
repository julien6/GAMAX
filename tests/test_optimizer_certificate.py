import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('certify',Path(__file__).resolve().parents[1]/'scripts/certify_optimizer.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def evidence():
    return {'reports':[{'case':'arbitrary','seed':seed,'passed':True,'modes':{'safe':{'comparison':{'passed':True,'snapshots':101,'float_atol':1e-12,'float_rtol':1e-10},'identities':{'passed':True,'snapshots':101},'kernel_comparison':None}}} for seed in range(1,11)]}


def test_certificate_requires_ten_distinct_seeds():
    proof=evidence();assert len(module.checked_rows(proof,'arbitrary','safe'))==10
    proof['reports'][-1]['seed']=1
    with pytest.raises(ValueError,match='seeds'):module.checked_rows(proof,'arbitrary','safe')


@pytest.mark.parametrize('change',('divergence','identity','horizon','tolerance','kernel'))
def test_certificate_rejects_incomplete_or_divergent_evidence(change):
    proof=evidence();value=proof['reports'][0]['modes']['safe']
    if change=='divergence':value['comparison']['passed']=False
    elif change=='identity':value['identities']['passed']=False
    elif change=='horizon':value['comparison']['snapshots']=100
    elif change=='tolerance':value['comparison']['float_atol']=1e-6
    else:value['kernel_comparison']={'passed':False}
    with pytest.raises(ValueError):module.checked_rows(proof,'arbitrary','safe')
