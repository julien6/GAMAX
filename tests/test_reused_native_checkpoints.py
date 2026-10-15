"""Never discard a real transition while handling native activation phases."""
from types import SimpleNamespace
import pytest
from benchmarks.reused_native_checkpoints import ReusedNativeExperiment


@pytest.mark.parametrize('activation_only',(False,True))
def test_activation_and_repeated_seeds_retain_every_model_cycle(activation_only):
    class Client:
        def __init__(self):self.socket=SimpleNamespace(settimeout=lambda seconds:None);self.cycle=0;self.first=True;self.steps=0
        def command(self,kind,**kwargs):
            if kind=='expression':self.cycle=0;self.first=True
            if kind=='step':
                count=kwargs['nb_step'];self.steps+=count
                if self.first and activation_only:count-=1
                self.first=False;self.cycle+=count
        def expression(self,exp,expr):return {'seed':2.,'rng':'mersenne','simulation_count':1}
    session=object.__new__(ReusedNativeExperiment);session.client=Client();session.exp='arbitrary'
    session.phase=lambda:{'cycle':session.client.cycle,'rng_usage':session.client.cycle*2}
    session.snapshot=lambda:{'value':session.client.cycle}
    for _ in range(2):
        result=session.run(seed=2,steps=10,checkpoints=[0,1,5,10])
        assert result['trace']==[{'value':v} for v in (0,1,5,10)]
        assert result['activation']['advanced']==(not activation_only)
    assert session.client.steps==20+2*activation_only
