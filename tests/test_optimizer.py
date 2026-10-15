from gamax.ir import Expr,Field,Scope,ModelIR,Token
from gamax.optimizer import optimize
import pytest

T=Token('x',1,1)
def model(inner):
    outer=Expr('name',['cells'],'agents:cells',T)
    value=Expr('accumulate',[outer,inner],'agents:people',T)
    ir=ModelIR('arbitrary',[Scope('global',[Field('result','agents:people',value,T)],[],[])],[],{'cells':5,'people':10})
    return ir,value

def spatial():return Expr('overlapping',[Expr('name',['people'],'agents:people',T),Expr('each_ref',[],'agent:cells',T)],'agents:people',T)

def test_safe_marks_only_verified_plain_spatial_accumulation():
    ir,value=model(spatial());report=optimize(ir,'safe')
    assert value.metadata['skip_inactive_entries'] is True
    assert next(p for p in report['passes'] if p['name']=='skip-inactive-spatial-accumulation')['applied_sites']==1
    assert report['validation'].startswith('requires differential')

def test_reference_mode_leaves_ir_untouched():
    ir,value=model(spatial());assert optimize(ir,'none')['passes']==[]
    assert value.metadata=={}

def test_action_or_getter_collection_is_not_optimized():
    inner=spatial();inner.args[0]=Expr('function_value',[],'agents:people',T,{'body':[]})
    ir,value=model(inner);assert next(p for p in optimize(ir,'safe')['passes'] if p['name']=='skip-inactive-spatial-accumulation')['applied_sites']==0
    assert value.metadata=={}

def test_unknown_mode_fails_closed():
    ir,_=model(spatial())
    with pytest.raises(ValueError,match='none.*safe'):optimize(ir,'aggressive')


def test_streaming_requires_checker_proof():
    from gamax.ir import Statement
    ir,value=model(spatial())
    loop=Statement('collection_loop','p',value,[],[],T,{'commutative_integer_reduction':False})
    ir.scopes[0].init=[loop]
    assert next(p for p in optimize(ir,'safe')['passes'] if p['name']=='stream-spatial-integer-reductions')['applied_sites']==0
    assert 'stream_integer_accumulation' not in loop.metadata


def test_streaming_marks_proven_consumer_without_changing_capacity():
    from gamax.ir import Statement
    ir,value=model(spatial())
    loop=Statement('collection_loop','p',value,[],[],T,{'commutative_integer_reduction':True})
    ir.scopes[0].init=[loop]
    capacities=dict(ir.populations)
    report=optimize(ir,'safe')
    assert next(p for p in report['passes'] if p['name']=='stream-spatial-integer-reductions')['applied_sites']==1
    assert loop.metadata['stream_integer_accumulation']
    assert ir.populations==capacities


def test_streaming_rejects_float_increment_even_with_integer_marker():
    from gamax.ir import Statement
    ir,value=model(spatial())
    increment=Expr('+',[Expr('name',['n'],'int',T),Expr('literal',[-0.5],'float',T)],'float',T)
    body=[Statement('assign','n',increment,[],[],T)]
    loop=Statement('collection_loop','p',value,body,[],T,{'commutative_integer_reduction':True});ir.scopes[0].init=[loop]
    optimize(ir,'safe')
    assert 'stream_integer_accumulation' not in loop.metadata
