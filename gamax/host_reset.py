"""Standalone NumPy execution of the emitted initializer, without JIT startup.

Only selected initializers run here; simulation transitions remain JAX.
A private function namespace avoids changing generated-module globals and
preserves concurrent resets. Every array update retains functional semantics.
"""
HOST_RESET_SOURCE = '''
import numpy as _host_np
import types as _host_types

class _HostArray(_host_np.ndarray):
    @property
    def at(self):
        return _HostAt(self)

class _HostAt:
    def __init__(self,array,index=None):self.array,self.index=array,index
    def __getitem__(self,index):return _HostAt(self.array,index)
    def set(self,value):
        result=self.array.copy();result[self.index]=value;return result
    def add(self,value):
        result=self.array.copy();_host_np.add.at(result,self.index,value);return result
    def max(self,value):
        result=self.array.copy();_host_np.maximum.at(result,self.index,value);return result
    def min(self,value):
        result=self.array.copy();_host_np.minimum.at(result,self.index,value);return result


def _host_wrap(value):
    # Scalar initialization values never use indexed functional updates.
    # Keep their NumPy dtype without allocating an ndarray subclass view.
    if isinstance(value,_host_np.generic):return value
    if isinstance(value,_host_np.ndarray) and value.ndim==0:return _host_np.asarray(value)
    if isinstance(value,_HostArray):return value
    if isinstance(value,_host_np.ndarray):
        return _host_np.asarray(value).view(_HostArray)
    if isinstance(value,tuple):return tuple(_host_wrap(v) for v in value)
    if isinstance(value,list):return [_host_wrap(v) for v in value]
    if isinstance(value,dict):return {k:_host_wrap(v) for k,v in value.items()}
    return value

class _HostNumpy:
    def __getattr__(self,name):
        value=getattr(_host_np,name)
        if isinstance(value,type) or not callable(value):resolved=value
        else:resolved=lambda *args,**kwargs:_host_wrap(value(*args,**kwargs))
        setattr(self,name,resolved)
        return resolved
    @staticmethod
    def nonzero(value,size=None,fill_value=0):
        indices=_host_np.nonzero(value)
        if size is None:return _host_wrap(indices)
        return tuple(_host_wrap(_host_np.pad(v[:size],(0,max(0,size-len(v))),constant_values=fill_value)) for v in indices)

class _HostLax:
    @staticmethod
    def optimization_barrier(value):return value
    @staticmethod
    def cond(predicate,true,false,*operands,operand=None):
        return (true if bool(predicate) else false)(*operands) if operands else (true if bool(predicate) else false)(operand)
    @staticmethod
    def switch(index,branches,*operands,operand=None):
        branch=branches[max(0,min(int(index),len(branches)-1))]
        return branch(*operands) if operands else branch(operand)
    @staticmethod
    def fori_loop(start,stop,body,state,**kwargs):
        dtype=getattr(start,'dtype',getattr(stop,'dtype',_host_np.dtype('int64')))
        for index in range(int(start),int(stop)):state=body(_host_wrap(_host_np.asarray(index,dtype=dtype)),state)
        return state
    @staticmethod
    def while_loop(condition,body,state):
        while bool(condition(state)):state=body(state)
        return state
    @staticmethod
    def bitcast_convert_type(value,dtype):return _host_wrap(_host_np.asarray(value).view(dtype))
    @staticmethod
    def dynamic_update_slice(array,update,starts):
        starts=[max(0,min(int(p)+n if int(p)<0 else int(p),n-m)) for p,n,m in zip(starts,array.shape,update.shape)]
        result=array.copy();result[tuple(slice(p,p+n) for p,n in zip(starts,update.shape))]=update;return result


def _initialize_on_host(original,namespace,seed,parameters):
    context=dict(namespace)
    for key,value in list(context.items()):
        if isinstance(value,jax.Array):context[key]=_host_wrap(jax.device_get(value))
    context['jnp']=_HostNumpy();context['jax']=_host_types.SimpleNamespace(lax=_HostLax())
    for key,value in list(context.items()):
        if isinstance(value,_host_types.FunctionType) and value.__globals__ is namespace:
            context[key]=_host_types.FunctionType(value.__code__,context,value.__name__,value.__defaults__,value.__closure__)
            context[key].__kwdefaults__=value.__kwdefaults__
    for helper,plain in context.get('OUTLINED_RNG_HELPERS',{}).items():context[helper]=context[plain]
    for name,function in context.get('PURE_GRID_INITIALIZERS',{}).items():
        context[name]=lambda state,function=function:_host_wrap(jax.device_get(function(state)))
    if '_reset_native' in context:context['_reset_native_jit']=context['_reset_native']
    initializer=_host_types.FunctionType(original.__code__,context,original.__name__,original.__defaults__,original.__closure__)
    initializer.__kwdefaults__=original.__kwdefaults__
    with _host_np.errstate(over='ignore',under='ignore',divide='ignore',invalid='ignore'):
        state=initializer(seed=seed,parameters=parameters)
    return jax.tree.map(lambda value:jnp.asarray(_host_np.asarray(value)),state)
'''
