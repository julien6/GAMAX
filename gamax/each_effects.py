"""Reject caller uses of each after a nested getter iterator clobbers it.

Pinned GAMA container lambdas set each without restoring the caller binding.
The current lowering uses lexical bindings. Diagnose the incompatible pattern
rather than silently compiling a different meaning.
"""
from dataclasses import fields,is_dataclass
from .ir import Expr,CompileError

ITERATORS={'count','collect','where','first_with','sort_by','min_of','max_of','with_max_of','with_min_of'}

def validate_each_effects(ir):
    scopes={s.name:s for s in ir.scopes}
    def has_iterator(value):
        if isinstance(value,Expr) and value.op in ITERATORS:return True
        return any(has_iterator(v) for v in children(value))
    def children(value):
        if isinstance(value,Expr):return [*value.args,value.metadata]
        if isinstance(value,dict):return list(value.values())
        if isinstance(value,(list,tuple)):return value
        if is_dataclass(value):return [getattr(value,f.name) for f in fields(value)]
        return []
    def effect(expr,scope,each_owner):
        if not isinstance(expr,Expr):return False,False
        owner=None;name=None;reads=False
        if expr.op=='each_ref':return True,False
        if expr.op=='each':
            owner=expr.metadata.get('owner_scope',each_owner);name=expr.args[0];reads=True
        elif expr.op=='name':
            owner=scope if any(f.name==expr.args[0] for f in scopes[scope].fields) else 'global';name=expr.args[0]
        elif expr.op.startswith('attribute:'):
            owner=expr.args[0].type.split(':')[-1];name=expr.op.split(':')[1]
            reads=expr.args[0].op=='each_ref'
        field=next((f for f in scopes.get(owner,scopes['global']).fields if f.name==name),None)
        if field is not None and field.getter is not None:
            return reads,has_iterator(field.getter)
        if expr.op in ITERATORS:
            # The operator binds its own each for its predicate. Container
            # evaluation can still read the caller's each.
            if expr.args:effect(expr.args[0],scope,each_owner)
            if len(expr.args)>1:
                target=expr.args[0].type.split(':')[-1]
                effect(expr.args[1],scope,target)
            return effect(expr.args[0],scope,each_owner)[0] if expr.args else False,True
        dirty=False;first_reads=reads
        for arg in expr.args:
            child_reads,child_dirty=effect(arg,scope,each_owner)
            if dirty and child_reads:
                raise CompileError(expr.token.source or '<IR>',expr.token,'A nested getter iterator can rebind native each before its next caller access; this expression requires native dynamic-each lowering','GAMAX_EACH_SCOPE')
            if not dirty:first_reads=first_reads or child_reads
            dirty=dirty or child_dirty
        return first_reads,dirty
    def walk(value,scope):
        if isinstance(value,Expr):effect(value,scope,scope)
        for child in children(value):walk(child,scope)
    for scope in ir.scopes:walk(scope,scope.name)
