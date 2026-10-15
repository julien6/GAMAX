"""Reusable resolution of species inheritance, before type/effect checking.

Integration is gated on native primitive validation. This module never dispatches
on model names or paths, and does not implement polymorphic reference lowering.
"""
from copy import deepcopy
from .ir import CompileError,Token,Expr


def resolve_field_dependencies(scopes):
    """Stable dependency-first initialization and update order, including facets.

    Dependencies on the attribute itself describe its previous value in update
    expressions and do not create a cycle. Cross-species references have their
    own scheduling; only current-scope variable dependencies order this pass.
    """
    for scope in scopes:
        by_name={field.name:field for field in scope.fields}
        visiting=set();done=set();ordered=[]
        def reads(expr):
            result=[]
            if expr.op=='name' and expr.args[0] in by_name:result.append(expr.args[0])
            for arg in expr.args:
                if isinstance(arg,Expr):result.extend(reads(arg))
            return result
        def visit(name):
            if name in done:return
            field=by_name[name]
            if name in visiting:raise CompileError(field.token.source or '<IR>',field.token,'Cyclic variable dependencies','GAMAX_EFFECT')
            visiting.add(name)
            for expr in [field.value,*field.facets.values(),*([field.getter] if field.getter is not None else [])]:
                for dependency in reads(expr):
                    if dependency!=name:visit(dependency)
            visiting.remove(name);done.add(name);ordered.append(field)
        for field in scope.fields:visit(field.name)
        scope.fields=ordered
    return scopes


def resolve_species(scopes):
    """Resolve named species DAGs and preserve slots of overridden members."""
    by_name={s.name:s for s in scopes}
    resolved={};visiting=set()
    def fail(scope,reason):
        token=scope.parent_token or (scope.fields[0].token if scope.fields else Token(scope.name,1,1))
        raise CompileError(token.source or '<IR>',token,reason,'GAMAX_INHERITANCE')
    def merge(base,local):
        result=deepcopy(base);positions={v.name:i for i,v in enumerate(result)}
        for value in local:
            if value.name in positions:result[positions[value.name]]=deepcopy(value)
            else:positions[value.name]=len(result);result.append(deepcopy(value))
        return result
    def visit(name):
        if name in resolved:return resolved[name]
        scope=by_name[name]
        if name in visiting:fail(scope,'Cyclic species inheritance')
        visiting.add(name);result=deepcopy(scope)
        parent=getattr(scope,'parent',None)
        if parent:
            if parent not in by_name or parent=='global':fail(scope,'Unknown/invalid parent species: '+parent)
            base=visit(parent)
            if scope.fsm_states or base.fsm_states:fail(scope,"FSM inheritance needs a separate native lifecycle contract")
            if scope.grid or base.grid:fail(scope,'Grid inheritance is outside the resolved species profile')
            result.fields=merge(base.fields,scope.fields)
            result.behaviors=merge(base.behaviors,scope.behaviors)
            result.actions=merge(base.actions,scope.actions)
            result.skills=list(dict.fromkeys([*base.skills,*scope.skills]))
            result.aspects={**deepcopy(base.aspects),**deepcopy(scope.aspects)}
            # Native init bodies compose parent-first; same-name reflexes override.
            result.init=deepcopy(base.init)+deepcopy(scope.init)
        def bind_self(items,owner):
            for item in items:
                if item.metadata.get('dynamic_self_species'):item.target=owner
                child=item.target if item.kind in ('create','birth','ask') else owner
                bind_self(item.body,child);bind_self(item.alternative,owner)
        bind_self(result.init,name)
        for member in [*result.behaviors,*result.actions]:bind_self(member.body,name)
        visiting.remove(name);resolved[name]=result
        return result
    return [visit(s.name) for s in scopes]


def resolve_experiments(ir):
    """Resolve experiment metadata DAGs, without executing GUI declarations."""
    done={};visiting=set()
    def visit(name):
        if name in done:return done[name]
        if name in visiting or name not in ir.presentation:
            raise CompileError('<experiments>',Token(name,1,1),'Unknown or cyclic experiment inheritance','GAMAX_EXPERIMENT')
        visiting.add(name);item=deepcopy(ir.presentation[name])
        parent=item.get('parent')
        if parent:
            base=visit(parent)
            merged=deepcopy(base['parameters']);positions={v['variable']:i for i,v in enumerate(merged)}
            for value in item['parameters']:
                if value['variable'] in positions:merged[positions[value['variable']]]=value
                else:positions[value['variable']]=len(merged);merged.append(value)
            item['parameters']=merged;item['outputs']=deepcopy(base['outputs'])+item['outputs']
            item['unsupported_effects']=deepcopy(base.get('unsupported_effects',[]))+item.get('unsupported_effects',[])
            if item.get('pacing') is None:item['pacing']=base.get('pacing')
        visiting.remove(name);done[name]=item;return item
    for name in ir.experiments:visit(name)
    ir.presentation=done


def resolve_monomorphic_references(ir):
    """Narrow references only when a parent has one instantiated descendant.

    Populations remain distinct. A reserved capacity that could instantiate the
    parent is rejected by compile_model, rather than changing dispatch silently.
    """
    parents={s.name:s.parent for s in ir.scopes if s.parent}
    aliases={}
    for parent in set(parents.values()):
        if ir.populations.get(parent,0):continue
        descendants=[]
        for scope in ir.scopes[1:]:
            ancestor=scope.name;lineage=set()
            while ancestor in parents and ancestor not in lineage:
                lineage.add(ancestor);ancestor=parents[ancestor]
                if ancestor==parent and ir.populations.get(scope.name,0):descendants.append(scope.name);break
        if len(descendants)==1:aliases[parent]=descendants[0]
    def narrow(value):
        if isinstance(value,(list,tuple)):
            for item in value:narrow(item)
        elif isinstance(value,dict):
            for item in value.values():narrow(item)
        elif hasattr(value,'__dataclass_fields__'):
            if hasattr(value,'type') and isinstance(value.type,str) and value.type.startswith(('agent:','agents:')):
                prefix,owner=value.type.split(':',1)
                if owner in aliases:value.type=prefix+':'+aliases[owner]
            for name in value.__dataclass_fields__:narrow(getattr(value,name))
    narrow(ir.scopes);ir.reference_aliases=aliases
