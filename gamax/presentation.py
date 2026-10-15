"""Classify GUI-input actions separately from headless simulation effects."""
from .ir import Expr,CompileError
from pathlib import Path
import ast
import re


def classify_presentation_actions(ir):
    resources=[(scope,field) for scope in ir.scopes for field in scope.fields if field.type=='presentation_resource']
    def references(value,name):
        if isinstance(value,Expr):return value.op=='name' and value.args==[name] or any(references(v,name) for v in value.args) or references(value.metadata,name)
        if isinstance(value,(list,tuple)):return any(references(v,name) for v in value)
        if isinstance(value,dict):return any(references(v,name) for v in value.values())
        if hasattr(value,'__dataclass_fields__'):return any(references(getattr(value,key),name) for key in value.__dataclass_fields__)
        return False
    for scope,field in resources:
        other_fields=[f for owner in ir.scopes for f in owner.fields if f is not field]
        simulation=[other_fields,*[owner.init for owner in ir.scopes],*[owner.behaviors for owner in ir.scopes],*[owner.actions for owner in ir.scopes]]
        if references(simulation,field.name):raise CompileError(field.token.source,field.token,'Render resource is referenced by simulation code; geometry/font/OBJ simulation semantics require a backend','GAMAX_OUTPUT')
        tokens=field.value.metadata['tokens']
        identifiers={value for value in tokens if re.fullmatch(r'[A-Za-z_]\w*',value)}
        allowed={'obj_file','font','geometry','svg_file','const','true','false'}
        if identifiers-allowed:raise CompileError(field.token.source,field.token,'Unknown or effectful render resource constructor','GAMAX_OUTPUT')
        if tokens.count('<-')!=1:raise CompileError(field.token.source,field.token,'Render resource needs one pure initializer','GAMAX_OUTPUT')
        strings=[ast.literal_eval(v) for v in tokens if v.startswith(('"',"'"))]
        paths=[] if field.value.type=='font' else strings
        if field.value.type=='geometry' and len(paths)!=1 or field.value.type=='obj_file' and len(paths)!=2 or field.value.type=='font' and len(strings)!=1:
            raise CompileError(field.token.source,field.token,'Unsupported render resource signature','GAMAX_OUTPUT')
        for path in paths:
            if Path(path).is_absolute() or not (Path(field.token.source).parent/path).is_file():raise CompileError(field.token.source,field.token,'Missing or nonportable render resource','GAMAX_ASSET')
        ir.presentation_resources.append({'scope':scope.name,'name':field.name,'type':field.value.type,'tokens':tokens,'source':field.token.source,'paths':paths,'execution':'pure resource declaration retained and bundled; constructor requires GUI backend and is excluded from headless simulation'})
        scope.fields.remove(field)

    actions={(scope.name,action.name):action for scope in ir.scopes for action in scope.actions}
    def gui(value):
        if isinstance(value,Expr):return value.op=='gui_input' or any(gui(arg) for arg in value.args)
        if isinstance(value,(list,tuple)):return any(gui(item) for item in value)
        if isinstance(value,dict):return any(gui(item) for item in value.values())
        if hasattr(value,'__dataclass_fields__'):return any(gui(getattr(value,name)) for name in value.__dataclass_fields__)
        return False
    def calls(items,owner):
        result=set()
        for item in items:
            if item.kind=='do':result.add((owner,item.target))
            result.update(calls(item.body,item.target if item.kind in ('ask','create','birth') else owner))
            result.update(calls(item.alternative,owner))
        return result
    edges={key:calls(action.body,key[0]) for key,action in actions.items()}
    marked={key for key,action in actions.items() if gui(action.body)}
    while True:
        updated=marked|{key for key,targets in edges.items() if targets&marked}
        if updated==marked:break
        marked=updated
    for key in marked:actions[key].presentation_only=True
