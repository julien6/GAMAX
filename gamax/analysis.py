"""Static type/effect checking; unsupported behavior fails before output mutation."""
from __future__ import annotations
from pathlib import Path
import copy
import json
import math
import re
import struct
from .ir import Token, CompileError, Expr, Field, Statement, Behavior, Action, Scope, ModelIR
from .lexing import TOKEN, PRECEDENCE, FUNCTIONS, RESERVED

class CheckerMixin:
    def resolve_creation_counts(self, ir):
        constants = {}
        collections={}
        def evaluate(expr):
            if expr.op == "literal": return expr.args[0]
            if expr.op == "name" and expr.args[0] in constants: return constants[expr.args[0]]
            if expr.op=='name' and expr.args[0] in collections:return collections[expr.args[0]]
            if expr.op=='empty_agents':return range(0)
            if expr.op=='list_literal':return range(len(expr.args))
            if expr.op in ('call:shuffle','as_list'):return evaluate(expr.args[0])
            if expr.op=='call:length':
                collection=evaluate(expr.args[0])
                if not isinstance(collection,range):raise ValueError('Unknown static collection length')
                return len(collection)
            if expr.op == "-_unary": return -evaluate(expr.args[0])
            if expr.op in ('call:int','call:float'):
                return (int if expr.op=='call:int' else float)(evaluate(expr.args[0]))
            if expr.op in ("+", "-", "*"):
                a,b = (evaluate(arg) for arg in expr.args)
                return {"+":lambda:a+b,"-":lambda:a-b,"*":lambda:a*b}[expr.op]()
            raise ValueError('not a statically resolvable count')
        for field in ir.scopes[0].fields:
            try:
                value = evaluate(field.getter if field.getter is not None else field.value)
                if field.type == "int" and type(value) is int: value = (value+2**31)%2**32-2**31
                if "min" in field.facets and value < evaluate(field.facets["min"]): value = evaluate(field.facets["min"])
                elif "max" in field.facets and value > evaluate(field.facets["max"]): value = evaluate(field.facets["max"])
                constants[field.name] = value
            except (ValueError,TypeError): pass
        for scope in ir.scopes:
            if not scope.grid:continue
            for dimension in ('width','height','frequency'):
                value=scope.grid[dimension]
                if not isinstance(value,Expr):continue
                try:resolved=evaluate(value)
                except (ValueError,TypeError,OverflowError):self.fail(value.token,'Grid dimensions must resolve from scalar default global parameters','GAMAX_SPATIAL')
                if type(resolved) is not int or not (0<=resolved<=100 if dimension=='frequency' else 0<resolved<=1000000):self.fail(value.token,'Resolved grid dimensions must be positive int32 capacities','GAMAX_SPATIAL')
                def freeze(expr):
                    if expr.op=='name':ir.fixed_grid_parameters[expr.args[0]]=constants[expr.args[0]]
                    for arg in expr.args:
                        if isinstance(arg,Expr):freeze(arg)
                freeze(value)
                scope.grid[dimension]=resolved;scope.grid['facets'][dimension]=resolved
            if scope.grid['width']*scope.grid['height']>1000000:self.fail(scope.fields[0].token if scope.fields else self.token,'Grid exceeds one million cells','GAMAX_CAPACITY')
        populations = {scope.name:scope.grid["width"]*scope.grid["height"] for scope in ir.scopes if scope.grid}
        collections.update({name:range(count) for name,count in populations.items()})
        globals_ = {f.name:f for f in ir.scopes[0].fields}
        def assign_constant(name, value, kind):
            try:
                result=evaluate(value)
                if kind=='int': result=(int(result)+2**31)%2**32-2**31
                field=globals_.get(name)
                if field:
                    if 'min' in field.facets and result<evaluate(field.facets['min']):result=evaluate(field.facets['min'])
                    elif 'max' in field.facets and result>evaluate(field.facets['max']):result=evaluate(field.facets['max'])
                constants[name]=result
            except (ValueError,TypeError,OverflowError): constants.pop(name,None)
        def walk(items):
            local_names=[]
            for item in items:
                if item.kind=='birth':item.kind='create'
                if item.kind.startswith('local:'):
                    local_names.append(item.target)
                    assign_constant(item.target,item.value,item.kind.split(':')[1])
                elif item.kind=='assign' and item.target in globals_:
                    assign_constant(item.target,item.value,globals_[item.target].type)
                elif item.kind == "create":
                    try: count = evaluate(item.value)
                    except ValueError: self.fail(item.token,"Initial creation count must be resolvable from scalar default parameters", "GAMAX_CAPACITY")
                    if type(count) is not int or not 0<=count<=1000000:
                        self.fail(item.token,"Initial creation count requires an int in [0, 1000000]", "GAMAX_CAPACITY")
                    if item.target in populations and any(s.name==item.target and s.grid for s in ir.scopes):
                        self.fail(item.token,"Grids are created automatically", "GAMAX_SPATIAL")
                    start = populations.get(item.target,0)
                    if start+count>1000000:self.fail(item.token,"Initial population exceeds capacity", "GAMAX_CAPACITY")
                    item.metadata.update(start=start,count=count)
                    populations[item.target]=start+count
                    collections[item.target]=range(start+count)
                    target=next((s for s in ir.scopes if s.name==item.target),None)
                    target_fields={f.name for f in target.fields} if target else set()
                    def invalidate(body):
                        for statement in body:
                            if statement.kind=='assign' and statement.target in globals_ and statement.target not in target_fields:
                                constants.pop(statement.target,None)
                            invalidate(statement.body);invalidate(statement.alternative)
                    if count: invalidate(item.body)
                else:
                    walk(item.body);walk(item.alternative)
            for name in local_names: constants.pop(name,None)
        walk(ir.scopes[0].init)
        ir.populations = self.populations = populations

    def check(self, ir):
        self.fixed_grid_parameters=ir.fixed_grid_parameters
        self.nullable_point_names=set()
        self.diffusion_targets=set()
        self.statement_depth=0
        self.output_count = 0
        self.scalar_collection = False
        self.caller_scope = None
        self.commutative_overlap = False
        def expanded_size(items):
            total = 0
            for item in items:
                size = 1 + expanded_size(item.body) + expanded_size(item.alternative)
                if item.kind == "static_loop": size *= max(1, len(item.value.args))
                total += size
                if total > 4096:
                    self.fail(item.token, "Expanded control flow exceeds 4096 statements", "GAMAX_LOOP")
            return total
        for scope in ir.scopes:
            expanded_size(scope.init)
            for behavior in scope.behaviors: expanded_size(behavior.body)
            for action in scope.actions: expanded_size(action.body)
        global_init = ir.scopes[0].init
        configuration = [i for i, stmt in enumerate(global_init) if stmt.kind == "assign" and stmt.target in ("rng", "seed")]
        if configuration:
            prefix = configuration == [0, 1]
            expected = global_init[:2] if prefix else global_init[-2:]
            if (not prefix and configuration != list(range(len(global_init)-2, len(global_init)))) or [stmt.target for stmt in expected] != ["rng", "seed"]:
                self.fail(global_init[configuration[0]].token, "RNG configuration must be a prefix or final rng/seed pair in global init", "GAMAX_RANDOM")
            rng, seed = [stmt.value for stmt in expected]
            if rng.op != "literal" or rng.args not in (["java"], ["mersenne"]) or seed.op != "literal" or seed.type not in ("int", "float"):
                self.fail(expected[0].token, "Random profile requires literal rng = java/mersenne and finite positive literal seed", "GAMAX_RANDOM")
            if seed.args[0] == 0: self.fail(seed.token, "GAMA seed zero requests a fresh random seed; a positive seed is required", "GAMAX_RANDOM")
            ir.random_seed = float(seed.args[0])
            ir.random_generator = rng.args[0]
            ir.random_initialization = prefix and len(global_init) > 2
            if ir.random_initialization:
                if any(scope.skills for scope in ir.scopes):
                    self.fail(expected[0].token, "Seed-before-creation with moving requires native implicit random initialization, not yet supported", "GAMAX_RANDOM")
                del global_init[:2]
            else:
                del global_init[-2:]
        if configuration and any(scope.grid for scope in ir.scopes):
            self.fail(global_init[configuration[0]].token, "Grid implicit initialization precedes global-init reseeding; combined RNG phases remain unsupported", "GAMAX_RANDOM")
        def implicit_random(value):
            if isinstance(value,Expr):return value.op in ('call:rnd','call:flip','call:one_of','call:shuffle','at_distance','overlapping') or any(implicit_random(a) for a in value.args)
            if isinstance(value,(list,tuple)):return any(implicit_random(v) for v in value)
            if isinstance(value,dict):return any(implicit_random(v) for v in value.values())
            if hasattr(value,'__dataclass_fields__'):return any(implicit_random(getattr(value,k)) for k in value.__dataclass_fields__)
            return False
        def requires_skill_defaults(scope):
            if not scope.skills:return False
            assigned={s.target for s in scope.init if s.kind=='assign'}
            if not {'location','speed','heading'}<=assigned:return True
            def reads_unknown(expr,known):
                return expr.op=='name' and expr.args[0] in {'location','heading'}-known or any(reads_unknown(a,known) for a in expr.args if isinstance(a,Expr))
            if any(reads_unknown(f.value,set()) for f in scope.fields if not f.value.metadata.get('native_moving_default')):return True
            def inspect(items,known):
                for statement in items:
                    if statement.value and reads_unknown(statement.value,known):return True
                    if inspect(statement.body,set(known)) or inspect(statement.alternative,set(known)):return True
                    if statement.kind=='assign':known.add(statement.target)
                return False
            return inspect(scope.init,set())
        implicit_skill=any(requires_skill_defaults(scope) for scope in ir.scopes)
        if not configuration and (implicit_skill or any(scope.grid for scope in ir.scopes) or bool(ir.scopes[1:]) and implicit_random(ir)):
            ir.random_seed = 0.0
            ir.random_generator = "mersenne"
            ir.random_initialization = True
            ir.random_implicit = True
        self.random_implicit = ir.random_implicit
        self.world_torus=ir.world_torus
        self.random_seed = ir.random_seed
        self.random_initialization = ir.random_initialization
        if configuration and implicit_skill:self.fail(self.token,'Moving defaults require the native implicit RNG profile','GAMAX_SPATIAL')
        if any(scope.skills for scope in ir.scopes) and any(f.name in ("step", "torus") for f in ir.scopes[0].fields):
            self.fail(ir.scopes[0].fields[0].token, "moving supports only the default world and one-second clock", "GAMAX_SPATIAL")
        if ir.random_initialization:
            for scope in ir.scopes[1:]:
                if scope.skills:
                    for field in scope.fields:
                        if not field.value.metadata.get('native_moving_default'):continue
                        if field.name=='location':field.value=Expr('implicit_location',[],'point',field.token)
                        elif field.name=='heading':field.value=Expr('call:rnd',[Expr('literal',[360.0],'float',field.token)],'float',field.token)
                if not scope.grid and not any(f.name == "location" for f in scope.fields):
                    token = scope.fields[0].token if scope.fields else Token(scope.name,1,1)
                    scope.fields.insert(0,Field("location","point",Expr("implicit_location",[],"point",token),token))
        for scope in ir.scopes[1:]:
            if scope.grid:
                if any(f.name=="location" for f in scope.fields):self.fail(scope.fields[0].token,"Redeclaring grid location is unsupported", "GAMAX_SPATIAL")
                scope.fields.insert(0,Field("location","point",Expr("grid_location",[],"point",Token(scope.name,1,1)),Token(scope.name,1,1)))
                def uses_coordinate(value,name):
                    if isinstance(value,Expr):return (value.op in ('name','each','myself') and value.args[0]==name or value.op=='attribute:'+name) or any(uses_coordinate(a,name) for a in value.args)
                    if isinstance(value,(list,tuple)):return any(uses_coordinate(v,name) for v in value)
                    if isinstance(value,dict):return any(uses_coordinate(v,name) for v in value.values())
                    if hasattr(value,'__dataclass_fields__'):return any(uses_coordinate(getattr(value,k),name) for k in value.__dataclass_fields__)
                    return False
                for coordinate in ('grid_x','grid_y'):
                    if any(f.name==coordinate for f in scope.fields):self.fail(scope.fields[0].token,'Redeclaring built-in grid coordinates is unsupported','GAMAX_SPATIAL')
                    if uses_coordinate(ir,coordinate):
                        token=Token(coordinate,1,1)
                        scope.fields.insert(1,Field(coordinate,'int',Expr(coordinate,[],'int',token),token))
        self.scopes = {scope.name: scope for scope in ir.scopes}
        globals_ = {f.name: f.type for f in ir.scopes[0].fields}
        for scope in ir.scopes:
            self.checked_scope = scope.name
            local = {f.name: f.type for f in scope.fields}
            initialized = dict(globals_) if scope.name != "global" else {}
            for name in local:
                initialized.pop(name, None)
            if scope.name != "global": initialized["index"] = "int"
            self.initializing = True
            self.checking_field=True
            for field in scope.fields:
                if field.type == "geometry" and (scope.name != "global" or field.name != "shape" or field.value.op not in ('call:square','call:rectangle')):
                    self.fail(field.token, "Current world geometry profile supports square/rectangle only", "GAMAX_SPATIAL")
                self.check_expr(field.value, initialized)
                if field.type=='collection':
                    if not field.value.type.startswith('values:'):self.fail(field.token,'Value-list element type cannot be inferred','GAMAX_TYPE')
                    field.type=field.value.type;local[field.name]=field.type
                    if scope.name=='global':globals_[field.name]=field.type
                self.compatible(field.type, field.value)
                if field.type=='point' and field.value.metadata.get('nullable_point'):self.fail(field.token,'Persistent nullable points require an explicit presence representation','GAMAX_TYPE')
                if field.type.startswith('values:'):
                    if scope.name!='global' or field.value.op!='list_literal':self.fail(field.token,'Persistent value lists currently require a global literal constructor','GAMAX_COLLECTION')
                    field.capacity=len(field.value.args)
                if field.type.startswith('agents:'):
                    if field.getter is not None:
                        owner=field.type.split(':')[1]
                        if owner not in self.scopes:self.fail(field.token,'Unknown computed collection species','GAMAX_TYPE')
                        field.capacity=self.populations[owner]
                    elif not scope.grid and scope.name!='global' and not ir.populations[scope.name] and field.value.op=='empty_agents':
                        field.capacity=self.populations[field.type.split(':')[1]]
                    elif not scope.grid:
                        if scope.name!='global' or field.value.op!='empty_agents':
                            self.fail(field.token,'Mutable agent lists currently require an empty global declaration','GAMAX_COLLECTION')
                        field.list_mutable=True
                        field.capacity=self.populations[field.type.split(':')[1]]
                    elif field.value.op != 'neighbors_at':
                        self.fail(field.token, 'Persistent agent lists currently require a grid neighborhood initializer', 'GAMAX_COLLECTION')
                    else:
                        owner = field.value.args[0].type.split(':')[1]
                        radius = field.value.metadata['max_radius']
                        field.capacity = (2*radius+1)**2-1 if self.scopes[owner].grid['neighbors']==8 else 2*radius*(radius+1)
                for facet_name,facet in field.facets.items():
                    if facet_name=='const':
                        if facet.op!='literal' or facet.type!='bool':self.fail(facet.token,'const requires a literal Boolean','GAMAX_FACET')
                        continue
                    if facet_name=='update':continue
                    self.check_expr(facet, initialized)
                    self.compatible(field.type, facet)
                    def effectful(expr):
                        return expr.op in ("call:rnd", "call:flip") or any(effectful(a) for a in expr.args if isinstance(a, Expr))
                    if effectful(facet): self.fail(facet.token, "Random declaration bounds are unsupported", "GAMAX_FACET")
                initialized[field.name] = field.type
            env = globals_ | local
            self.checking_field=False
            if scope.name != "global": env["index"] = "int"
            self.check_statements(scope.init, env, local, scope.name, unconditional=True,allow_global_writes=scope.name!='global')
            self.initializing = False
            for field in scope.fields:
                if field.getter is None:continue
                if 'update' in field.facets:self.fail(field.token,'Computed attributes cannot have update facets','GAMAX_FACET')
                if field.type not in ('int','float','bool','point','rgb') and not field.type.startswith('agents:'):self.fail(field.token,'Unsupported computed attribute type','GAMAX_FACET')
                def effectful_getter(expr):
                    return expr.op in ('call:rnd','call:flip','call:one_of','call:shuffle','neighbors_at','overlapping','at_distance','implicit_location','inside') or any(effectful_getter(arg) for arg in expr.args if isinstance(arg,Expr))
                field.getter_random=effectful_getter(field.getter)
                self.checking_getter=True
                self.check_expr(field.getter,env)
                self.checking_getter=False
                self.compatible(field.type,field.getter)
            for field in scope.fields:
                if 'update' in field.facets:
                    if field.type.startswith(('agent:','agents:','values:')) or field.type=='geometry':self.fail(field.token,'Update attributes currently require scalar/point/RGBA values','GAMAX_FACET')
                    self.check_expr(field.facets['update'],env)
                    self.compatible(field.type,field.facets['update'])
            for action in scope.actions:
                # Used definitions are checked in every fully expanded call site,
                # preserving init/runtime context. Unused definitions still fail
                # closed instead of hiding unsupported operations.
                if action.phases or action.presentation_only:continue
                action_env = dict(env); action_local = dict(local)
                for name, type_ in action.parameters:
                    if name in action_env: self.fail(action.token, "Action parameter shadowing is unsupported", "GAMAX_NAME")
                    action_env[name] = action_local[name] = type_
                saved_return=getattr(self,"return_type",None);self.return_type=action.return_type
                self.check_statements(action.body, action_env, action_local, scope.name, allow_global_writes=True)
                self.return_type=saved_return
            for behavior in scope.behaviors:
                self.check_expr(behavior.guard, env)
                self.compatible("bool", behavior.guard)
                self.check_statements(behavior.body, env, local, scope.name, allow_global_writes=True)
                behavior.reads, behavior.writes = self.effects(behavior, scope.name, local)
        getters={s.name+'.'+f.name:f for s in ir.scopes for f in s.fields if f.getter is not None}
        visiting=set();done=set()
        def verify(key):
            if key in done:return
            field=getters[key]
            if key in visiting:self.fail(field.token,'Cyclic computed attribute dependencies','GAMAX_EFFECT')
            visiting.add(key);owner=key.split('.')[0]
            def dependencies(expr):
                targets=[]
                if expr.op=='name':
                    selected=owner if any(f.name==expr.args[0] for f in self.scopes[owner].fields) else 'global'
                    targets=[selected+'.'+expr.args[0]]
                elif expr.op.startswith('attribute:'):targets=[expr.args[0].type.split(':')[1]+'.'+expr.op.split(':')[1]]
                elif expr.op in ('each','myself'):targets=[expr.metadata.get('owner_scope',expr.metadata.get('caller_scope',owner))+'.'+expr.args[0]]
                for target in targets:
                    if target in getters:
                        verify(target);field.getter_random=field.getter_random or getters[target].getter_random
                        break
                for arg in expr.args:
                    if isinstance(arg,Expr):dependencies(arg)
            dependencies(field.getter);visiting.remove(key);done.add(key)
        for key in getters:verify(key)
        for field in getters.values():
            if field.getter_random and field.type in ('point','rgb'):
                self.fail(field.token,'Random point/RGBA getters require single-evaluation native observation bindings','GAMAX_EFFECT')


    def effects(self, behavior, scope, local):
        reads, writes = set(), set()
        def qualified(name):
            return scope + "." + name if name in local else "global." + name
        def expression(expr):
            if expr.op == "name":
                if expr.args[0] in local or any(f.name == expr.args[0] for f in self.scopes["global"].fields): reads.add(qualified(expr.args[0]))
            elif expr.op=="function_value":
                statements(expr.metadata["body"])
            elif expr.op == "each":
                # Conservative whole-population dependencies for every candidate scope.
                for name, target in self.scopes.items():
                    if name != 'global' and any(f.name == expr.args[0] for f in target.fields): reads.add(name+'.'+expr.args[0])
            elif expr.op == "myself": reads.add(expr.metadata['caller_scope']+'.'+expr.args[0])
            elif expr.op not in ("literal", "iterations"):
                for arg in expr.args: expression(arg)
        def statements(items):
            for item in items:
                if item.value: expression(item.value)
                if item.kind in ('ask','birth','create'):
                    target = self.scopes[item.target]
                    reads.add('$present.'+target.name)
                    if item.kind in ('birth','create'):
                        writes.update({'$present.'+target.name,'$next.'+target.name})
                        writes.update(target.name+'.'+f.name for f in target.fields)
                    nested = Behavior('ask', Expr('literal', [True], 'bool', item.token), item.body, item.token)
                    nr, nw = self.effects(nested, target.name, {f.name: f.type for f in target.fields})
                    reads.update(nr); writes.update(nw)
                else:
                    if item.kind=='diffuse':reads.add(item.metadata['on']+'.'+item.metadata['var']);writes.add(item.metadata['on']+'.'+item.metadata['var'])
                    if item.kind=="attribute_assign":
                        expression(item.metadata["reference"]);writes.add(item.metadata["owner_scope"]+"."+item.target)
                    if item.kind == 'die': writes.add('$present.'+scope)
                    if item.kind in ("skill_move","skill_goto"):
                        reads.update(scope + "." + name for name in ("location", "heading", "speed"))
                        writes.update(scope + "." + name for name in ("location", "heading", "real_speed"))
                    if item.kind == "assign" and item.target == "location" and "moving" in self.scopes[scope].skills:
                        reads.add(scope + ".location"); writes.add(scope + ".heading")
                    if item.kind == "assign" and (item.target in local or any(f.name == item.target for f in self.scopes["global"].fields)): writes.add(qualified(item.target))
                    if item.kind == "assign":
                        owner = self.scopes[scope] if item.target in local else self.scopes["global"]
                        field = next((f for f in owner.fields if f.name == item.target), None)
                        if field:
                            for facet in field.facets.values(): expression(facet)
                    statements(item.body); statements(item.alternative)
        expression(behavior.guard); statements(behavior.body)
        return sorted(reads), sorted(writes)


    def compatible(self, type_, expr):
        if type_=="point" and expr.type=="nil":expr.type="point";expr.op="point_nil";expr.args=[];expr.metadata["nullable_point"]=True
        if type_=='image_file' and expr.type=='nil':expr.type='image_file'
        if type_=='rgb' and expr.type=='nil':expr.type='rgb';expr.args=[[-1,-1,-1,-1]]
        if type_.startswith('agent:') and expr.type=='nil':
            expr.type=type_
        if type_ != expr.type and not (type_ == "float" and expr.type == "int"):
            self.fail(expr.token, f"Cannot assign {expr.type} to {type_}", "GAMAX_TYPE")


    def check_statements(self, statements, env, local, scope, unconditional=False, allow_global_writes=False):
        env, local = dict(env), dict(local)
        nullable_before=set(self.nullable_point_names)
        self.statement_depth+=1
        for statement in statements:
            if statement.metadata.get('initial_attributes'):
                fields={f.name:f for f in self.scopes[statement.target].fields}
                def effectful(expr):return expr.op=='function_value' or expr.op in ('call:rnd','call:flip','call:one_of','call:shuffle','neighbors_at','inside','overlapping','at_distance') or any(effectful(v) for v in expr.args if isinstance(v,Expr))
                for name,value in statement.metadata['initial_attributes'].items():
                    if name not in fields or fields[name].getter is not None:self.fail(statement.token,'Unknown/read-only creation attribute','GAMAX_TYPE')
                    if effectful(value):self.fail(value.token,'Effectful creation attributes need a native evaluation-order contract','GAMAX_EFFECT')
                    self.check_expr(value,env);self.compatible(fields[name].type,value)
            if statement.kind=="return":
                kind=getattr(self,"return_type",None)
                if kind is None:self.fail(statement.token,"return requires a typed action","GAMAX_EFFECT")
                self.check_expr(statement.value,env);self.compatible(kind,statement.value)
            elif statement.kind.startswith("local:"):
                if statement.target in env: self.fail(statement.token, "Local shadowing is unsupported", "GAMAX_NAME")
                self.check_expr(statement.value, env)
                type_ = statement.kind.split(":",1)[1]
                self.compatible(type_, statement.value)
                env[statement.target] = local[statement.target] = type_
                if type_=="point" and statement.value.metadata.get("nullable_point"):self.nullable_point_names.add(statement.target)
            elif statement.kind=='skill_goto':
                if self.world_torus:self.fail(statement.token,'Toroidal goto requires a separate native path contract','GAMAX_SPATIAL')
                if self.initializing or scope=='global':self.fail(statement.token,'goto requires a moving species transition','GAMAX_SPATIAL')
                self.check_expr(statement.value,env)
                if not statement.value.type.startswith('agent:'):self.compatible('point',statement.value)
                speed=statement.metadata.get('speed')
                if speed is not None:
                    self.check_expr(speed,env)
                    if speed.type not in ('int','float'):self.fail(speed.token,'goto speed must be numeric','GAMAX_TYPE')
                def effectful(value):return value.op in ('call:rnd','call:flip','call:one_of','call:shuffle','at_distance') or any(effectful(a) for a in value.args if isinstance(a,Expr))
                if speed is not None and effectful(speed):self.fail(statement.token,'Random goto speed requires a native evaluation-order contract','GAMAX_EFFECT')
            elif statement.kind == "skill_move":
                if self.initializing or scope == "global": self.fail(statement.token, "move requires a species transition", "GAMAX_SPATIAL")
                if statement.value is not None:
                    self.check_expr(statement.value,env)
                    if statement.value.type not in ('int','float'):self.fail(statement.token,'Moving speed facet requires a numeric scalar','GAMAX_TYPE')
            elif statement.kind == "block":
                self.check_statements(statement.body, env, local, scope, unconditional=unconditional, allow_global_writes=allow_global_writes)
            elif statement.kind == 'collection_loop':
                if statement.target in env:
                    self.fail(statement.token, 'Loop variable shadowing is unsupported', 'GAMAX_NAME')
                saved_overlap=self.commutative_overlap
                self.commutative_overlap=self.commutative_integer_loop(statement,local,scope)
                statement.metadata['commutative_integer_reduction']=self.commutative_overlap
                self.check_expr(statement.value, env)
                self.commutative_overlap=saved_overlap
                if not statement.value.type.startswith('agents:'):
                    self.fail(statement.token, 'Collection loops currently require ordered agent references', 'GAMAX_LOOP')
                type_ = 'agent:'+statement.value.type.split(':')[1]
                self.check_statements(statement.body, env|{statement.target:type_},local|{statement.target:type_},scope,allow_global_writes=allow_global_writes)
            elif statement.kind in ("static_loop", "dynamic_loop"):
                if statement.kind == "dynamic_loop":
                    for bound in statement.value.args:
                        self.check_expr(bound, env)
                        self.compatible("int", bound)
                if statement.target in env:
                    self.fail(statement.token, "Loop variable shadowing is unsupported", "GAMAX_NAME")
                variable = {statement.target: "int"} if statement.target is not None else {}
                self.check_statements(statement.body, env | variable,
                    local | variable, scope,
                    allow_global_writes=allow_global_writes)
            elif statement.kind=='matrix_assign':
                kind=env.get(statement.target,'')
                if scope!='global' or not kind.startswith('matrix:'):self.fail(statement.token,'Matrix assignment requires a declared global matrix','GAMAX_EFFECT')
                for index in statement.metadata['indices']:self.check_expr(index,env);self.compatible('int',index)
                self.check_expr(statement.value,env);self.compatible(kind.split(':')[1],statement.value)
            elif statement.kind=='attribute_assign':
                reference=statement.metadata['reference'];self.check_expr(reference,env)
                if not reference.type.startswith('agent:'):self.fail(statement.token,'Attribute assignment requires a statically typed agent reference','GAMAX_TYPE')
                owner=reference.type.split(':')[1]
                field=next((f for f in self.scopes[owner].fields if f.name==statement.target),None)
                if field is None or field.getter is not None or field.type not in ('int','float','bool','rgb','point'):
                    self.fail(statement.token,'Unsupported or read-only remote attribute','GAMAX_EFFECT')
                if statement.target=='location':self.fail(statement.token,'Remote spatial mutation requires a native indexing contract','GAMAX_SPATIAL')
                self.check_expr(statement.value,env);self.compatible(field.type,statement.value)
                statement.metadata['owner_scope']=owner
            elif statement.kind == "assign":
                if statement.target in self.fixed_grid_parameters:self.fail(statement.token,"Changing a fixed grid declaration parameter requires recompilation","GAMAX_SPATIAL")
                field=next((f for f in self.scopes[scope].fields if f.name==statement.target),None) or next((f for f in self.scopes['global'].fields if f.name==statement.target),None)
                if field is not None and field.facets.get("const") is not None and field.facets["const"].args==[True]:self.fail(statement.token,"Constant attribute is read-only","GAMAX_EFFECT")
                if field is not None and field.getter is not None:self.fail(statement.token,'Computed attributes are read-only','GAMAX_EFFECT')
                if env.get(statement.target)=="fsm_state":self.fail(statement.token,"Direct FSM state writes require transition lifecycle semantics","GAMAX_EFFECT")
                if statement.target not in local and not (allow_global_writes and statement.target in env and statement.target != "index"):
                    self.fail(statement.token, "Init writes must be local; reflex writes may also target declared globals", "GAMAX_EFFECT")
                self.check_expr(statement.value, env)
                self.compatible(env[statement.target], statement.value)
                if env[statement.target]=="point" and statement.value.metadata.get("nullable_point"):
                    if statement.target=="location" and self.scopes[scope].skills:statement.metadata["ignore_nil_location"]=True
                    elif statement.target in local and statement.target not in {f.name for f in self.scopes[scope].fields}:self.nullable_point_names.add(statement.target)
                    else:self.fail(statement.token,"Nullable persistent point requires a native storage contract","GAMAX_TYPE")
                if env[statement.target].startswith('values:'):
                    self.fail(statement.token,'Persistent value lists are currently read-only','GAMAX_COLLECTION')
                if env[statement.target].startswith('agents:'):
                    if statement.target not in {f.name for f in self.scopes[scope].fields} and statement.target in local and statement.target not in {f.name for f in self.scopes["global"].fields}:continue
                    field=next((f for f in self.scopes['global'].fields if f.name==statement.target),None)
                    if statement.target in {f.name for f in self.scopes[scope].fields} and scope!='global' or field is None or not field.list_mutable:
                        self.fail(statement.token, 'Persistent neighborhood lists are currently read-only', 'GAMAX_COLLECTION')
            elif statement.kind == "if":
                self.check_expr(statement.value, env); self.compatible("bool", statement.value)
                self.check_statements(statement.body, env, local, scope, allow_global_writes=allow_global_writes)
                self.check_statements(statement.alternative, env, local, scope, allow_global_writes=allow_global_writes)
            elif statement.kind=='fsm_dispatch':
                if self.initializing:self.fail(statement.token,'FSM dispatch requires a transition','GAMAX_EFFECT')
                self.check_statements(statement.body,env,local,scope,allow_global_writes=allow_global_writes)
            elif statement.kind=='fsm_state':
                saved=getattr(self,'inside_fsm_state',False);self.inside_fsm_state=True
                self.check_statements(statement.body,env,local,scope,allow_global_writes=allow_global_writes)
                self.inside_fsm_state=saved
            elif statement.kind=='fsm_transition':
                if not getattr(self,'inside_fsm_state',False):self.fail(statement.token,'Transition outside FSM state','GAMAX_EFFECT')
                names=[state.name for state in self.scopes[scope].fsm_states]
                if statement.target not in names:self.fail(statement.token,'Unknown target state','GAMAX_NAME')
                self.check_expr(statement.value,env);self.compatible('bool',statement.value)
                statement.metadata['state_index']=names.index(statement.target)
            elif statement.kind=='diffuse':
                facets=statement.metadata
                required={'var','on','proportion','radius','propagation','method'}
                if set(facets)!=required or facets['propagation']!='gradient' or facets['method']!='convolution':self.fail(statement.token,'Diffusion currently requires explicit gradient convolution facets','GAMAX_DIFFUSION')
                owner=facets['on'];variable=facets['var']
                target=self.scopes.get(owner)
                field=next((f for f in target.fields if f.name==variable),None) if target else None
                if target is None or not target.grid or field is None or field.type!='float' or target.grid['torus']:self.fail(statement.token,'Gradient diffusion requires a non-toroidal rectangular grid float attribute','GAMAX_DIFFUSION')
                if self.initializing or scope!='global' or self.statement_depth!=1:self.fail(statement.token,'Diffusion currently requires a global reflex statement outside nested control flow','GAMAX_DIFFUSION')
                key=owner+'.'+variable
                if key in self.diffusion_targets:self.fail(statement.token,'Multiple diffusion requests for one field require native kernel merging','GAMAX_DIFFUSION')
                self.diffusion_targets.add(key)
                self.check_expr(facets['proportion'],env)
                if facets['proportion'].type not in ('int','float'):self.fail(statement.token,'Diffusion proportion must be numeric','GAMAX_TYPE')
                radius=facets['radius']
                if radius.op!='literal' or radius.type!='int' or not 0<=radius.args[0]<=8:self.fail(radius.token,'Diffusion radius must be a literal within [0,8]','GAMAX_DIFFUSION')
                facets['deferred_gradient']=True;facets['radius_value']=max(1,radius.args[0])

            elif statement.kind=="pause":
                if self.initializing:self.fail(statement.token,"pause initialization requires an experiment-control contract","GAMAX_EFFECT")
            elif statement.kind in ("write","save"):
                if statement.kind=="save":
                    destination=statement.metadata["destination"]
                    if destination.op!="literal" or destination.type!="string" or Path(destination.args[0]).is_absolute() or ".." in Path(destination.args[0]).parts:self.fail(statement.token,"save requires a relative literal path without parent traversal","GAMAX_OUTPUT")
                    self.check_expr(statement.metadata["rewrite"],env);self.compatible("bool",statement.metadata["rewrite"])
                    statement.metadata["path"]=destination.args[0]
                if statement.kind=="write" and (not self.initializing or scope!="global" or not unconditional):
                    self.fail(statement.token,"Current output profile accepts unconditional global-init write only", "GAMAX_OUTPUT")
                parts=[]
                def flatten(expr):
                    if expr.op=="literal" and expr.type=="string":parts.append({'literal':expr.args[0]});return
                    def text(value):return value.op=="literal" and value.type=="string" or value.op=="+" and any(text(a) for a in value.args)
                    if expr.op=="+" and text(expr):
                        for a in expr.args:flatten(a)
                        return
                    self.check_expr(expr,env)
                    if expr.type not in ("int","float","bool"):
                        self.fail(expr.token,"write accepts literal text and scalar values only", "GAMAX_OUTPUT")
                    key=f'$output.{self.output_count}.{len(parts)}'
                    parts.append({'key':key,'type':expr.type,'expression':expr})
                flatten(statement.value)
                statement.metadata.update(output_id=self.output_count,parts=parts)
                self.output_count+=1
            elif statement.kind == "create":
                if not unconditional:
                    self.fail(statement.token, "Conditional/runtime creation is not supported", "GAMAX_EFFECT")
                self.check_expr(statement.value,env);self.compatible("int",statement.value)
                if statement.target not in self.scopes or self.scopes[statement.target].grid:
                    self.fail(statement.token,"Creation target requires a declared species", "GAMAX_NAME")
                target=self.scopes[statement.target]
                target_local={f.name:f.type for f in target.fields}
                globals_={f.name:f.type for f in self.scopes['global'].fields}
                captured={name:type_ for name,type_ in local.items() if name not in {f.name for f in self.scopes[scope].fields} and name!='index'}
                if set(captured)&set(target_local):self.fail(statement.token,'Creation captures shadow target attributes','GAMAX_NAME')
                saved_scope,saved_caller=self.checked_scope,self.caller_scope
                self.checked_scope=target.name;self.caller_scope=scope
                self.check_statements(statement.body,globals_|captured|target_local|{'index':'int'},captured|target_local,target.name,allow_global_writes=True)
                self.checked_scope,self.caller_scope=saved_scope,saved_caller
            elif statement.kind == 'birth':
                if self.initializing or not self.random_initialization:
                    self.fail(statement.token,'Runtime birth requires seeded nonmoving initialization and a transition context','GAMAX_EFFECT')
                self.check_expr(statement.value,env);self.compatible('int',statement.value)
                if statement.target not in self.scopes or self.scopes[statement.target].grid:
                    self.fail(statement.token,'Birth target requires a non-grid species','GAMAX_NAME')
                target=self.scopes[statement.target]
                fields={f.name:f.type for f in target.fields}
                captured={n:k for n,k in local.items() if n not in {f.name for f in self.scopes[scope].fields} and n!='index'}
                if set(captured)&set(fields):self.fail(statement.token,'Birth captures shadow target attributes','GAMAX_NAME')
                globals_={f.name:f.type for f in self.scopes['global'].fields}
                saved_scope,saved_caller,saved_init=self.checked_scope,self.caller_scope,self.initializing
                self.checked_scope,self.caller_scope,self.initializing=target.name,scope,True
                self.check_statements(statement.body,globals_|captured|fields|{'index':'int'},captured|fields,target.name,allow_global_writes=True)
                self.checked_scope,self.caller_scope,self.initializing=saved_scope,saved_caller,saved_init
            elif statement.kind=='die':
                if scope=='global':self.fail(statement.token,'Death requires an agent context','GAMAX_EFFECT')
            elif statement.kind == "ask":
                self.check_expr(statement.value,env)
                if statement.value.type not in ('agent:'+statement.target,'agents:'+statement.target):self.fail(statement.token,'Ask selection type differs from resolved species','GAMAX_TYPE')
                target=self.scopes[statement.target]
                target_local={f.name:f.type for f in target.fields}
                globals_={f.name:f.type for f in self.scopes['global'].fields}
                captured={name:kind for name,kind in local.items() if name not in {f.name for f in self.scopes[scope].fields} and name!='index'}
                if set(captured)&set(target_local):self.fail(statement.token,'Captured variables shadow target attributes','GAMAX_NAME')
                saved_scope,saved_caller=self.checked_scope,self.caller_scope
                self.checked_scope,self.caller_scope=target.name,scope
                self.check_statements(statement.body,globals_|target_local|captured|{'index':'int'},target_local|captured,target.name,allow_global_writes=True)
                self.checked_scope,self.caller_scope=saved_scope,saved_caller
            else:self.fail(statement.token,"Unsupported simulation statement "+statement.kind,"GAMAX_EFFECT")
        self.nullable_point_names=nullable_before
        self.statement_depth-=1


    def commutative_integer_loop(self, statement, local, scope):
        """Prove that collection order cannot affect independent local int sums.

        This narrow optimization never admits persistent writes, RNG, floating
        sums, dependent accumulators, or effects in the predicate/increment.
        """
        persistent={f.name for s in (self.scopes[scope],self.scopes['global']) for f in s.fields}
        targets=set()
        def gather(items):
            for item in items:
                if item.kind=='assign': targets.add(item.target)
                gather(item.body);gather(item.alternative)
        gather(statement.body)
        if not targets or any(local.get(n)!='int' or n in persistent for n in targets):return False
        def pure(expr):
            if expr.op=='name':return expr.args[0] not in targets
            if expr.op in ('literal','each','myself','each_ref'):return True
            if expr.op.startswith('call:') or expr.op in ('accumulate','overlapping','neighbors_at'):return False
            return all(pure(a) for a in expr.args)
        def verify(items):
            for item in items:
                if item.kind=='if':
                    if not pure(item.value) or not verify(item.body) or not verify(item.alternative):return False
                elif item.kind=='assign':
                    v=item.value
                    if v.op!='+' or v.args[0].op!='name' or v.args[0].args[0]!=item.target or not pure(v.args[1]):return False
                else:return False
            return True
        return verify(statement.body)

    def check_expr(self, expr, env):
        if expr.op=="point_nil":expr.type="point";return
        if expr.op=="name" and expr.args[0] in self.nullable_point_names:expr.metadata["nullable_point"]=True
        if expr.op.startswith('grid_at:'):
            owner=expr.op.split(':')[1]
            if owner not in self.scopes or not self.scopes[owner].grid or len(expr.args)!=1:self.fail(expr.token,'Spatial cast requires one point and a rectangular grid','GAMAX_SPATIAL')
            self.check_expr(expr.args[0],env);self.compatible('point',expr.args[0])
            expr.type='agent:'+owner;return
        if expr.op=="name" and expr.args[0]=="cycle":expr.op="cycle";expr.args=[];expr.type="int";return
        if expr.op=='function_value':
            if self.collection_scope is not None and not self.scalar_collection:self.fail(expr.token,'Effectful action in vectorized collection requires scalar evaluation','GAMAX_EFFECT')
            if self.checking_field or getattr(self,'checking_getter',False):self.fail(expr.token,'Function-valued field initializer/getter requires a native initialization/purity contract','GAMAX_EFFECT')
            if expr.metadata['return_type'] not in ('int','float','bool','point','rgb') and not expr.metadata['return_type'].startswith('agent:'):self.fail(expr.token,'Unsupported function return type','GAMAX_TYPE')
            def terminates(items):
                return any(item.kind=='return' or item.kind=='if' and terminates(item.body) and terminates(item.alternative) for item in items)
            if not terminates(expr.metadata['body']):self.fail(expr.token,'Typed action must return on all paths','GAMAX_EFFECT')
            saved=getattr(self,'return_type',None)
            self.return_type=expr.metadata['return_type']
            owner=self.checked_scope
            fields={f.name:f.type for f in self.scopes[owner].fields}
            function_env={f.name:f.type for f in self.scopes['global'].fields}|fields
            if owner!='global':function_env['index']='int'
            params=expr.metadata['body'][:expr.metadata.get('parameter_count',0)]
            for parameter in params:
                self.check_expr(parameter.value,env);kind=parameter.kind.split(':',1)[1];self.compatible(kind,parameter.value)
                function_env[parameter.target]=fields[parameter.target]=kind
            self.check_statements(expr.metadata['body'][len(params):],function_env,fields,owner,allow_global_writes=True)
            self.return_type=saved
            expr.type=expr.metadata['return_type']
            if expr.type=='point':expr.metadata['nullable_point']=True
            return
        if expr.op=='grid_distance':
            owner=expr.metadata['grid']
            if owner not in self.scopes or not self.scopes[owner].grid:self.fail(expr.token,'Explicit grid distance requires a rectangular grid topology','GAMAX_SPATIAL')
            for arg in expr.args:
                self.check_expr(arg,env)
                if arg.type!='point' and not arg.type.startswith('agent:'):self.fail(arg.token,'Grid distance operands must be points or agents','GAMAX_TYPE')
                if arg.type.startswith('agent:') and arg.type.split(':')[1]!=owner:self.fail(arg.token,'Cross-species grid distance requires geometry projection','GAMAX_SPATIAL')
            expr.type='float';return
        if expr.op=='distance_to':
            if self.scopes[self.checked_scope].grid:
                expr.op='grid_distance';expr.metadata['grid']=self.checked_scope
                self.check_expr(expr,env);return
            owners=[]
            for arg in expr.args:
                if arg.op=='each_ref':owner=self.collection_scope
                elif arg.op=='name' and arg.args==['self']:owner=self.checked_scope
                else:
                    self.check_expr(arg,env)
                    owner=arg.type.split(':')[1] if arg.type.startswith('agent:') else None
                if owner and self.scopes[owner].grid:owners.append(owner)
            if owners and self.world_torus:self.fail(expr.token,'Toroidal grid geometry distance in a continuous scope requires a periodic envelope contract','GAMAX_SPATIAL')
            if owners:expr.metadata['grid_geometry_operands']=owners
            for i,arg in enumerate(expr.args):
                if arg.op=='each_ref':expr.args[i]=Expr('each',['location'],'',arg.token)
                elif arg.op=='name' and arg.args==['self']:expr.args[i]=Expr('name',['location'],'',arg.token)
                else:
                    self.check_expr(arg,env)
                    if arg.type.startswith('agent:'):expr.args[i]=Expr('attribute:location',[arg],'',arg.token)
        if expr.op=='raster_matrix':expr.type='matrix:int';return
        if expr.op.startswith('matrix_cast:'):
            self.check_expr(expr.args[0],env)
            if not expr.args[0].type.startswith('matrix:'):self.fail(expr.token,'Matrix cast requires a matrix','GAMAX_TYPE')
            expr.type='matrix:'+expr.op.split(':')[1];return
        if expr.op=='matrix_at':
            for arg in expr.args:self.check_expr(arg,env)
            if not expr.args[0].type.startswith('matrix:') or any(arg.type!='int' for arg in expr.args[1:]):self.fail(expr.token,'Matrix indices must be integers','GAMAX_TYPE')
            expr.type=expr.args[0].type.split(':')[1];return
        if expr.op=='inside':
            for arg in expr.args:self.check_expr(arg,env)
            collection,target=expr.args
            if not collection.type.startswith('agents:') or not target.type.startswith('agent:'):self.fail(expr.token,'inside requires an agent collection and agent/grid target','GAMAX_SPATIAL')
            owner=collection.type.split(':')[1];target_owner=target.type.split(':')[1]
            if self.scopes[owner].grid or any(f.type=='polygon' for f in self.scopes[owner].fields) or any(f.type=='polygon' for f in self.scopes[target_owner].fields):self.fail(expr.token,'Toroidal inside currently supports point candidates and point/rectangular-cell targets','GAMAX_SPATIAL')
            expr.type=collection.type;expr.metadata['rectangle_query']=not self.world_torus;return
        if expr.op=='overlaps':
            for arg in expr.args:self.check_expr(arg,env)
            if any(not a.type.startswith('agent:') for a in expr.args):self.fail(expr.token,'overlaps requires agent references','GAMAX_SPATIAL')
            owners=[a.type.split(':')[1] for a in expr.args]
            if sum(bool(self.scopes[owner].grid) for owner in owners)!=1 or any(f.type=='polygon' for owner in owners for f in self.scopes[owner].fields):self.fail(expr.token,'overlaps currently supports rectangular cells against point agents','GAMAX_SPATIAL')
            expr.type='bool';return
        if expr.op=='sort_by':
            collection,score=expr.args
            self.check_expr(collection,env)
            if not collection.type.startswith('agents:'):self.fail(expr.token,'sort_by requires ordered agents','GAMAX_COLLECTION')
            saved=self.collection_scope,self.scalar_collection
            self.collection_scope=collection.type.split(':')[1];self.scalar_collection=True
            self.check_expr(score,env);self.collection_scope,self.scalar_collection=saved
            def effectful(value):return value.op=='function_value' or value.op in ('call:rnd','call:flip','call:one_of','call:shuffle','neighbors_at','inside','at_distance','overlapping') or any(effectful(a) for a in value.args if isinstance(a,Expr))
            if score.type not in ('int','float') or effectful(score):self.fail(score.token,'sort_by currently requires a pure numeric key','GAMAX_EFFECT')
            expr.type=collection.type;return
        if expr.op in ('with_max_of','with_min_of'):

            collection,score=expr.args
            self.check_expr(collection,env)
            if not collection.type.startswith('agents:'):self.fail(expr.token,'Extremum selection requires ordered agents','GAMAX_COLLECTION')
            saved=self.collection_scope,self.scalar_collection
            self.collection_scope=collection.type.split(':')[1];self.scalar_collection=True
            self.check_expr(score,env)
            self.collection_scope,self.scalar_collection=saved
            if score.type not in ('int','float'):self.fail(score.token,'Extremum score must be numeric','GAMAX_TYPE')
            expr.type='agent:'+collection.type.split(':')[1];return
        if expr.op=='first_with':
            self.check_expr(expr.args[0],env)
            if not expr.args[0].type.startswith('agents:'):self.fail(expr.token,'first_with requires an agent collection','GAMAX_COLLECTION')
            saved=self.collection_scope,self.scalar_collection
            self.collection_scope=expr.args[0].type.split(':')[1];self.scalar_collection=True
            self.check_expr(expr.args[1],env);self.compatible('bool',expr.args[1])
            self.collection_scope,self.scalar_collection=saved
            expr.type='agent:'+expr.args[0].type.split(':')[1];return
        if expr.op=='agent_shape':
            if env.get('location')!='point':self.fail(expr.token,'Agent shape requires an initialized location','GAMAX_SPATIAL')
            expr.type='polygon';return
        if expr.op=='my':
            expr.op='myself' if self.caller_scope is not None else 'name'
            self.check_expr(expr,env);return
        if expr.op=='at_distance':
            if self.world_torus:self.fail(expr.token,'Toroidal spatial queries require a separate distance/order contract','GAMAX_SPATIAL')
            if self.initializing:self.fail(expr.token,'Spatial queries during initialization require a separate native population-visibility contract','GAMAX_SPATIAL')
            for arg in expr.args:self.check_expr(arg,env)
            collection,radius=expr.args
            owner=collection.type.split(':')[1] if collection.type.startswith('agents:') else None
            if owner is None or self.scopes[owner].grid or self.checked_scope=='global' or self.scopes[self.checked_scope].grid or self.collection_scope is not None or radius.type not in ('int','float') or self.random_seed is None:
                self.fail(expr.token,'at_distance currently requires a seeded scalar point-agent caller and a nongrid agent collection','GAMAX_SPATIAL')
            expr.type=collection.type;return
        if expr.op=='empty_agents':return
        if expr.op=='list_literal':
            for arg in expr.args:self.check_expr(arg,env)
            if not expr.args:
                if not expr.type.startswith('values:'):self.fail(expr.token,'Empty value lists require an explicit element type','GAMAX_TYPE')
                return
            types={arg.type for arg in expr.args}
            if len(types)!=1 or next(iter(types)) not in ('int','float','bool','point','rgb','image_file'):self.fail(expr.token,'Value literal lists require one homogeneous supported element type','GAMAX_TYPE')
            expr.type='values:'+next(iter(types));return
        if expr.op=='overlapping' and expr.args[1].op=='call:circle':
            if self.world_torus or self.initializing or self.collection_scope is not None or self.checked_scope=='global' or self.random_seed is None:self.fail(expr.token,'Circle queries require a seeded non-toroidal species transition','GAMAX_SPATIAL')
            for arg in expr.args:self.check_expr(arg,env)
            owner=expr.args[0].type.split(':')[1] if expr.args[0].type.startswith('agents:') else None
            if owner is None or self.scopes[owner].grid:self.fail(expr.token,'Circle queries currently require point-agent collections','GAMAX_SPATIAL')
            expr.type=expr.args[0].type;expr.metadata['circle_query']=True;return
        if expr.op == 'overlapping':
            if not self.commutative_overlap or self.initializing or self.random_seed is None:
                self.fail(expr.token,'overlapping currently requires a proven order-independent local integer reduction in a seeded transition','GAMAX_SPATIAL')
            for arg in expr.args:self.check_expr(arg,env)
            a,b=expr.args
            target=a.type.split(':')[1] if a.type.startswith('agents:') else None
            grid=b.type.split(':')[1] if b.type.startswith('agent:') else None
            if target is None or self.scopes[target].grid or self.scopes[target].skills or grid is None or not self.scopes[grid].grid:
                self.fail(expr.token,'overlapping reduction currently requires point agents and a minimal rectangular grid cell','GAMAX_SPATIAL')
            expr.type=a.type
            expr.metadata['order_elided']='proved independent int32 sums; both native collection shuffle draw sequences retained'
            return
        if expr.op == 'accumulate':
            if self.initializing:
                self.fail(expr.token, 'accumulate initialization remains unsupported', 'GAMAX_COLLECTION')
            self.check_expr(expr.args[0], env)
            if not expr.args[0].type.startswith('agents:'):
                self.fail(expr.token, 'accumulate requires an ordered agent collection', 'GAMAX_COLLECTION')
            saved = self.collection_scope, self.scalar_collection
            self.collection_scope = expr.args[0].type.split(':')[1]
            self.scalar_collection = True
            self.check_expr(expr.args[1],env)
            self.collection_scope, self.scalar_collection = saved
            if not expr.args[1].type.startswith('agents:'):
                self.fail(expr.token, 'accumulate mapping must return agent references', 'GAMAX_COLLECTION')
            expr.type = expr.args[1].type
            return
        if expr.op == 'each_ref':
            if self.collection_scope is None or not self.scalar_collection:
                self.fail(expr.token, 'Bare each requires an ordered scalar collection mapping', 'GAMAX_COLLECTION')
            expr.type = 'agent:'+self.collection_scope
            return
        if expr.op == 'as_list':
            self.check_expr(expr.args[0], env)
            if expr.args[0].type=='rgb':expr.type='values:int';expr.metadata['rgba_list']=True;return
            if not expr.args[0].type.startswith(('agents:', 'values:')):
                self.fail(expr.token, 'as list requires an ordered collection', 'GAMAX_TYPE')
            expr.type = expr.args[0].type
            return
        if expr.op in ('grid_location','grid_x','grid_y'):
            expr.type='point' if expr.op=='grid_location' else 'int';return
        if expr.op.startswith("attribute:"):
            self.check_expr(expr.args[0],env)
            base=expr.args[0]
            if expr.op=='attribute:neighbors' and base.type.startswith('agent:') and self.scopes[base.type.split(':')[1]].grid:
                expr.op='neighbors_at';expr.args=[base,Expr('literal',[1],'int',expr.token)]
                self.check_expr(expr,env);return
            if not base.type.startswith("agent:") or (self.collection_scope is not None and not self.scalar_collection):
                self.fail(expr.token,"Attribute lookup requires a scalar typed agent reference", "GAMAX_TYPE")
            owner=base.type.split(":")[1];name=expr.op.split(":")[1]
            if owner not in self.scopes:self.fail(expr.token,"Unknown reference species", "GAMAX_NAME")
            fields={f.name:f.type for f in self.scopes[owner].fields}|{'index':'int'}
            if name not in fields:self.fail(expr.token,"Unknown referenced attribute", "GAMAX_NAME")
            expr.type=fields[name];return
        if expr.op == "implicit_location":
            if not self.random_initialization or self.checked_scope=="global":
                self.fail(expr.token,"Implicit location requires seeded agent initialization", "GAMAX_RANDOM")
            expr.type="point";return
        if expr.op == "literal":
            if expr.type == "string": self.fail(expr.token, "String dynamics are outside this profile", "GAMAX_TYPE")
            return
        if expr.op in ('each', 'myself'):
            if expr.op == 'myself':
                if self.caller_scope is None:self.fail(expr.token,'myself requires a creation-body caller','GAMAX_NAME')
                fields={f.name:f.type for f in self.scopes[self.caller_scope].fields}
                if self.caller_scope!='global':fields['index']='int'
                if expr.args[0] not in fields:self.fail(expr.token,'Unknown caller attribute','GAMAX_NAME')
                expr.type=fields[expr.args[0]];expr.metadata['caller_scope']=self.caller_scope
                return
            if self.initializing and self.checking_field:self.fail(expr.token,'Agent-context aliases in field initializers require a separate population-visibility contract')
            owner = self.collection_scope if expr.op == 'each' else self.checked_scope
            if owner is None or owner == 'global' or expr.args[0] == 'index' and expr.op == 'myself':
                self.fail(expr.token, 'Agent expression outside its supported context', 'GAMAX_NAME')
            fields = {f.name: f.type for f in self.scopes[owner].fields} | {'index': 'int'}
            if expr.args[0] not in fields: self.fail(expr.token, 'Unknown agent attribute', 'GAMAX_NAME')
            expr.metadata['owner_scope']=owner
            expr.type = fields[expr.args[0]]; return
        if expr.op in ('where', 'collect', 'count'):
            self.check_expr(expr.args[0], env)
            if not expr.args[0].type.startswith('agents:'): self.fail(expr.token, 'Collection operator requires fixed agents', 'GAMAX_TYPE')
            owner = expr.args[0].type.split(':')[1]
            saved = self.collection_scope
            if saved is not None: self.fail(expr.token, 'Nested each bindings are outside this profile')
            self.collection_scope = owner
            self.check_expr(expr.args[1], env)
            self.collection_scope = saved
            if expr.op in ('where', 'count'):
                self.compatible('bool', expr.args[1]); expr.type = 'int' if expr.op == 'count' else expr.args[0].type
            else:
                if expr.args[1].type not in ('int','float','bool','point'): self.fail(expr.token, 'Collect requires scalar values', 'GAMAX_TYPE')
                expr.type = 'values:'+expr.args[1].type
            return
        if expr.op == "conditional":
            for arg in expr.args: self.check_expr(arg, env)
            self.compatible("bool", expr.args[0])
            a,b = expr.args[1].type, expr.args[2].type
            if {a,b}=={'rgb','nil'}:
                for arg in expr.args[1:]:self.compatible('rgb',arg)
                expr.type='rgb';return
            if a == b and a in ("int", "float", "bool", "point","rgb"): expr.type = a
            elif {a,b} <= {"int", "float"}: expr.type = "float"
            else: self.fail(expr.token, "Incompatible conditional branch types", "GAMAX_TYPE")
            return
        if expr.op == "point":
            for arg in expr.args:
                self.check_expr(arg, env)
                if arg.type not in ("int", "float"): self.fail(arg.token, "Point coordinates must be numeric", "GAMAX_TYPE")
            expr.type = "point"; return
        if expr.op.startswith("component:"):
            self.check_expr(expr.args[0], env)
            if expr.args[0].type != "point": self.fail(expr.token, "Coordinate access requires a point", "GAMAX_TYPE")
            expr.type = "float"; return
        if expr.op == "neighbors_at":
            self.check_expr(expr.args[0],env); self.check_expr(expr.args[1],env)
            owner = expr.args[0].type.split(":")[1] if expr.args[0].type.startswith("agent:") else None
            if owner is None or not self.scopes[owner].grid or (self.collection_scope is not None and not self.scalar_collection):
                self.fail(expr.token, "neighbors_at requires a scalar grid self reference", "GAMAX_SPATIAL")
            radius = expr.args[1]
            low = high = None
            if radius.op == 'literal' and radius.type == 'int':
                low = high = radius.args[0]
            elif radius.op == 'name' and radius.type == 'int':
                field = next((f for f in self.scopes['global'].fields if f.name == radius.args[0]), None)
                if field and all(n in field.facets and field.facets[n].op=='literal' and field.facets[n].type=='int' for n in ('min','max')):
                    low,high = field.facets['min'].args[0],field.facets['max'].args[0]
            if low is None or not 1 <= low <= high <= min(5,self.scopes[owner].grid['width']):
                self.fail(radius.token, 'Grid radius requires a literal or a global int with verified min/max bounds within [1,min(width,5)]', 'GAMAX_SPATIAL')
            expr.metadata['max_radius'] = high
            expr.type = "agents:"+owner
            return
        if expr.op == "name":
            if expr.args[0] == "self" and self.checked_scope != "global":
                expr.type = "agent:"+self.checked_scope; return
            if expr.args[0] in self.populations:
                if self.initializing and not self.scopes[expr.args[0]].grid:
                    if self.checking_field:self.fail(expr.token,'Initial population reads in field declarations require a separate visibility contract','GAMAX_COLLECTION')
                expr.type = 'agents:'+expr.args[0]; return
            if expr.args[0] not in env:
                self.fail(expr.token, f"Unknown or unsupported name {expr.args[0]!r}", "GAMAX_NAME")
            expr.type = env[expr.args[0]]; return
        for arg in expr.args: self.check_expr(arg, env)
        types = [arg.type for arg in expr.args]
        if expr.op=='at':
            if types==['image_file','point']:
                expr.metadata['raster_pixel']=True;expr.type='int';return
            if not types[0].startswith(('agents:','values:')) or types[1]!='int' or self.collection_scope is not None:
                self.fail(expr.token,'at requires a scalar checked integer rank into an ordered collection','GAMAX_COLLECTION')
            expr.type=('agent:' if types[0].startswith('agents:') else '')+types[0].split(':')[1];return
        if expr.op in ('append_agent','remove_agent'):
            if not types[0].startswith('agents:') or types[1]!='agent:'+types[0].split(':')[1]:
                self.fail(expr.token,'List insertion/removal requires matching agent reference types','GAMAX_COLLECTION')
            expr.type=types[0];return
        if expr.op in ('=','!=') and any(t.startswith('agent:') for t in types):
            owner=next(t for t in types if t.startswith('agent:'))
            if any(t not in (owner,'nil') for t in types):
                self.fail(expr.token, 'Agent equality requires matching reference species or nil', 'GAMAX_TYPE')
            for arg in expr.args:
                if arg.type=='nil':arg.type=owner
            expr.type='bool'
            return
        if expr.op=='-' and types[0].startswith('agents:') and types[1]=='agent:'+types[0].split(':')[1]:
            expr.op='remove_agent';expr.type=types[0];return
        if expr.op == '+' and types[0].startswith('agents:') and types[1] in (types[0], 'agent:'+types[0].split(':')[1]):
            expr.type = types[0]
            return
        if expr.op.startswith('call:'):
            function = expr.op.split(':')[1]
            if function=='shuffle':
                if len(types)!=1 or not types[0].startswith(('agents:','values:')) or self.random_seed is None or self.collection_scope is not None:
                    self.fail(expr.token,'shuffle requires a seeded scalar ordered collection','GAMAX_RANDOM')
                if self.initializing and not self.random_initialization:self.fail(expr.token,'shuffle initialization requires a pre-initialization seed','GAMAX_RANDOM')
                expr.type=types[0];return
            if function=='circle':
                if len(types)!=1 or types[0] not in ('int','float') or self.checked_scope=='global':self.fail(expr.token,'Supported circle constructor takes a scalar radius in an agent scope','GAMAX_SPATIAL')
                expr.type='circle';return
            if function in ('first','last'):
                if len(types)!=1 or not types[0].startswith('agents:') or self.collection_scope is not None:self.fail(expr.token,'first/last requires a scalar agent collection','GAMAX_COLLECTION')
                expr.type='agent:'+types[0].split(':')[1];return
            if function=='copy':
                if len(types)!=1 or types[0] not in ('int','float','bool','point') and not types[0].startswith('agents:'):self.fail(expr.token,'Unsupported copy operand','GAMAX_TYPE')
                expr.type=types[0];return
            if function == "one_of":
                if len(types)!=1 or not types[0].startswith('agents:') or self.random_seed is None or self.collection_scope is not None:
                    self.fail(expr.token,"one_of requires a seeded scalar agent collection", "GAMAX_RANDOM")
                if self.initializing and not self.random_initialization:self.fail(expr.token,"one_of initialization requires pre-initialization seed", "GAMAX_RANDOM")
                if expr.args[0].op == 'name' and expr.args[0].args[0] in self.populations:
                    owner=expr.args[0].args[0]
                    if self.scopes[owner].grid:expr.metadata['direct_grid_sampling']=dict(self.scopes[owner].grid)
                expr.type='agent:'+types[0].split(':')[1];return
            if function=='point':
                if len(types)!=1 or not (types[0]=='point' or types[0].startswith('agent:')):self.fail(expr.token,'point cast requires an agent or point','GAMAX_TYPE')
                expr.type='point';expr.metadata['nullable_point']=types[0].startswith('agent:');return
            if function=="round":
                if len(types)!=1 or types[0] not in ("int","float"):self.fail(expr.token,"round requires one numeric value","GAMAX_TYPE")
                expr.type="int";return
            if function == "rgb":
                if types==["int"]:expr.metadata["packed_argb"]=True;expr.type="rgb";return
                if len(types) not in (3,4) or any(t!="int" for t in types):
                    self.fail(expr.token,"rgb requires a packed int or three/four integer channels", "GAMAX_TYPE")
                expr.type="rgb";return
            if function == "square":
                if self.checked_scope != "global" or len(types)!=1 or types[0] not in ("int","float"):
                    self.fail(expr.token, "square requires a global numeric dimension", "GAMAX_SPATIAL")
                expr.type = "geometry";return
            if function=='rectangle':
                if self.checked_scope!='global' or len(types)!=2 or any(t not in ('int','float') for t in types):
                    self.fail(expr.token,'rectangle requires two global numeric dimensions','GAMAX_SPATIAL')
                expr.type='geometry';return
            if function=='empty':
                if len(types)!=1 or not types[0].startswith(('agents:','values:')):self.fail(expr.token,'empty requires a supported collection','GAMAX_TYPE')
                expr.type='bool';return
            if function=='signum':
                if len(types)!=1 or types[0] not in ('int','float'):self.fail(expr.token,'signum requires one numeric value','GAMAX_TYPE')
                expr.type='int';return
            if function=='triangle':
                if self.checked_scope=='global' or len(types)!=1 or types[0] not in ('int','float'):
                    self.fail(expr.token,'triangle requires one numeric side in an agent scope','GAMAX_SPATIAL')
                expr.type='polygon';return
            if function in ("rnd", "flip"):
                if self.random_seed is None: self.fail(expr.token, "Runtime randomness requires explicit final global-init rng/seed configuration", "GAMAX_RANDOM")
                if self.collection_scope is not None or (self.initializing and (not self.random_initialization or (self.checked_scope == "global" and not self.random_implicit))):
                    self.fail(expr.token, "Random initialization requires seed-before-creation in a species without skills; random collections/global fields are unsupported", "GAMAX_RANDOM")
                if function == "flip":
                    if len(types) != 1 or types[0] not in ("int", "float"): self.fail(expr.token, "flip requires one numeric probability", "GAMAX_TYPE")
                    expr.type = "bool"
                else:
                    if len(types) not in (1, 2) or len(set(types)) != 1 or types[0] not in ("int", "float"):
                        self.fail(expr.token, "rnd requires one or two same-type numeric bounds", "GAMAX_TYPE")
                    def literal_bound(arg):
                        if arg.op=='literal':return arg.args[0]
                        if arg.op=='-_unary' and arg.args[0].op=='literal':return -arg.args[0].args[0]
                        return None
                    literals=[literal_bound(arg) for arg in expr.args]
                    if all(value is not None for value in literals):
                        bounds = [0, literals[0]] if len(types) == 1 else literals
                        if bounds[0] > bounds[1]: self.fail(expr.token, "rnd bounds must be ordered", "GAMAX_RANDOM")
                    else:
                        expr.metadata['checked_bounds']=True
                    expr.type = types[0]
                return
            arity = 2 if function in ('min','max') else 1
            if function in ('min','max') and len(types)==1 and types[0].startswith('matrix:'):expr.type=types[0].split(':')[1];return
            if function in ('min','max') and len(types) == 1 and types[0].startswith('values:'): arity = 1
            if len(types) != arity: self.fail(expr.token, f'{function} requires {arity} operands', 'GAMAX_TYPE')
            if function in ('length','sum','mean') or function in ('min','max') and arity == 1:
                if not types[0].startswith(('agents:', 'values:')): self.fail(expr.token, 'Reducer requires a fixed collection', 'GAMAX_TYPE')
                if function != 'length' and types[0] not in ('values:int','values:float') and not (function in ('sum','mean') and types[0] == 'values:point'): self.fail(expr.token, 'Numeric collection required', 'GAMAX_TYPE')
                expr.type = 'int' if function == 'length' else ('point' if types[0] == 'values:point' else 'float' if function == 'mean' else types[0].split(':')[1])
            else:
                if any(t not in ('int','float') for t in types): self.fail(expr.token, 'Numeric scalar required', 'GAMAX_TYPE')
                expr.type = 'int' if function == 'int' else ('float' if function in ('float','sqrt','sin','cos','exp') else ('float' if 'float' in types else 'int'))
            return
        if expr.op in ('=','!=') and set(types)=={'rgb','nil'}:
            for arg in expr.args:self.compatible('rgb',arg)
            types=['rgb','rgb']
        if expr.op in ("+","-") and types==["rgb","int"]:
            expr.type="rgb";expr.metadata["color_scalar_arithmetic"]=True;return
        if expr.op in ("=","!=") and types==["rgb","rgb"]:
            expr.type="bool";return
        if expr.op in ('=','!=') and set(types)=={'point','nil'}:
            expr.op='point_is_nil' if expr.op=='=' else 'point_is_not_nil';expr.args=[arg for arg in expr.args if arg.type=='point'];expr.type='bool';return
        if "point" in types:
            if expr.op == "distance_to" and types == ["point", "point"]: expr.type = "float"; return
            if expr.op in ("+", "-") and types == ["point", "point"]: expr.type = "point"; return
            if expr.op == "*" and len(types) == 2 and all(t in ("point", "int", "float") for t in types) and types.count("point") == 1: expr.type = "point"; return
            if expr.op == "/" and types[0] == "point" and types[1] in ("int", "float"):
                if expr.args[1].op == "literal" and expr.args[1].args[0] == 0: self.fail(expr.token, "Point division by zero is unsupported")
                if self.collection_scope is not None and not self.scalar_collection and expr.args[1].op != "literal": self.fail(expr.token, "Checked dynamic division inside collections requires ordered scalar evaluation")
                expr.type = "point"; return
            self.fail(expr.token, "Unsupported point operation", "GAMAX_TYPE")
        if expr.op == "distance_to": self.fail(expr.token, "distance_to requires points", "GAMAX_TYPE")
        if any(t not in ('int','float','bool') for t in types): self.fail(expr.token, 'Collections must be reduced before arithmetic', 'GAMAX_TYPE')
        if expr.op in ("and", "or", "not_unary"):
            if any(t != "bool" for t in types): self.fail(expr.token, "Boolean operands required", "GAMAX_TYPE")
            expr.type = "bool"
        elif expr.op in ("=", "!="):
            if "float" in types:
                self.fail(expr.token, "GAMA floating equality has special rounding semantics; not lowered in this profile", "GAMAX_UNSUPPORTED")
            if "bool" in types and types[0] != types[1]: self.fail(expr.token, "Incompatible equality operands", "GAMAX_TYPE")
            expr.type = "bool"
        else:
            if "bool" in types: self.fail(expr.token, "Numeric operands required", "GAMAX_TYPE")
            if expr.op in ('div','mod') and types != ['int','int']: self.fail(expr.token, 'Integer div/mod profile requires int operands', 'GAMAX_TYPE')
            if expr.op in ('/','div','mod'):
                denominator = expr.args[1]
                safe = denominator.op == 'literal' and denominator.args[0] != 0
                if denominator.op == 'call:max':
                    safe = any(a.op == 'literal' and a.args[0] > 0 for a in denominator.args)
                if denominator.op == 'literal' and denominator.args[0] == 0:
                    self.fail(expr.token, 'Literal division by zero is unsupported')
                if not safe and self.collection_scope is not None and not self.scalar_collection:
                    self.fail(expr.token, 'Checked dynamic division inside collections requires ordered scalar evaluation')
            expr.type = "bool" if expr.op in ("<", ">", "<=", ">=") else ("float" if "float" in types or expr.op == '/' else "int")


