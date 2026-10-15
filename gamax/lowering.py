"""Standalone Python/JAX generation from ordered typed IR."""
from __future__ import annotations
from pathlib import Path
import copy
import json
import math
import re
import struct
from .ir import Token, CompileError, Expr, Field, Statement, Behavior, Action, Scope, ModelIR
from .lexing import TOKEN, PRECEDENCE, FUNCTIONS, RESERVED
from .randomness import MERSENNE_SOURCE, MERSENNE_OPTIMIZED_SOURCE, MERSENNE_CACHED_SOURCE, MERSENNE_BATCH_SOURCE, mersenne_initial_state
from .spatial import GRID_SOURCE,GRID_CACHED_SOURCE,GRID_BATCH_INITIAL_SOURCE,GRID_REFERENCE_SOURCE,implicit_grid_shuffle_draws,grid_neighbor_geometry
from .spatial_index import QUADTREE_SOURCE,QUADTREE_OPTIMIZED_SOURCE,QUADTREE_COUNTED_SOURCE,CIRCLE_QUERY_SOURCE,RECT_QUERY_SOURCE,RECT_UNORDERED_SOURCE,DOUBLE_SHUFFLE_SOURCE,INCREMENTAL_ORDER_SOURCE,UNORDERED_DISTANCE_SOURCE,CONTACT_CACHE_SOURCE,CONTACT_ROW_CACHE_SOURCE
from .native_math import ATAN_SOURCE,TRIG_SOURCE
from .hotspot_math import HOTSPOT_TRIG_SOURCE
from .host_reset import HOST_RESET_SOURCE
from .output_format import OUTPUT_FORMAT_SOURCE
from .spatial_index import DOUBLE_SHUFFLE_BATCH_SOURCE
from .diffusion import GRADIENT_SOURCE
import inspect

class Generator:
    def computed_value(self,owner,field,index):
        saved=self.agent_index,self.each_scope,self.each_indices,self.locals
        self.agent_index=index;self.each_scope=None;self.each_indices=None;self.locals={}
        value=self.expression(field.getter,owner)
        self.agent_index,self.each_scope,self.each_indices,self.locals=saved
        return value

    def __init__(self, ir, reverse_agents=False):
        self.ir = ir
        self.scopes = {s.name:s for s in ir.scopes}
        self.reverse_agents = reverse_agents
        self.agent_index = None
        self.locals = {}
        self.local_capacities = {}
        self.eval_indent = "    "
        self.active_mask = "jnp.asarray(True)"
        self.each_scope = None
        self.each_indices = None
        self.caller_index = None
        targets=set();self.circle_queries=False;self.rectangle_queries=False
        def spatial_queries(value):
            if isinstance(value,Expr):
                if value.op=='at_distance' or value.op=='overlapping' and value.metadata.get('circle_query'):targets.add(value.args[0].type.split(':')[1])
                if value.metadata.get('circle_query'):self.circle_queries=True
                if value.metadata.get('rectangle_query'):targets.add(value.args[0].type.split(':')[1]);self.rectangle_queries=True
                for arg in value.args:spatial_queries(arg)
                spatial_queries(value.metadata)
            elif isinstance(value,(list,tuple)):
                for item in value:spatial_queries(item)
            elif isinstance(value,dict):
                for item in value.values():spatial_queries(item)
            elif hasattr(value,'__dataclass_fields__'):
                for key in value.__dataclass_fields__:spatial_queries(getattr(value,key))
        spatial_queries(ir);self.spatial_targets=sorted(targets)
        def nodes(value):
            yield value
            if isinstance(value,Expr):children=[*value.args,value.metadata]
            elif isinstance(value,(list,tuple)):children=value
            elif isinstance(value,dict):children=value.values()
            elif hasattr(value,'__dataclass_fields__'):children=[getattr(value,key) for key in value.__dataclass_fields__]
            else:children=[]
            for child in children:yield from nodes(child)
        all_nodes=list(nodes(ir.scopes))
        self.grid_reference_pairs={}
        self.grid_reference_sites=0
        if getattr(ir,'optimize_pure_mapper',False):
            for node in all_nodes:
                if not isinstance(node,Statement) or not node.metadata.get('stream_integer_accumulation'):continue
                inner=node.value.args[1]
                if inner.op!='overlapping' or inner.metadata.get('circle_query'):continue
                owner=inner.args[0].type.split(':')[1];grid_owner=inner.args[1].type.split(':')[1]
                if not self.scopes[grid_owner].grid:continue
                references=sorted(field.name for field in self.scopes[owner].fields if field.type=='agent:'+grid_owner and field.getter is None)
                if references:self.grid_reference_pairs[owner,grid_owner]=references[0]
        self.grid_geometry={}
        self.broadcast_grid_sites=0
        self.pure_mapper_sites=0
        self.independent_grid_sites=0
        self.unordered_rectangle_sites=0
        self.first_match_sites=0
        self.single_return_sites=0
        self.grid_random_sites=0
        self.grid_neighborhood_sites=0
        self.random_pair_sites=0
        self.deferred_query_reuse_sites=0
        self.pure_grid_initializer_sites=0
        self.pure_grid_initializer_helpers=[]
        self.pure_grid_initializer_names=[]
        if getattr(ir,'optimize_grid_geometry',False):
            requested=set()
            for node in all_nodes:
                if isinstance(node,Expr) and node.op=='neighbors_at':
                    grid=self.scopes[node.args[0].type.split(':')[1]].grid
                    requested.add((grid['width'],grid['height'],node.metadata['max_radius'],grid['neighbors']==8,grid['torus']))
            used=0
            for key in sorted(requested):
                width,height,bound,moore,torus=key;entries=width*height*((2*bound+1)**2-1 if moore else 2*bound*(bound+1))
                if used+entries<=500000:self.grid_geometry[key]=len(self.grid_geometry);used+=entries
        self.native_trig=getattr(ir,'optimize_native_trig',False) and any(isinstance(n,Statement) and n.kind=='skill_move' for n in all_nodes)
        self.contact_cache_targets=set()
        if getattr(ir,'optimize_contact_cache',False):
            for scope in ir.scopes:
                for node in nodes(scope):
                    if not isinstance(node,Expr) or node.op not in ('count','call:length'):continue
                    query=self.unordered_distance_query(node,scope.name)
                    if query is not None and self.cacheable_distance_query(query,scope.name):self.contact_cache_targets.add(scope.name)
        self.strict_constant_division=any(isinstance(n,Expr) and n.op=='/' and len(n.args)==2 and n.args[1].op=='literal' and n.args[1].type in ('int','float') and n.args[1].args[0]!=0 and math.frexp(abs(n.args[1].args[0]))[0]!=0.5 for n in all_nodes)
        self.has_clock=any(isinstance(n,Expr) and n.op=='cycle' for n in all_nodes) or any(scope.grid and scope.grid['frequency']>1 for scope in ir.scopes)
        self.has_pause=any(isinstance(n,Statement) and n.kind=='pause' for n in all_nodes)
        self.diffusions=[n for n in all_nodes if isinstance(n,Statement) and n.kind=='diffuse']
        for i,statement in enumerate(self.diffusions):statement.metadata['diffusion_id']=i
        self.save_statements={n.metadata['output_id']:n for n in all_nodes if isinstance(n,Statement) and n.kind=='save'}
        if self.spatial_targets:
            def static_world(items,owner):
                for item in items:
                    if owner=='global' and item.kind=='assign' and item.target=='shape':raise CompileError(item.token.source or '<IR>',item.token,'Ordered spatial queries require an immutable world shape','GAMAX_SPATIAL')
                    static_world(item.body,owner);static_world(item.alternative,owner)
            for scope in ir.scopes:
                static_world(scope.init,scope.name)
                for behavior in scope.behaviors:static_world(behavior.body,scope.name)
        self.lines = ["# Automatically generated by GAMAX. Do not edit model dynamics.",
                      "import math", "import struct", "import jax", "jax.config.update('jax_enable_x64', True)",
                      "import jax.numpy as jnp", "DTYPES = {'int': jnp.int32, 'bool': jnp.bool_, 'float': jnp.float64, 'point': jnp.float64, 'geometry': jnp.float64, 'rgb': jnp.int32}", ""]
        asset_fields={scope.name+'.'+f.name:f.type for scope in ir.scopes for f in scope.fields if f.type in ('image_file','values:image_file')}
        fsm_fields={scope.name+'.state':[state.name for state in scope.fsm_states] for scope in ir.scopes if scope.fsm_states}
        if fsm_fields:self.lines.extend(["DTYPES['fsm_state']=jnp.int32","FSM_FIELDS="+repr(fsm_fields)])
        if any(f.type=='polygon' for scope in ir.scopes for f in scope.fields):self.lines.append("DTYPES['polygon'] = jnp.float64")
        if any(scope.skills for scope in ir.scopes):self.lines.append(ATAN_SOURCE)
        if self.native_trig:self.lines.append(HOTSPOT_TRIG_SOURCE)
        self.raster_tables={}
        if getattr(ir,'optimize_raster_tables',False):
            for number,record in enumerate(ir.assets):
                if 'raster_pixels' in record:
                    self.raster_tables[record['key']]=number
                    self.lines.append(f"RASTER_PIXELS_{number}=jnp.asarray({record['raster_pixels']!r},jnp.int32)")
        if self.diffusions:self.lines.append(GRADIENT_SOURCE)
        if asset_fields:
            self.lines += ["from pathlib import Path", "import hashlib", "DTYPES['image_file'] = jnp.int32",
                           'ASSET_FIELDS = '+repr(asset_fields),'ASSET_KEYS = '+repr([record['key'] for record in ir.assets]),
                           'ASSET_FILES = '+repr({record['asset_bundle_path']:record['sha256'] for record in ir.assets}),
                           'def verify_assets():',"    root = Path(__file__).resolve().parent",
                           '    for relative, expected in ASSET_FILES.items():',
                           "        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected: raise ValueError('Changed asset: '+relative)",'']
        reference_types = {f.type for scope in ir.scopes for f in scope.fields if f.type.startswith(('agent:', 'agents:'))}
        reference_types.update('agent:'+scope.name for scope in ir.scopes[1:])
        matrix_types={f.type for scope in ir.scopes for f in scope.fields if f.type.startswith("matrix:")}
        if matrix_types:self.lines.append("MATRIX_FIELDS="+repr([scope.name+"."+f.name for scope in ir.scopes for f in scope.fields if f.type.startswith("matrix:")]))
        if matrix_types:self.lines.append("DTYPES.update({"+",".join(repr(t)+":DTYPES["+repr(t.split(":")[1])+"]" for t in sorted(matrix_types))+"})")
        if reference_types:self.lines.append("DTYPES.update("+repr({t:None for t in sorted(reference_types)}).replace('None','jnp.int32')+")")
        value_types={f.type for scope in ir.scopes for f in scope.fields if f.type.startswith('values:')}
        if value_types:self.lines.append('DTYPES.update({'+','.join(repr(t)+':DTYPES['+repr(t.split(':')[1])+']' for t in sorted(value_types))+'})')
        self.lines += [
            "JIT_OPTIONS = " + repr({'xla_cpu_enable_fast_math': False, 'xla_disable_hlo_passes': '' if getattr(ir,'optimize_algebra',False) else 'algsimp', **({'xla_cpu_max_isa':'AVX'} if getattr(ir,'optimize_no_fma_isa',False) else {}), **({'xla_backend_optimization_level':2} if getattr(ir,'optimize_quick_compile',False) else {})} if getattr(ir,'optimize_fusion',False) else {'xla_cpu_enable_fast_math': False, **({'xla_disable_hlo_passes': 'fusion,algsimp' if any(s.skills for s in ir.scopes) or self.strict_constant_division else 'fusion'} if any(f.type in ('float', 'point', 'geometry') for s in ir.scopes for f in s.fields) else {})}),
            '',
            'def ordered_sum(values, selected, dtype):',
            '    values = values.astype(dtype)',
            '    if values.shape[0] == 0: return jnp.zeros(values.shape[1:], dtype=dtype)',
            *(['    if jnp.issubdtype(dtype,jnp.integer):', '        mask=selected.reshape(selected.shape+(1,)*(values.ndim-selected.ndim))', '        return jnp.sum(jnp.where(mask,values,jnp.zeros_like(values)),axis=0,dtype=dtype)'] if getattr(ir,'optimize_integer_sum',False) else []),
            *(['    indices=jnp.nonzero(selected,size=values.shape[0],fill_value=0)[0]', '    count=jnp.sum(selected,dtype=jnp.int32)', '    def add_selected(i,total):', '        return jax.lax.optimization_barrier(total+values[indices[i]])', '    return jax.lax.fori_loop(0,count,add_selected,jnp.zeros(values.shape[1:],dtype=dtype))'] if getattr(ir,'optimize_sparse_sum',False) else []),
            '    def accumulate(i, total):',
            '        return jax.lax.cond(selected[i], lambda t: jax.lax.optimization_barrier(t + values[i]), lambda t: t, total)',
            '    return jax.lax.fori_loop(0, values.shape[0], accumulate, jnp.zeros(values.shape[1:], dtype=dtype))',
            '',
        ]
        if self.grid_reference_pairs:self.lines.append(GRID_REFERENCE_SOURCE)
        if ir.random_seed is not None:
            if self.spatial_targets:
                self.lines.append(QUADTREE_COUNTED_SOURCE if getattr(ir,'optimize_spatial_order',False) else QUADTREE_SOURCE)
                if getattr(ir,'optimize_spatial_order',False):self.lines.append(INCREMENTAL_ORDER_SOURCE)
                if getattr(ir,'optimize_unordered_distance',False):self.lines.append(UNORDERED_DISTANCE_SOURCE)
                if self.contact_cache_targets:self.lines.append(CONTACT_ROW_CACHE_SOURCE)
            if getattr(ir,'optimize_double_shuffle',False) and (self.circle_queries or self.rectangle_queries):self.lines.append(DOUBLE_SHUFFLE_BATCH_SOURCE if getattr(ir,'optimize_prefetch_shuffles',False) else DOUBLE_SHUFFLE_SOURCE)
            def envelope_source(source):
                if not getattr(ir,'optimize_double_shuffle',False):return source
                source=source.replace('rng,indices,present=collection_shuffle_forward(rng,indices,mask[indices],uniform)\n    rng,indices,present=collection_shuffle_forward(rng,indices,present,uniform)','rng,indices,present=collection_shuffle_twice_forward(rng,indices,mask[indices],uniform)')
                return source.replace('rng,indices,present=collection_shuffle_forward(rng,indices,mask[indices],uniform)\n    # Native envelope search shuffles twice before the polygon predicate.\n    rng,indices,present=collection_shuffle_forward(rng,indices,present,uniform)','rng,indices,present=collection_shuffle_twice_forward(rng,indices,mask[indices],uniform)')
            if self.circle_queries:self.lines.append(envelope_source(CIRCLE_QUERY_SOURCE))
            if self.rectangle_queries:
                self.lines.append(envelope_source(RECT_QUERY_SOURCE))
                if getattr(ir,'optimize_unordered_rectangle',False):self.lines.append(RECT_UNORDERED_SOURCE)
            self.lines += [
                "def java_uniform(state):",
                "    mask = jnp.uint64((1 << 48) - 1)",
                "    first = (state * jnp.uint64(0x5DEECE66D) + jnp.uint64(11)) & mask",
                "    second = (first * jnp.uint64(0x5DEECE66D) + jnp.uint64(11)) & mask",
                "    bits = ((first >> jnp.uint64(22)) << jnp.uint64(27)) + (second >> jnp.uint64(21))",
                "    return second, bits.astype(jnp.float64) / jnp.asarray(1 << 53, jnp.float64)", "",
            ]
            if ir.random_generator == "mersenne":
                self.lines += inspect.getsource(mersenne_initial_state).splitlines() + [""] + (MERSENNE_CACHED_SOURCE if getattr(self.ir,'optimize_mersenne',False) else MERSENNE_SOURCE).splitlines() + [""]
                if getattr(ir,'optimize_grid_random_fields',False):self.lines.append(MERSENNE_BATCH_SOURCE)
        if any(scope.grid for scope in ir.scopes):
            self.lines += GRID_SOURCE.replace("uniform=mersenne_uniform", "uniform=" + ("mersenne_uniform" if ir.random_generator == "mersenne" else "java_uniform")).splitlines()+[""]
            if self.grid_geometry:
                self.lines.append(GRID_CACHED_SOURCE)
                if getattr(ir,'optimize_grid_random_fields',False):self.lines.append(GRID_BATCH_INITIAL_SOURCE)
                for key,number in self.grid_geometry.items():
                    width,height,bound,moore,torus=key
                    values,radii=grid_neighbor_geometry(width,height,bound,moore=moore,torus=torus)
                    self.lines.append(f'GRID_GEOMETRY_{number}=jnp.asarray({values!r},jnp.int32)')
                    self.lines.append(f'GRID_GEOMETRY_RADII_{number}=jnp.asarray({radii!r},jnp.int32)')
        self.output_records = []
        self.map = []
        self.counter = 0
        self.scopes = {s.name: s for s in ir.scopes}
        for owner in self.spatial_targets:
            if self.capacity(owner)>0 and any(f.type=='polygon' for f in self.scopes[owner].fields):
                token=next(f.token for f in self.scopes[owner].fields if f.type=='polygon')
                raise CompileError(token.source or '<IR>',token,'Ordered queries over nonpoint agent shapes require a native polygon intersection contract','GAMAX_SPATIAL')
        mutable=[f for f in ir.scopes[0].fields if f.list_mutable]
        self.list_alias_fields=[f for f in mutable if sum(g.type==f.type for g in mutable)>1]

    def emit(self, text, token=None):
        self.lines.append(text)
        if token: self.map.append(dict(generated_line=len(self.lines), source_line=token.line, source_column=token.column, source_file=token.source))

    def capacity(self, scope):
        return self.ir.capacities.get(scope,self.ir.populations[scope])

    def reserved(self, scope):
        return scope in self.ir.capacities and not self.scopes[scope].grid

    def field_capacity(self, field):
        return self.capacity(field.type.split(':')[1]) if field.list_mutable or field.getter is not None and field.type.startswith('agents:') else field.capacity

    def record_spatial_position(self,owner,index,changed,indent,mask):
        if any(f.type=='polygon' for f in self.scopes[owner].fields):
            center='$shape_center.'+owner
            self.emit(f"{indent}s[{(owner+'.shape')!r}] = s[{(owner+'.shape')!r}].at[{index}].set(jnp.where({mask},s[{(owner+'.shape')!r}][{index}] + (s[{(owner+'.location')!r}][{index}]-s[{center!r}][{index}]),s[{(owner+'.shape')!r}][{index}]))")
            self.emit(f"{indent}s[{center!r}] = s[{center!r}].at[{index}].set(jnp.where({mask},s[{(owner+'.location')!r}][{index}],s[{center!r}][{index}]))")
        if owner not in self.spatial_targets:return
        prefix='$spatial.'+owner+'.'
        keys=[prefix+suffix for suffix in ('split','leaf','order','tick')]
        if getattr(self.ir,'optimize_spatial_order',False):keys.append(prefix+'counts')
        values='('+','.join(f's[{key!r}]' for key in keys)+')'
        if any(f.name=='shape' and f.type=='geometry' for f in self.scopes['global'].fields):
            width="jnp.max(s['global.shape'][:,0])-jnp.min(s['global.shape'][:,0])"
            height="jnp.max(s['global.shape'][:,1])-jnp.min(s['global.shape'][:,1])"
        else:width=height='jnp.asarray(100.,jnp.float64)'
        if getattr(self.ir,'optimize_spatial_order',False):self.emit(f"{indent}previous_leaf = s[{keys[1]!r}]")
        self.emit(f"{indent}{', '.join('s['+repr(key)+']' for key in keys)} = jax.lax.cond(({mask}) & (({changed}) | (s[{keys[1]!r}][{index}]<0)),lambda args:quadtree_insert(*args,s[{(owner+'.location')!r}],{index},{width},{height}),lambda args:args,{values})")
        # Cache DFS/leaf order at mutations; read-only queries reuse it. The
        # sorted order is independent of query bounds and the allowed collection.
        sort=f"quadtree_update_order(s[{(prefix+'sorted')!r}],previous_leaf,s[{keys[1]!r}],s[{keys[2]!r}],{index})" if getattr(self.ir,'optimize_spatial_order',False) else f"jnp.lexsort((s[{keys[2]!r}],QUADTREE_DFS_KEYS[jnp.maximum(s[{keys[1]!r}],0)])).astype(jnp.int32)"
        self.emit(f"{indent}s[{(prefix+'sorted')!r}] = jax.lax.cond(({mask}) & ({changed}),lambda _:{sort},lambda _:s[{(prefix+'sorted')!r}],operand=None)")
        self.update_contacts(owner,index,f'({mask}) & (({changed}) | (previous_leaf[{index}]<0))',indent)

    def unordered_distance_query(self,expr,scope):
        if not self.locals and expr.op in ('count','call:length'):
            collection=expr.args[0]
            query=collection
            if collection.op=='name':
                field=next((f for f in self.scopes[scope].fields if f.name==collection.args[0]),None)
                if field is not None and field.getter is not None:query=field.getter
            def pure_predicate(value):
                if value.op not in ('literal','name','each','=','!=','<','<=','>','>=','and','or','not'):return False
                if value.op in ('name','each'):
                    owner=value.metadata.get('owner_scope',scope)
                    fields=self.scopes[owner].fields
                    field=next((f for f in fields if f.name==value.args[0]),None)
                    if field is None and value.op=='name':field=next((f for f in self.scopes['global'].fields if f.name==value.args[0]),None)
                    if field is None or field.getter is not None:return False
                return all(pure_predicate(arg) for arg in value.args if isinstance(arg,Expr))
            if query.op=='at_distance' and (expr.op=='call:length' or pure_predicate(expr.args[1])):return query
        return None

    def cacheable_distance_query(self,query,scope):
        if query.args[0].op!='name' or query.args[0].args[0]!=scope or not 512<=self.capacity(scope)<=4096:return False
        radius=query.args[1]
        if radius.op=='literal':return radius.type in ('float','int')
        if radius.op!='name' or any(f.name==radius.args[0] for f in self.scopes[scope].fields):return False
        field=next((f for f in self.scopes['global'].fields if f.name==radius.args[0]),None)
        return field is not None and field.getter is None and field.type in ('int','float')

    def contact_cache_keys(self,owner):
        return ['$contacts.'+owner+'.'+name for name in ('boxes','nearby','box_counts','counts')]

    def discard_only_mapper(self,expr,scope,each_owner,seen=None):
        """Conservative expression effect check; getters may discard shuffles."""
        seen=set() if seen is None else seen
        query=self.unordered_distance_query(expr,scope)
        if query is not None:
            return query.args[0].op=='name' and query.args[0].args[0]==scope and self.cacheable_distance_query(query,scope)
        if expr.op in ('literal','each_ref'):return True
        if expr.op in ('name','each') or expr.op.startswith('attribute:'):
            if expr.op.startswith('attribute:'):
                if expr.args[0].op!='each_ref':return False
                owner=expr.args[0].type.split(':')[1];name=expr.op.split(':')[1]
            else:
                name=expr.args[0]
                if name=='self':return True
                owner=expr.metadata.get('owner_scope',each_owner if expr.op=='each' else scope)
                if not any(f.name==name for f in self.scopes[owner].fields):owner='global'
            field=next((f for f in self.scopes[owner].fields if f.name==name),None)
            if field is None:return False
            if field.getter is None:return field.type in ('int','bool','float','point','rgb')
            key=(owner,name)
            if key in seen:return False
            return self.discard_only_mapper(field.getter,owner,owner,seen|{key})
        if expr.op=='/':
            if expr.args[1].op!='literal' or expr.args[1].args[0]==0:return False
        elif expr.op not in ('+','-','*','negate','=','!=','<','<=','>','>=','and','or','not'):return False
        return all(self.discard_only_mapper(arg,scope,each_owner,seen) for arg in expr.args if isinstance(arg,Expr))

    def independent_cell_expression(self,expr,scope):
        if expr.op=='literal':return True
        if expr.op=='name':
            field=next((f for owner in (scope,'global') for f in self.scopes[owner].fields if f.name==expr.args[0]),None)
            return field is not None and field.getter is None and field.type in ('int','float','bool','point','rgb')
        if expr.op=='/':
            if expr.args[1].op!='literal' or expr.args[1].args[0]==0:return False
        elif expr.op not in ('+','-','*','negate','=','!=','<','<=','>','>=','and','or','not','point','call:rgb','call:int','call:float','call:abs'):return False
        return all(self.independent_cell_expression(arg,scope) for arg in expr.args if isinstance(arg,Expr))

    def pure_grid_initializer(self,expr,scope):
        if expr.op=='grid_location':return True
        if expr.op=='grid_distance' and expr.metadata['grid']==scope:
            return all(arg.op=='name' and arg.args[0]=='self' or arg.type=='point' and self.pure_grid_initializer(arg,scope) for arg in expr.args)
        if expr.op in ('literal','name'):return self.independent_cell_expression(expr,scope)
        if expr.op=='/':
            if expr.args[1].op!='literal' or expr.args[1].args[0]==0:return False
        elif expr.op not in ('conditional','+','-','*','negate','=','!=','<','<=','>','>=','and','or','not','point','call:rgb','call:int','call:float','call:abs'):return False
        return all(self.pure_grid_initializer(arg,scope) for arg in expr.args if isinstance(arg,Expr))

    def contact_alive(self,owner):
        return f"s[{('$present.'+owner)!r}]" if self.reserved(owner) else f'jnp.ones(({self.capacity(owner)},),jnp.bool_)'

    def update_contacts(self,owner,index,mask,indent):
        if owner not in self.contact_cache_targets or getattr(self,'initializing',False):return
        key='$contacts.'+owner+'.epoch'
        self.emit(f"{indent}s[{key!r}] = s[{key!r}] + jnp.where({mask},jnp.asarray(1,jnp.int64),jnp.asarray(0,jnp.int64))")

    def expression(self, expr, scope):
        if getattr(self.ir,'optimize_unordered_rectangle',False) and expr.op in ('call:empty','call:length') and not expr.metadata.get('unordered_rectangle_lowered'):
            query=expr.args[0]
            if query.op=='inside' and query.metadata.get('rectangle_query'):
                query=copy.deepcopy(query);query.metadata['unordered_rectangle_consumer']=True
                replacement=copy.copy(expr);replacement.args=[query,*expr.args[1:]];replacement.metadata={**expr.metadata,'unordered_rectangle_lowered':True}
                self.unordered_rectangle_sites+=1
                return self.expression(replacement,scope)
        if getattr(self.ir,'optimize_unordered_distance',False) and not expr.metadata.get('unordered_consumer_lowered'):
            query=self.unordered_distance_query(expr,scope)
            if query is not None:
                query=copy.deepcopy(query);query.metadata['unordered_consumer']=True
                query.metadata['cache_contacts']=not getattr(self,'initializing',False) and scope in self.contact_cache_targets and self.cacheable_distance_query(query,scope)
                replacement=copy.copy(expr);replacement.args=[query,*expr.args[1:]]
                replacement.metadata={**expr.metadata,'unordered_consumer_lowered':True}
                return self.expression(replacement,scope)
        if expr.op=='point_nil':return 'jnp.full((3,),jnp.nan,jnp.float64)'
        if expr.op=='sort_by':
            collection=self.expression(expr.args[0],scope);cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
            if not cap:return collection
            self.counter+=1;prefix=f'sorted_collection_{self.counter}';base=self.eval_indent
            self.emit(f'{base}{prefix}_values,{prefix}_mask={collection}')
            self.emit(f'{base}def {prefix}_score(i,carry):')
            self.emit(f'{base}    s,keys=carry')
            saved=self.each_scope,self.each_indices,self.eval_indent,self.active_mask
            self.each_scope=expr.args[0].type.split(':')[1];self.each_indices=prefix+'_values[i]';self.eval_indent=base+'    ';self.active_mask=f'({saved[3]}) & {prefix}_mask[i]'
            score=self.expression(expr.args[1],scope)
            self.emit(f'{base}    return s,keys.at[i].set(jnp.where({prefix}_mask[i],({score}).astype(jnp.float64),jnp.inf))')
            self.each_scope,self.each_indices,self.eval_indent,self.active_mask=saved
            self.emit(f'{base}s,{prefix}_keys=jax.lax.fori_loop(0,{cap},{prefix}_score,(s,jnp.full(({cap},),jnp.inf,jnp.float64)))')
            self.emit(f'{base}{prefix}_indices=jnp.lexsort((jnp.arange({cap}),~jnp.signbit({prefix}_keys),{prefix}_keys,~{prefix}_mask))')
            return f'({prefix}_values[{prefix}_indices],{prefix}_mask[{prefix}_indices])'
        if expr.op.startswith('grid_at:'):
            owner=expr.op.split(':')[1];grid=self.scopes[owner].grid;point=self.expression(expr.args[0],scope)
            self.counter+=1;prefix=f'grid_place_{self.counter}';base=self.eval_indent
            world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
            self.emit(f'{base}{prefix}_point={point}')
            self.emit(f'{base}{prefix}_world={world}')
            self.emit(f'{base}{prefix}_dimensions=jnp.asarray([{grid["width"]},{grid["height"]}],jnp.int32)')
            self.emit(f'{base}{prefix}_xy=jnp.trunc({prefix}_point[:2]/({prefix}_world[:2]/{prefix}_dimensions.astype(jnp.float64))).astype(jnp.int32)')
            self.emit(f'{base}{prefix}_xy=jnp.where({prefix}_point[:2]=={prefix}_world[:2],{prefix}_dimensions-1,{prefix}_xy)')
            if grid['torus']:
                self.emit(f'{base}{prefix}_xy=jnp.where({prefix}_xy<0,{prefix}_xy+{grid["width"]},{prefix}_xy)')
                self.emit(f'{base}{prefix}_xy=jnp.where({prefix}_xy<0,-((-{prefix}_xy)%{prefix}_dimensions),{prefix}_xy%{prefix}_dimensions)')
            self.emit(f'{base}{prefix}_valid=jnp.all(jnp.isfinite({prefix}_point)) & jnp.all({prefix}_xy>=0) & jnp.all({prefix}_xy<{prefix}_dimensions)')
            return f'jnp.where({prefix}_valid,{prefix}_xy[1]*{grid["width"]}+{prefix}_xy[0],jnp.asarray(-1,jnp.int32))'
        if expr.op in ('point_is_nil','point_is_not_nil'):
            value=self.expression(expr.args[0],scope);isnil=f'jnp.all(jnp.isnan({value}),axis=-1)'
            return isnil if expr.op=='point_is_nil' else f'~({isnil})'
        if expr.op=='grid_distance':
            owner=expr.metadata['grid'];grid=self.scopes[owner].grid
            world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
            self.counter+=1;prefix=f'grid_distance_{self.counter}';base=self.eval_indent
            coordinates=[]
            for i,arg in enumerate(expr.args):
                value=self.expression(arg,scope)
                if arg.type.startswith('agent:'):
                    self.emit(f'{base}{prefix}_{i}=jnp.stack([({value})%{grid["width"]},({value})//{grid["width"]}])')
                else:self.emit(f'{base}{prefix}_{i}=jnp.floor(({value})[:2]/({world})[:2]*jnp.asarray([{grid["width"]},{grid["height"]}])).astype(jnp.int32)')
                coordinates.append(prefix+'_'+str(i))
            self.emit(f'{base}{prefix}_delta=jnp.abs({coordinates[0]}-{coordinates[1]})')
            return f'jnp.{"sum" if grid["neighbors"]==4 else "max"}({prefix}_delta).astype(jnp.float64)'
        if expr.op=='raster_matrix':
            return f"jnp.asarray({expr.metadata['pixels']!r},jnp.int32).reshape(({expr.metadata['height']},{expr.metadata['width']}))"
        if expr.op.startswith('matrix_cast:'):return f"({self.expression(expr.args[0],scope)}).astype(DTYPES[{expr.type!r}])"
        if expr.op=='matrix_at':
            matrix=self.expression(expr.args[0],scope);x=self.expression(expr.args[1],scope);y=self.expression(expr.args[2],scope)
            self.counter+=1;prefix=f'matrix_index_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}_matrix={matrix}')
            self.emit(f'{self.eval_indent}{prefix}_x,{prefix}_y={x},{y}')
            self.emit(f"{self.eval_indent}s['$error']=jnp.where(({self.active_mask}) & (({prefix}_x<0)|({prefix}_y<0)|({prefix}_x>={prefix}_matrix.shape[1])|({prefix}_y>={prefix}_matrix.shape[0])) & (s['$error']==0),jnp.asarray(14,jnp.int32),s['$error'])")
            return f'{prefix}_matrix[jnp.clip({prefix}_y,0,{prefix}_matrix.shape[0]-1),jnp.clip({prefix}_x,0,{prefix}_matrix.shape[1]-1)]'
        if expr.op=="cycle":return "s['$cycle']"
        if expr.op in ('with_max_of','with_min_of'):
            container=self.expression(expr.args[0],scope);cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
            if not cap:return 'jnp.asarray(-1,jnp.int32)'
            self.counter+=1;prefix=f'extremum_{self.counter}';base=self.eval_indent;active=self.active_mask
            self.emit(f'{base}{prefix}_values,{prefix}_mask={container}')
            self.emit(f'{base}def {prefix}_select(i,state):')
            self.emit(f'{base}    s,result,best=state')
            saved=self.each_scope,self.each_indices,self.eval_indent,self.active_mask
            self.each_scope=expr.args[0].type.split(':')[1];self.each_indices=prefix+'_values[i]'
            self.eval_indent=base+'    ';self.active_mask=f'({active}) & {prefix}_mask[i]'
            score=self.expression(expr.args[1],scope)
            self.emit(f'{base}    score=({score}).astype(jnp.float64)')
            operator='>' if expr.op=='with_max_of' else '<'
            self.emit(f'{base}    replace={prefix}_mask[i] & ((result<0) | (score {operator} best))')
            self.emit(f'{base}    return s,jnp.where(replace,{prefix}_values[i],result),jnp.where(replace,score,best)')
            self.each_scope,self.each_indices,self.eval_indent,self.active_mask=saved
            self.emit(f'{base}s,{prefix}_result,_=jax.lax.fori_loop(0,{cap},{prefix}_select,(s,jnp.asarray(-1,jnp.int32),jnp.asarray(0.,jnp.float64)))')
            return prefix+'_result'
        if expr.op=='function_value':
            body=expr.metadata['body']
            if getattr(self.ir,'optimize_single_return',False) and len(body)==1 and body[0].kind=='return':
                kind=expr.metadata['return_type'];value=self.expression(body[0].value,scope)
                initial='[0.,0.,0.]' if kind=='point' else '-1' if kind.startswith('agent:') else 'False' if kind=='bool' else '0'
                self.single_return_sites+=1
                return f'jnp.where({self.active_mask},({value}).astype(DTYPES[{kind!r}]),jnp.asarray({initial},DTYPES[{kind!r}]))'
            self.counter+=1;prefix=f'function_value_{self.counter}'
            value_key='$return.'+prefix;flag_key='$returned.'+prefix
            kind=expr.metadata['return_type'];base=self.eval_indent;active=self.active_mask
            initial='[0.,0.,0.]' if kind=='point' else '-1' if kind.startswith('agent:') else 'False' if kind=='bool' else '0'
            self.emit(f"{base}s[{value_key!r}]=jnp.asarray({initial},DTYPES[{kind!r}])")
            self.emit(f"{base}s[{flag_key!r}]=jnp.asarray(False)")
            saved=getattr(self,'return_context',None)
            self.return_context=(value_key,flag_key,kind)
            self.statements(expr.metadata['body'],scope,active,base)
            self.return_context=saved
            self.emit(f"{base}{prefix}=s.pop({value_key!r})")
            self.emit(f"{base}s.pop({flag_key!r})")
            self.eval_indent,self.active_mask=base,active
            return prefix
        op = expr.op
        if op=='first_with':
            container=self.expression(expr.args[0],scope)
            self.counter+=1;prefix=f'first_matching_{self.counter}';base=self.eval_indent
            cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
            if not cap:return 'jnp.asarray(-1,jnp.int32)'
            self.emit(f'{base}{prefix}_container = {container}')
            saved=self.each_scope,self.each_indices,self.active_mask,self.eval_indent
            self.emit(f'{base}def {prefix}(position,carry):')
            self.emit(f'{base}    s,result = carry');self.emit(f'{base}    s = dict(s)')
            self.each_scope=expr.args[0].type.split(':')[1];self.each_indices=f'{prefix}_container[0][position]'
            self.active_mask=f'({saved[2]}) & (result<0) & {prefix}_container[1][position]'
            self.eval_indent=base+'    '
            value=self.expression(expr.args[1],scope)
            self.emit(f'{base}    result = jnp.where(({self.active_mask}) & ({value}),{self.each_indices},result)')
            self.emit(f'{base}    return s,result')
            self.each_scope,self.each_indices,self.active_mask,self.eval_indent=saved
            if getattr(self.ir,'optimize_first_match',False):
                self.emit(f'{base}_,(s,{prefix}_result)=jax.lax.while_loop(lambda carry:(carry[0]<{cap}) & (carry[1][1]<0),lambda carry:(carry[0]+1,{prefix}(carry[0],carry[1])),(jnp.asarray(0,jnp.int32),(s,jnp.asarray(-1,jnp.int32))))')
                self.first_match_sites+=1
            else:self.emit(f'{base}s,{prefix}_result = jax.lax.fori_loop(0,{cap},{prefix},(s,jnp.asarray(-1,jnp.int32)))')
            return prefix+'_result'
        if op=='inside' and expr.metadata.get('rectangle_query'):
            owner=expr.args[0].type.split(':')[1];target_owner=expr.args[1].type.split(':')[1];cap=self.capacity(owner);target_cap=self.capacity(target_owner)
            container=self.expression(expr.args[0],scope);target=self.expression(expr.args[1],scope)
            if not cap or not target_cap:return '(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))'
            self.counter+=1;prefix=f'rectangle_query_{self.counter}';base=self.eval_indent
            self.emit(f'{base}{prefix}_values,{prefix}_mask={container}')
            self.emit(f'{base}{prefix}_target={target}')
            self.emit(f"{base}s['$error']=jnp.where(({self.active_mask}) & ({prefix}_target>={target_cap}) & (s['$error']==0),jnp.asarray(5,jnp.int32),s['$error'])")
            self.emit(f'{base}{prefix}_allowed=jnp.zeros(({cap},),jnp.bool_).at[jnp.maximum(0,{prefix}_values)].max({prefix}_mask)')
            self.emit(f"{base}{prefix}_source=s[{(target_owner+'.location')!r}][jnp.clip({prefix}_target,0,{target_cap-1})]")
            grid=self.scopes[target_owner].grid
            world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
            half=f'({world})[:2]/jnp.asarray([{grid["width"]},{grid["height"]}],jnp.float64)/2.' if grid else 'jnp.zeros((2,),jnp.float64)'
            self.emit(f'{base}{prefix}_half={half}')
            exclude=prefix+'_target' if owner==target_owner else '-1'
            uniform='mersenne_uniform' if self.ir.random_generator=='mersenne' else 'java_uniform';tree='$spatial.'+owner+'.'
            call=f'point_agents_inside_rectangle(rng,s[{(owner+".location")!r}],s[{(tree+"leaf")!r}],s[{(tree+"sorted")!r}],{prefix}_allowed,{prefix}_source,{prefix}_half,{exclude},{uniform})'
            if expr.metadata.get('unordered_rectangle_consumer'):
                call=f'point_agents_inside_rectangle_unordered(rng,s[{(owner+".location")!r}],s[{(tree+"leaf")!r}],{prefix}_allowed,{prefix}_source,{prefix}_half,{exclude})'
            self.emit(f"{base}s['$rng'],{prefix}_result=jax.lax.cond(({self.active_mask}) & ({prefix}_target>=0) & (s['$error']==0),lambda rng:{call},lambda rng:(rng,(jnp.zeros(({cap},),jnp.int32),jnp.zeros(({cap},),jnp.bool_))),s['$rng'])")
            return prefix+'_result'
        if op in ('inside','overlaps'):
            self.counter+=1;prefix=f'spatial_relation_{self.counter}';base=self.eval_indent
            world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
            self.emit(f'{base}{prefix}_world = {world}')
            if op=='inside':
                owner=expr.args[0].type.split(':')[1];target_owner=expr.args[1].type.split(':')[1]
                container=self.expression(expr.args[0],scope);target=self.expression(expr.args[1],scope)
                cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
                if not cap:return '(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))'
                self.emit(f'{base}{prefix}_values,{prefix}_mask = {container}')
                self.emit(f'{base}{prefix}_target = {target}')
                self.emit(f'{base}{prefix}_positions = s[{(owner+".location")!r}][jnp.maximum(0,{prefix}_values)]')
                self.emit(f'{base}{prefix}_equal = jnp.all({prefix}_positions[:,None,:]=={prefix}_positions[None,:,:],axis=-1) & {prefix}_mask[None,:]')
                self.emit(f'{base}{prefix}_ranks = jnp.arange({cap},dtype=jnp.int32)')
                self.emit(f'{base}{prefix}_last = jnp.max(jnp.where({prefix}_equal,{prefix}_ranks[None,:],0),axis=1)')
                self.emit(f'{base}{prefix}_first = jnp.min(jnp.where({prefix}_equal,{prefix}_ranks[None,:],{cap}),axis=1)')
                self.emit(f'{base}{prefix}_mask = {prefix}_mask & ({prefix}_first=={prefix}_ranks)')
                self.emit(f'{base}{prefix}_values = {prefix}_values[{prefix}_last]')
                if owner==target_owner:self.emit(f'{base}{prefix}_mask = {prefix}_mask & ({prefix}_values!={prefix}_target)')
                self.emit(f'{base}{prefix}_center = s[{(target_owner+".location")!r}][jnp.maximum(0,{prefix}_target)]')
                self.emit(f'{base}{prefix}_delta = {prefix}_positions-{prefix}_center')
                grid=self.scopes[target_owner].grid
                half=f'{prefix}_world[:2]/jnp.asarray([{grid["width"]},{grid["height"]}],jnp.float64)/2.' if grid else 'jnp.zeros((2,),jnp.float64)'
                self.emit(f'{base}{prefix}_half = {half}')
                self.emit(f'{base}{prefix}_shifts = jnp.asarray([[0,0],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]],jnp.float64)*{prefix}_world[:2]')
                self.emit(f'{base}{prefix}_included = jnp.any(jnp.all(jnp.abs({prefix}_delta[:,None,:2]+{prefix}_shifts[None,:,:])<={prefix}_half,axis=-1),axis=-1) & ({prefix}_delta[:,2]==0)')
                return f'({prefix}_values,{prefix}_mask & {prefix}_included & ({prefix}_target>=0))'
            left,right=expr.args
            if not self.scopes[left.type.split(':')[1]].grid:left,right=right,left
            grid_owner=left.type.split(':')[1];point_owner=right.type.split(':')[1];grid=self.scopes[grid_owner].grid
            a,b=self.expression(left,scope),self.expression(right,scope)
            self.emit(f'{base}{prefix}_cell = {a}');self.emit(f'{base}{prefix}_agent = {b}')
            self.emit(f'{base}{prefix}_delta = s[{(point_owner+".location")!r}][jnp.maximum(0,{prefix}_agent)]-s[{(grid_owner+".location")!r}][jnp.maximum(0,{prefix}_cell)]')
            self.emit(f'{base}{prefix}_half = {prefix}_world[:2]/jnp.asarray([{grid["width"]},{grid["height"]}],jnp.float64)/2.')
            # Boolean overlaps compares the actual shapes, independently of
            # topology. Collection inside instead uses the toroidal geometry.
            included=f'jnp.all(jnp.abs({prefix}_delta[:2])<={prefix}_half)'
            return f'({included} & ({prefix}_delta[2]==0) & ({prefix}_cell>=0) & ({prefix}_agent>=0))'
        if op=='agent_shape':return f"jnp.broadcast_to(s[{(scope+'.location')!r}][{self.agent_index}],(4,3))"
        if op=='at_distance' or op=='overlapping' and expr.metadata.get('circle_query'):
            owner=expr.type.split(':')[1];cap=self.capacity(owner)
            container=self.expression(expr.args[0],scope);radius=self.expression(expr.args[1].args[0] if expr.metadata.get('circle_query') else expr.args[1],scope)
            self.counter+=1;name=f'distance_query_{self.counter}'
            self.emit(f'{self.eval_indent}{name}_values, {name}_mask = {container}')
            self.emit(f'{self.eval_indent}{name}_radius = ({radius}).astype(jnp.float64)')
            self.emit(f"{self.eval_indent}s['$error'] = jnp.where(({self.active_mask}) & (({name}_radius<0) | ~jnp.isfinite({name}_radius)) & (s['$error']==0),jnp.asarray(10,jnp.int32),s['$error'])")
            if not cap:return '(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))'
            self.emit(f'{self.eval_indent}{name}_allowed = jnp.zeros(({cap},),jnp.bool_).at[jnp.maximum(0,{name}_values)].max({name}_mask)')
            source=f's[{(scope+".location")!r}][{self.agent_index}]'
            exclude=self.agent_index if scope==owner else '-1'
            uniform='mersenne_uniform' if self.ir.random_generator=='mersenne' else 'java_uniform'
            prefix='$spatial.'+owner+'.'
            call=f'point_neighbors_at_distance(rng,s[{(owner+".location")!r}],s[{(prefix+"leaf")!r}],s[{(prefix+"sorted")!r}],{name}_allowed,{source},{exclude},{name}_radius,{uniform})'
            if expr.metadata.get('unordered_consumer'):
                call=f'point_neighbors_unordered(rng,s[{(owner+".location")!r}],s[{(prefix+"leaf")!r}],{name}_allowed,{source},{exclude},{name}_radius)'
            if expr.metadata.get('circle_query'):
                call=f'point_neighbors_in_circle(rng,s[{(owner+".location")!r}],s[{(prefix+"leaf")!r}],s[{(prefix+"sorted")!r}],{name}_allowed,{source},{name}_radius,{uniform})'
            if getattr(self,'deferred_mapper_rng',False) and expr.metadata.get('unordered_consumer'):
                base=self.eval_indent
                key=(owner,source,exclude,repr(expr.args[1])) if base==self.deferred_query_base and self.cacheable_distance_query(expr,scope) else None
                previous=self.deferred_query_cache.get(key) if key is not None else None
                if previous is not None:
                    for suffix in ('nearby','box_count','near_count'):self.emit(f'{base}{name}_{suffix}={previous}_{suffix}')
                    self.deferred_query_reuse_sites+=1
                else:
                    self.emit(f"{base}{name}_delta=s[{(owner+'.location')!r}]-{source}")
                    self.emit(f'{base}{name}_envelope={name}_radius*jnp.asarray(1.4142135623730951,jnp.float64)')
                    self.emit(f"{base}{name}_box={name}_allowed & (s[{(prefix+'leaf')!r}]>=0) & (jnp.arange({cap})!={exclude}) & jnp.all(jnp.abs({name}_delta[:,:2])<={name}_envelope,axis=1)")
                    self.emit(f'{base}{name}_nearby={name}_box & (jnp.sqrt(jnp.sum({name}_delta*{name}_delta,axis=1))<={name}_radius)')
                    self.emit(f'{base}{name}_box_count=jnp.sum({name}_box,dtype=jnp.int64)')
                    self.emit(f'{base}{name}_near_count=jnp.sum({name}_nearby,dtype=jnp.int64)')
                    if key is not None:self.deferred_query_cache[key]=name
                self.emit(f"{base}{name}_enabled=({self.active_mask}) & (s['$error']==0)")
                self.emit(f"{base}s['$deferred.draws']=s['$deferred.draws']+jnp.where({name}_enabled,jnp.where({name}_box_count>=2,{name}_box_count,0)+jnp.where({name}_near_count>=2,{name}_near_count,0),0)")
                return f'(jnp.arange({cap},dtype=jnp.int32),{name}_nearby & {name}_enabled)'
            if expr.metadata.get('cache_contacts'):
                keys=self.contact_cache_keys(owner);cache='('+','.join(f's[{key!r}]' for key in keys)+')';cache_prefix='$contacts.'+owner+'.'
                valid=f's[{(cache_prefix+"valid")!r}]';cached_radius=f's[{(cache_prefix+"radius")!r}]'
                cached_source=f's[{(cache_prefix+"source")!r}]';cached_epoch=f's[{(cache_prefix+"cached_epoch")!r}]';epoch=f's[{(cache_prefix+"epoch")!r}]'
                call=f'memoized_point_neighbors(rng,{valid},{cached_radius},{cache},{cached_source},{cached_epoch},{epoch},s[{(owner+".location")!r}],{self.contact_alive(owner)},s[{(prefix+"leaf")!r}],{self.agent_index},{name}_radius)'
                result=f"s['$rng'],{name}_result,{valid},{cached_radius},({','.join('s['+repr(key)+']' for key in keys)}),{cached_source},{cached_epoch}"
                empty=f'(rng,(jnp.zeros(({cap},),jnp.int32),jnp.zeros(({cap},),jnp.bool_)),{valid},{cached_radius},{cache},{cached_source},{cached_epoch})'
                self.emit(f"{self.eval_indent}{result} = jax.lax.cond(({self.active_mask}) & (s['$error']==0),lambda rng:{call},lambda rng:{empty},s['$rng'])")
                return name+'_result'
            self.emit(f"{self.eval_indent}s['$rng'],{name}_result = jax.lax.cond(({self.active_mask}) & (s['$error']==0),lambda rng:{call},lambda rng:(rng,(jnp.zeros(({cap},),jnp.int32),jnp.zeros(({cap},),jnp.bool_))),s['$rng'])")
            return name+'_result'
        if op=='list_literal':
            if expr.args:return '(jnp.stack(['+', '.join(self.expression(a,scope) for a in expr.args)+']),jnp.ones(('+str(len(expr.args))+',),jnp.bool_))'
            element=expr.type.split(':')[1];tail=(3,) if element=='point' else (4,) if element=='rgb' else ()
            return f'(jnp.zeros({(0,*tail)!r},DTYPES[{element!r}]),jnp.zeros((0,),jnp.bool_))'
        if op=='at' and expr.metadata.get('raster_pixel'):
            handle=self.expression(expr.args[0],scope);point=self.expression(expr.args[1],scope)
            self.counter+=1;prefix=f'raster_pixel_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}_handle={handle}')
            self.emit(f'{self.eval_indent}{prefix}_point={point}')
            self.emit(f'{self.eval_indent}{prefix}_xy=jnp.trunc({prefix}_point[:2]).astype(jnp.int32)')
            branches=[]
            for record in self.ir.assets:
                if 'raster_size' not in record:raise CompileError(expr.token.source,expr.token,'Undecoded raster resource','GAMAX_ASSET')
                width,height=record['raster_size'];pixels=record['raster_pixels']
                data=f'RASTER_PIXELS_{self.raster_tables[record["key"]]}' if record['key'] in self.raster_tables else f'jnp.asarray({pixels!r},jnp.int32)'
                branches.append(f'lambda xy:({data}[jnp.clip(xy[1],0,{height-1})*{width}+jnp.clip(xy[0],0,{width-1})],jnp.all(xy>=0)&(xy[0]<{width})&(xy[1]<{height}))')
            if not branches:raise CompileError(expr.token.source,expr.token,'Raster indexing requires a bundled image','GAMAX_ASSET')
            self.emit(f'{self.eval_indent}{prefix}, {prefix}_valid=jax.lax.switch({prefix}_handle,('+','.join(branches)+',),'+prefix+'_xy)')
            self.emit(f"{self.eval_indent}s['$error']=jnp.where(({self.active_mask}) & (~{prefix}_valid | ({prefix}_handle<0) | ({prefix}_handle>={len(branches)}) | ~jnp.all(jnp.isfinite({prefix}_point))) & (s['$error']==0),jnp.asarray(13,jnp.int32),s['$error'])")
            return prefix
        if op=='at':
            container=self.expression(expr.args[0],scope);rank=self.expression(expr.args[1],scope)
            self.counter+=1;prefix=f'indexed_collection_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}_values, {prefix}_mask = {container}')
            self.emit(f'{self.eval_indent}{prefix}_rank = {rank}')
            self.emit(f'{self.eval_indent}{prefix}_count = jnp.sum({prefix}_mask,dtype=jnp.int32)')
            self.emit(f"{self.eval_indent}s['$error'] = jnp.where(({self.active_mask}) & (({prefix}_rank<0) | ({prefix}_rank>={prefix}_count)) & (s['$error']==0),jnp.asarray(8,jnp.int32),s['$error'])",expr.token)
            cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
            if not cap:
                shape=(3,) if expr.type=='point' else (4,) if expr.type=='rgb' else ()
                return f'jnp.zeros({shape!r},DTYPES[{expr.type!r}])'
            self.emit(f'{self.eval_indent}{prefix}_index = jnp.argmax({prefix}_mask & (jnp.cumsum({prefix}_mask,dtype=jnp.int32)-1=={prefix}_rank))')
            return f'{prefix}_values[{prefix}_index]'
        if op=='empty_agents':
            size=self.capacity(expr.type.split(':')[1])
            return f'(jnp.full(({size},),-1,jnp.int32),jnp.zeros(({size},),jnp.bool_))'
        if op in ('append_agent','remove_agent'):
            container=self.expression(expr.args[0],scope);element=self.expression(expr.args[1],scope)
            self.counter+=1;prefix=f'list_operation_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}_values, {prefix}_mask = {container}')
            self.emit(f'{self.eval_indent}{prefix}_element = {element}')
            if op=='append_agent':
                return f'(jnp.concatenate([{prefix}_values,jnp.reshape({prefix}_element,(1,))]),jnp.concatenate([{prefix}_mask,jnp.ones((1,),jnp.bool_)]))'
            self.emit(f'{self.eval_indent}{prefix}_matches = {prefix}_mask & ({prefix}_values=={prefix}_element)')
            self.emit(f'{self.eval_indent}{prefix}_first = jnp.argmax({prefix}_matches)' if self.collection_capacity(expr.args[0],scope,self.each_scope) else f'{self.eval_indent}{prefix}_first = -1')
            return f'({prefix}_values,{prefix}_mask & ~((jnp.arange({prefix}_values.shape[0])=={prefix}_first) & jnp.any({prefix}_matches)))'
        if op == 'overlapping':
            owner=expr.args[0].type.split(':')[1]
            grid_owner=expr.args[1].type.split(':')[1];grid=self.scopes[grid_owner].grid
            container=self.expression(expr.args[0],scope);cell=self.expression(expr.args[1],scope)
            self.counter+=1;prefix=f'overlap_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}_values, {prefix}_present = {container}')
            self.emit(f'{self.eval_indent}{prefix}_cell = {cell}')
            center=f"s[{(grid_owner+'.location')!r}][{prefix}_cell]"
            world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
            self.emit(f'{self.eval_indent}{prefix}_half = ({world})[:2]/jnp.asarray([{grid["width"]},{grid["height"]}])/2')
            geometric=f'jnp.all(jnp.abs(s[{(owner+".location")!r}][{prefix}_values,:2]-({center})[:2])<={prefix}_half,axis=-1)'
            self.emit(f'{self.eval_indent}{prefix}_present = {prefix}_present & {geometric}')
            self.emit(f'{self.eval_indent}{prefix}_count = jnp.sum({prefix}_present,dtype=jnp.int32)')
            uniform='mersenne_uniform' if self.ir.random_generator=='mersenne' else 'java_uniform'
            self.emit(f'{self.eval_indent}def {prefix}_draw(_,rng):')
            self.emit(f'{self.eval_indent}    rng,_ = {uniform}(rng)')
            self.emit(f'{self.eval_indent}    return rng')
            if getattr(self.ir,'optimize_mersenne',False):
                self.emit(f"{self.eval_indent}s['$rng'] = mersenne_discard_uniforms(s['$rng'],jnp.where(({self.active_mask}) & ({prefix}_count>=2),2*{prefix}_count,0))")
            else:
                self.emit(f"{self.eval_indent}s['$rng'] = jax.lax.fori_loop(0,jnp.where(({self.active_mask}) & ({prefix}_count>=2),2*{prefix}_count,0),{prefix}_draw,s['$rng'])")
            return f'({prefix}_values,{prefix}_present)'
        if op == 'each_ref':
            return self.each_indices
        if op == 'accumulate':
            container = self.expression(expr.args[0],scope)
            owner = expr.args[0].type.split(':')[1]
            outer_capacity = self.collection_capacity(expr.args[0],scope,self.each_scope)
            inner_capacity = self.collection_capacity(expr.args[1],scope,owner)
            if outer_capacity == 0:
                return '(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))'
            self.counter += 1; prefix=f'accumulate_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}_values, {prefix}_present = {container}')
            self.emit(f'{self.eval_indent}def {prefix}_evaluate(iteration,state):' if expr.metadata.get('skip_inactive_entries') else f'{self.eval_indent}def {prefix}(iteration,state):')
            self.emit(f'{self.eval_indent}    s, values, present = state')
            self.emit(f'{self.eval_indent}    s = dict(s)')
            self.emit(f'{self.eval_indent}    {prefix}_agent = {prefix}_values[iteration]')
            saved = self.each_scope,self.each_indices,self.eval_indent,self.active_mask
            self.each_scope,self.each_indices=owner,prefix+'_agent'
            self.eval_indent += '    '
            self.active_mask=f'({saved[3]}) & {prefix}_present[iteration]'
            value = self.expression(expr.args[1],scope)
            self.emit(f'{self.eval_indent}{prefix}_mapped_values, {prefix}_mapped_present = {value}')
            self.emit(f'{self.eval_indent}values = jax.lax.dynamic_update_slice(values,{prefix}_mapped_values,(iteration*{inner_capacity},))')
            self.emit(f'{self.eval_indent}present = jax.lax.dynamic_update_slice(present, {prefix}_mapped_present & ({self.active_mask}),(iteration*{inner_capacity},))')
            self.emit(f'{self.eval_indent}return s,values,present')
            self.each_scope,self.each_indices,self.eval_indent,self.active_mask=saved
            if expr.metadata.get('skip_inactive_entries'):
                self.emit(f'{self.eval_indent}def {prefix}(iteration,state):')
                self.emit(f'{self.eval_indent}    return jax.lax.cond(({self.active_mask}) & {prefix}_present[iteration],lambda pair:{prefix}_evaluate(iteration,pair),lambda pair:pair,state)')
            capacity=outer_capacity*inner_capacity
            self.emit(f'{self.eval_indent}s, {prefix}_result_values, {prefix}_result_present = jax.lax.fori_loop(0,{outer_capacity},{prefix},(s,jnp.full(({capacity},),-1,jnp.int32),jnp.zeros(({capacity},),jnp.bool_)))')
            return f'({prefix}_result_values,{prefix}_result_present)'
        if op == 'as_list':
            value=self.expression(expr.args[0],scope)
            if expr.metadata.get('rgba_list'):return f'({value},jnp.ones((4,),jnp.bool_))'
            return value
        if op in ('grid_x','grid_y'):
            return f'({self.agent_index} % {self.scopes[scope].grid["width"]})' if op=='grid_x' else f'({self.agent_index} // {self.scopes[scope].grid["width"]})'
        if op == "grid_location":
            grid = self.scopes[scope].grid
            index = self.agent_index
            if any(f.name=="shape" and f.type=="geometry" for f in self.scopes['global'].fields):
                width="jnp.max(s['global.shape'][:,0])-jnp.min(s['global.shape'][:,0])"
                height="jnp.max(s['global.shape'][:,1])-jnp.min(s['global.shape'][:,1])"
            else:width=height="jnp.asarray(100.,jnp.float64)"
            return f"jnp.stack([(({index} % {grid['width']}).astype(jnp.float64)+0.5)*(({width})/{grid['width']}), (({index} // {grid['width']}).astype(jnp.float64)+0.5)*(({height})/{grid['height']}), jnp.asarray(0.,jnp.float64)])"
        if op.startswith("attribute:"):
            index = self.expression(expr.args[0],scope)
            self.counter += 1; name=f"reference_{self.counter}"
            self.emit(f"{self.eval_indent}{name} = {index}")
            owner = expr.args[0].type.split(':')[1];attribute = op.split(':')[1]
            count = self.capacity(owner)
            self.emit(f"{self.eval_indent}s['$error'] = jnp.where(jnp.any(({self.active_mask}) & (({name}<0) | ({name}>={count}))) & (s['$error']==0),jnp.asarray(5,jnp.int32),s['$error'])")
            if attribute=='index':return name
            if count == 0:
                shape = (3,) if expr.type == 'point' else (4,) if expr.type == 'rgb' else ()
                return f"jnp.zeros({shape!r}, DTYPES[{expr.type!r}])"
            field=next(f for f in self.scopes[owner].fields if f.name==attribute)
            if field.getter is not None:return self.computed_value(owner,field,f'jnp.clip({name},0,{count-1})')
            if expr.type.startswith(('agents:','values:')):
                key = owner+'.'+attribute
                return f"(s[{key!r}][jnp.clip({name},0,{count-1})], s[{('$listmask.'+key)!r}][jnp.clip({name},0,{count-1})])"
            return f"s[{(owner+'.'+attribute)!r}][jnp.clip({name},0,{count-1})]"
        if op == "implicit_location":
            self.counter += 1; prefix = f"implicit_position_{self.counter}"
            uniform = "mersenne_uniform" if self.ir.random_generator=="mersenne" else "java_uniform"
            if getattr(self.ir,'optimize_grid_random_fields',False):
                self.random_pair_sites+=1
                self.emit(f"{self.eval_indent}s['$rng'], {prefix}_pair = jax.lax.cond({self.active_mask}, mersenne_uniform_pair, lambda rng:(rng,jnp.zeros((2,),jnp.float64)), s['$rng'])",expr.token)
                for i,coordinate in enumerate(('x','y')):self.emit(f'{self.eval_indent}{prefix}_{coordinate} = {prefix}_pair[{i}]')
            else:
                for coordinate in ("x","y"):
                    self.emit(f"{self.eval_indent}s['$rng'], {prefix}_{coordinate} = jax.lax.cond({self.active_mask}, {uniform}, lambda rng:(rng,jnp.asarray(0.,jnp.float64)), s['$rng'])",expr.token)
            if any(f.name=="shape" and f.type=="geometry" for f in self.scopes['global'].fields):
                x = f"jnp.min(s['global.shape'][:,0]) + {prefix}_x * (jnp.max(s['global.shape'][:,0])-jnp.min(s['global.shape'][:,0]))"
                y = f"jnp.min(s['global.shape'][:,1]) + {prefix}_y * (jnp.max(s['global.shape'][:,1])-jnp.min(s['global.shape'][:,1]))"
            else:
                x,y=f"{prefix}_x * 100.0",f"{prefix}_y * 100.0"
            return f"jnp.stack([{x}, {y}, jnp.asarray(0.,jnp.float64)])"
        if op == "conditional":
            condition = self.expression(expr.args[0], scope)
            self.counter += 1; name = f"condition_{self.counter}"
            self.emit(f"{self.eval_indent}{name} = {condition}")
            saved = self.active_mask
            self.active_mask = f"({saved}) & ({name})"
            yes = self.expression(expr.args[1], scope)
            self.active_mask = f"({saved}) & ~({name})"
            no = self.expression(expr.args[2], scope)
            self.active_mask = saved
            selected = f"jnp.expand_dims({name}, -1)" if expr.type in ('point','rgb') else name
            return f"jnp.where({selected}, {yes}, {no})"
        if op == "point":
            values = [self.expression(arg, scope) for arg in expr.args]
            return f"jnp.stack(jnp.broadcast_arrays({', '.join(values)}), axis=-1).astype(jnp.float64)"
        if op.startswith("component:"):
            value = self.expression(expr.args[0], scope)
            return f"({value})[..., {('x','y','z').index(op.split(':')[1])}]"
        if op == "literal": return f"jnp.asarray({expr.args[0]!r}, dtype=DTYPES[{expr.type!r}])"
        if op == "name":
            name = expr.args[0]
            if name == "self": return self.agent_index
            if name in self.locals:
                key,kind=self.locals[name]
                if kind.startswith('agents:'):return f"(s[{key!r}],s[{('$listmask.'+key)!r}])"
                return f"s[{key!r}]"
            if name in self.ir.populations:
                count = self.capacity(name)
                if self.reserved(name):
                    return f'(jnp.arange({count},dtype=jnp.int32),s[{("$present."+name)!r}])'
                return f'(jnp.arange({count}, dtype=jnp.int32), jnp.ones(({count},), dtype=jnp.bool_))'
            if name == "index":
                return self.agent_index if self.agent_index is not None else f"jnp.arange({self.capacity(scope)}, dtype=jnp.int32)"
            owner = scope if any(f.name == name for f in self.scopes[scope].fields) else "global"
            field=next(f for f in self.scopes[owner].fields if f.name==name)
            if field.getter is not None:return self.computed_value(owner,field,self.agent_index if owner!='global' else None)
            value = f"s[{(owner+'.'+name)!r}]"
            if expr.type.startswith(('agents:','values:')):
                mask = f"s[{('$listmask.'+owner+'.'+name)!r}]"
                if owner != 'global' and self.agent_index is not None:
                    value += f'[{self.agent_index}]'; mask += f'[{self.agent_index}]'
                return f'({value}, {mask})'
            return f"{value}[{self.agent_index}]" if owner != "global" and self.agent_index is not None else value
        if op in ('each','myself'):
            owner = self.each_scope if op == 'each' else expr.metadata['caller_scope']
            name = expr.args[0]
            if op=='myself' and owner=='global':
                field=next(f for f in self.scopes['global'].fields if f.name==name)
                if field.getter is not None:return self.computed_value(owner,field,None)
                value=f's[{("global."+name)!r}]'
                return f'({value},s[{("$listmask.global."+name)!r}])' if expr.type.startswith(('agents:','values:')) else value
            if op=='myself' and self.caller_index is None:
                raise CompileError(expr.token.source or '<IR>',expr.token,'Missing caller index in backend','GAMAX_BACKEND')
            if op=='each':
                self.emit(f"{self.eval_indent}s['$error'] = jnp.where(jnp.any(({self.active_mask}) & (({self.each_indices}<0) | ({self.each_indices}>={self.capacity(owner)}))) & (s['$error']==0),jnp.asarray(5,jnp.int32),s['$error'])",expr.token)
            if name == 'index': return self.each_indices if op == 'each' else self.caller_index
            if op=='each' and self.capacity(owner)==0:
                tail=(3,) if expr.type=='point' else (4,) if expr.type=='rgb' else ()
                return f'jnp.zeros(jnp.shape({self.each_indices})+{tail!r},DTYPES[{expr.type!r}])'
            field=next(f for f in self.scopes[owner].fields if f.name==name)
            if field.getter is not None:return self.computed_value(owner,field,self.each_indices if op=='each' else self.caller_index)
            value = f's[{(owner+"."+name)!r}]'
            if expr.type.startswith('agents:'):
                mask = f's[{("$listmask."+owner+"."+name)!r}]'
                index = self.each_indices if op=='each' else self.caller_index
                return f'({value}[jnp.maximum(0,{index})],{mask}[jnp.maximum(0,{index})])'
            return f'{value}[jnp.maximum(0,{self.each_indices})]' if op == 'each' else f'{value}[{self.caller_index}]'
        if op == "neighbors_at":
            owner = expr.args[0].type.split(':')[1]
            grid = self.scopes[owner].grid
            index = self.expression(expr.args[0],scope)
            self.counter += 1; name = f"neighbors_{self.counter}"
            uniform = "mersenne_uniform" if self.ir.random_generator == "mersenne" else "java_uniform"
            radius = self.expression(expr.args[1],scope)
            bound = expr.metadata['max_radius']
            call = f"grid_neighbors(rng, {index}, {grid['width']}, {grid['height']}, {radius}, max_radius={bound}, moore={grid['neighbors']==8}, torus={grid['torus']}, uniform={uniform})"
            key=(grid['width'],grid['height'],bound,grid['neighbors']==8,grid['torus'])
            if key in self.grid_geometry:
                table=self.grid_geometry[key]
                self.emit(f'{self.eval_indent}{name}_index={index}')
                cached=f'grid_neighbors_from_geometry(rng,GRID_GEOMETRY_{table},GRID_GEOMETRY_RADII_{table},{name}_index,{radius},uniform={uniform})'
                if expr.args[0].op=='name' and expr.args[0].args[0]=='self':call=cached
                else:call=f'jax.lax.cond(({name}_index>=0) & ({name}_index<{grid["width"]*grid["height"]}),lambda rng:{cached},lambda rng:{call},rng)'
            capacity = (2*bound+1)**2-1 if grid['neighbors']==8 else 2*bound*(bound+1)
            self.emit(f"{self.eval_indent}s['$rng'], {name} = jax.lax.cond({self.active_mask}, lambda rng: {call}, lambda rng: (rng, (jnp.full(({capacity},), -1, jnp.int32), jnp.zeros(({capacity},), jnp.bool_))), s['$rng'])",expr.token)
            return name
        if op in ('where','collect','count'):
            container = self.expression(expr.args[0], scope)
            def reads_getter(value):
                if value.op in ('each','myself','name'):
                    owner=value.metadata.get('owner_scope',scope)
                    if value.op=='name' and not any(f.name==value.args[0] for f in self.scopes[owner].fields):owner='global'
                    if any(f.name==value.args[0] and f.getter is not None for f in self.scopes[owner].fields):return True
                if value.op.startswith('attribute:') and value.args[0].type.startswith('agent:'):
                    owner=value.args[0].type.split(':')[1]
                    if any(f.name==value.op.split(':')[1] and f.getter is not None for f in self.scopes[owner].fields):return True
                return any(reads_getter(a) for a in value.args if isinstance(a,Expr))
            if reads_getter(expr.args[1]):
                owner=expr.args[0].type.split(':')[1]
                cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
                if getattr(self.ir,'optimize_pure_mapper',False) and scope=='global' and not self.initializing and not getattr(self,'deferred_mapper_rng',False) and op in ('count','collect') and expr.args[1].type in ('int','bool') and cap*self.capacity(owner)<=2000000 and self.discard_only_mapper(expr.args[1],scope,owner):
                    self.counter+=1;prefix=f'parallel_mapper_{self.counter}';base=self.eval_indent
                    self.emit(f'{base}{prefix}_container={container}')
                    saved=self.each_scope,self.each_indices,self.active_mask,self.eval_indent
                    self.emit(f'{base}def {prefix}(position):')
                    self.emit(f'{base}    local_state=dict(s)')
                    self.emit(f"{base}    local_state['$deferred.draws']=jnp.asarray(0,jnp.int64)")
                    # A local name keeps the captured state separate from the
                    # mapper's functional error/budget updates.
                    self.emit(f'{base}    def evaluate(s):')
                    self.each_scope=owner;self.each_indices=f'{prefix}_container[0][position]'
                    self.active_mask=f'({saved[2]}) & {prefix}_container[1][position]';self.eval_indent=base+'        '
                    self.deferred_mapper_rng=True
                    self.deferred_query_cache={}
                    self.deferred_query_base=self.eval_indent
                    value=self.expression(expr.args[1],scope)
                    self.deferred_mapper_rng=False
                    self.deferred_query_cache={}
                    self.emit(f"{base}        return {value},s['$deferred.draws'],s['$error']")
                    self.emit(f'{base}    return evaluate(local_state)')
                    self.each_scope,self.each_indices,self.active_mask,self.eval_indent=saved
                    self.emit(f'{base}{prefix}_result,{prefix}_draws,{prefix}_errors=jax.vmap({prefix})(jnp.arange({cap},dtype=jnp.int32))')
                    self.emit(f"{base}s['$rng']=mersenne_discard_uniforms(s['$rng'],jnp.sum({prefix}_draws,dtype=jnp.int64))")
                    if cap:
                        self.emit(f'{base}{prefix}_first=jnp.nonzero({prefix}_errors!=0,size={cap},fill_value=0)[0][0]')
                        self.emit(f"{base}s['$error']=jnp.where((s['$error']==0) & jnp.any({prefix}_errors!=0),{prefix}_errors[{prefix}_first],s['$error'])")
                    self.pure_mapper_sites+=1
                    if op=='count':return f'jnp.sum({prefix}_container[1] & {prefix}_result,dtype=jnp.int32)'
                    return f'({prefix}_result,{prefix}_container[1])'
                self.counter+=1;prefix=f'ordered_mapper_{self.counter}'
                cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
                self.emit(f'{self.eval_indent}{prefix}_container = {container}')
                saved=self.each_scope,self.each_indices,self.active_mask,self.eval_indent
                base=self.eval_indent
                self.emit(f'{base}def {prefix}(position,carry):')
                self.emit(f'{base}    s, result = carry');self.emit(f'{base}    s = dict(s)')
                self.each_scope=expr.args[0].type.split(':')[1];self.each_indices=f'{prefix}_container[0][position]'
                self.active_mask=f'({saved[2]}) & {prefix}_container[1][position]';self.eval_indent=base+'    '
                value=self.expression(expr.args[1],scope)
                self.emit(f'{base}    result = result.at[position].set({value})')
                self.emit(f'{base}    return s,result')
                tail=(3,) if expr.args[1].type=='point' else ()
                self.emit(f"{base}s,{prefix}_result = jax.lax.fori_loop(0,{cap},{prefix},(s,jnp.zeros({(cap,*tail)!r},DTYPES[{expr.args[1].type!r}])))")
                self.each_scope,self.each_indices,self.active_mask,self.eval_indent=saved
                if op=='count':return f'jnp.sum({prefix}_container[1] & {prefix}_result,dtype=jnp.int32)'
                if op=='where':return f'({prefix}_container[0],{prefix}_container[1] & {prefix}_result)'
                return f'({prefix}_result,{prefix}_container[1])'
            saved = self.each_scope, self.each_indices, self.active_mask
            self.each_scope = expr.args[0].type.split(':')[1]
            self.each_indices = f'{container}[0]'
            self.active_mask = f'({saved[2]}) & {container}[1]'
            values = self.expression(expr.args[1], scope)
            self.each_scope, self.each_indices, self.active_mask = saved
            if op == 'count': return f'jnp.sum({container}[1] & ({values}), dtype=jnp.int32)'
            if op == 'where': return f'({container}[0], {container}[1] & ({values}))'
            shape = f'{container}[1].shape + (3,)' if expr.args[1].type == 'point' else f'{container}[1].shape'
            return f'(jnp.broadcast_to({values}, {shape}), {container}[1])'
        if op in ("and", "or"):
            left = self.expression(expr.args[0], scope)
            self.counter += 1; name = f"boolean_{self.counter}"
            self.emit(f"{self.eval_indent}{name} = {left}")
            saved = self.active_mask
            self.active_mask = f"({saved}) & ({name})" if op == "and" else f"({saved}) & ~({name})"
            right = self.expression(expr.args[1], scope)
            self.active_mask = saved
            return f"({name} {'&' if op == 'and' else '|'} ({right}))"
        values = [self.expression(arg, scope) for arg in expr.args]
        if expr.metadata.get('color_scalar_arithmetic'):
            self.counter+=1;prefix=f'color_arithmetic_{self.counter}'
            self.emit(f'{self.eval_indent}{prefix}={values[0]}')
            return f'jnp.concatenate([jnp.clip({prefix}[:3] {op} ({values[1]}).astype(jnp.int32),0,255),{prefix}[3:4]])'
        if op == '+' and expr.type.startswith('agents:'):
            left = values[0]
            if expr.args[1].type.startswith('agent:'):
                right = f'(jnp.atleast_1d({values[1]}), jnp.ones((1,),jnp.bool_))'
            else:
                right = values[1]
            return f'(jnp.concatenate(({left}[0],{right}[0])), jnp.concatenate(({left}[1],{right}[1])))'
        if op.startswith('call:'):
            function = op.split(':')[1]
            if function=='shuffle':
                cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
                if cap==0:return values[0]
                self.counter+=1;prefix=f'shuffle_{self.counter}'
                self.emit(f'{self.eval_indent}{prefix}_values, {prefix}_mask = {values[0]}')
                self.emit(f'{self.eval_indent}{prefix}_count = jnp.sum({prefix}_mask,dtype=jnp.int32)')
                self.emit(f'{self.eval_indent}{prefix}_indices = jnp.nonzero({prefix}_mask,size={cap},fill_value=0)[0]')
                self.emit(f'{self.eval_indent}{prefix}_values = {prefix}_values[{prefix}_indices]')
                self.emit(f'{self.eval_indent}def {prefix}_swap(iteration,state):')
                self.emit(f'{self.eval_indent}    rng,array = state')
                self.emit(f'{self.eval_indent}    position = {prefix}_count-1-iteration')
                uniform='mersenne_uniform' if self.ir.random_generator=='mersenne' else 'java_uniform'
                self.emit(f'{self.eval_indent}    rng,sample = {uniform}(rng)')
                self.emit(f'{self.eval_indent}    choice = jnp.trunc((position+1).astype(jnp.float64)*sample).astype(jnp.int32)')
                self.emit(f'{self.eval_indent}    a,b = array[position],array[choice]')
                self.emit(f'{self.eval_indent}    return rng,array.at[position].set(b).at[choice].set(a)')
                self.emit(f"{self.eval_indent}s['$rng'], {prefix}_values = jax.lax.fori_loop(0,jnp.where({self.active_mask},jnp.maximum(0,{prefix}_count-1),0),{prefix}_swap,(s['$rng'],{prefix}_values))")
                return f'({prefix}_values,jnp.arange({cap})<{prefix}_count)'
            if function in ('first','last'):
                cap=self.collection_capacity(expr.args[0],scope,self.each_scope)
                if not cap:return 'jnp.asarray(-1,jnp.int32)'
                container=values[0]
                rank=f'jnp.argmax({container}[1])' if function=='first' else f'({cap}-1-jnp.argmax({container}[1][::-1]))'
                return f'jnp.where(jnp.any({container}[1]),{container}[0][{rank}],jnp.asarray(-1,jnp.int32))'
            if function=='copy':return values[0]
            if function == "one_of":
                if expr.metadata.get('direct_grid_sampling'):
                    grid=expr.metadata['direct_grid_sampling'];self.counter+=1;prefix=f'grid_selection_{self.counter}'
                    uniform='mersenne_uniform' if self.ir.random_generator=='mersenne' else 'java_uniform'
                    if getattr(self.ir,'optimize_grid_random_fields',False):
                        self.random_pair_sites+=1
                        self.emit(f"{self.eval_indent}s['$rng'],{prefix}_pair = jax.lax.cond({self.active_mask},mersenne_uniform_pair,lambda rng:(rng,jnp.zeros((2,),jnp.float64)),s['$rng'])")
                        for i,axis in enumerate(('x','y')):self.emit(f'{self.eval_indent}{prefix}_{axis} = {prefix}_pair[{i}]')
                    else:
                        for axis in ('x','y'):self.emit(f"{self.eval_indent}s['$rng'],{prefix}_{axis} = jax.lax.cond({self.active_mask},{uniform},lambda rng:(rng,jnp.asarray(0.,jnp.float64)),s['$rng'])")
                    return f'(jnp.trunc({prefix}_y*{grid["height"]}).astype(jnp.int32)*{grid["width"]}+jnp.trunc({prefix}_x*{grid["width"]}).astype(jnp.int32))'
                if self.capacity(expr.type.split(':')[1]) == 0:
                    return "jnp.asarray(-1,jnp.int32)"
                self.counter += 1; name=f"selection_{self.counter}"
                container = values[0]
                self.emit(f"{self.eval_indent}{name}_length = jnp.sum({container}[1],dtype=jnp.int32)")
                uniform = "mersenne_uniform" if self.ir.random_generator=='mersenne' else "java_uniform"
                self.emit(f"{self.eval_indent}s['$rng'], {name}_random = jax.lax.cond(({self.active_mask}) & ({name}_length>0),{uniform},lambda rng:(rng,jnp.asarray(0.,jnp.float64)),s['$rng'])",expr.token)
                self.emit(f"{self.eval_indent}{name}_rank = jnp.trunc({name}_length.astype(jnp.float64)*{name}_random).astype(jnp.int32)")
                self.emit(f"{self.eval_indent}{name}_offset = jnp.argmax({container}[1] & (jnp.cumsum({container}[1],dtype=jnp.int32)-1 == {name}_rank))")
                return f"jnp.where({name}_length>0,{container}[0][{name}_offset],jnp.asarray(-1,jnp.int32))"
            if function=='point':
                if expr.args[0].type=='point':return values[0]
                owner=expr.args[0].type.split(':')[1];count=self.capacity(owner)
                if not count:return 'jnp.full((3,),jnp.nan,jnp.float64)'
                self.counter+=1;prefix=f'point_cast_{self.counter}'
                self.emit(f'{self.eval_indent}{prefix}_index={values[0]}')
                self.emit(f"{self.eval_indent}s['$error']=jnp.where(({self.active_mask}) & ({prefix}_index>={count}) & (s['$error']==0),jnp.asarray(5,jnp.int32),s['$error'])")
                return f"jnp.where({prefix}_index>=0,s[{(owner+'.location')!r}][jnp.clip({prefix}_index,0,{count-1})],jnp.full((3,),jnp.nan,jnp.float64))"
            if function=="round":return f"jnp.clip(jnp.floor(({values[0]}).astype(jnp.float64)+0.5),-2147483648,2147483647).astype(jnp.int32)"
            if function in ("min","max") and expr.args[0].type.startswith("matrix:"):return f"jnp.{function}({values[0]})"
            if function == "rgb":
                if expr.metadata.get("packed_argb"):
                    self.counter+=1;prefix=f"packed_color_{self.counter}"
                    self.emit(f"{self.eval_indent}{prefix}={values[0]}")
                    return f"jnp.stack([({prefix}>>16)&255,({prefix}>>8)&255,{prefix}&255,({prefix}>>24)&255]).astype(jnp.int32)"
                channels=values if len(values)==4 else values+["jnp.asarray(255,jnp.int32)"]
                if any(a.op!="literal" or not 0<=a.args[0]<=255 for a in expr.args):channels=[f"jnp.clip({channel},0,255)" for channel in channels]
                return f"jnp.stack([{', '.join(channels)}])"
            if function=='triangle':
                self.counter+=1;prefix=f'triangle_{self.counter}'
                self.emit(f'{self.eval_indent}{prefix}_side = ({values[0]}).astype(jnp.float64)')
                self.emit(f"{self.eval_indent}s['$error'] = jnp.where(({self.active_mask}) & (({prefix}_side<=0) | ~jnp.isfinite({prefix}_side)) & (s['$error']==0),jnp.asarray(12,jnp.int32),s['$error'])")
                self.emit(f'{self.eval_indent}{prefix}_height = jax.lax.optimization_barrier(jnp.asarray(math.sqrt(3.)/2.,jnp.float64)*{prefix}_side)')
                location=f"s[{(scope+'.location')!r}][{self.agent_index}]"
                self.emit(f'{self.eval_indent}{prefix}_left = {location}+jnp.stack([-{prefix}_side/2.,{prefix}_height/3.,jnp.asarray(0.,jnp.float64)])')
                self.emit(f'{self.eval_indent}{prefix}_right = {location}+jnp.stack([{prefix}_side/2.,{prefix}_height/3.,jnp.asarray(0.,jnp.float64)])')
                self.emit(f'{self.eval_indent}{prefix}_top = {location}+jnp.stack([jnp.asarray(0.,jnp.float64),-2.*{prefix}_height/3.,jnp.asarray(0.,jnp.float64)])')
                return f'jnp.stack([{prefix}_left,{prefix}_right,{prefix}_top,{prefix}_left])'
            if function == "square":
                self.counter += 1; dimension = f"world_size_{self.counter}"
                self.emit(f"{self.eval_indent}{dimension} = ({values[0]}).astype(jnp.float64)")
                self.emit(f"{self.eval_indent}s['$error'] = jnp.where(({self.active_mask}) & (({dimension} <= 0) | ~jnp.isfinite({dimension})) & (s['$error'] == 0), jnp.asarray(3, jnp.int32), s['$error'])")
                return f"jnp.asarray([[0.,1.,0.],[1.,1.,0.],[1.,0.,0.],[0.,0.,0.],[0.,1.,0.]],jnp.float64) * {dimension}"
            if function=='rectangle':
                self.counter+=1;prefix=f'world_rectangle_{self.counter}'
                self.emit(f'{self.eval_indent}{prefix} = jnp.stack([({values[0]}).astype(jnp.float64),({values[1]}).astype(jnp.float64),jnp.asarray(0.,jnp.float64)])')
                self.emit(f"{self.eval_indent}s['$error'] = jnp.where(({self.active_mask}) & (jnp.any({prefix}[:2]<=0) | ~jnp.all(jnp.isfinite({prefix}))) & (s['$error']==0),jnp.asarray(3,jnp.int32),s['$error'])")
                return f'jnp.asarray([[0.,1.,0.],[1.,1.,0.],[1.,0.,0.],[0.,0.,0.],[0.,1.,0.]],jnp.float64)*{prefix}'
            if function=='empty':return f'(~jnp.any({values[0]}[1]))'
            if function=='signum':return f'jnp.where({values[0]}<0,-1,jnp.where({values[0]}>0,1,0)).astype(jnp.int32)'
            if function in ("rnd", "flip"):
                self.counter += 1
                uniform = f"random_{self.counter}"
                uniform_function = "mersenne_uniform" if self.ir.random_generator == "mersenne" else "java_uniform"
                active=self.active_mask
                if expr.metadata.get('checked_bounds'):
                    low,high=('jnp.asarray(0,jnp.float64)',values[0]) if len(values)==1 else values
                    valid=f"random_bounds_valid_{self.counter}"
                    self.emit(f"{self.eval_indent}{valid} = ({low} <= {high}) & jnp.isfinite({low}) & jnp.isfinite({high})")
                    self.emit(f"{self.eval_indent}s['$error'] = jnp.where(({active}) & ~{valid} & (s['$error']==0),jnp.asarray(9,jnp.int32),s['$error'])")
                    active=f'({active}) & {valid}'
                self.emit(f"{self.eval_indent}s['$rng'], {uniform} = jax.lax.cond({active}, {uniform_function}, lambda r: (r, jnp.asarray(0.0, jnp.float64)), s['$rng'])", expr.token)
                if function == "flip": return f"({uniform} < {values[0]})"
                low, high = ("jnp.asarray(0, jnp.float64)", values[0]) if len(values) == 1 else values
                if expr.type == "int": return f"(({low}).astype(jnp.int64) + jnp.trunc((({high}).astype(jnp.int64) + 1 - ({low}).astype(jnp.int64)).astype(jnp.float64) * {uniform}).astype(jnp.int64)).astype(jnp.int32)"
                return f"jax.lax.optimization_barrier(({low}) + jax.lax.optimization_barrier((({high}) - ({low})) * {uniform}))"
            if function == 'length': return f'jnp.sum({values[0]}[1], dtype=jnp.int32)'
            if function in ('sum','mean'):
                dtype = 'point' if expr.args[0].type == 'values:point' else 'float' if function == 'mean' else expr.args[0].type.split(':')[1]
                total = f'ordered_sum({values[0]}[0], {values[0]}[1], DTYPES[{dtype!r}])'
                if function == 'sum': return total
                # GAMA numeric mean of an empty container is zero (checked natively).
                return f'(({total}).astype(jnp.float64) / jnp.maximum(1, jnp.sum({values[0]}[1], dtype=jnp.int32)).astype(jnp.float64))'
            if function in ('min','max') and len(values) == 1:
                dtype = expr.args[0].type.split(':')[1]
                bound = ('2147483647' if function == 'min' else '-2147483648') if dtype == 'int' else ("jnp.inf" if function == 'min' else "-jnp.inf")
                reduction = f'jnp.{function}(jnp.where({values[0]}[1], {values[0]}[0], {bound}))'
                return f'jnp.where(jnp.any({values[0]}[1]), {reduction}, 0).astype(DTYPES[{dtype!r}])'
            if function == 'int':
                # Java's narrowing conversion saturates overflow and maps NaN to zero.
                return f'jnp.nan_to_num(jnp.clip({values[0]}, -2147483648, 2147483647), nan=0).astype(jnp.int32)'
            if function == 'float': return f'({values[0]}).astype(jnp.float64)'
            if function in ('sin','cos'): return f'jnp.{function}(jnp.deg2rad(({values[0]}).astype(jnp.float64)))'
            if function in ('sqrt','exp'): return f'jnp.{function}(({values[0]}).astype(jnp.float64))'
            function = {'min':'minimum','max':'maximum'}.get(function,function)
            return f'jnp.{function}({", ".join(values)})'
        if op == "distance_to":
            difference = f"(({values[0]}) - ({values[1]}))"
            if expr.metadata.get('grid_geometry_operands'):
                widths=[];heights=[]
                for owner in expr.metadata['grid_geometry_operands']:
                    grid=self.scopes[owner].grid;widths.append(grid['width']);heights.append(grid['height'])
                world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
                extent=f"(({world})*jnp.asarray([{sum(0.5/w for w in widths)!r},{sum(0.5/h for h in heights)!r},0.],jnp.float64))"
                difference=f'jnp.maximum(jnp.abs({difference})-{extent},0.)'
            return f"jnp.sqrt(jnp.sum(({difference}) * ({difference}), axis=-1))"
        if expr.type == "point":
            a, b = values
            if expr.args[0].type != "point": a = f"jnp.expand_dims({a}, -1)"
            if expr.args[1].type != "point": b = f"jnp.expand_dims({b}, -1)"
            if op=='/' and getattr(self.ir,'optimize_algebra',False):b=f'jax.lax.optimization_barrier({b})'
            return f"jax.lax.optimization_barrier(({a}) {op} ({b}))"
        if op in ("=","!=") and expr.args[0].type=="rgb":
            return f"jnp.{'all' if op=='=' else 'any'}(({values[0]}) {'==' if op=='=' else '!='} ({values[1]}),axis=-1)"
        if op in ("/", "div", "mod") and expr.args[1].op != "literal":
            self.counter += 1; denominator = f"denominator_{self.counter}"
            self.emit(f"{self.eval_indent}{denominator} = {values[1]}")
            self.emit(f"{self.eval_indent}s['$error'] = jnp.where(jnp.any(({self.active_mask}) & ({denominator} == 0)) & (s['$error'] == 0), jnp.asarray(2, jnp.int32), s['$error'])")
            values[1] = f"jnp.where({denominator} == 0, 1, {denominator})"
        if op in ('mod','div'):
            # Widen before abs/division to handle Java MIN_VALUE safely.
            a, b = [f'({v}).astype(jnp.int64)' for v in values]
            quotient = f'((jnp.abs({a}) // jnp.abs({b})) * jnp.where(({a} < 0) != ({b} < 0), -1, 1))'
            return f'({quotient}).astype(jnp.int32)' if op == 'div' else f'({a} - {quotient} * {b}).astype(jnp.int32)'
        if op in ("not_unary", "-_unary"): return f"({'~' if op == 'not_unary' else '-'}{values[0]})"
        if op == '/':
            divisor=f'({values[1]}).astype(jnp.float64)'
            if getattr(self.ir,'optimize_algebra',False):divisor=f'jax.lax.optimization_barrier({divisor})'
            return f'jax.lax.optimization_barrier(({values[0]}).astype(jnp.float64) / {divisor})'
        operator = {"and": "&", "or": "|", "=": "=="}.get(op, op)
        operation = f"({values[0]} {operator} {values[1]})"
        return f'jax.lax.optimization_barrier({operation})' if expr.type == 'float' or getattr(self.ir,'optimize_fusion',False) and expr.type in ('point','geometry','polygon','matrix:float') else operation

    def collection_capacity(self, expr, scope, each_owner=None):
        if expr.op=='as_list' and expr.metadata.get('rgba_list'):return 4
        if expr.op in ('as_list','where','sort_by','collect','overlapping','inside','at_distance','remove_agent','call:shuffle','call:copy'): return self.collection_capacity(expr.args[0],scope,each_owner)
        if expr.op=='append_agent':return self.collection_capacity(expr.args[0],scope,each_owner)+1
        if expr.op=='empty_agents':return self.capacity(expr.type.split(':')[1])
        if expr.op=='list_literal':return len(expr.args)
        if expr.op == 'name':
            name=expr.args[0]
            if name in self.ir.populations:return self.capacity(name)
            if name in self.locals and self.locals[name][1].startswith('agents:'):return self.local_capacities[self.locals[name][0]]
            owner=scope if any(f.name==name for f in self.scopes[scope].fields) else 'global'
        elif expr.op.startswith('attribute:'):
            owner=expr.args[0].type.split(':')[1]; name=expr.op.split(':')[1]
        elif expr.op=='each':
            owner=each_owner; name=expr.args[0]
        elif expr.op=='neighbors_at':
            owner=expr.args[0].type.split(':')[1];radius=expr.metadata['max_radius']
            return (2*radius+1)**2-1 if self.scopes[owner].grid['neighbors']==8 else 2*radius*(radius+1)
        elif expr.op=='+':
            return self.collection_capacity(expr.args[0],scope,each_owner)+(1 if expr.args[1].type.startswith('agent:') else self.collection_capacity(expr.args[1],scope,each_owner))
        elif expr.op=='accumulate':
            return self.collection_capacity(expr.args[0],scope,each_owner)*self.collection_capacity(expr.args[1],scope,expr.args[0].type.split(':')[1])
        else:
            raise ValueError('No verified collection capacity for '+expr.op)
        field=next(f for f in self.scopes[owner].fields if f.name==name)
        if field.capacity is None:raise ValueError('Missing fixed collection capacity')
        return self.field_capacity(field)

    def mask(self, expression, indent):
        self.counter += 1
        name = f"mask_{self.counter}"
        self.emit(f"{indent}{name} = {expression}")
        return name

    def statements(self, statements, scope, mask, indent="    "):
        saved_locals = dict(self.locals)
        for statement in statements:
            if getattr(self,"return_context",None):
                mask=self.mask(f'({mask}) & ~s[{self.return_context[1]!r}]',indent)
            if scope != 'global' and self.reserved(scope) and self.agent_index is not None:
                mask = self.mask(f'({mask}) & s[{("$present."+scope)!r}][{self.agent_index}]',indent)
            self.eval_indent, self.active_mask = indent, mask
            if statement.kind=="return":
                value_key,flag_key,kind=self.return_context
                value=self.expression(statement.value,scope)
                self.emit(f"{indent}s[{value_key!r}]=jnp.where({mask},({value}).astype(DTYPES[{kind!r}]),s[{value_key!r}])",statement.token)
                self.emit(f"{indent}s[{flag_key!r}]=s[{flag_key!r}] | ({mask})")
            elif statement.kind.startswith("local:"):
                self.counter += 1
                key = f"$local{self.counter}"
                type_ = statement.kind.split(":",1)[1]
                value = self.expression(statement.value, scope)
                if type_.startswith('agents:'):
                    capacity=self.collection_capacity(statement.value,scope,self.each_scope)
                    self.local_capacities[key]=capacity
                    self.emit(f"{indent}s[{key!r}],s[{('$listmask.'+key)!r}] = {value}",statement.token)
                else:self.emit(f"{indent}s[{key!r}] = ({value}).astype(DTYPES[{type_!r}])", statement.token)
                self.locals[statement.target] = (key, type_)
            elif statement.kind == "block":
                self.statements(statement.body, scope, mask, indent)
            elif statement.kind == 'collection_loop':
                if statement.metadata.get('stream_integer_accumulation'):
                    self.stream_integer_accumulation(statement,scope,mask,indent)
                    continue
                container = self.expression(statement.value, scope)
                if self.collection_capacity(statement.value,scope,self.each_scope)==0:
                    continue
                self.counter += 1; prefix = f'collection_loop_{self.counter}'
                key = '$iteration'+str(self.counter)
                type_ = 'agent:'+statement.value.type.split(':')[1]
                self.emit(f'{indent}{prefix}_values, {prefix}_mask = {container}')
                self.emit(f'{indent}{prefix}_indices = jnp.nonzero({prefix}_mask,size={prefix}_values.shape[0],fill_value=0)[0]')
                self.emit(f'{indent}{prefix}_count = jnp.sum({prefix}_mask,dtype=jnp.int32)')
                self.emit(f"{indent}s[{key!r}] = jnp.asarray(-1,jnp.int32)")
                saved = dict(self.locals)
                self.locals[statement.target] = (key,type_)
                if statement.metadata.get('commutative_integer_reduction'):
                    self.emit(f"{indent}s[{key!r}] = {prefix}_values")
                    self.integer_reductions(statement.body,scope,f'({mask}) & {prefix}_mask',indent)
                    self.emit(f"{indent}s.pop({key!r})")
                    self.locals=saved
                    continue
                self.emit(f'{indent}def {prefix}(iteration,state):',statement.token)
                self.emit(f'{indent}    s = dict(state)')
                self.emit(f"{indent}    s[{key!r}] = {prefix}_values[{prefix}_indices[iteration]]")
                self.statements(statement.body,scope,mask,indent+'    ')
                self.emit(f'{indent}    return s')
                self.emit(f'{indent}s = jax.lax.fori_loop(0,{prefix}_count,{prefix},s)')
                self.emit(f'{indent}s.pop({key!r})')
                self.locals = saved
            elif statement.kind == "static_loop":
                for iteration in statement.value.args:
                    variable = Statement("local:int", statement.target,
                        Expr("literal", [iteration], "int", statement.token), [], [], statement.token)
                    self.statements(([variable] if statement.target else []) + statement.body, scope, mask, indent)
            elif statement.kind == "dynamic_loop":
                self.counter += 1
                prefix = f"loop_{self.counter}"
                bounds = []
                for i, bound in enumerate(statement.value.args):
                    value = self.expression(bound, scope)
                    name = f"{prefix}_bound_{i}"
                    self.emit(f"{indent}{name} = ({value}).astype(jnp.int64)", statement.token)
                    bounds.append(name)
                if statement.value.type == "times":
                    start, stride = "jnp.asarray(0, jnp.int64)", "jnp.asarray(1, jnp.int64)"
                    count = bounds[0]
                    invalid = f"({count} < 0) | ({count} > 1000000)"
                else:
                    start, stop, stride = bounds
                    count = f"jnp.where(({stride} != 0) & jnp.where({stride} > 0, {start} <= {stop}, {start} >= {stop}), jnp.abs({stop} - {start}) // jnp.maximum(jnp.abs({stride}), 1) + 1, 0)"
                    invalid = f"({stride} == 0) | ({count} > 1000000)"
                self.emit(f"{indent}s['$error'] = jnp.where(({mask}) & ({invalid}) & (s['$error'] == 0), jnp.asarray(1, jnp.int32), s['$error'])")
                self.emit(f"{indent}{prefix}_count = jnp.where(({mask}) & (s['$error'] == 0), {count}, 0)")
                saved = dict(self.locals)
                if statement.target:
                    key = f"$iteration{self.counter}"
                    self.emit(f"{indent}s[{key!r}] = jnp.asarray(0, jnp.int32)")
                    self.locals[statement.target] = (key, "int")
                function = prefix + "_body"
                self.emit(f"{indent}def {function}(iteration, state):", statement.token)
                self.emit(f"{indent}    s = dict(state)")
                if statement.target:
                    self.emit(f"{indent}    s[{key!r}] = ({start} + iteration * {stride}).astype(jnp.int32)")
                self.statements(statement.body, scope, "jnp.asarray(True)", indent + "    ")
                self.emit(f"{indent}    return s")
                self.emit(f"{indent}s = jax.lax.fori_loop(0, {prefix}_count, {function}, s)")
                if statement.target: self.emit(f"{indent}s.pop({key!r})")
                self.locals = saved
            elif statement.kind=='matrix_assign':
                x,y=[self.expression(v,scope) for v in statement.metadata['indices']]
                self.counter+=1;prefix=f'matrix_assignment_{self.counter}';key='global.'+statement.target
                self.emit(f'{indent}{prefix}_x,{prefix}_y={x},{y}')
                self.emit(f"{indent}s['$error']=jnp.where(({mask}) & (({prefix}_x<0)|({prefix}_y<0)|({prefix}_x>=s[{key!r}].shape[1])|({prefix}_y>=s[{key!r}].shape[0])) & (s['$error']==0),jnp.asarray(14,jnp.int32),s['$error'])")
                value=self.expression(statement.value,scope)
                self.emit(f"{indent}s[{key!r}]=s[{key!r}].at[{prefix}_y,{prefix}_x].set(jnp.where({mask},{value},s[{key!r}][{prefix}_y,{prefix}_x]))")
            elif statement.kind=='attribute_assign':
                self.counter+=1;prefix=f'remote_assignment_{self.counter}'
                reference=self.expression(statement.metadata['reference'],scope)
                owner=statement.metadata['owner_scope'];count=self.capacity(owner)
                self.emit(f'{indent}{prefix}_index={reference}')
                self.emit(f"{indent}s['$error']=jnp.where(({mask}) & (({prefix}_index<0)|({prefix}_index>={count})) & (s['$error']==0),jnp.asarray(5,jnp.int32),s['$error'])")
                value=self.expression(statement.value,scope)
                if count:
                    field=next(f for f in self.scopes[owner].fields if f.name==statement.target)
                    saved_index=self.agent_index;self.agent_index=f'jnp.clip({prefix}_index,0,{count-1})'
                    value=self.coerce_field(value,field,owner);self.agent_index=saved_index
                    key=owner+'.'+statement.target
                    self.emit(f"{indent}s[{key!r}]=s[{key!r}].at[{prefix}_index].set(jnp.where({mask},({value}).astype(DTYPES[{field.type!r}]),s[{key!r}][{prefix}_index]))",statement.token)
            elif statement.kind == "assign" and statement.target in self.locals:
                if self.locals[statement.target][1].startswith('agents:'):
                    key,kind=self.locals[statement.target];cap=self.local_capacities[key]
                    value=self.expression(statement.value,scope);self.counter+=1;prefix=f'local_list_assignment_{self.counter}'
                    source_cap=self.collection_capacity(statement.value,scope,self.each_scope)
                    self.emit(f'{indent}{prefix}_values,{prefix}_mask={value}')
                    self.emit(f'{indent}{prefix}_count=jnp.sum({prefix}_mask,dtype=jnp.int32)')
                    self.emit(f"{indent}s['$error']=jnp.where(({mask}) & ({prefix}_count>{cap}) & (s['$error']==0),jnp.asarray(7,jnp.int32),s['$error'])")
                    self.emit(f'{indent}{prefix}_indices=jnp.nonzero({prefix}_mask,size={cap},fill_value=0)[0]')
                    replacement=f'{prefix}_values[{prefix}_indices]' if source_cap else f'jnp.full(({cap},),-1,jnp.int32)'
                    self.emit(f"{indent}s[{key!r}]=jnp.where({mask},{replacement},s[{key!r}])")
                    self.emit(f"{indent}s[{('$listmask.'+key)!r}]=jnp.where({mask},jnp.arange({cap})<{prefix}_count,s[{('$listmask.'+key)!r}])")
                    continue
                key, type_ = self.locals[statement.target]
                value = self.expression(statement.value, scope)
                selected = f"jnp.expand_dims({mask}, -1)" if type_ == "point" else mask
                self.emit(f"{indent}s[{key!r}] = jnp.where({selected}, {value}, s[{key!r}]).astype(DTYPES[{type_!r}])", statement.token)
            elif statement.kind == "assign":
                owner = scope if any(f.name == statement.target for f in self.scopes[scope].fields) else "global"
                field = next(f for f in self.scopes[owner].fields if f.name == statement.target)
                key = f"{owner}.{field.name}"
                value = self.expression(statement.value, scope)
                assignment_mask=mask
                if statement.metadata.get('ignore_nil_location'):
                    self.counter+=1;temporary=f'nullable_location_{self.counter}'
                    self.emit(f'{indent}{temporary}={value}')
                    mask=self.mask(f'({mask}) & ~jnp.all(jnp.isnan({temporary}))',indent)
                    self.active_mask=mask;value=temporary
                spatial_change=None
                if field.name=='location' and (owner in self.spatial_targets or any(f.type=='polygon' for f in self.scopes[owner].fields)):
                    self.counter+=1;temporary=f'spatial_position_{self.counter}'
                    self.emit(f'{indent}{temporary} = {value}')
                    value=temporary;spatial_change=temporary+'_changed'
                    self.emit(f'{indent}{spatial_change} = jnp.any({temporary} != s[{key!r}][{self.agent_index}])')
                if field.list_mutable:
                    self.counter+=1;prefix=f'list_assignment_{self.counter}'
                    cap=self.field_capacity(field);mask_key='$listmask.'+key
                    self.emit(f'{indent}{prefix}_values, {prefix}_mask = {value}')
                    self.emit(f'{indent}{prefix}_length = jnp.sum({prefix}_mask,dtype=jnp.int32)')
                    self.emit(f"{indent}s['$error'] = jnp.where(({mask}) & ({prefix}_length>{cap}) & (s['$error']==0),jnp.asarray(7,jnp.int32),s['$error'])",statement.token)
                    self.emit(f'{indent}{prefix}_indices = jnp.nonzero({prefix}_mask,size={cap},fill_value=0)[0]')
                    new=f'{prefix}_values[{prefix}_indices]' if self.collection_capacity(statement.value,scope,self.each_scope) else f'jnp.full(({cap},),-1,jnp.int32)'
                    self.emit(f'{indent}s[{key!r}] = jnp.where({mask},{new},s[{key!r}])')
                    self.emit(f'{indent}s[{mask_key!r}] = jnp.where({mask},jnp.arange({cap})<{prefix}_length,s[{mask_key!r}])')
                    if field in self.list_alias_fields:
                        alias_key='$listalias.'+key
                        source=statement.value
                        while source.op=='as_list':source=source.args[0]
                        source_field=next((f for f in self.list_alias_fields if source.op=='name' and f.name==source.args[0] and not any(g.name==f.name for g in self.scopes[scope].fields if scope!='global')),None)
                        if statement.value.op in ('append_agent','remove_agent'):
                            # << and >> mutate the referenced list, including its aliases.
                            pass
                        elif source_field is not None:
                            self.emit(f"{indent}s[{alias_key!r}] = jnp.where({mask},s[{('$listalias.global.'+source_field.name)!r}],s[{alias_key!r}])")
                        else:
                            self.emit(f"{indent}s['$list_generation'] = s['$list_generation']+({mask}).astype(jnp.int64)")
                            self.emit(f"{indent}s[{alias_key!r}] = jnp.where({mask},s['$list_generation'],s[{alias_key!r}])")
                        for other in self.list_alias_fields:
                            if other is field or other.type!=field.type:continue
                            other_key='global.'+other.name
                            condition=f'({mask}) & (s[{("$listalias."+other_key)!r}]==s[{alias_key!r}])'
                            self.emit(f"{indent}s[{other_key!r}] = jnp.where({condition},s[{key!r}],s[{other_key!r}])")
                            self.emit(f"{indent}s[{('$listmask.'+other_key)!r}] = jnp.where({condition},s[{mask_key!r}],s[{('$listmask.'+other_key)!r}])")
                    continue
                if field.name == "location" and "moving" in self.scopes[scope].skills and self.agent_index is not None and not self.ir.world_torus:
                    self.counter += 1; delta = f"location_delta_{self.counter}"; angle = f"location_angle_{self.counter}"
                    evaluated = f"location_value_{self.counter}"
                    self.emit(f"{indent}{evaluated} = {value}")
                    value = evaluated
                    self.emit(f"{indent}{delta} = ({value}) - s[{key!r}][{self.agent_index}]")
                    self.emit(f"{indent}{angle} = native_atan2({delta}[1], {delta}[0]) * jnp.asarray(57.29577951308232, jnp.float64)")
                    self.emit(f"{indent}{angle} = jnp.mod(jnp.where({angle} < 0.0, {angle} + 360.0, {angle}), 360.0)")
                    heading_key = scope + ".heading"
                    self.emit(f"{indent}s[{heading_key!r}] = s[{heading_key!r}].at[{self.agent_index}].set(jnp.where(({mask}) & ~jnp.all(jnp.abs({delta}) <= 0.01), {angle}, s[{heading_key!r}][{self.agent_index}]))")
                value = self.coerce_field(value, field, scope)
                old = f"s[{key!r}]"
                if owner != "global" and self.agent_index is not None:
                    old += f"[{self.agent_index}]"
                selected = f"jnp.expand_dims({mask}, -1)" if field.type in ("point","rgb") else mask
                if field.name == "heading" and "moving" in self.scopes[scope].skills: value = f"jnp.mod({value}, 360.0)"
                update = f"jnp.where({selected}, {value}, {old}).astype(DTYPES[{field.type!r}])"
                if owner != "global" and self.agent_index is not None:
                    update = f"s[{key!r}].at[{self.agent_index}].set({update})"
                self.emit(f"{indent}s[{key!r}] = {update}", statement.token)
                if field.type=='polygon':
                    self.emit(f"{indent}s[{('$listmask.'+key)!r}] = s[{('$listmask.'+key)!r}].at[{self.agent_index}].set(jnp.where({mask},jnp.ones((4,),jnp.bool_),s[{('$listmask.'+key)!r}][{self.agent_index}]))")
                    self.emit(f"{indent}s[{('$shape_center.'+owner)!r}] = s[{('$shape_center.'+owner)!r}].at[{self.agent_index}].set(jnp.where({mask},s[{(owner+'.location')!r}][{self.agent_index}],s[{('$shape_center.'+owner)!r}][{self.agent_index}]))")
                if spatial_change is not None:self.record_spatial_position(owner,self.agent_index,spatial_change,indent,mask)
                if statement.metadata.get('ignore_nil_location'):mask=assignment_mask;self.active_mask=mask
            elif statement.kind=='skill_goto':
                self.counter+=1;prefix=f'goto_{self.counter}'
                def attr(name):return f"s[{(scope+'.'+name)!r}][{self.agent_index}]"
                speed=self.expression(statement.metadata['speed'],scope) if statement.metadata.get('speed') is not None else attr('speed')
                target=self.expression(statement.value,scope)
                if statement.value.type.startswith('agent:'):
                    target_owner=statement.value.type.split(':')[1]
                    self.emit(f'{indent}{prefix}_agent = {target}')
                    self.emit(f"{indent}s['$error'] = jnp.where(({mask}) & (({prefix}_agent<0) | ({prefix}_agent>={self.capacity(target_owner)})) & (s['$error']==0),jnp.asarray(5,jnp.int32),s['$error'])")
                    target=f"s[{(target_owner+'.location')!r}][jnp.maximum(0,{prefix}_agent)]"
                self.emit(f'{indent}{prefix}_target = {target}')
                self.emit(f'{indent}{prefix}_delta = jax.lax.optimization_barrier({prefix}_target-{attr("location")})')
                distance=f'jnp.sqrt(jnp.sum({prefix}_delta*{prefix}_delta))'
                if getattr(self.ir,'optimize_fusion',False):distance=f'jax.lax.optimization_barrier(jnp.sqrt(jax.lax.optimization_barrier(jnp.sum(jax.lax.optimization_barrier({prefix}_delta*{prefix}_delta)))))'
                self.emit(f'{indent}{prefix}_distance = {distance}')
                self.emit(f'{indent}{prefix}_speed = ({speed}).astype(jnp.float64)')
                bounds=f"jnp.all(({prefix}_target[:2]>=jnp.min(s['global.shape'][:,:2],axis=0)) & ({prefix}_target[:2]<=jnp.max(s['global.shape'][:,:2],axis=0)))" if any(f.type=='geometry' and f.name=='shape' for f in self.scopes['global'].fields) else f'jnp.all(({prefix}_target[:2]>=0) & ({prefix}_target[:2]<=100))'
                # Native continuous goto follows the path even outside the
                # world's envelope; unlike move/wander it does not normalize it.
                valid=f'jnp.isfinite({prefix}_speed) & ({prefix}_speed>=0) & jnp.all(jnp.isfinite({prefix}_target)) & ({prefix}_target[2]==0)'
                self.emit(f"{indent}s['$error'] = jnp.where(({mask}) & ~({valid}) & (s['$error']==0),jnp.asarray(11,jnp.int32),s['$error'])")
                self.emit(f'{indent}{prefix}_moving = ({mask}) & ({valid}) & ~jnp.all(jnp.abs({prefix}_delta)<=0.01)')
                fraction=f'{prefix}_speed/jnp.where({prefix}_distance>0,{prefix}_distance,1.)'
                if getattr(self.ir,'optimize_fusion',False):fraction=f'jax.lax.optimization_barrier({fraction})'
                self.emit(f'{indent}{prefix}_fraction = jnp.minimum(1.,{fraction})')
                self.emit(f'{indent}{prefix}_destination = jnp.where({prefix}_speed<{prefix}_distance,jax.lax.optimization_barrier({attr("location")}+jax.lax.optimization_barrier({prefix}_fraction*{prefix}_delta)),{prefix}_target)')
                self.emit(f'{indent}{prefix}_heading = jnp.mod(native_atan2({prefix}_delta[1],{prefix}_delta[0])*jnp.asarray(57.29577951308232,jnp.float64),360.)')
                self.emit(f"{indent}s[{(scope+'.real_speed')!r}] = s[{(scope+'.real_speed')!r}].at[{self.agent_index}].set(jnp.where({mask},jnp.where({prefix}_moving,jax.lax.optimization_barrier({prefix}_distance*{prefix}_fraction),0.),{attr('real_speed')}))")
                self.emit(f"{indent}s[{(scope+'.heading')!r}] = s[{(scope+'.heading')!r}].at[{self.agent_index}].set(jnp.where({prefix}_moving,{prefix}_heading,{attr('heading')}))")
                self.emit(f"{indent}s[{(scope+'.location')!r}] = s[{(scope+'.location')!r}].at[{self.agent_index}].set(jnp.where({prefix}_moving,{prefix}_destination,{attr('location')}))")
                self.record_spatial_position(scope,self.agent_index,f'{prefix}_moving',indent,mask)
            elif statement.kind == "skill_move":
                self.counter += 1; suffix = str(self.counter)
                def attr(name): return self.expression(Expr("name", [name], "point" if name == "location" else "float", statement.token), scope)
                position, heading, speed = attr("location"), attr("heading"), attr("speed")
                if statement.value is not None:speed=self.expression(statement.value,scope)
                destination, valid = "destination_" + suffix, "valid_" + suffix
                if self.native_trig:
                    self.emit(f"{indent}trig_{suffix}=native_hotspot_heading_sincos({heading})")
                    sine,cosine=f'trig_{suffix}[0]',f'trig_{suffix}[1]'
                else:sine,cosine=f'jnp.sin(jnp.deg2rad({heading}))',f'jnp.cos(jnp.deg2rad({heading}))'
                self.emit(f"{indent}{destination} = jax.lax.optimization_barrier({position} + jnp.stack([jax.lax.optimization_barrier({speed} * {cosine}), jax.lax.optimization_barrier({speed} * {sine}), jnp.asarray(0.0)]))", statement.token)
                if self.ir.world_torus:
                    world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if any(f.type=='geometry' and f.name=='shape' for f in self.scopes['global'].fields) else 'jnp.asarray([100.,100.,0.])'
                    minimum="jnp.min(s['global.shape'],axis=0)" if any(f.type=='geometry' and f.name=='shape' for f in self.scopes['global'].fields) else 'jnp.zeros((3,),jnp.float64)'
                    self.emit(f'{indent}torus_candidates_{suffix} = {destination}[None,:] + jnp.asarray([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[1,-1,0],[0,-1,0],[-1,-1,0],[-1,0,0],[-1,1,0]],jnp.float64)*({world})')
                    self.emit(f'{indent}torus_valid_{suffix} = jnp.all((torus_candidates_{suffix}[:,:2]>=({minimum})[:2]) & (torus_candidates_{suffix}[:,:2]<=(({minimum})+({world}))[:2]),axis=-1)')
                    self.emit(f'{indent}{destination} = jnp.where(jnp.any(torus_valid_{suffix}),torus_candidates_{suffix}[jnp.argmax(torus_valid_{suffix})],{destination})')
                if any(f.name=='shape' and f.type=='geometry' for f in self.scopes['global'].fields):
                    self.emit(f"{indent}{valid} = jnp.all(({destination}[:2] >= jnp.min(s['global.shape'][:,:2],axis=0)) & ({destination}[:2] <= jnp.max(s['global.shape'][:,:2],axis=0)))")
                else:self.emit(f"{indent}{valid} = jnp.all(({destination}[:2] >= 0.0) & ({destination}[:2] <= 100.0))")
                failed_speed=attr('real_speed') if statement.metadata.get('wander') else '0.0'
                self.emit(f"{indent}s[{(scope+'.real_speed')!r}] = s[{(scope+'.real_speed')!r}].at[{self.agent_index}].set(jnp.where({mask}, jnp.where({valid}, jnp.sqrt(jnp.sum(({destination} - {position}) ** 2)), {failed_speed}), {attr('real_speed')}))")
                delta = "delta_" + suffix
                orientation = "orientation_" + suffix
                self.emit(f"{indent}{delta} = {destination} - {position}")
                self.emit(f"{indent}{orientation} = native_atan2({delta}[1], {delta}[0]) * jnp.asarray(57.29577951308232, jnp.float64)")
                self.emit(f"{indent}{orientation} = jnp.mod(jnp.where({orientation} < 0.0, {orientation} + 360.0, {orientation}), 360.0)")
                self.emit(f"{indent}{orientation} = jnp.where(jnp.all(jnp.abs({delta}) <= 0.01), {heading}, {orientation})")
                if self.ir.world_torus:self.emit(f'{indent}{orientation} = {heading}')
                self.emit(f"{indent}s[{(scope+'.heading')!r}] = s[{(scope+'.heading')!r}].at[{self.agent_index}].set(jnp.where({mask}, jnp.where({valid}, {orientation}, jnp.mod({heading} - 180.0, 360.0)), {heading}))")
                self.emit(f"{indent}s[{(scope+'.location')!r}] = s[{(scope+'.location')!r}].at[{self.agent_index}].set(jnp.where(({mask}) & {valid}, {destination}, {position}))")
                self.record_spatial_position(scope,self.agent_index,f'jnp.any({delta}!=0)',indent,f'({mask}) & {valid}')
            elif statement.kind == "if":
                condition = self.mask(self.expression(statement.value, scope), indent)
                yes = self.mask(f"({mask}) & {condition}", indent)
                no = self.mask(f"({mask}) & ~{condition}", indent)
                self.statements(statement.body, scope, yes, indent)
                self.statements(statement.alternative, scope, no, indent)
            elif statement.kind=='diffuse':
                i=statement.metadata['diffusion_id'];enabled='$diffusion.enabled.'+str(i);proportion='$diffusion.proportion.'+str(i)
                value=self.expression(statement.metadata['proportion'],scope)
                self.emit(f"{indent}s[{enabled!r}]=s[{enabled!r}] | ({mask})")
                self.emit(f"{indent}s[{proportion!r}]=jnp.where({mask},({value}).astype(jnp.float64),s[{proportion!r}])")
                self.emit(f"{indent}s['$error']=jnp.where(({mask}) & (~jnp.isfinite(s[{proportion!r}]) | (s[{proportion!r}]<0.) | (s[{proportion!r}]>1.)) & (s['$error']==0),jnp.asarray(15,jnp.int32),s['$error'])")
            elif statement.kind=='fsm_dispatch':
                self.counter+=1;prefix=f'fsm_dispatch_{self.counter}';key=scope+'.state';taken='$fsm_taken.'+prefix
                self.emit(f"{indent}{prefix}_state=s[{key!r}][{self.agent_index}]")
                self.emit(f"{indent}s[{taken!r}]=jnp.asarray(False)")
                saved=getattr(self,'fsm_context',None);self.fsm_context=(prefix+'_state',taken,key)
                self.statements(statement.body,scope,mask,indent)
                self.fsm_context=saved;self.emit(f"{indent}s.pop({taken!r})")
            elif statement.kind=='fsm_state':
                state,taken,key=self.fsm_context
                index=[state.name for state in self.scopes[scope].fsm_states].index(statement.target)
                selected=self.mask(f'({mask}) & ({state}=={index})',indent)
                self.statements(statement.body,scope,selected,indent)
            elif statement.kind=='fsm_transition':
                state,taken,key=self.fsm_context
                saved_mask=self.active_mask;self.active_mask=f'({mask}) & ~s[{taken!r}]'
                condition=self.expression(statement.value,scope);self.active_mask=saved_mask
                select=self.mask(f'({mask}) & ~s[{taken!r}] & ({condition})',indent)
                self.emit(f"{indent}s[{key!r}]=s[{key!r}].at[{self.agent_index}].set(jnp.where({select},jnp.asarray({statement.metadata['state_index']},jnp.int32),s[{key!r}][{self.agent_index}]))")
                self.emit(f"{indent}s[{taken!r}]=s[{taken!r}] | ({select})")
            elif statement.kind=='pause':
                self.emit(f"{indent}s['$paused']=s['$paused'] | ({mask})",statement.token)
            elif statement.kind in ('write','save'):
                record=dict(id=statement.metadata['output_id'],parts=[])
                if statement.kind=='save':
                    record.update(kind='save',path=statement.metadata['path'])
                    record['generation']='$save.generation.'+str(record['id']);record['rewrite']='$save.rewrite.'+str(record['id'])
                    rewrite=self.expression(statement.metadata['rewrite'],scope)
                    self.emit(f"{indent}s[{record['generation']!r}]=s[{record['generation']!r}]+jnp.asarray({mask},jnp.int64)")
                    self.emit(f"{indent}s[{record['rewrite']!r}]=jnp.where({mask},{rewrite},s[{record['rewrite']!r}])")
                for part in statement.metadata['parts']:
                    if 'literal' in part:record['parts'].append(part)
                    else:
                        value=self.expression(part['expression'],scope)
                        if statement.kind=='save':self.emit(f"{indent}s[{part['key']!r}]=jnp.where({mask},({value}).astype(DTYPES[{part['type']!r}]),s[{part['key']!r}])",statement.token)
                        else:self.emit(f"{indent}s[{part['key']!r}] = ({value}).astype(DTYPES[{part['type']!r}])",statement.token)
                        record['parts'].append({k:v for k,v in part.items() if k!='expression'})
                self.output_records.append(record)
            elif statement.kind in ("create", "birth"):
                count = self.expression(statement.value,scope)
                creation_with={}
                for name,expression in statement.metadata.get('initial_attributes',{}).items():
                    value=self.expression(expression,scope);self.counter+=1;temporary=f'creation_attribute_{self.counter}'
                    self.emit(f'{indent}{temporary}={value}');creation_with[name]=temporary
                if self.reserved(statement.target):
                    self.counter += 1; prefix=f'creation_{self.counter}'
                    cursor='$next.'+statement.target
                    present='$present.'+statement.target
                    capacity=self.capacity(statement.target)
                    self.emit(f'{indent}{prefix}_count = {count}')
                    self.emit(f'{indent}{prefix}_start = s[{cursor!r}]')
                    self.emit(f"{indent}s['$error'] = jnp.where(({mask}) & (({prefix}_count<0) | ({prefix}_count>{capacity}-{prefix}_start)) & (s['$error']==0),jnp.asarray(6,jnp.int32),s['$error'])",statement.token)
                    self.emit(f'{indent}{prefix}_count = jnp.where({mask},jnp.clip({prefix}_count,0,{capacity}-{prefix}_start),0)')
                    self.emit(f'{indent}{prefix}_end = {prefix}_start+{prefix}_count')
                    self.emit(f"{indent}s[{present!r}] = s[{present!r}] | ((jnp.arange({capacity})>={prefix}_start) & (jnp.arange({capacity})<{prefix}_end))")
                    self.emit(f'{indent}s[{cursor!r}] = {prefix}_end')
                    if capacity:
                        self.initialize(self.scopes[statement.target],start=prefix+'_start',count=prefix+'_count',creation_body=statement.body,creation_with=creation_with,base=indent)
                    continue
                if statement.kind == 'birth':
                    raise CompileError(statement.token.source or '<IR>',statement.token,'Runtime birth requires an explicit reserved capacity','GAMAX_CAPACITY')
                expected = statement.metadata['count']
                self.emit(f"{indent}s['$error'] = jnp.where(({mask}) & (({count}) != {expected}) & (s['$error'] == 0), jnp.asarray(4,jnp.int32),s['$error'])",statement.token)
                self.initialize(self.scopes[statement.target], start=statement.metadata['start'], count=expected, creation_body=statement.body,creation_with=creation_with)
            elif statement.kind == 'die':
                key='$present.'+scope
                self.emit(f'{indent}s[{key!r}] = s[{key!r}].at[{self.agent_index}].set(s[{key!r}][{self.agent_index}] & ~({mask}))',statement.token)
                if scope in self.spatial_targets:
                    leaf='$spatial.'+scope+'.leaf'
                    if getattr(self.ir,'optimize_spatial_order',False):self.emit(f"{indent}previous_leaf = s[{leaf!r}]")
                    self.emit(f"{indent}s[{leaf!r}] = s[{leaf!r}].at[{self.agent_index}].set(jnp.where({mask},-1,s[{leaf!r}][{self.agent_index}]))")
                    if getattr(self.ir,'optimize_spatial_order',False):
                        count_key='$spatial.'+scope+'.counts'
                        self.emit(f"{indent}s[{count_key!r}] = s[{count_key!r}].at[jnp.maximum(previous_leaf[{self.agent_index}],0)].add(jnp.where(({mask}) & (previous_leaf[{self.agent_index}]>=0),-1,0))")
                        sorted_key='$spatial.'+scope+'.sorted';order_key='$spatial.'+scope+'.order'
                        self.emit(f"{indent}s[{sorted_key!r}] = jax.lax.cond({mask},lambda _:quadtree_update_order(s[{sorted_key!r}],previous_leaf,s[{leaf!r}],s[{order_key!r}],{self.agent_index}),lambda _:s[{sorted_key!r}],operand=None)")
                    self.update_contacts(scope,self.agent_index,mask,indent)
            elif statement.kind == 'ask':
                self.counter+=1;function=f'ask_{self.counter}'
                selection=self.expression(statement.value,scope)
                if statement.value.type.startswith('agent:'):
                    count=1
                    self.emit(f'{indent}{function}_reference = {selection}')
                    self.emit(f'{indent}{function}_values = jnp.asarray([{function}_reference],jnp.int32)')
                    self.emit(f'{indent}{function}_mask = jnp.asarray([{function}_reference>=0],jnp.bool_)')
                else:
                    count=self.collection_capacity(statement.value,scope,self.each_scope)
                    self.emit(f'{indent}{function}_values,{function}_mask = {selection}')
                if not count:continue
                snapshot=function+'_indices';limit=function+'_count'
                self.emit(f'{indent}{snapshot} = {function}_values[jnp.nonzero({function}_mask,size={count},fill_value=0)[0]]')
                self.emit(f'{indent}{limit} = jnp.sum({function}_mask,dtype=jnp.int32)')
                saved=self.agent_index,self.caller_index,self.each_scope,self.each_indices
                if self.agent_index is not None:
                    self.emit(f'{indent}{function}_caller = {self.agent_index}')
                    self.caller_index=function+'_caller'
                else:self.caller_index=None
                self.emit(f'{indent}def {function}(iteration,state):',statement.token)
                self.emit(f'{indent}    s = dict(state)')
                self.agent_index=function+'_agent';self.each_scope=None;self.each_indices=None
                index=f'{limit}-1-iteration' if self.reverse_agents else 'iteration'
                self.emit(f'{indent}    {self.agent_index} = {snapshot}[{index}]')
                ask_mask=f'({mask}) & s[{("$present."+statement.target)!r}][{self.agent_index}]' if self.reserved(statement.target) else mask
                self.statements(statement.body,statement.target,ask_mask,indent+'    ')
                self.emit(f'{indent}    return s')
                self.agent_index,self.caller_index,self.each_scope,self.each_indices=saved
                self.emit(f'{indent}s = jax.lax.fori_loop(0,{limit},{function},s)')
            else:
                raise CompileError(statement.token.source or '<IR>',statement.token,'No backend implementation for statement '+statement.kind,'GAMAX_BACKEND')

        for name in self.locals:
            if name in saved_locals: continue
            self.emit(f"{indent}s.pop({self.locals[name][0]!r})")
            if self.locals[name][1].startswith('agents:'):self.emit(f"{indent}s.pop({('$listmask.'+self.locals[name][0])!r})")
        self.locals = saved_locals

    def stream_integer_accumulation(self, statement, scope, mask, indent):
        """Fuse a proven int32 consumer with accumulation; never build the product list."""
        expr=statement.value
        container=self.expression(expr.args[0],scope)
        owner=expr.args[0].type.split(':')[1]
        self.counter+=1;prefix=f'stream_{self.counter}'
        self.emit(f'{indent}{prefix}_values,{prefix}_present={container}')
        capacity=self.collection_capacity(expr.args[0],scope,self.each_scope)
        if not capacity:return
        inner=expr.args[1];pair=(inner.args[0].type.split(':')[1],owner) if inner.op=='overlapping' else None
        fast=False
        def readonly(value):
            if value.op=='literal':return True
            if value.op.startswith('attribute:'):
                reference=value.args[0]
                if not (reference.op=='each_ref' or reference.op=='name' and reference.args[0]==statement.target):return False
                field=next((f for f in self.scopes[reference.type.split(':')[1]].fields if f.name==value.op.split(':')[1]),None)
                return field is not None and field.getter is None and field.type in ('int','bool','float')
            if value.op in ('name','each'):
                owner=value.metadata.get('owner_scope',scope)
                field=next((f for candidate in (owner,'global') for f in self.scopes[candidate].fields if f.name==value.args[0]),None)
                return field is not None and field.getter is None and field.type in ('int','bool','float')
            return value.op in ('+','-','*','negate','=','!=','<','<=','>','>=','and','or','not') and all(readonly(arg) for arg in value.args if isinstance(arg,Expr))
        def body_safe(items):
            return all(item.kind=='assign' and readonly(item.value.args[1]) or item.kind=='if' and readonly(item.value) and body_safe(item.body) and body_safe(item.alternative) for item in items)
        if pair in self.grid_reference_pairs and body_safe(statement.body):
            population,grid_owner=pair;grid=self.scopes[grid_owner].grid
            references=f'{prefix}_references';valid=f'{prefix}_reference_valid'
            self.emit(f'{indent}{references}=s[{(population+"."+self.grid_reference_pairs[pair])!r}]')
            world="jnp.max(s['global.shape'],axis=0)-jnp.min(s['global.shape'],axis=0)" if self.has_geometry() else 'jnp.asarray([100.,100.,0.])'
            self.emit(f'{indent}{prefix}_half=({world})[:2]/jnp.asarray([{grid["width"]},{grid["height"]}])/2')
            self.emit(f'{indent}{valid}=grid_reference_membership_valid(s[{(population+".location")!r}],{references},s[{("$present."+population)!r}],s[{(grid_owner+".location")!r}],{prefix}_half,{grid["width"]},{grid["height"]})')
            cells=grid['width']*grid['height'];original_indent=indent;fast=True
            self.emit(f'{indent}{valid}={valid} & jnp.all(~{prefix}_present | (({prefix}_values>=0) & ({prefix}_values<{cells})))')
            self.emit(f'{indent}def {prefix}_fast(state):')
            fast_indent=indent+'    ';self.emit(f'{fast_indent}s=dict(state)')
            self.emit(f'{fast_indent}{prefix}_population_counts=jnp.bincount(jnp.clip({references},0,{cells-1}),weights=s[{("$present."+population)!r}].astype(jnp.int32),length={cells})')
            self.emit(f'{fast_indent}{prefix}_counts={prefix}_population_counts[jnp.clip({prefix}_values,0,{cells-1})]')
            self.emit(f"{fast_indent}s['$rng']=mersenne_discard_uniforms(s['$rng'],jnp.sum(jnp.where(({mask}) & {prefix}_present & ({prefix}_counts>=2),2*{prefix}_counts,0),dtype=jnp.int64))")
            weights=f'{prefix}_weights'
            self.emit(f'{fast_indent}{weights}=jnp.bincount(jnp.clip({prefix}_values,0,{cells-1}),weights={prefix}_present.astype(jnp.int32),length={cells})[jnp.clip({references},0,{cells-1})]')
            local_key='$reference_stream'+str(self.counter)
            self.emit(f'{fast_indent}s[{local_key!r}]=jnp.arange({self.capacity(population)},dtype=jnp.int32)')
            saved=self.each_scope,self.each_indices,self.eval_indent,self.active_mask,dict(self.locals)
            self.locals[statement.target]=(local_key,'agent:'+population)
            self.integer_reductions(statement.body,scope,f'({mask}) & s[{("$present."+population)!r}]',fast_indent,weights=weights)
            self.each_scope,self.each_indices,self.eval_indent,self.active_mask,self.locals=saved
            self.emit(f'{fast_indent}s.pop({local_key!r})');self.emit(f'{fast_indent}return s')
            self.emit(f'{indent}def {prefix}_slow(state):');indent+='    ';self.emit(f'{indent}s=dict(state)')
            self.grid_reference_sites+=1
        saved=(self.each_scope,self.each_indices,self.eval_indent,self.active_mask,dict(self.locals))
        self.emit(f'{indent}def {prefix}_evaluate(iteration,state):')
        self.emit(f'{indent}    s=dict(state)')
        self.each_scope=owner;self.each_indices=f'{prefix}_values[iteration]'
        self.eval_indent=indent+'    ';self.active_mask=f'({mask}) & {prefix}_present[iteration]'
        mapped=self.expression(expr.args[1],scope)
        self.emit(f'{self.eval_indent}{prefix}_mapped,{prefix}_mask={mapped}')
        key='$stream_iteration'+str(self.counter)
        self.emit(f'{self.eval_indent}s[{key!r}]={prefix}_mapped')
        self.locals[statement.target]=(key,'agent:'+expr.type.split(':')[1])
        # each bindings in the consumer belong to its own expressions, rather
        # than to the outer grid binding used by the spatial mapper.
        self.each_scope,self.each_indices=saved[:2]
        self.integer_reductions(statement.body,scope,f'({mask}) & {prefix}_present[iteration] & {prefix}_mask',self.eval_indent)
        self.emit(f'{self.eval_indent}s.pop({key!r})')
        self.emit(f'{self.eval_indent}return s')
        self.each_scope,self.each_indices,self.eval_indent,self.active_mask,self.locals=saved
        self.emit(f'{indent}def {prefix}(iteration,state):')
        self.emit(f'{indent}    return jax.lax.cond(({mask}) & {prefix}_present[iteration],lambda s:{prefix}_evaluate(iteration,s),lambda s:s,state)')
        self.emit(f'{indent}{prefix}_limit=jnp.where({mask},jnp.max(jnp.where({prefix}_present,jnp.arange({capacity},dtype=jnp.int32)+1,0)),0)')
        self.emit(f'{indent}s=jax.lax.fori_loop(0,{prefix}_limit,{prefix},s)')
        if fast:
            self.emit(f'{indent}return s')
            self.emit(f'{original_indent}s=jax.lax.cond({valid},{prefix}_fast,{prefix}_slow,s)')

    def integer_reductions(self, items, scope, mask, indent, weights=None):
        """Lower only the type/effect-proven independent int32 sums."""
        for item in items:
            self.eval_indent,self.active_mask=indent,mask
            if item.kind=='assign':
                key,_=self.locals[item.target]
                increment=self.expression(item.value.args[1],scope)
                if weights is not None:increment=f'({increment})*{weights}'
                self.emit(f"{indent}s[{key!r}] = (s[{key!r}] + jnp.sum(jnp.where({mask},{increment},0),dtype=jnp.int32)).astype(jnp.int32)",item.token)
            elif item.kind=='if':
                condition=self.mask(self.expression(item.value,scope),indent)
                self.integer_reductions(item.body,scope,f'({mask}) & ({condition})',indent,weights)
                self.integer_reductions(item.alternative,scope,f'({mask}) & ~({condition})',indent,weights)
            else:
                raise CompileError(item.token.source or '<IR>',item.token,'Invalid integer reduction proof','GAMAX_BACKEND')

    def coerce_field(self, value, field, scope):
        if not field.facets: return value
        # Native NumberVariable first converts to the declared type, then returns
        # min immediately if below it; max is checked only otherwise.
        self.counter += 1
        name = f"coerced_{self.counter}"
        self.emit(f"{self.eval_indent}{name} = ({value}).astype(DTYPES[{field.type!r}])")
        low = self.expression(field.facets["min"], scope) if "min" in field.facets else None
        high = self.expression(field.facets["max"], scope) if "max" in field.facets else None
        above = f"jnp.where({name} > ({high}), ({high}), {name})" if high else name
        return f"jnp.where({name} < ({low}), ({low}), {above})" if low else above

    def initialize_grid_fields(self, scope):
        count = self.capacity(scope.name)
        for field in scope.fields:
            shape = (count,3) if field.type == "point" else (count,4) if field.type == "rgb" else (count,)
            if field.type.startswith('agents:'):
                shape = (count,field.capacity)
                self.emit(f"    s[{('$listmask.'+scope.name+'.'+field.name)!r}] = jnp.zeros({shape!r},jnp.bool_)")
            self.emit(f"    s[{(scope.name+'.'+field.name)!r}] = jnp.zeros({shape!r}, DTYPES[{field.type!r}])",field.token)
            def invariant(expr):
                if expr.op=='literal':return True
                if expr.op=='name':
                    name=expr.args[0]
                    return not any(f.name==name for f in scope.fields) and any(f.name==name and f.getter is None for f in self.scopes['global'].fields)
                if expr.op=='/':
                    if expr.args[1].op!='literal' or expr.args[1].args[0]==0:return False
                elif expr.op not in ('+','-','*','negate','point','==','!=','<','<=','>','>=','and','or','not'):return False
                return all(invariant(arg) for arg in expr.args if isinstance(arg,Expr))
            if getattr(self.ir,'optimize_constant_grid_initializers',False) and field.type in ('int','float','bool','point','rgb') and invariant(field.value) and all(invariant(value) for name,value in field.facets.items() if name in ('min','max') and isinstance(value,Expr)):
                self.agent_index=None;self.eval_indent,self.active_mask='    ','jnp.asarray(True)'
                value=self.coerce_field(self.expression(field.value,scope.name),field,scope.name)
                self.emit(f"    s[{(scope.name+'.'+field.name)!r}] = jnp.broadcast_to(({value}).astype(DTYPES[{field.type!r}]),{shape!r}).copy()")
                self.broadcast_grid_sites+=1
                continue
            if getattr(self.ir,'optimize_grid_random_fields',False) and field.type in ('int','float') and field.value.op=='call:rnd' and all(arg.type in ('int','float') and invariant(arg) for arg in field.value.args) and all(self.independent_cell_expression(value,scope.name) for value in field.facets.values() if isinstance(value,Expr)):
                self.counter+=1;prefix=f'grid_uniforms_{self.counter}'
                self.agent_index=None;self.eval_indent,self.active_mask='    ','jnp.asarray(True)'
                args=[self.expression(arg,scope.name) for arg in field.value.args]
                low,high=('jnp.asarray(0,jnp.float64)',args[0]) if len(args)==1 else args
                self.emit(f'    {prefix}_low={low}');self.emit(f'    {prefix}_high={high}')
                self.emit(f'    {prefix}_valid=({prefix}_low<={prefix}_high) & jnp.isfinite({prefix}_low) & jnp.isfinite({prefix}_high)')
                self.emit(f"    s['$error']=jnp.where(~{prefix}_valid & (s['$error']==0),jnp.asarray(9,jnp.int32),s['$error'])")
                self.emit(f"    s['$rng'],{prefix}=jax.lax.cond({prefix}_valid,lambda rng:mersenne_uniforms(rng,{count}),lambda rng:(rng,jnp.zeros(({count},),jnp.float64)),s['$rng'])")
                if field.value.type=='int':value=f'(({prefix}_low).astype(jnp.int64)+jnp.trunc((({prefix}_high).astype(jnp.int64)+1-({prefix}_low).astype(jnp.int64)).astype(jnp.float64)*{prefix}).astype(jnp.int64)).astype(jnp.int32)'
                else:value=f'jax.lax.optimization_barrier({prefix}_low+jax.lax.optimization_barrier(({prefix}_high-{prefix}_low)*{prefix}))'
                self.agent_index=f'jnp.arange({count},dtype=jnp.int32)'
                value=self.coerce_field(value,field,scope.name)
                self.emit(f"    s[{(scope.name+'.'+field.name)!r}]=({value}).astype(DTYPES[{field.type!r}])")
                self.agent_index=None;self.grid_random_sites+=1
                continue
            if getattr(self.ir,'optimize_grid_random_fields',False) and field.value.op=='neighbors_at' and field.value.args[0].op=='name' and field.value.args[0].args[0]=='self' and invariant(field.value.args[1]):
                grid=scope.grid;bound=field.value.metadata['max_radius'];key=(grid['width'],grid['height'],bound,grid['neighbors']==8,grid['torus'])
                if key in self.grid_geometry:
                    number=self.grid_geometry[key];self.counter+=1;prefix=f'grid_neighborhoods_{self.counter}'
                    self.agent_index=None;self.eval_indent,self.active_mask='    ','jnp.asarray(True)'
                    radius=self.expression(field.value.args[1],scope.name)
                    self.emit(f"    s['$rng'],({prefix}_values,{prefix}_mask)=initialized_grid_neighborhoods(s['$rng'],GRID_GEOMETRY_{number},GRID_GEOMETRY_RADII_{number},{radius})")
                    self.emit(f"    s[{(scope.name+'.'+field.name)!r}]={prefix}_values")
                    self.emit(f"    s[{('$listmask.'+scope.name+'.'+field.name)!r}]={prefix}_mask")
                    self.grid_neighborhood_sites+=1
                    continue
            if getattr(self.ir,'optimize_independent_grid',False) and count>=1024 and field.type in ('int','float','bool','point','rgb') and self.pure_grid_initializer(field.value,scope.name) and all(self.pure_grid_initializer(value,scope.name) for value in field.facets.values() if isinstance(value,Expr)):
                self.counter+=1;function=f'pure_grid_initializer_{self.counter}'
                outer=self.lines;self.lines=[]
                self.emit(f'def {function}(state):')
                self.emit('    def cell(agent_i):');self.emit('        s=dict(state)')
                self.agent_index='agent_i';self.eval_indent,self.active_mask='        ','jnp.asarray(True)'
                value=self.coerce_field(self.expression(field.value,scope.name),field,scope.name)
                self.emit(f"        return ({value}).astype(DTYPES[{field.type!r}])")
                self.emit(f'    return jax.vmap(cell)(jnp.arange({count},dtype=jnp.int32))')
                self.emit(f'{function}_jit=jax.jit({function},compiler_options=JIT_OPTIONS)')
                self.pure_grid_initializer_helpers.extend(self.lines);self.lines=outer
                self.pure_grid_initializer_names.append(function+'_jit')
                self.emit(f"    s[{(scope.name+'.'+field.name)!r}]={function}_jit(s)")
                self.agent_index=None;self.eval_indent,self.active_mask='    ','jnp.asarray(True)'
                self.pure_grid_initializer_sites+=1
                continue
            self.counter += 1; function = f"initialize_grid_field_{self.counter}"
            self.emit(f"    def {function}(agent_i, state):")
            self.emit("        s = dict(state)")
            self.agent_index = "agent_i"
            self.eval_indent, self.active_mask = "        ", "jnp.asarray(True)"
            value = self.coerce_field(self.expression(field.value,scope.name),field,scope.name)
            if field.type.startswith('agents:'):
                self.counter += 1; temporary = f'list_initializer_{self.counter}'
                self.emit(f'        {temporary} = {value}')
                self.emit(f"        s[{('$listmask.'+scope.name+'.'+field.name)!r}] = s[{('$listmask.'+scope.name+'.'+field.name)!r}].at[agent_i].set({temporary}[1])")
                value = temporary+'[0]'
            self.emit(f"        s[{(scope.name+'.'+field.name)!r}] = s[{(scope.name+'.'+field.name)!r}].at[agent_i].set(({value}).astype(DTYPES[{field.type!r}]))",field.token)
            self.emit("        return s")
            self.emit(f"    s = jax.lax.fori_loop(0, {count}, {function}, s)")
        self.counter += 1; function = f"initialize_grid_bodies_{self.counter}"
        self.emit(f"    def {function}(agent_i, state):")
        self.emit("        s = dict(state)")
        self.agent_index = "agent_i"
        self.statements(scope.init,scope.name,"jnp.asarray(True)","        ")
        self.emit("        return s")
        self.emit(f"    s = jax.lax.fori_loop(0, {count}, {function}, s)")
        self.agent_index = None

    def initialize(self, scope, *, start=0, count=None, creation_body=None, creation_with=None, base="    "):
        if scope.grid:
            self.initialize_grid_fields(scope)
            return
        count = self.ir.populations.get(scope.name,0) if count is None else count
        stop=start+count if type(start) is int and type(count) is int else f'({start})+({count})'
        if scope.name != "global" and count == 0: return
        creation_with={} if creation_with is None else creation_with
        caller_index=self.agent_index
        saved_agent_index=self.agent_index
        saved_locals=dict(self.locals)
        self.locals={}
        saved_caller=self.caller_index
        creation_body = [] if creation_body is None else creation_body
        multiple = scope.name != "global" and (start != 0 or count != self.capacity(scope.name))
        def global_writes(items):
            local={f.name for f in scope.fields}
            return any(item.kind=='assign' and item.target not in local or global_writes(item.body) or global_writes(item.alternative) for item in items)
        sequential = scope.name != "global" and (self.ir.random_initialization or self.has_dynamic(scope.init) or multiple or bool(creation_body) or global_writes(scope.init))
        indent = base+"    " if sequential else base
        if sequential:
            self.counter += 1
            function = f"initialize_agents_{self.counter}"
            self.emit(f"{base}def {function}(agent_i, state):")
            self.emit(f"{base}    s = dict(state)")
            self.agent_index = "agent_i"
        for field in scope.fields:
            self.eval_indent, self.active_mask = indent, "jnp.asarray(True)"
            if field.getter is not None and field.type.startswith('agents:'):
                key=scope.name+'.'+field.name;cap=self.field_capacity(field)
                if sequential:
                    self.emit(f"{indent}s[{key!r}] = s[{key!r}].at[agent_i].set(jnp.zeros(({cap},),jnp.int32))")
                    self.emit(f"{indent}s[{('$listmask.'+key)!r}] = s[{('$listmask.'+key)!r}].at[agent_i].set(jnp.zeros(({cap},),jnp.bool_))")
                else:
                    shape=(cap,) if scope.name=='global' else (self.capacity(scope.name),cap)
                    self.emit(f"{indent}s[{key!r}] = jnp.zeros({shape!r},jnp.int32)")
                    self.emit(f"{indent}s[{('$listmask.'+key)!r}] = jnp.zeros({shape!r},jnp.bool_)")
                continue
            if scope.name == "global":
                self.counter+=1; value=f'global_initializer_{self.counter}'
                self.emit(f'{indent}if {field.name!r} in parameters:')
                self.emit(f"{indent}    {value} = jnp.asarray(parameters[{field.name!r}],dtype=DTYPES[{field.type!r}])")
                self.emit(f'{indent}else:')
                self.eval_indent=indent+'    '
                default=self.expression(field.value,scope.name)
                self.emit(f'{indent}    {value} = {default}')
                self.eval_indent=indent
            else:
                value = creation_with[field.name] if field.name in creation_with else self.expression(field.value, scope.name)
            value = self.coerce_field(value, field, scope.name)
            if field.list_mutable or field.type.startswith('values:'):
                self.emit(f"{indent}s[{('$listmask.'+scope.name+'.'+field.name)!r}] = {value}[1]")
                self.emit(f"{indent}s[{(scope.name+'.'+field.name)!r}] = {value}[0].astype(DTYPES[{field.type!r}])",field.token)
                continue
            if sequential:
                key = scope.name + '.' + field.name
                self.emit(f"{indent}s[{key!r}] = s[{key!r}].at[agent_i].set(({value}).astype(DTYPES[{field.type!r}]))", field.token)
                if field.type=='polygon':
                    self.emit(f"{indent}s[{('$listmask.'+key)!r}] = s[{('$listmask.'+key)!r}].at[agent_i].set(jnp.asarray([True,False,False,False]))")
                    self.emit(f"{indent}s[{('$shape_center.'+scope.name)!r}] = s[{('$shape_center.'+scope.name)!r}].at[agent_i].set(s[{(scope.name+'.location')!r}][agent_i])")
                if field.name=='location':self.record_spatial_position(scope.name,'agent_i','jnp.asarray(True)',indent,'jnp.asarray(True)')
            else:
                if scope.name != "global":
                    shape = (self.capacity(scope.name),3) if field.type=="point" else (self.capacity(scope.name),4) if field.type=="rgb" else (self.capacity(scope.name),)
                    value=f"jnp.broadcast_to({value}, {shape!r})"
                self.emit(f"{indent}s[{(scope.name+'.'+field.name)!r}] = ({value}).astype(DTYPES[{field.type!r}])", field.token)
        if sequential:
            # Native create initializes attributes of the entire population first,
            # then invokes each agent's init block in creation order.
            self.emit(f"{base}    return s")
            self.emit(f"{base}s = jax.lax.fori_loop({start}, {stop}, {function}, s)")
            self.counter += 1
            function = f"initialize_bodies_{self.counter}"
            self.emit(f"{base}def {function}(agent_i, state):")
            self.emit(f"{base}    s = dict(state)")
        if scope.name == "global":
            for species in self.ir.scopes[1:]:
                if species.grid:continue
                if self.reserved(species.name):
                    self.emit(f"{base}s[{('$present.'+species.name)!r}] = jnp.zeros(({self.capacity(species.name)},),jnp.bool_)")
                    self.emit(f"{base}s[{('$next.'+species.name)!r}] = jnp.asarray(0,jnp.int32)")
                for field in species.fields:
                    count = self.capacity(species.name)
                    shape = (count,3) if field.type == "point" else (count,4) if field.type == "rgb" else (count,)
                    if field.type=='polygon':
                        shape=(count,4,3)
                        self.emit(f"{base}s[{('$listmask.'+species.name+'.'+field.name)!r}] = jnp.zeros(({count},4),jnp.bool_)")
                        self.emit(f"{base}s[{('$shape_center.'+species.name)!r}] = jnp.zeros(({count},3),jnp.float64)")
                    if field.getter is not None and field.type.startswith('agents:'):
                        shape=(count,self.field_capacity(field))
                        self.emit(f"{base}s[{('$listmask.'+species.name+'.'+field.name)!r}] = jnp.zeros({shape!r},jnp.bool_)")
                    self.emit(f"{base}s[{(species.name+'.'+field.name)!r}] = jnp.zeros({shape!r},DTYPES[{field.type!r}])")
            for grid_scope in self.ir.scopes:
                if not grid_scope.grid: continue
                grid = grid_scope.grid
                if getattr(self.ir,'optimize_grid_init_draws',False):
                    draws=implicit_grid_shuffle_draws(grid['width'],grid['height'],moore=grid['neighbors']==8,torus=grid['torus'])
                    self.emit(f"{base}s['$rng'] = mersenne_discard_uniforms(s['$rng'],jnp.asarray({draws},jnp.int32))")
                    self.initialize(grid_scope)
                    continue
                uniform = "mersenne_uniform" if self.ir.random_generator == "mersenne" else "java_uniform"
                self.counter += 1; name = f"implicit_grid_neighbors_{self.counter}"
                self.emit(f"{base}def {name}(grid_i, rng):")
                self.emit(f"{base}    rng, _ = grid_neighbors(rng, grid_i, {grid['width']}, {grid['height']}, 1, moore={grid['neighbors']==8}, torus={grid['torus']}, uniform={uniform})")
                self.emit(f"{base}    return rng")
                self.emit(f"{base}s['$rng'] = jax.lax.fori_loop(0, {self.capacity(grid_scope.name)}, {name}, s['$rng'])")
                self.initialize(grid_scope)
        self.statements(scope.init, scope.name, "jnp.asarray(True)", indent)
        if sequential:
            self.emit(f"{base}    return s")
            self.emit(f"{base}s = jax.lax.fori_loop({start}, {stop}, {function}, s)")
            if creation_body:
                self.counter += 1; function = f"initialize_creation_body_{self.counter}"
                self.emit(f"{base}def {function}(agent_i,state):")
                self.emit(f"{base}    s = dict(state)")
                self.caller_index=caller_index
                self.locals=dict(saved_locals)
                self.statements(creation_body,scope.name,"jnp.asarray(True)",base+"    ")
                self.caller_index=saved_caller
                self.emit(f"{base}    return s")
                self.emit(f"{base}s = jax.lax.fori_loop({start}, {stop}, {function}, s)")
            self.agent_index = saved_agent_index

        self.locals=saved_locals
        self.agent_index=saved_agent_index

    def has_checked_arithmetic(self):
        def walk(value):
            if isinstance(value, Expr):
                return (value.op in ("/", "div", "mod") and value.args[1].op != "literal") or any(walk(arg) for arg in value.args)
            if isinstance(value, (list, tuple)): return any(walk(v) for v in value)
            if isinstance(value, dict): return any(walk(v) for v in value.values())
            if hasattr(value, "__dataclass_fields__"): return any(walk(getattr(value,key)) for key in value.__dataclass_fields__)
            return False
        return walk(self.ir)

    def has_geometry(self):
        return any(field.type == "geometry" for scope in self.ir.scopes for field in scope.fields)

    def has_checked_index(self):
        def walk(value):
            if isinstance(value,Expr):return value.op in ('at','matrix_at','raster_matrix') or any(walk(a) for a in value.args)
            if isinstance(value,(list,tuple)):return any(walk(v) for v in value)
            if hasattr(value,'__dataclass_fields__'):return any(walk(getattr(value,k)) for k in value.__dataclass_fields__)
            return False
        return walk(self.ir)

    def has_checked_random(self):
        def walk(value):
            if isinstance(value,Expr):return value.metadata.get('checked_bounds',False) or any(walk(a) for a in value.args)
            if isinstance(value,(list,tuple)):return any(walk(v) for v in value)
            if hasattr(value,'__dataclass_fields__'):return any(walk(getattr(value,k)) for k in value.__dataclass_fields__)
            return False
        return walk(self.ir)

    def generate(self):
        self.initializing=True
        optimized_reset=getattr(self.ir,'optimize_reset',False) and self.strict_constant_division
        self.emit("def _reset_native(initial_rng, parameters=None):" if optimized_reset else "def _reset_native(seed=None, parameters=None):" if self.strict_constant_division else "def reset(seed=None, parameters=None):")
        if any(f.type in ('image_file','values:image_file') for scope in self.ir.scopes for f in scope.fields):self.emit("    verify_assets()")
        self.emit("    parameters = {} if parameters is None else dict(parameters)")
        if self.ir.fixed_grid_parameters:
            self.emit(f"    fixed_grid_parameters = {self.ir.fixed_grid_parameters!r}")
            self.emit("    for name, value in fixed_grid_parameters.items():")
            self.emit("        if name in parameters and parameters[name] != value: raise ValueError('Grid parameter requires recompilation: ' + name)")
        schema = {f.name: f.type for f in self.ir.scopes[0].fields}
        computed_parameters=[f.name for f in self.ir.scopes[0].fields if f.getter is not None]
        if computed_parameters:self.emit(f"    if set(parameters) & set({computed_parameters!r}): raise ValueError('Computed parameters are read-only')")
        self.emit(f"    schema = {schema!r}")
        self.emit("    unknown = set(parameters) - set(schema)")
        self.emit("    if unknown: raise ValueError('Unknown parameters: ' + ', '.join(sorted(unknown)))")
        self.emit("    for name, value in parameters.items():")
        self.emit("        kind = schema[name]")
        self.emit("        valid = (kind == 'bool' and type(value) is bool) or (kind == 'int' and type(value) is int and -2147483648 <= value <= 2147483647) or (kind == 'float' and type(value) in (int, float) and math.isfinite(value)) or (kind == 'point' and isinstance(value, (tuple, list)) and len(value) == 3 and all(type(x) in (int, float) and math.isfinite(x) for x in value))")
        self.emit("        if not valid: raise ValueError('Invalid parameter type/range: ' + name)")
        self.emit("    s = {}")
        if self.has_clock:self.emit("    s['$cycle']=jnp.asarray(0,jnp.int32)")
        if self.has_pause:self.emit("    s['$paused']=jnp.asarray(False)")
        for i,_ in enumerate(self.diffusions):
            self.emit(f"    s[{('$diffusion.enabled.'+str(i))!r}]=jnp.asarray(False)")
            self.emit(f"    s[{('$diffusion.proportion.'+str(i))!r}]=jnp.asarray(0.,jnp.float64)")
        for output_id,statement in self.save_statements.items():
            self.emit(f"    s[{('$save.generation.'+str(output_id))!r}]=jnp.asarray(0,jnp.int64)")
            self.emit(f"    s[{('$save.rewrite.'+str(output_id))!r}]=jnp.asarray(False)")
            for part in statement.metadata['parts']:
                if 'key' in part:self.emit(f"    s[{part['key']!r}]=jnp.asarray(0,DTYPES[{part['type']!r}])")
        if self.list_alias_fields:
            self.emit(f"    s['$list_generation'] = jnp.asarray({len(self.list_alias_fields)},jnp.int64)")
            for identity,field in enumerate(self.list_alias_fields):self.emit(f"    s[{('$listalias.global.'+field.name)!r}] = jnp.asarray({identity},jnp.int64)")
        for owner in self.spatial_targets:
            prefix='$spatial.'+owner+'.'
            split_capacity=1 if getattr(self.ir,'optimize_spatial_order',False) and self.capacity(owner)<=100 else (4**8-1)//3
            self.emit(f"    s[{(prefix+'split')!r}] = jnp.zeros(({split_capacity},),jnp.bool_)")
            if getattr(self.ir,'optimize_spatial_order',False):self.emit(f"    s[{(prefix+'counts')!r}] = jnp.zeros(({split_capacity},),jnp.int32)")
            self.emit(f"    s[{(prefix+'leaf')!r}] = jnp.full(({self.capacity(owner)},),-1,jnp.int32)")
            self.emit(f"    s[{(prefix+'order')!r}] = jnp.zeros(({self.capacity(owner)},),jnp.int64)")
            self.emit(f"    s[{(prefix+'tick')!r}] = jnp.asarray(0,jnp.int64)")
            self.emit(f"    s[{(prefix+'sorted')!r}] = jnp.arange({self.capacity(owner)},dtype=jnp.int32)")
        for owner in sorted(self.contact_cache_targets):
            prefix='$contacts.'+owner+'.';size=self.capacity(owner)
            self.emit(f"    s[{(prefix+'valid')!r}]=jnp.asarray(False)")
            self.emit(f"    s[{(prefix+'radius')!r}]=jnp.asarray(jnp.nan,jnp.float64)")
            self.emit(f"    s[{(prefix+'source')!r}]=jnp.asarray(-1,jnp.int32)")
            self.emit(f"    s[{(prefix+'cached_epoch')!r}]=jnp.asarray(-1,jnp.int64)")
            self.emit(f"    s[{(prefix+'epoch')!r}]=jnp.asarray(0,jnp.int64)")
            for name in ('boxes','nearby'):self.emit(f"    s[{(prefix+name)!r}]=jnp.zeros(({size},),jnp.bool_)")
            for name in ('box_counts','counts'):self.emit(f"    s[{(prefix+name)!r}]=jnp.asarray(0,jnp.int32)")
        if self.has_dynamic_model() or self.has_checked_arithmetic() or self.has_geometry() or self.has_checked_index() or self.has_checked_random() or self.ir.populations or any(f.type.startswith('agent:') for s in self.ir.scopes for f in s.fields): self.emit("    s['$error'] = jnp.asarray(0, jnp.int32)")
        if not self.ir.random_initialization: self.initialize(self.ir.scopes[0])
        if self.ir.random_seed is not None:
            if not optimized_reset:
                self.emit(f"    seed = {self.ir.random_seed!r} if seed is None else float(seed)")
                self.emit("    if not math.isfinite(seed) or seed == 0: raise ValueError('A finite nonzero seed is required')")
            if self.ir.random_generator == "mersenne":
                if getattr(self.ir,'optimize_mersenne',False):
                    self.emit("    s['$rng'] = (initial_rng, jnp.asarray(624, jnp.int32),jnp.zeros((312,),jnp.float64))" if optimized_reset else "    s['$rng'] = (jnp.asarray(mersenne_initial_state(seed), jnp.uint32), jnp.asarray(624, jnp.int32),jnp.zeros((312,),jnp.float64))")
                else:
                    self.emit("    s['$rng'] = (jnp.asarray(mersenne_initial_state(seed), jnp.uint32), jnp.asarray(624, jnp.int32))")
            else:
                self.emit("    seed = abs(seed)")
                self.emit("    if seed < 1: seed *= 9223372036854775808.0")
                self.emit("    native_seed = int.from_bytes(struct.pack('<d', seed), 'big')")
                self.emit("    s['$rng'] = jnp.asarray((native_seed ^ 0x5DEECE66D) & ((1 << 48) - 1), jnp.uint64)")
        else:
            self.emit("    if seed is not None: raise ValueError('This model has no supported RNG declaration')")
        if self.ir.random_initialization: self.initialize(self.ir.scopes[0])
        self.emit("    return s"); self.emit("")
        self.lines.extend(self.pure_grid_initializer_helpers)
        if self.pure_grid_initializer_names:self.emit('PURE_GRID_INITIALIZERS={'+','.join(repr(name)+':'+name for name in self.pure_grid_initializer_names)+'}')
        if self.strict_constant_division or optimized_reset:
            self.emit("_reset_native_jit=jax.jit(_reset_native,static_argnames=('parameters',),compiler_options=JIT_OPTIONS)" if optimized_reset else "_reset_native_jit=jax.jit(_reset_native,static_argnames=('seed','parameters'),compiler_options=JIT_OPTIONS)")
            self.emit("def reset(seed=None,parameters=None):")
            if self.ir.assets:self.emit("    verify_assets()")
            self.emit("    entries=None if parameters is None else tuple(sorted((name,tuple(value) if isinstance(value,(list,tuple)) else value) for name,value in parameters.items()))")
            if optimized_reset:
                self.emit(f"    seed = {self.ir.random_seed!r} if seed is None else float(seed)")
                self.emit("    if not math.isfinite(seed) or seed == 0: raise ValueError('A finite nonzero seed is required')")
                self.emit("    return _reset_native_jit(jnp.asarray(mersenne_initial_state(seed),jnp.uint32),parameters=entries)")
            else:self.emit("    return _reset_native_jit(seed=seed,parameters=entries)")
            self.emit('')
        self.initializing=False
        self.emit("def step(state):"); self.emit("    s = dict(state)")
        if self.has_pause:self.emit("    s['$paused']=jnp.asarray(False)")
        for i,_ in enumerate(self.diffusions):self.emit(f"    s[{('$diffusion.enabled.'+str(i))!r}]=jnp.asarray(False)")
        for scope in sorted(self.ir.scopes,key=lambda item:bool(item.grid)):
            if scope.name == "global":
                updates=[Statement('assign',f.name,f.facets['update'],[],[],f.token) for f in scope.fields if 'update' in f.facets]
                if updates:self.statements(updates,scope.name,'jnp.asarray(True)','    ')
                for behavior in scope.behaviors:
                    self.eval_indent, self.active_mask = "    ", "jnp.asarray(True)"
                    guard = self.mask(self.expression(behavior.guard, scope.name), "    ")
                    self.statements(behavior.body, scope.name, guard)
                continue
            updates=[Statement('assign',f.name,f.facets['update'],[],[],f.token) for f in scope.fields if 'update' in f.facets]
            if not scope.behaviors and not updates or self.capacity(scope.name)==0 or scope.grid and scope.grid['frequency']==0: continue
            if getattr(self.ir,'optimize_independent_grid',False) and scope.grid and not scope.behaviors and all(self.independent_cell_expression(item.value,scope.name) for item in updates) and all(self.independent_cell_expression(value,scope.name) for f in scope.fields for value in f.facets.values() if isinstance(value,Expr)):
                self.counter+=1;function=f'parallel_grid_{self.counter}'
                self.emit(f'    def {function}(state):');self.emit('        s=dict(state)')
                for item in updates:
                    field=next(f for f in scope.fields if f.name==item.target)
                    self.counter+=1;mapper=f'cell_value_{self.counter}'
                    self.emit(f'        def {mapper}(agent_i):');self.emit('            local_state=dict(s)');self.emit('            def evaluate(s):')
                    self.agent_index='agent_i';self.eval_indent,self.active_mask='                ','jnp.asarray(True)'
                    value=self.coerce_field(self.expression(item.value,scope.name),field,scope.name)
                    self.emit(f"                return ({value}).astype(DTYPES[{field.type!r}])")
                    self.emit('            return evaluate(local_state)')
                    self.emit(f"        s[{(scope.name+'.'+field.name)!r}]=jax.vmap({mapper})(jnp.arange({self.capacity(scope.name)},dtype=jnp.int32))")
                self.emit('        return s')
                if scope.grid['frequency']>1:self.emit(f"    s=jax.lax.cond(s['$cycle']%{scope.grid['frequency']}==0,{function},lambda state:state,s)")
                else:self.emit(f'    s={function}(s)')
                self.agent_index=None;self.eval_indent,self.active_mask='    ','jnp.asarray(True)'
                self.independent_grid_sites+=1
                continue
            function = f"advance_{scope.name}"
            count = self.capacity(scope.name)
            if self.reserved(scope.name):
                snapshot=function+'_indices'; limit=function+'_count'
                self.emit(f'    {snapshot} = jnp.nonzero(s[{("$present."+scope.name)!r}],size={count},fill_value=0)[0]')
                self.emit(f'    {limit} = jnp.sum(s[{("$present."+scope.name)!r}],dtype=jnp.int32)')
            else: limit=str(count)
            self.emit(f"    def {function}(iteration, state):")
            self.emit("        s = dict(state)")
            self.agent_index = "agent_index"
            index=f'{limit}-1-iteration' if self.reverse_agents else 'iteration'
            self.emit(f'        agent_index = {snapshot}[{index}]' if self.reserved(scope.name) else f'        agent_index = {index}')
            if updates:
                present=f's[{("$present."+scope.name)!r}][agent_index]' if self.reserved(scope.name) else 'jnp.asarray(True)'
                self.statements(updates,scope.name,present,'        ')
            for behavior in scope.behaviors:
                present=f's[{("$present."+scope.name)!r}][agent_index]' if self.reserved(scope.name) else 'jnp.asarray(True)'
                self.eval_indent, self.active_mask = "        ", present
                guard = self.mask(f'({self.expression(behavior.guard, scope.name)}) & ({present})', "        ")
                if getattr(self.ir,'optimize_reflex_guards',False) and behavior.guard.op!='literal':
                    self.counter+=1;conditional=f'guarded_behavior_{self.counter}'
                    self.emit(f'        def {conditional}(state):')
                    self.emit('            s = dict(state)')
                    self.statements(behavior.body,scope.name,'jnp.asarray(True)','            ')
                    self.emit('            return s')
                    self.emit(f'        s = jax.lax.cond({guard},{conditional},lambda state:state,s)')
                else:self.statements(behavior.body, scope.name, guard, "        ")
            self.emit("        return s")
            self.agent_index = None
            if scope.grid and scope.grid["frequency"]>1:self.emit(f"    s = jax.lax.cond(s['$cycle']%{scope.grid['frequency']}==0,lambda array:jax.lax.fori_loop(0,{limit},{function},array),lambda array:array,s)")
            else:self.emit(f"    s = jax.lax.fori_loop(0, {limit}, {function}, s)")
        for i,statement in enumerate(self.diffusions):
            owner=statement.metadata['on'];grid=self.scopes[owner].grid;key=owner+'.'+statement.metadata['var'];proportion='$diffusion.proportion.'+str(i);enabled='$diffusion.enabled.'+str(i)
            self.emit(f"    s[{key!r}]=jax.lax.cond(s[{enabled!r}] & (s['$error']==0),lambda array:gradient_convolution(array,s[{proportion!r}],{grid['width']},{grid['height']},{statement.metadata['radius_value']},{grid['neighbors']==8}),lambda array:array,s[{key!r}])")
        if self.has_clock:self.emit("    s['$cycle']=s['$cycle']+jnp.asarray(1,jnp.int32)")
        self.emit("    return s"); self.emit(""); self.emit("step_jit = jax.jit(step, compiler_options=JIT_OPTIONS)")
        if any(f.getter is not None for scope in self.ir.scopes for f in scope.fields):
            self.emit('');self.emit('def observe(state):');self.emit('    s = dict(state)')
            for scope in self.ir.scopes:
                fields=[f for f in scope.fields if f.getter is not None]
                if not fields:continue
                if scope.name=='global':
                    self.agent_index=None;self.eval_indent='    ';self.active_mask='jnp.asarray(True)'
                    for field in fields:
                        value=self.computed_value(scope.name,field,None)
                        if field.type.startswith('agents:'):
                            self.counter+=1;temporary=f'observed_list_{self.counter}'
                            self.emit(f'    {temporary} = {value}')
                            self.emit(f"    s[{('global.'+field.name)!r}] = {temporary}[0]")
                            self.emit(f"    s[{('$listmask.global.'+field.name)!r}] = {temporary}[1]")
                        else:self.emit(f"    s[{('global.'+field.name)!r}] = ({value}).astype(DTYPES[{field.type!r}])")
                elif self.capacity(scope.name):
                    # Native observation is field-major: each collect completes before
                    # reading the next computed field. This matters for random getters.
                    for field in fields:
                        self.counter+=1;function=f'observe_computed_{self.counter}'
                        self.emit(f'    def {function}(agent_i,state):');self.emit('        s = dict(state)')
                        self.agent_index='agent_i';self.eval_indent='        '
                        self.active_mask=f's[{("$present."+scope.name)!r}][agent_i]' if self.reserved(scope.name) else 'jnp.asarray(True)'
                        value=self.computed_value(scope.name,field,'agent_i')
                        if field.type.startswith('agents:'):
                            self.counter+=1;temporary=f'observed_list_{self.counter}'
                            self.emit(f'        {temporary} = {value}')
                            self.emit(f"        s[{(scope.name+'.'+field.name)!r}] = s[{(scope.name+'.'+field.name)!r}].at[agent_i].set({temporary}[0])")
                            self.emit(f"        s[{('$listmask.'+scope.name+'.'+field.name)!r}] = s[{('$listmask.'+scope.name+'.'+field.name)!r}].at[agent_i].set({temporary}[1])")
                        else:self.emit(f"        s[{(scope.name+'.'+field.name)!r}] = s[{(scope.name+'.'+field.name)!r}].at[agent_i].set(({value}).astype(DTYPES[{field.type!r}]))")
                        self.emit('        return s');self.emit(f'    s = jax.lax.fori_loop(0,{self.capacity(scope.name)},{function},s)')
            if any(f.getter_random for scope in self.ir.scopes for f in scope.fields):self.emit("    s['$rng'] = state['$rng']")
            self.agent_index=None;self.emit('    return s');self.emit('observe_jit = jax.jit(observe,compiler_options=JIT_OPTIONS)')
        self.emit('')
        self.emit('def identities(state):')
        self.emit('    result = {}')
        for scope in self.ir.scopes[1:]:
            if self.reserved(scope.name):
                self.emit(f"    result[{scope.name!r}] = [i for i, present in enumerate(jax.device_get(state[{('$present.'+scope.name)!r}]).tolist()) if present]")
            else:
                self.emit(f"    result[{scope.name!r}] = list(range({self.capacity(scope.name)}))")
        self.emit('    return result')
        if self.output_records:
            self.emit("")
            if getattr(self.ir,'optimize_fusion',False):self.lines.append(OUTPUT_FORMAT_SOURCE)
            self.emit(f"OUTPUT_SCHEMA = {self.output_records!r}")
            self.emit("def outputs(state):")
            self.emit("    events = []")
            self.emit("    for output in OUTPUT_SCHEMA:")
            if self.save_statements:
                self.emit("        if output.get('kind')=='save' and not int(jax.device_get(state[output['generation']])): continue")
            self.emit("        parts = []")
            self.emit("        for part in output['parts']:")
            self.emit("            if 'literal' in part: parts.append(part['literal'])")
            self.emit("            else:")
            self.emit("                value = jax.device_get(state[part['key']]).item()")
            if getattr(self.ir,'optimize_fusion',False):self.emit("                parts.append(native_scalar_text(value))")
            else:self.emit("                parts.append(str(value).lower() if type(value) is bool else str(value))")
            if self.save_statements:
                self.emit("        if output.get('kind')=='save': events.append({'kind':'save','phase':'simulation','id':output['id'],'text':''.join(parts),'path':output['path'],'rewrite':bool(jax.device_get(state[output['rewrite']])),'generation':int(jax.device_get(state[output['generation']]))})")
                self.emit("        else: events.append({'kind':'write','phase':'init','id':output['id'],'text':''.join(parts)})")
            else:self.emit("        events.append({'kind': 'write', 'phase': 'init', 'id': output['id'], 'text': ''.join(parts)})")
            self.emit("    return events")
        if getattr(self.ir,'optimize_outlined_rng',False):
            self.lines.append("OUTLINED_RNG_HELPERS={}")
            for helper in ('mersenne_twist','mersenne_cache','mersenne_refill','mersenne_uniform','mersenne_discard_uniforms'):
                self.lines.extend([f'_plain_{helper}={helper}',f'OUTLINED_RNG_HELPERS[{helper!r}]="_plain_{helper}"',f'{helper}=jax.jit({helper},inline=False)'])
        if getattr(self.ir,'optimize_host_reset',False):
            self.lines.append(HOST_RESET_SOURCE)
            self.lines.extend(['_reset_on_jax = reset','def reset(seed=None, parameters=None):','    return _initialize_on_host(_reset_on_jax, globals(), seed, parameters)'])
        return "\n".join(self.lines)+"\n"

    @staticmethod
    def has_dynamic(items):
        return any(item.kind == "dynamic_loop" or Generator.has_dynamic(item.body) or Generator.has_dynamic(item.alternative) for item in items)

    def has_dynamic_model(self):
        return any(self.has_dynamic(scope.init) or any(self.has_dynamic(b.body) for b in scope.behaviors) for scope in self.ir.scopes)


