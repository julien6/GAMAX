"""Public compiler API: dependency resolution, validation, and artifact generation."""
from __future__ import annotations
from pathlib import Path
import copy
import json
import math
import re
import struct
from .ir import Token, CompileError, Expr, Field, Statement, Behavior, Action, Scope, ModelIR
from .lexing import TOKEN, PRECEDENCE, FUNCTIONS, RESERVED
from dataclasses import asdict
import hashlib
import os
from .presentation import classify_presentation_actions
from .assets import resolve_assets
from .frontend import Parser
from .optimizer import optimize
from .each_effects import validate_each_effects
from .lowering import Generator
from .inheritance import resolve_species,resolve_field_dependencies,resolve_experiments,resolve_monomorphic_references

def implementation_identity():
    """Identify all compiler components, not only this public API module."""
    names = ("compiler.py", "ir.py", "lexing.py", "frontend.py", "analysis.py", "each_effects.py", "lowering.py", "randomness.py", "spatial.py", "spatial_index.py", "inheritance.py", "assets.py", "presentation.py", "native_math.py", "hotspot_math.py", "movement_trig_table.py", "host_reset.py", "output_format.py", "raster_decode.py", "diffusion.py", "optimizer.py")
    root = Path(__file__).parent
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files, fingerprint



def compile_model(source_path, output_dir, *, experiment=None, capacities=None, optimization="none"):
    path = Path(source_path).resolve()
    source = path.read_text()
    ir, dependencies = load_model(path,experiment=experiment)
    validate_each_effects(ir)
    if experiment is not None and experiment not in ir.experiments:
        raise CompileError(path, Token(str(experiment), 1, 1), "Unknown experiment", "GAMAX_EXPERIMENT")
    if experiment is not None and ir.presentation[experiment].get('virtual'):
        raise CompileError(path,Token(experiment,1,1),'Cannot run a virtual experiment','GAMAX_EXPERIMENT')
    selected=[ir.presentation[experiment]] if experiment is not None else list(ir.presentation.values())
    if any(item.get('unsupported_effects') for item in selected):raise CompileError(path,Token(str(experiment),1,1),'Selected experiment has unsupported initialization/state effects; choose a supported headless experiment','GAMAX_EXPERIMENT')
    capacities = {} if capacities is None else dict(capacities)
    for parent in ir.reference_aliases:
        if capacities.get(parent,0):raise CompileError(path,Token(parent,1,1),'Reserved parent population invalidates monomorphic reference resolution','GAMAX_INHERITANCE')
    explicit_capacities=set(capacities)
    def lifecycle(items, owner):
        for item in items:
            if item.kind == 'birth' and item.target not in explicit_capacities:
                raise CompileError(path,item.token,'Runtime birth requires an explicit reserved capacity for '+item.target,'GAMAX_CAPACITY')
            if item.kind == 'die':
                capacities.setdefault(owner,ir.populations[owner])
            def calls(value):
                if isinstance(value,Expr):
                    if value.op=='function_value':lifecycle(value.metadata['body'],owner)
                    for arg in value.args:calls(arg)
                elif isinstance(value,dict):
                    for arg in value.values():calls(arg)
            calls(item.value);calls(item.metadata)
            lifecycle(item.body,item.target if item.kind in ('ask','birth','create') else owner)
            lifecycle(item.alternative,owner)
    for scope in ir.scopes:
        if scope.name!='global' and not scope.grid and not capacities.get(scope.name,ir.populations[scope.name]):continue
        for behavior in scope.behaviors:
            lifecycle(behavior.body,scope.name)
    for field in ir.scopes[0].fields:
        if field.list_mutable:
            owner=field.type.split(':')[1]
            if not next(s for s in ir.scopes if s.name==owner).grid:capacities.setdefault(owner,ir.populations[owner])
    for species, capacity in capacities.items():
        scope = next((s for s in ir.scopes if s.name==species),None)
        if species not in ir.populations or type(capacity) is not int or not ir.populations[species]<=capacity<=1000000 or scope.grid and capacity!=ir.populations[species]:
            raise CompileError(path, Token(str(species), 1, 1), "Capacity must be an int between the default population and 1000000; grid capacity must equal its declared cell count", "GAMAX_CAPACITY")
    def initial_reads(value):
        if isinstance(value,Expr):
            if value.op=='name' and value.args[0] in ir.populations:
                owner=value.args[0]
                if not next(scope for scope in ir.scopes if scope.name==owner).grid:capacities.setdefault(owner,ir.populations[owner])
            for arg in value.args:initial_reads(arg)
            initial_reads(value.metadata)
        elif isinstance(value,(list,tuple)):
            for item in value:initial_reads(item)
        elif isinstance(value,dict):
            for item in value.values():initial_reads(item)
        elif hasattr(value,'__dataclass_fields__'):
            for name in value.__dataclass_fields__:initial_reads(getattr(value,name))
    for scope in ir.scopes:initial_reads(scope.init)
    ir.capacities=capacities
    optimization_report = optimize(ir, optimization)
    generator = Generator(ir)
    for transformation in optimization_report['passes']:
        if transformation['name']=='dynamic-seeded-compiled-reset':
            transformation['applied_sites']=int(getattr(ir,'optimize_reset',False) and generator.strict_constant_division)
            transformation['proof_condition']+='; restricted to initializers already requiring compiled reset'
        if transformation['name']=='memoized-unordered-contact-row':transformation['applied_sites']=len(generator.contact_cache_targets)
        if transformation['name']=='unsplittable-small-population-quadtree':
            transformation['applied_sites']=sum(generator.capacity(owner)<=100 for owner in generator.spatial_targets)
        if transformation['name']=='hotspot-normalized-movement-trigonometry':transformation['applied_sites']=int(generator.native_trig)
        if transformation['name']=='precomputed-ordered-grid-geometry':transformation['applied_sites']=len(generator.grid_geometry)
        if transformation['name']=='hoisted-read-only-raster-pixels':transformation['applied_sites']=len(generator.raster_tables)
        if transformation['name']=='single-body-double-envelope-shuffle':transformation['applied_sites']=int(generator.circle_queries)+int(generator.rectangle_queries)
    code = generator.generate()
    for transformation in optimization_report['passes']:
        if transformation['name']=='broadcast-invariant-grid-initializers':transformation['applied_sites']=generator.broadcast_grid_sites
        if transformation['name']=='vectorized-discard-only-integer-getters':transformation['applied_sites']=generator.pure_mapper_sites
        if transformation['name']=='vectorized-independent-grid-updates':transformation['applied_sites']=generator.independent_grid_sites
        if transformation['name']=='elide-unobserved-rectangle-permutations':transformation['applied_sites']=generator.unordered_rectangle_sites
        if transformation['name']=='short-circuit-first-matching-agent':transformation['applied_sites']=generator.first_match_sites
        if transformation['name']=='inline-single-return-actions':transformation['applied_sites']=generator.single_return_sites
        if transformation['name']=='batch-native-uniform-grid-initializers':transformation['applied_sites']=generator.grid_random_sites
        if transformation['name']=='shared-loop-native-uniform-pairs':transformation['applied_sites']=generator.random_pair_sites
        if transformation['name']=='reuse-pure-getter-query-geometry':transformation['applied_sites']=generator.deferred_query_reuse_sites
        if transformation['name']=='guarded-grid-reference-overlap':transformation['applied_sites']=generator.grid_reference_sites
        if transformation['name']=='compiled-pure-cell-grid-initializers':transformation['applied_sites']=generator.pure_grid_initializer_sites
        if transformation['name']=='prefetch-exact-envelope-shuffle-draws':transformation['applied_sites']=int(generator.circle_queries)+int(generator.rectangle_queries) if getattr(ir,'optimize_prefetch_shuffles',False) else 0
        if transformation['name']=='batch-ordered-grid-neighborhood-initializers':transformation['applied_sites']=generator.grid_neighborhood_sites
    # All validation precedes filesystem mutation.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module = output / "model_jax.py"
    module.write_text(code)
    manifest = dict(schema_version=1, compiler_version="0.4.0", compiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model=ir.name,
                    source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    parsed_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    generated_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    profile="bounded-native-coupled-gaml-jax-v0.4", experiments=ir.experiments,
                    populations=ir.populations, fields={f"{s.name}.{f.name}": f.type for s in ir.scopes for f in s.fields},
                    schedule=[f"{s.name}.{b.name}" for s in sorted(ir.scopes,key=lambda item:bool(item.grid)) for b in s.behaviors],
                    effects={f"{s.name}.{b.name}": dict(reads=b.reads, writes=b.writes) for s in ir.scopes for b in s.behaviors},
                    numeric_profile="Java int32 wraparound and truncating div/mod; float64 division; source-order numeric reductions and floating arithmetic barriers; GAMA degree trigonometry; nonzero-divisor checks",
                    capabilities=['fixed species where/collect and scalar reductions', 'global reflex ask over fixed species in index order',
                                  'explicit coordinate/adjacency attributes; restricted moving profile; no built-in graph skills'],
                    jit_contract=dict(cpu_fast_math=False, floating_models_disable_hlo_fusion=not getattr(ir,'optimize_fusion',False) and any(f.type in ('float', 'point', 'geometry') for s in ir.scopes for f in s.fields),
                                      entrypoint='step_jit or outer jit with generated JIT_OPTIONS; arbitrary re-jitting may violate the numeric contract'),
                    limitations=["no nested asks, inheritance/overrides, general geometry/graph operations, FSM, diffusion, or effectful experiments",
                                 "no general mutable lists, nested each bindings or unchecked division; collection reads during init restricted to grids",
                                 "agent transitions read/write self attributes and declared globals; init writes are local",
                                 "global first, ordinary populations then grids in stable declaration order; ascending native identity; agent updates then source-order reflexes/FSM; deferred diffusion after populations; finite native evidence, not a full-language proof"])
    manifest['declaration_facets'] = {f'{scope.name}.{f.name}': {key: asdict(value) for key,value in f.facets.items()} for scope in ir.scopes for f in scope.fields if f.facets}
    manifest['capabilities'] += ['native scalar defaults; numeric min/max coercion on initialization and assignments; step metadata']
    manifest['optimization'] = optimization_report
    manifest['selected_experiment'] = experiment
    manifest['presentation']={'experiments':ir.presentation,'aspects':{s.name:s.aspects for s in ir.scopes if s.aspects},'gui_actions':{scope.name+'.'+action.name:asdict(action) for scope in ir.scopes for action in scope.actions if action.presentation_only}}
    if ir.presentation_resources:manifest['presentation']['resources']=ir.presentation_resources
    manifest['presentation_contract']='GUI tokens, render RNG and GUI-input callbacks preserved separately and explicitly excluded from headless simulation execution. Selected experiment state/init effects require a backend and are rejected. No native GUI reproduction claim.'
    manifest['capacities'] = {**ir.populations, **capacities}
    manifest['reset_contract'] = 'reset(seed=None, parameters=None): typed global initializer overrides in declaration order; init assignments retain their effects; explicit Java/Mersenne or native implicit Mersenne profile; seed zero rejected'
    manifest['implementation_files'], manifest['implementation_sha256'] = implementation_identity()
    manifest['outputs'] = generator.output_records
    manifest['output_contract'] = 'Unconditional global init write events and bounded relative text save events: evaluated into hidden state, exposed by outputs(state), exported by runtime; native text save has no automatic newline. GUI declarations/manual callbacks explicitly excluded; unsupported simulation effects rejected.'
    manifest['grid_profile'] = {s.name:s.grid for s in ir.scopes if s.grid}
    manifest['fixed_grid_parameters']=ir.fixed_grid_parameters
    manifest['rng_observation_required']=any(f.getter_random for scope in ir.scopes for f in scope.fields)
    manifest['computed_fields']=[s.name+'.'+f.name for s in ir.scopes for f in s.fields if f.getter is not None]
    manifest['reference_aliases']=ir.reference_aliases
    manifest['world_torus']=ir.world_torus
    manifest['updated_fields']=[s.name+'.'+f.name for s in ir.scopes for f in s.fields if 'update' in f.facets]
    manifest['variable_order']={s.name:[f.name for f in s.fields] for s in ir.scopes}
    manifest['implicit_rng'] = ir.random_implicit
    manifest['random_profile'] = None if ir.random_seed is None else dict(generator=ir.random_generator, seed=ir.random_seed,
        reseed='external seed required before globals/grids; field-major grid initialization' if ir.random_implicit else 'prefix global init; species attributes then init bodies, without skills' if ir.random_initialization else 'final global init; runtime only', draws='GAMA nextDouble (26+27 bits), scalar rnd/flip, lazy guard/branch/and/or, source and agent order')
    if ir.random_seed is not None: manifest['capabilities'].append('native-compatible Java/Mersenne runtime rnd/flip with explicit reseeding')
    manifest['dependencies'] = dependencies
    common = Path(os.path.commonpath([str(Path(item['path']).parent) for item in [*dependencies,*ir.assets]]))
    manifest['source_bundle_entry'] = str(Path('sources') / path.relative_to(common))
    for item in dependencies:
        original = Path(item['path'])
        relative = Path('sources') / original.relative_to(common)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes())
        item['bundle_path'] = str(relative)
    manifest['assets']=ir.assets
    for item in ir.assets:
        original=Path(item['path']);relative=Path('sources')/original.relative_to(common)
        for destination in (output/relative,output/item['asset_bundle_path']):
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(original.read_bytes())
        item['source_bundle_path']=str(relative)
    manifest['asset_fields']={s.name+'.'+f.name:f.type for s in ir.scopes for f in s.fields if f.type in ('image_file','values:image_file')}
    manifest['asset_contract']='Bundled SHA-256 verified immutable image handles; packed ARGB pixel reads and bounded compile-time Java ImageIO/AWT raster matrices; explicitly classified GUI-only SVG/OBJ/font resources archived separately, not executed by the headless backend'
    manifest['spatial_profile'] = dict(world='default non-toroidal square [0,100]²', step_seconds=1, skills=['moving: do move without facets'], unsupported=['custom shapes/topologies', 'wander/goto', 'geometry neighborhoods', 'graphs/grids']) if any(scope.skills for scope in ir.scopes) else None
    manifest['capabilities'] += ['point coordinates/arithmetic/distance_to; restricted 2D moving']
    manifest['capabilities'] += ['recursive relative imports with global member overrides and entry-first init composition; species DAG inheritance with parent-first init and member overrides; resolved virtual actions', 'nonrecursive same-scope scalar actions and block locals']
    manifest['capabilities'] += ['else if chains; literal times and inclusive from/to/step loops (maximum 1024 iterations)']
    manifest['capabilities'] += ['same-scope action arguments named as facets, evaluated in declaration order']
    manifest['limitations'].append('no value-collection loops or break/continue; static expansion limited to 4096 statements per behavior/action/init; dynamic loops limited to 1000000 iterations with explicit runtime failure')
    manifest['capabilities'] += ['Java/Mersenne rnd/flip in species field initializers and init bodies with explicit prefix reseeding, without skills', 'scalar dynamic times/from/to/step loops; checked runtime trip count and block-local state']
    manifest['runtime_errors'] = {}
    if generator.spatial_targets:manifest['runtime_errors']['10']='invalid spatial query radius'
    if generator.has_checked_index(): manifest['runtime_errors']['8']='collection index outside its checked domain'
    if generator.has_checked_random(): manifest['runtime_errors']['9']='invalid dynamic random bounds'
    if any(f.list_mutable for s in ir.scopes for f in s.fields): manifest['runtime_errors']['7']='ordered agent-list capacity exceeded'
    if generator.has_dynamic_model(): manifest['runtime_errors']['1'] = 'invalid dynamic-loop count/step or trip-count capacity exceeded'
    if generator.has_checked_arithmetic(): manifest['runtime_errors']['2'] = 'division by zero'
    if any(f.type.startswith('agent:') for s in ir.scopes for f in s.fields): manifest['runtime_errors']['5']='nil agent reference dereference'
    manifest['capabilities'] += ['typed scalar agent references, checked attribute lookup, ordered one_of, grid center locations']
    if ir.populations: manifest['runtime_errors']['4'] = 'initial creation count differs from reserved fixed population'
    manifest['capabilities'] += ['RGBA constants/construction/equality; preserved unconditional init write events']
    manifest['capabilities'] += ['seeded native implicit location for nonmoving agents in rectangular worlds']
    manifest['capabilities'] += ['multiple initial creation batches, statically resolvable demographic counts, remote initialization bodies and zero-sized species']
    if generator.has_geometry(): manifest['runtime_errors']['3'] = 'invalid square world dimension'
    manifest['capabilities'] += ['square world polygon; literal rectangular minimal-agent grids; native cached VN/Moore neighborhoods and initialization draws']
    manifest['capabilities'] += ['read-only persistent ordered grid neighborhoods; bounded external radius; ordered loops over agent collections']
    manifest['collection_fields'] = {s.name+'.'+f.name: {'capacity':ir.capacities.get(f.type.split(':')[1],ir.populations.get(f.type.split(':')[1])) if f.list_mutable else f.capacity,'mutability':'ordered append/remove/replace' if f.list_mutable else 'read-only'} for s in ir.scopes for f in s.fields if f.type.startswith(('agents:','values:'))}
    manifest['capacities'] = ir.capacities
    if ir.capacities:
        manifest['runtime_errors']['6']='reserved population capacity exceeded or invalid creation count'
        manifest['capabilities'].append('reserved populations, external demographic counts, nonmoving runtime birth/death, stable monotonically allocated identities, snapshot scheduling; explicit capacity required for birth')
        manifest['lifecycle_contract']='Capacity bounds total identities allocated during a run, including dead agents; no identity reuse. Runtime birth requires seeded nonmoving initialization. Remote creation bodies support myself attributes. Agent init cannot create/kill agents.'
    manifest['capabilities'] += ['lazy ternary and !, ordered count and point sum/mean, checked scalar/point dynamic division']
    manifest['capabilities'] += ['immutable global typed value lists with checked indexing; ordered global agent lists with append/remove-first/replace and explicit overflow; native backward list shuffle; verified CSS named colors; checked dynamic rnd bounds']
    manifest['capabilities'] += ['dependency-first variable initialization and update facets before reflexes; pure lazy scalar/point/RGBA getters with nonmutating observe_jit snapshots', 'native moving defaults (location, heading, speed=1); square-world move speed/heading facets and wander amplitude/speed facets; random facets rejected']
    manifest['limitations'][0] = "No general polymorphism, graph skills, dynamic assets, effectful experiment init, recursive actions, moving births, FSM enter/exit/state memory, toroidal or repeated-target diffusion, general mutable value lists, or polygon-agent spatial queries with positive population. The bounded supported subset is explicit; no whole-GAMA compatibility claim."
    manifest['capabilities'] += ['nested ordered asks with lexical captures and stable references; checked remote scalar/RGBA attribute writes', 'typed returning actions with lazy early returns and conditional draws', 'pure numeric ordered sort_by and first-tie extrema; bounded local reference-list reassignment', 'selected inherited experiment parameter literal initializers before global initialization; external reset overrides take precedence', 'pure creation with arguments replacing initial field values before agent init', 'first-match FSM transitions after state bodies and regular reflexes; canonical state names', 'context-sensitive grid/continuous geometry distances and checked grid point casts; nullable local point casts', 'bounded indexed int/float matrices and packed ARGB/RGBA conversion', 'deferred pointwise-maximum gradient convolution on non-toroidal rectangular grids with fixed radius']
    manifest['jit_contract']['reset_native_options']=generator.strict_constant_division
    manifest['jit_contract']['disable_algebraic_simplification']=bool(any(scope.skills for scope in ir.scopes) or generator.strict_constant_division)
    manifest['runtime_errors'].update({'5':'nil or out-of-domain agent reference dereference','11':'invalid goto target/speed','12':'invalid triangle side','13':'invalid image pixel index/handle','14':'matrix index outside its checked domain','15':'invalid diffusion proportion'})
    manifest['spatial_profile'] = dict(world='static rectangular world; explicitly bounded continuous/grid topology contracts',world_torus=ir.world_torus,step_seconds=1,skills=['moving defaults, move/wander/goto with pure facets'],unsupported=['graph movement','positive-population polygon-agent ordered queries','toroidal diffusion']) if any(scope.skills for scope in ir.scopes) else None
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output / "source_map.json").write_text(json.dumps(generator.map, indent=2)+"\n")
    (output / "source.gaml").write_text(source)
    (output / "ir.json").write_text(json.dumps(asdict(ir), indent=2)+"\n")
    return module, manifest


def load_model(path, *, experiment=None):
    """Resolve a dependency DAG once; preserve original per-file token locations."""
    path = Path(path).resolve()
    ordered, visiting, seen = [], [], set()
    def visit(current, via=None):
        if current in visiting:
            raise CompileError(current, via or Token("import", 1, 1), "Cyclic import dependency", "GAMAX_IMPORT")
        if current in seen: return
        if not current.is_file():
            raise CompileError(current, via or Token("import", 1, 1), "Imported source does not exist", "GAMAX_IMPORT")
        visiting.append(current)
        parser = Parser(current.read_text(), str(current))
        ir = parser.parse(validate=False)
        for relative, token in parser.imports:
            if Path(relative).is_absolute(): parser.fail(token, "Imports must be relative", "GAMAX_IMPORT")
            visit((current.parent / relative).resolve(), token)
        visiting.pop(); seen.add(current); ordered.append((current, parser, ir))
    visit(path)
    entry_parser = ordered[-1][1]
    merged = ModelIR(ordered[-1][2].name, [Scope("global", [], [], [])], [], {})
    for current, parser, ir in ordered:
        if ir.world_torus_explicit:merged.world_torus=ir.world_torus;merged.world_torus_explicit=True
        for scope in ir.scopes:
            if scope.name == "global":
                target = merged.scopes[0]
                for attribute in ("fields", "behaviors", "actions"):
                    existing = {member.name:index for index,member in enumerate(getattr(target, attribute))}
                    for member in getattr(scope, attribute):
                        if member.name in existing:getattr(target,attribute)[existing[member.name]]=member
                        else:
                            existing[member.name]=len(getattr(target,attribute));getattr(target, attribute).append(member)
                # Native global init composes entry-first, then imported init;
                # unlike species init, which composes parent-first.
                target.init = scope.init + target.init
            else:
                if any(s.name == scope.name for s in merged.scopes): parser.fail(Token(scope.name, 1, 1, str(current)), "Species override/inheritance is unsupported", "GAMAX_IMPORT")
                merged.scopes.append(scope)
        for species, count in ir.populations.items():
            if species in merged.populations: parser.fail(Token(species, 1, 1, str(current)), "Duplicate imported creation", "GAMAX_IMPORT")
            merged.populations[species] = count
        for experiment in ir.experiments:
            if experiment in merged.experiments: parser.fail(Token(experiment, 1, 1, str(current)), "Experiment override is unsupported", "GAMAX_IMPORT")
            merged.experiments.append(experiment)
        merged.presentation.update(ir.presentation)
    entry_parser.populations = merged.populations
    for scope in merged.scopes:
        if scope.grid:scope.grid['torus']=merged.world_torus
    merged.scopes=resolve_species(merged.scopes)
    resolve_field_dependencies(merged.scopes)
    entry_parser.resolve_ask_targets(merged)
    classify_presentation_actions(merged)
    entry_parser.expand_actions(merged)
    resolve_experiments(merged)
    if experiment is not None:
        if experiment not in merged.presentation:
            entry_parser.fail(Token(str(experiment),1,1),"Unknown experiment","GAMAX_EXPERIMENT")
        fields={f.name:f for f in merged.scopes[0].fields}
        for parameter in merged.presentation[experiment]['parameters']:
            if 'initial' not in parameter:continue
            field=fields.get(parameter['variable'])
            if field is None or field.getter is not None:
                entry_parser.fail(Token(parameter['variable'],1,1),'Experiment initializer needs a writable global variable','GAMAX_EXPERIMENT')
            value=parameter['initial'];kind='bool' if type(value) is bool else 'int' if type(value) is int else 'float'
            field.value=Expr('literal',[value],kind,field.token)
    entry_parser.resolve_creation_counts(merged)
    for parent in {s.parent for s in merged.scopes if s.parent}:merged.populations.setdefault(parent,0)
    if set(merged.populations) != {s.name for s in merged.scopes[1:]}:
        entry_parser.fail(entry_parser.token, "Every non-grid species must be created in global init")
    resolve_assets(merged)
    resolve_experiments(merged)
    resolve_monomorphic_references(merged)
    entry_parser.check(merged)
    dependencies = [dict(path=str(current), sha256=hashlib.sha256(current.read_bytes()).hexdigest()) for current, _, _ in ordered]
    return merged, dependencies
