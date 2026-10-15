"""Conservative, model-independent transformations of the typed GAML IR.

Never relax floating arithmetic, ordering, or RNG consumption. Opt-in until
native differential evidence covers the optimized profile.
"""
from dataclasses import fields,is_dataclass
from .ir import Expr,Statement


def optimize(ir, mode='none'):
    if mode not in ('none','safe','fusion','fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined'):
        raise ValueError('Optimization must be none, safe, fusion, fusion-avx, fusion-avx-math, fusion-avx-host fusion-avx-batch fusion-avx-quick fusion-avx-algebra or fusion-avx-outlined; received '+mode)
    report={'mode':mode,'passes':[],'preserves':['agent scheduling','ordered collections','conditional RNG draws','float64 arithmetic barriers'],'validation':'requires differential validation; compilation is not certification'}
    if mode=='none':return report
    ir.optimize_fusion=mode in ('fusion','fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    ir.optimize_no_fma_isa=mode in ('fusion-avx','fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    if ir.optimize_no_fma_isa:report['passes'].append({'name':'fusion-with-avx-isa-cap','applied_sites':1,'proof_condition':'experimental CPU ISA cap to investigate contraction differences; requires native differential validation, not a fidelity guarantee'})
    ir.optimize_outlined_rng=mode=='fusion-avx-outlined' and ir.random_generator=='mersenne' and ir.random_seed is not None
    if ir.optimize_outlined_rng:report['passes'].append({'name':'outlined-exact-mersenne-helpers','applied_sites':1,'proof_condition':'same standalone RNG functions behind non-inline JAX calls; keep the normal HLO call-inlining pipeline while exposing shared JAX helper calls; restore plain functions in the NumPy initializer; native validation required'})
    ir.optimize_algebra=mode=='fusion-avx-algebra'
    if ir.optimize_algebra:report['passes'].append({'name':'algebraic-simplification-with-division-barriers','applied_sites':1,'proof_condition':'experimental HLO simplification with explicit IR division operand barriers, AVX cap and CPU fast-math disabled; must pass full native traces before use'})
    ir.optimize_quick_compile=mode=='fusion-avx-quick'
    if ir.optimize_quick_compile:report['passes'].append({'name':'llvm-optimization-level-two','applied_sites':1,'proof_condition':'experimental backend code-generation tradeoff; retain fast-math prohibition, AVX cap and arithmetic barriers; renewed native validation required'})
    ir.optimize_spatial_order=True
    ir.optimize_native_trig=mode in ('fusion-avx-math','fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    ir.optimize_host_reset=mode in ('fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined') and ir.random_generator=='mersenne' and ir.random_seed is not None
    ir.optimize_grid_geometry=mode in ('fusion-avx-host','fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    ir.optimize_pure_mapper=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined') and ir.random_generator=='mersenne' and ir.random_seed is not None
    report['passes'].append({'name':'reuse-pure-getter-query-geometry','applied_sites':0,'proof_condition':'within one read-only mapper, repeated full-population unordered distance query with identical source and stored radius; reuse membership only, retaining each conditional draw budget, radius validation and first-error handling'})
    ir.optimize_independent_grid=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    report['passes'].append({'name':'compiled-pure-cell-grid-initializers','applied_sites':0,'proof_condition':'at least 1024 cells; initializer/facets read only this cell and stored globals, no actions/getters/RNG or other-cell references; preserve field order and coercion; reuse standalone JAX maps from the private NumPy initialization context'})
    ir.optimize_raster_tables=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    ir.optimize_unordered_rectangle=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined') and ir.random_generator=='mersenne' and ir.random_seed is not None
    ir.optimize_first_match=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    ir.optimize_single_return=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    report['passes'].append({'name':'inline-single-return-actions','applied_sites':0,'proof_condition':'one return statement, same expression effects/active mask and declared return coercion; omit unused return-control state'})
    ir.optimize_double_shuffle=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined')
    ir.optimize_prefetch_shuffles=mode=='fusion-avx-outlined' and ir.random_generator=='mersenne' and ir.random_seed is not None
    report['passes'].append({'name':'single-body-double-envelope-shuffle','applied_sites':0,'proof_condition':'compact once, preserve both ordered forward shuffle phases and all native draws, restore identical second-phase padding; shared loop body avoids duplicated RNG code'})
    report['passes'].append({'name':'prefetch-exact-envelope-shuffle-draws','applied_sites':0,'proof_condition':'two pure ordered permutation phases with no intervening effects; consume the exact bounded native sample stream, then run swaps without carrying the RNG arrays; preserve all padding and conditional draw counts'})
    report['passes'].append({'name':'short-circuit-first-matching-agent','applied_sites':0,'proof_condition':'same ordered predicate evaluation until first true selected entry; eliminate only remaining masked evaluations and their unused arithmetic'})
    report['passes'].append({'name':'elide-unobserved-rectangle-permutations','applied_sites':0,'proof_condition':'direct empty/length consumer of point-agent rectangular inside query; same envelope/predicate and both shuffle draw counts, preserve all observable ordered query paths'})
    report['passes'].append({'name':'hoisted-read-only-raster-pixels','applied_sites':0,'proof_condition':'reuse immutable decoded RGBA data by asset identity; preserve pixel indexing, coercions, bounds checks and asset checksums'})
    report['passes'].append({'name':'vectorized-independent-grid-updates','applied_sites':0,'proof_condition':'grid has no reflexes; updates and facets read only this cell and stored globals, no getters/actions/RNG or other cells; preserve field update/coercion order'})
    report['passes'].append({'name':'vectorized-discard-only-integer-getters','applied_sites':0,'proof_condition':'global read-only integer/boolean maps; only unobserved at-distance permutations may use RNG; compute masked per-entry draw budgets and advance exact MT words once; bounded pairwise allocation, no reordering of actions or floating reductions'})
    ir.optimize_constant_grid_initializers=ir.optimize_host_reset
    report['passes'].append({'name':'broadcast-invariant-grid-initializers','applied_sites':0,'proof_condition':'scalar/point/color initialization from literals and unchanged stored globals only; no agent/grid reads, getters, actions or RNG; same typed coercion and facets'})
    report['passes'].append({'name':'precomputed-ordered-grid-geometry','applied_sites':0,'proof_condition':'frozen grid dimensions and bounded radii; identical ordered distinct candidates before shuffle, retain every runtime shuffle/draw; bounded table budget and original fallback for invalid references'})
    report['passes'].append({'name':'functional-numpy-initialization','applied_sites':int(ir.optimize_host_reset),'proof_condition':'execute unchanged emitted initializer in a private NumPy namespace, preserve functional array updates and ordered native RNG/effects, transfer resulting state to JAX; requires full native initialization/trajectory validation'})
    report['passes'].append({'name':'avoid-scalar-initializer-array-views','applied_sites':int(ir.optimize_host_reset),'proof_condition':'retain NumPy scalar/zero-dimensional values and dtypes without allocating indexed-update wrappers; functional copies remain for indexed arrays; full private initialization state and native trajectories must agree'})
    report['passes'].append({'name':'hotspot-normalized-movement-trigonometry','applied_sites':int(ir.optimize_native_trig),'proof_condition':'experimental one-turn HotSpot range reduction and compensated polynomial sums; mathematically generated table, no model branches; requires primitive and full native trajectory validation on the configured JVM'})
    report['passes'].append({'name':'incremental-quadtree-leaf-occupancy','applied_sites':1,'proof_condition':'exact int32 count of identities per leaf; decrement before movement/death, redistribute only on subdivision, increment after insertion; identical subdivision threshold and traversal'})
    report['passes'].append({'name':'unsplittable-small-population-quadtree','applied_sites':1,'proof_condition':'at most 100 reserved identities; removing the inserted identity leaves fewer than the native subdivision threshold; root remains unsplit for births and movements'})
    report['passes'].append({'name':'lazy-quadtree-subdivision','applied_sites':1,'proof_condition':'skip occupancy counts on known subdivided nodes; redistribute population only when a previously unsplit node splits; same traversal arithmetic and persistent flags'})
    ir.optimize_integer_sum=True
    ir.optimize_sparse_sum=True
    report['passes'].append({'name':'compact-ordered-float-reduction','applied_sites':1,'proof_condition':'visit selected entries in original order; identical casts, additions, initial zero and arithmetic barriers; no parallel floating reduction'})
    ir.optimize_reflex_guards=True
    report['passes'].append({'name':'short-circuit-reflex-guards','applied_sites':sum(1 for scope in ir.scopes if scope.name!='global' for behavior in scope.behaviors if behavior.guard.op!='literal'),'proof_condition':'evaluate native guard exactly once; execute body only when true; retain agent order and local cleanup'})
    report['passes'].append({'name':'parallel-masked-integer-sum','applied_sites':1,'proof_condition':'fixed-width integer addition modulo 2^w; retain ordered floating reductions'})
    report['passes'].append({'name':'incremental-quadtree-order','applied_sites':1,'proof_condition':'integer lexicographic insertion of the moved identity; full stable sort whenever a split changes another leaf; no change to query order or RNG'})
    if ir.optimize_fusion:report['passes'].append({'name':'guarded-hlo-fusion','applied_sites':1,'proof_condition':'retain source arithmetic barriers, CPU fast-math disabled and algebraic simplification disabled; experimental backend profile requiring native differential certification'})
    def stream_body_safe(items):
        def expression_safe(expr):
            if expr.op.startswith('attribute:'):
                owner=expr.args[0].type.split(':')[-1];name=expr.op.split(':')[1]
                target=next((scope for scope in ir.scopes if scope.name==owner),None)
                if target and any(f.name==name and f.getter is not None for f in target.fields):return False
            if expr.op=='name' and any(f.name==expr.args[0] and f.getter is not None for scope in ir.scopes for f in scope.fields):return False
            return all(expression_safe(a) for a in expr.args if isinstance(a,Expr))
        for item in items:
            if item.kind=='assign':
                if item.value.args[1].type!='int' or not expression_safe(item.value.args[1]):return False
            elif item.kind=='if':
                if not expression_safe(item.value) or not stream_body_safe(item.body) or not stream_body_safe(item.alternative):return False
            else:return False
        return True
    ir.optimize_mersenne=ir.random_generator=='mersenne' and ir.random_seed is not None
    ir.optimize_grid_random_fields=mode in ('fusion-avx-batch','fusion-avx-quick','fusion-avx-algebra','fusion-avx-outlined') and ir.optimize_mersenne
    report['passes'].append({'name':'shared-loop-native-uniform-pairs','applied_sites':0,'proof_condition':'two consecutive native draws under the same active mask, with no intervening effects; retain coordinate order and full RNG state with one loop body'})
    report['passes'].append({'name':'batch-native-uniform-grid-initializers','applied_sites':0,'proof_condition':'direct numeric rnd initializer with invariant scalar bounds; same ordered MT samples and per-cell typed coercion/facets, cached block copies with odd-word fallback'})
    report['passes'].append({'name':'batch-ordered-grid-neighborhood-initializers','applied_sites':0,'proof_condition':'invariant radius and static grid geometry; draw the exact concatenated native sample stream, preserve every row forward permutation and padding, no other initialization effects interleaved'})
    ir.optimize_grid_init_draws=ir.optimize_mersenne
    report['passes'].append({'name':'batch-discard-implicit-grid-shuffles','applied_sites':sum(bool(scope.grid) for scope in ir.scopes) if ir.optimize_grid_init_draws else 0,'proof_condition':'initial radius-one grid neighborhoods are discarded before field initialization; count distinct native candidates including boundaries/rectangular torus, preserve every consumed Mersenne word'})
    ir.optimize_reset=ir.optimize_mersenne
    report['passes'].append({'name':'dynamic-seeded-compiled-reset','applied_sites':int(ir.optimize_reset),'proof_condition':'unchanged host seed conversion passed as 624 uint32 words; same ordered initializer IR; cache initialization executable across seeds; native validation required for previously eager initialization'})
    ir.optimize_unordered_distance=ir.optimize_mersenne
    ir.optimize_contact_cache=ir.optimize_mersenne
    report['passes'].append({'name':'memoized-unordered-contact-row','applied_sites':0,'proof_condition':'same-species full population query, shared stored global/literal radius, pure count/length; reuse one row only for the same identity, radius and spatial/lifecycle epoch; retain native shuffle draw counts; bounded 512–4096 slots'})
    report['passes'].append({'name':'elide-unobserved-spatial-permutations','applied_sites':int(ir.optimize_unordered_distance),'proof_condition':'length or count with pure stored-field predicate; membership unchanged and both native shuffle draw counts advanced exactly'})
    report['passes'].append({'name':'cached-mersenne-uniforms','applied_sites':int(ir.optimize_mersenne),'proof_condition':'dependency-preserving uint32 twist, exact binary paired-uniform cache, batch advance of discarded draws; preserve raw state/index'})
    seen=set();count=0;streamed=0
    def visit(value):
        nonlocal count,streamed
        if isinstance(value,(dict,list,tuple)) or is_dataclass(value):
            if id(value) in seen:return
            seen.add(id(value))
        if isinstance(value,Expr) and value.op=='accumulate':
            inner=value.args[1]
            # This lowering has no user actions/getters or declaration writes:
            # only masked RNG advancement and masked error checks. Missing
            # outer entries cannot contribute values or draws.
            if inner.op=='overlapping' and inner.args[0].op=='name' and inner.args[0].args[0] in ir.populations and inner.args[1].op=='each_ref':
                value.metadata['skip_inactive_entries']=True;count+=1
        if isinstance(value,Statement) and value.kind=='collection_loop' and value.metadata.get('commutative_integer_reduction') and value.value.op=='accumulate' and stream_body_safe(value.body):
            inner=value.value.args[1]
            if inner.op=='overlapping' and inner.args[0].op=='name' and inner.args[0].args[0] in ir.populations and inner.args[1].op=='each_ref':
                value.metadata['stream_integer_accumulation']=True;streamed+=1
        if is_dataclass(value):children=[getattr(value,f.name) for f in fields(value)]
        elif isinstance(value,dict):children=value.values()
        elif isinstance(value,(list,tuple)):children=value
        else:return
        for child in children:visit(child)
    visit(ir)
    report['passes'].append({'name':'skip-inactive-spatial-accumulation','applied_sites':count,'proof_condition':'plain population overlapping each_ref; absent/masked outer entries have no observable effects'})
    report['passes'].append({'name':'stream-spatial-integer-reductions','applied_sites':streamed,'proof_condition':'checker-proven independent local int32 sums over plain spatial accumulation; preserve query RNG order'})
    report['passes'].append({'name':'guarded-grid-reference-overlap','applied_sites':0,'proof_condition':'within a readonly integer stream, verify rectilinear separated cell centers and all live point/reference alignments; aggregate independent int32 sums with exact cell multiplicities and discarded shuffle budgets; otherwise preserve the original geometric stream'})
    return report
