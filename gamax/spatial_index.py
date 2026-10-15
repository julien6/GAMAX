"""Standalone point-only ordered quadtree primitives for GAMA 2025.06.4.

Leaf insertion order and persistent splits are observable through native query
shuffles. The minimum leaf size is width/100, with a distinct root split test.
"""
QUADTREE_SOURCE = '''def quadtree_dfs_keys():
    keys=[]
    for original in range((4**8-1)//3):
        node=original;value=0;factor=1;depth=0
        while node:
            value+=((node-1)%4)*factor
            node=(node-1)//4;factor*=4;depth+=1
        keys.append(value*4**(7-depth))
    return jnp.asarray(keys,jnp.int32)

QUADTREE_DFS_KEYS=quadtree_dfs_keys()

def quadtree_insert(split,leaf,order,tick,positions,index,width,height):
    leaf=leaf.at[index].set(-1)
    position=positions[index,:2]
    def can_split(depth,lower,upper):
        return jnp.where(depth==0,(width>10.) & (height>10.),jnp.all(upper-lower>width/100.)) & (depth<7)
    def condition(state):
        flags,leaves,node,depth,lower,upper=state
        return flags[node] | (can_split(depth,lower,upper) & (jnp.sum(leaves==node)>=100))
    def descend(state):
        flags,leaves,node,depth,lower,upper=state
        middle=(lower+upper)/2.
        top=(positions[:,1]>=lower[1]) & (positions[:,1]<middle[1])
        left=(positions[:,0]>=lower[0]) & (positions[:,0]<middle[0])
        quadrants=(~left).astype(jnp.int32)+2*(~top).astype(jnp.int32)
        leaves=jnp.where((leaves==node) & ~flags[node],node*4+1+quadrants,leaves)
        flags=flags.at[node].set(True)
        top=(position[1]>=lower[1]) & (position[1]<middle[1])
        left=(position[0]>=lower[0]) & (position[0]<middle[0])
        quadrant=(~left).astype(jnp.int32)+2*(~top).astype(jnp.int32)
        low=jnp.where(jnp.stack([left,top]),lower,middle)
        high=jnp.where(jnp.stack([left,top]),middle,upper)
        return flags,leaves,node*4+1+quadrant,depth+1,low,high
    split,leaf,node,_,_,_=jax.lax.while_loop(condition,descend,(split,leaf,jnp.asarray(0,jnp.int32),jnp.asarray(0,jnp.int32),jnp.zeros((2,),jnp.float64),jnp.asarray([width,height],jnp.float64)))
    return split,leaf.at[index].set(node),order.at[index].set(tick),tick+jnp.asarray(1,jnp.int64)

def collection_shuffle_forward(rng,values,mask,uniform):
    size=values.shape[0]
    if size==0:return rng,values,mask
    count=jnp.sum(mask,dtype=jnp.int32)
    indices=jnp.nonzero(mask,size=size,fill_value=0)[0]
    values=values[indices]
    def swap(i,state):
        rng,values=state
        rng,sample=uniform(rng)
        choice=i+jnp.trunc((count-i).astype(jnp.float64)*sample).astype(jnp.int32)
        a,b=values[i],values[choice]
        return rng,values.at[i].set(b).at[choice].set(a)
    rng,values=jax.lax.fori_loop(0,jnp.where(count>=2,count,0),swap,(rng,values))
    return rng,values,jnp.arange(size)<count

def point_neighbors_at_distance(rng,positions,leaf,indices,allowed,source,exclude,radius,uniform):
    size=positions.shape[0]
    if size==0:return rng,(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))
    delta=positions-source
    envelope=radius*jnp.asarray(1.4142135623730951,jnp.float64)
    mask=allowed & (leaf>=0) & (jnp.arange(size)!=exclude) & jnp.all(jnp.abs(delta[:,:2])<=envelope,axis=1)
    rng,indices,present=collection_shuffle_forward(rng,indices,mask[indices],uniform)
    distances=jnp.sqrt(jnp.sum(delta[indices]*delta[indices],axis=1))
    present=present & (distances<=radius)
    rng,indices,present=collection_shuffle_forward(rng,indices,present,uniform)
    return rng,(indices,present)
'''

CIRCLE_QUERY_SOURCE = '''
def circle_unit_vertices():
    points=[[math.cos(-i*(math.pi/16)),math.sin(-i*(math.pi/16))] for i in range(32)]
    for index,point in ((0,[1.,0.]),(8,[0.,-1.]),(16,[-1.,0.]),(24,[0.,1.])):points[index]=point
    return jnp.asarray(points,jnp.float64)

CIRCLE_UNIT_VERTICES=circle_unit_vertices()

def point_neighbors_in_circle(rng,positions,leaf,indices,allowed,source,radius,uniform):
    size=positions.shape[0]
    if size==0:return rng,(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))
    radius=jnp.maximum(radius,0.)
    delta=positions-source
    mask=allowed & (leaf>=0) & jnp.all(jnp.abs(delta[:,:2])<=radius,axis=1)
    rng,indices,present=collection_shuffle_forward(rng,indices,mask[indices],uniform)
    # Native envelope search shuffles twice before the polygon predicate.
    rng,indices,present=collection_shuffle_forward(rng,indices,present,uniform)
    vertices=jax.lax.optimization_barrier(source[:2]+jax.lax.optimization_barrier(CIRCLE_UNIT_VERTICES*radius))
    edges=jax.lax.optimization_barrier(jnp.roll(vertices,-1,axis=0)-vertices)
    point_delta=jax.lax.optimization_barrier(positions[indices,None,:2]-vertices[None,:,:])
    cross=jax.lax.optimization_barrier(edges[None,:,0]*point_delta[:,:,1])-jax.lax.optimization_barrier(edges[None,:,1]*point_delta[:,:,0])
    present=present & jnp.where(radius>0,jnp.all(cross<=0,axis=1),jnp.all(delta[indices,:2]==0,axis=1))
    return rng,(indices,present)
'''

RECT_QUERY_SOURCE = """
def point_agents_inside_rectangle(rng,positions,leaf,indices,allowed,source,half,exclude,uniform):
    size=positions.shape[0]
    if size==0:return rng,(jnp.zeros((0,),jnp.int32),jnp.zeros((0,),jnp.bool_))
    delta=positions-source
    mask=allowed & (leaf>=0) & (jnp.arange(size)!=exclude) & jnp.all(jnp.abs(delta[:,:2])<=half,axis=1)
    rng,indices,present=collection_shuffle_forward(rng,indices,mask[indices],uniform)
    rng,indices,present=collection_shuffle_forward(rng,indices,present,uniform)
    present=present & (delta[indices,2]==0)
    return rng,(indices,present)
"""

RECT_UNORDERED_SOURCE = '''
def point_agents_inside_rectangle_unordered(rng,positions,leaf,allowed,source,half,exclude):
    size=positions.shape[0]
    delta=positions-source
    box=allowed & (leaf>=0) & (jnp.arange(size)!=exclude) & jnp.all(jnp.abs(delta[:,:2])<=half,axis=1)
    count=jnp.sum(box,dtype=jnp.int32)
    rng=mersenne_discard_uniforms(rng,2*jnp.where(count>=2,count,0))
    return rng,(jnp.arange(size,dtype=jnp.int32),box & (delta[:,2]==0))
'''

DOUBLE_SHUFFLE_SOURCE = '''
def collection_shuffle_twice_forward(rng,values,mask,uniform):
    size=values.shape[0]
    if size==0:return rng,values,mask
    count=jnp.sum(mask,dtype=jnp.int32)
    packed=values[jnp.nonzero(mask,size=size,fill_value=0)[0]]
    packed=jnp.where(count<2,jnp.full_like(packed,packed[0]),packed)
    def swap(iteration,state):
        rng,values=state
        values=jax.lax.cond(iteration==count,lambda v:jnp.where(jnp.arange(size)<count,v,v[0]),lambda v:v,values)
        index=jnp.where(iteration<count,iteration,iteration-count)
        rng,sample=uniform(rng)
        choice=index+jnp.trunc((count-index).astype(jnp.float64)*sample).astype(jnp.int32)
        a,b=values[index],values[choice]
        return rng,values.at[index].set(b).at[choice].set(a)
    rng,packed=jax.lax.fori_loop(0,jnp.where(count>=2,count*2,0),swap,(rng,packed))
    return rng,packed,jnp.arange(size)<count
'''

DOUBLE_SHUFFLE_BATCH_SOURCE = '''
def collection_shuffle_twice_forward(rng,values,mask,uniform):
    size=values.shape[0]
    if size==0:return rng,values,mask
    count=jnp.sum(mask,dtype=jnp.int32)
    packed=values[jnp.nonzero(mask,size=size,fill_value=0)[0]]
    packed=jnp.where(count<2,jnp.full_like(packed,packed[0]),packed)
    draws=jnp.where(count>=2,count*2,0)
    rng,samples=mersenne_uniforms_bounded(rng,draws,size*2)
    def swap(iteration,values):
        values=jax.lax.cond(iteration==count,lambda v:jnp.where(jnp.arange(size)<count,v,v[0]),lambda v:v,values)
        index=jnp.where(iteration<count,iteration,iteration-count)
        choice=index+jnp.trunc((count-index).astype(jnp.float64)*samples[iteration]).astype(jnp.int32)
        a,b=values[index],values[choice]
        return values.at[index].set(b).at[choice].set(a)
    packed=jax.lax.fori_loop(0,draws,swap,packed)
    return rng,packed,jnp.arange(size)<count
'''

INCREMENTAL_ORDER_SOURCE = '''
def quadtree_update_order(indices,previous_leaf,leaf,order,index):
    size=indices.shape[0]
    if size==0:return indices
    keys=jnp.zeros(size,jnp.int32) if size<=100 else QUADTREE_DFS_KEYS[jnp.maximum(leaf,0)]
    # A split redistributes other agents; retain the general stable sort then.
    split_changed=jnp.any((previous_leaf!=leaf) & (jnp.arange(size)!=index))
    def insert(_):
        positions=jnp.arange(size)
        old_position=jnp.argmax(indices==index)
        before=(keys<keys[index]) | ((keys==keys[index]) & ((order<order[index]) | ((order==order[index]) & (positions<index))))
        rank=jnp.sum(before,dtype=jnp.int32)
        remaining=indices[jnp.minimum(positions+(positions>=old_position),size-1)]
        return jnp.where(positions==rank,index,remaining[jnp.maximum(positions-(positions>rank),0)]).astype(jnp.int32)
    return jax.lax.cond(split_changed,lambda _:jnp.lexsort((order,keys)).astype(jnp.int32),insert,operand=None)
'''

UNORDERED_DISTANCE_SOURCE = '''
def point_neighbors_unordered(rng,positions,leaf,allowed,source,exclude,radius):
    size=positions.shape[0]
    delta=positions-source
    envelope=radius*jnp.asarray(1.4142135623730951,jnp.float64)
    selected=allowed & (leaf>=0) & (jnp.arange(size)!=exclude) & jnp.all(jnp.abs(delta[:,:2])<=envelope,axis=1)
    count=jnp.sum(selected,dtype=jnp.int32)
    distances=jnp.sqrt(jnp.sum(delta*delta,axis=1))
    present=selected & (distances<=radius)
    inside=jnp.sum(present,dtype=jnp.int32)
    draws=jnp.where(count>=2,count,0)+jnp.where(inside>=2,inside,0)
    rng=mersenne_discard_uniforms(rng,draws)
    return rng,(jnp.arange(size,dtype=jnp.int32),present)
'''

# Reuse the reference arithmetic and insertion contract verbatim. Only avoid
# population-wide work whose result is already known from the split flag.
QUADTREE_OPTIMIZED_SOURCE = QUADTREE_SOURCE.replace(
    '    leaf=leaf.at[index].set(-1)\n',
    '''    # Removing the inserted identity leaves fewer than 100 occupants.
    # With at most 100 reserved slots the root can never subdivide.
    if positions.shape[0]<=100:
        return split,leaf.at[index].set(0),order.at[index].set(tick),tick+jnp.asarray(1,jnp.int64)
    leaf=leaf.at[index].set(-1)
''',
).replace(
    'return flags[node] | (can_split(depth,lower,upper) & (jnp.sum(leaves==node)>=100))',
    'return jax.lax.cond(flags[node],lambda _:jnp.asarray(True),lambda _:jax.lax.cond(can_split(depth,lower,upper),lambda _:jnp.sum(leaves==node)>=100,lambda _:jnp.asarray(False),operand=None),operand=None)',
).replace(
    '''        top=(positions[:,1]>=lower[1]) & (positions[:,1]<middle[1])
        left=(positions[:,0]>=lower[0]) & (positions[:,0]<middle[0])
        quadrants=(~left).astype(jnp.int32)+2*(~top).astype(jnp.int32)
        leaves=jnp.where((leaves==node) & ~flags[node],node*4+1+quadrants,leaves)
        flags=flags.at[node].set(True)''',
    '''        def redistribute(state):
            flags,leaves=state
            top=(positions[:,1]>=lower[1]) & (positions[:,1]<middle[1])
            left=(positions[:,0]>=lower[0]) & (positions[:,0]<middle[0])
            quadrants=(~left).astype(jnp.int32)+2*(~top).astype(jnp.int32)
            leaves=jnp.where(leaves==node,node*4+1+quadrants,leaves)
            return flags.at[node].set(True),leaves
        flags,leaves=jax.lax.cond(flags[node],lambda state:state,redistribute,(flags,leaves))''',
)

_COUNTED_INSERT = '''def quadtree_insert(split,leaf,order,tick,counts,positions,index,width,height):
    previous=leaf[index]
    if positions.shape[0]<=100:
        counts=counts.at[0].add((previous<0).astype(jnp.int32))
        return split,leaf.at[index].set(0),order.at[index].set(tick),tick+jnp.asarray(1,jnp.int64),counts
    counts=counts.at[jnp.maximum(previous,0)].add(jnp.where(previous>=0,-1,0))
    leaf=leaf.at[index].set(-1)
    position=positions[index,:2]
    def can_split(depth,lower,upper):
        return jnp.where(depth==0,(width>10.) & (height>10.),jnp.all(upper-lower>width/100.)) & (depth<7)
    def condition(state):
        flags,leaves,occupancy,node,depth,lower,upper=state
        return flags[node] | (can_split(depth,lower,upper) & (occupancy[node]>=100))
    def descend(state):
        flags,leaves,occupancy,node,depth,lower,upper=state
        middle=(lower+upper)/2.
        def redistribute(state):
            flags,leaves,occupancy=state
            top=(positions[:,1]>=lower[1]) & (positions[:,1]<middle[1])
            left=(positions[:,0]>=lower[0]) & (positions[:,0]<middle[0])
            quadrants=(~left).astype(jnp.int32)+2*(~top).astype(jnp.int32)
            selected=leaves==node
            child_counts=jnp.stack([jnp.sum(selected & (quadrants==q),dtype=jnp.int32) for q in range(4)])
            occupancy=occupancy.at[node].set(0).at[node*4+1+jnp.arange(4)].set(child_counts)
            leaves=jnp.where(selected,node*4+1+quadrants,leaves)
            return flags.at[node].set(True),leaves,occupancy
        flags,leaves,occupancy=jax.lax.cond(flags[node],lambda state:state,redistribute,(flags,leaves,occupancy))
        top=(position[1]>=lower[1]) & (position[1]<middle[1])
        left=(position[0]>=lower[0]) & (position[0]<middle[0])
        quadrant=(~left).astype(jnp.int32)+2*(~top).astype(jnp.int32)
        low=jnp.where(jnp.stack([left,top]),lower,middle)
        high=jnp.where(jnp.stack([left,top]),middle,upper)
        return flags,leaves,occupancy,node*4+1+quadrant,depth+1,low,high
    split,leaf,counts,node,_,_,_=jax.lax.while_loop(condition,descend,(split,leaf,counts,jnp.asarray(0,jnp.int32),jnp.asarray(0,jnp.int32),jnp.zeros((2,),jnp.float64),jnp.asarray([width,height],jnp.float64)))
    counts=counts.at[node].add(1)
    return split,leaf.at[index].set(node),order.at[index].set(tick),tick+jnp.asarray(1,jnp.int64),counts

'''
_insert_start=QUADTREE_OPTIMIZED_SOURCE.index('def quadtree_insert(')
_insert_end=QUADTREE_OPTIMIZED_SOURCE.index('def collection_shuffle_forward(')
QUADTREE_COUNTED_SOURCE=QUADTREE_OPTIMIZED_SOURCE[:_insert_start]+_COUNTED_INSERT+QUADTREE_OPTIMIZED_SOURCE[_insert_end:]

CONTACT_CACHE_SOURCE = '''
def contact_row(positions,alive,leaf,index,radius):
    delta=positions-positions[index]
    active=alive & (leaf>=0)
    envelope=radius*jnp.asarray(1.4142135623730951,jnp.float64)
    box=active & active[index] & (jnp.arange(positions.shape[0])!=index) & jnp.all(jnp.abs(delta[:,:2])<=envelope,axis=1)
    nearby=box & (jnp.sqrt(jnp.sum(delta*delta,axis=1))<=radius)
    return box,nearby

def build_contact_cache(positions,alive,leaf,radius):
    size=positions.shape[0]
    def row(index,cache):
        boxes,nearby,box_counts,counts=cache
        box,selected=contact_row(positions,alive,leaf,index,radius)
        return boxes.at[index].set(box),nearby.at[index].set(selected),box_counts.at[index].set(jnp.sum(box,dtype=jnp.int32)),counts.at[index].set(jnp.sum(selected,dtype=jnp.int32))
    return jax.lax.fori_loop(0,size,row,(jnp.zeros((size,size),jnp.bool_),jnp.zeros((size,size),jnp.bool_),jnp.zeros(size,jnp.int32),jnp.zeros(size,jnp.int32)))

def update_contact_cache(cache,positions,alive,leaf,index,radius):
    boxes,nearby,box_counts,counts=cache
    box,selected=contact_row(positions,alive,leaf,index,radius)
    box_counts=box_counts+box.astype(jnp.int32)-boxes[:,index].astype(jnp.int32)
    counts=counts+selected.astype(jnp.int32)-nearby[:,index].astype(jnp.int32)
    boxes=boxes.at[index].set(box).at[:,index].set(box)
    nearby=nearby.at[index].set(selected).at[:,index].set(selected)
    return boxes,nearby,box_counts.at[index].set(jnp.sum(box,dtype=jnp.int32)),counts.at[index].set(jnp.sum(selected,dtype=jnp.int32))

def cached_point_neighbors(rng,valid,cached_radius,cache,positions,alive,leaf,index,radius):
    def cached(_):
        updated=jax.lax.cond(valid & (cached_radius==radius),lambda _:cache,lambda _:build_contact_cache(positions,alive,leaf,radius),operand=None)
        boxes,nearby,box_counts,counts=updated
        count,inside=box_counts[index],counts[index]
        advanced=mersenne_discard_uniforms(rng,jnp.where(count>=2,count,0)+jnp.where(inside>=2,inside,0))
        return advanced,(jnp.arange(positions.shape[0],dtype=jnp.int32),nearby[index]),jnp.asarray(True),radius,updated
    def uncached(_):
        advanced,result=point_neighbors_unordered(rng,positions,leaf,alive,positions[index],index,radius)
        return advanced,result,valid,cached_radius,cache
    return jax.lax.cond(alive[index] & (leaf[index]>=0),cached,uncached,operand=None)
'''

CONTACT_ROW_CACHE_SOURCE = '''
def memoized_point_neighbors(rng,valid,cached_radius,cache,cached_source,cached_epoch,epoch,positions,alive,leaf,index,radius):
    index=jnp.asarray(index,jnp.int32)
    def calculate(_):
        delta=positions-positions[index]
        envelope=radius*jnp.asarray(1.4142135623730951,jnp.float64)
        box=alive & (leaf>=0) & (jnp.arange(positions.shape[0])!=index) & jnp.all(jnp.abs(delta[:,:2])<=envelope,axis=1)
        nearby=box & (jnp.sqrt(jnp.sum(delta*delta,axis=1))<=radius)
        return box,nearby,jnp.sum(box,dtype=jnp.int32),jnp.sum(nearby,dtype=jnp.int32)
    updated=jax.lax.cond(valid & (cached_radius==radius) & (cached_source==index) & (cached_epoch==epoch),lambda _:cache,calculate,operand=None)
    boxes,nearby,count,inside=updated
    rng=mersenne_discard_uniforms(rng,jnp.where(count>=2,count,0)+jnp.where(inside>=2,inside,0))
    return rng,(jnp.arange(positions.shape[0],dtype=jnp.int32),nearby),jnp.asarray(True),radius,updated,index,epoch
'''
