"""Standalone ordered rectangular grid primitives for the pinned native oracle.

GAMA shuffles the ordered neighborhood on every getNeighborsIn call. Caching
geometric indices does not remove those random draws. A sorted set would change
both the list order and every later draw in the model.
"""

GRID_REFERENCE_SOURCE = '''
def grid_reference_membership_valid(locations,references,present,centers,half,width,height):
    # Runtime proof: rectilinear centers with monotone, separated axes. A
    # center can then overlap only its own closed cell box, including borders.
    rows=centers.reshape((height,width,3))
    regular=jnp.all(rows[:,:,0]==rows[0,:,0][None,:]) & jnp.all(rows[:,:,1]==rows[:,0,1][:,None])
    x_delta=jnp.diff(rows[0,:,0]);y_delta=jnp.diff(rows[:,0,1])
    x_separated=jnp.all(x_delta>half[0]) | jnp.all(-x_delta>half[0])
    y_separated=jnp.all(y_delta>half[1]) | jnp.all(-y_delta>half[1])
    valid_references=(references>=0) & (references<width*height)
    selected=centers[jnp.clip(references,0,width*height-1),:2]
    aligned=jnp.all(~present | (valid_references & jnp.all(locations[:,:2]==selected,axis=1)))
    finite=jnp.all(jnp.isfinite(rows[0,:,0])) & jnp.all(jnp.isfinite(rows[:,0,1]))
    return regular & finite & x_separated & y_separated & jnp.all(half>=0) & aligned
'''

def implicit_grid_shuffle_draws(width, height, *, moore=False, torus=False):
    """Count the discarded radius-one shuffles using the native index rules.

    Grid dimensions are frozen in the IR. Preserve degenerate-grid duplicates
    and the pinned rectangular-torus negative-Y behavior, too.
    """
    def remainder(value, modulus):
        return -((-value)%modulus) if value<0 else value%modulus
    total=0
    offsets=[(0,-1),(0,1),(-1,1),(1,-1),(-1,0),(1,0),(-1,-1),(1,1)] if moore else [(1,0),(-1,0),(0,1),(0,-1)]
    for index in range(width*height):
        x,y=index%width,index//width;neighbors=set()
        for dx,dy in offsets:
            a,b=x+dx,y+dy
            if torus:
                if a<0:a+=width
                if b<0:b+=width
                a,b=remainder(a,width),remainder(b,height)
            elif not (0<=a<width and 0<=b<height):continue
            neighbors.add(b*width+a)
        if len(neighbors)>=2:total+=len(neighbors)
    return total


def grid_neighbor_geometry(width,height,bound,*,moore=False,torus=False):
    """Ordered, unique native candidates and their first appearance radius."""
    capacity=(2*bound+1)**2-1 if moore else 2*bound*(bound+1)
    values,distances=[],[]
    def remainder(value,modulus):return -((-value)%modulus) if value<0 else value%modulus
    for index in range(width*height):
        x,y=index%width,index//width;seen=set();row=[];radii=[]
        for distance in range(1,bound+1):
            if moore:
                offsets=[p for k in range(1-distance,distance) for p in ((k,-distance),(-k,distance))]
                offsets += [p for k in range(-distance,distance+1) for p in ((-distance,-k),(distance,k))]
            else:offsets=[p for k in range(-distance,distance) for p in ((-k,-abs(k)+distance),(k,abs(k)-distance))]
            for dx,dy in offsets:
                a,b=x+dx,y+dy
                if torus:
                    if a<0:a+=width
                    if b<0:b+=width
                    a,b=remainder(a,width),remainder(b,height)
                elif not (0<=a<width and 0<=b<height):continue
                candidate=b*width+a
                if candidate not in seen:
                    row.append(candidate);radii.append(distance);seen.add(candidate)
        values.append(row+[-1]*(capacity-len(row)))
        distances.append(radii+[-1]*(capacity-len(row)))
    return values,distances


GRID_CACHED_SOURCE = '''
def grid_neighbors_from_geometry(rng,geometry,radii,index,radius,*,uniform):
    values=geometry[index];distance=radii[index];capacity=values.shape[0]
    selected=(distance>=1) & (distance<=radius)
    used=jnp.sum(selected,dtype=jnp.int32)
    order=jnp.nonzero(selected,size=capacity,fill_value=0)[0]
    values=jnp.where(jnp.arange(capacity)<used,values[order],-1).astype(jnp.int32)
    def swap(iteration,state):
        rng,values=state
        rng,sample=uniform(rng)
        choice=iteration+jnp.trunc((used-iteration).astype(jnp.float64)*sample).astype(jnp.int32)
        a,b=values[iteration],values[choice]
        return rng,values.at[iteration].set(b).at[choice].set(a)
    rng,values=jax.lax.fori_loop(0,jnp.where(used>=2,used,0),swap,(rng,values))
    return rng,(values,jnp.arange(capacity)<used)
'''
GRID_SOURCE = '''def grid_neighbors(rng, index, width, height, radius, *, moore=False, torus=False, uniform=mersenne_uniform, max_radius=None):
    bound = radius if max_radius is None else max_radius
    capacity = (2 * bound + 1) ** 2 - 1 if moore else 2 * bound * (bound + 1)
    values = jnp.full((capacity,), -1, jnp.int32)
    used = jnp.asarray(0, jnp.int32)
    x, y = index % width, index // width
    def add(values, used, a, b):
        if torus:
            # Pinned native getPlaceIndexAt adds numCols to negative Y too.
            # Preserve that rectangular-torus behavior, not mathematical modulo.
            a = jnp.where(a < 0, a + width, a)
            b = jnp.where(b < 0, b + width, b)
            a = jnp.where(a < 0, -((-a) % width), a % width)
            b = jnp.where(b < 0, -((-b) % height), b % height)
            valid = jnp.asarray(True)
        else:
            valid = (a >= 0) & (a < width) & (b >= 0) & (b < height)
        candidate = b * width + a
        new = valid & (distance <= radius) & ~jnp.any((jnp.arange(capacity) < used) & (values == candidate))
        values = jax.lax.cond(new, lambda v: v.at[used].set(candidate), lambda v: v, values)
        return values, used + new.astype(jnp.int32)
    for distance in range(1, bound + 1):
        if moore:
            for k in range(1-distance, distance):
                values, used = add(values, used, x+k, y-distance)
                values, used = add(values, used, x-k, y+distance)
            for k in range(-distance, distance+1):
                values, used = add(values, used, x-distance, y-k)
                values, used = add(values, used, x+distance, y+k)
        else:
            for k in range(-distance, distance):
                values, used = add(values, used, x-k, y-abs(k)+distance)
                values, used = add(values, used, x+k, y+abs(k)-distance)
    def swap(iteration, state):
        rng, values = state
        remaining = used - iteration
        rng, sample = uniform(rng)
        choice = iteration + jnp.trunc(remaining.astype(jnp.float64) * sample).astype(jnp.int32)
        last = iteration
        a, b = values[last], values[choice]
        values = values.at[last].set(b).at[choice].set(a)
        return rng, values
    rng, values = jax.lax.fori_loop(0, jnp.where(used >= 2, used, 0), swap, (rng, values))
    return rng, (values, jnp.arange(capacity) < used)
'''

GRID_BATCH_INITIAL_SOURCE = '''
def initialized_grid_neighborhoods(rng,geometry,radii,radius):
    rows,capacity=geometry.shape
    if capacity==0:return rng,(geometry,jnp.zeros(geometry.shape,jnp.bool_))
    present=(radii>=1) & (radii<=radius)
    counts=jnp.sum(present,axis=1,dtype=jnp.int32)
    draws=jnp.where(counts>=2,counts,0)
    offsets=jnp.cumsum(draws,dtype=jnp.int32)-draws
    rng,samples=mersenne_uniforms_bounded(rng,jnp.sum(draws,dtype=jnp.int32),rows*capacity)
    values=jnp.where(present,geometry,-1).astype(jnp.int32)
    row_indices=jnp.arange(rows,dtype=jnp.int32)
    def swap(index,values):
        active=(counts>=2) & (index<counts)
        sample=samples[jnp.minimum(offsets+index,rows*capacity-1)]
        choice=jnp.clip(index+jnp.trunc((counts-index).astype(jnp.float64)*sample).astype(jnp.int32),0,capacity-1)
        a,b=values[row_indices,index],values[row_indices,choice]
        values=values.at[row_indices,index].set(jnp.where(active,b,a))
        return values.at[row_indices,choice].set(jnp.where(active,a,values[row_indices,choice]))
    values=jax.lax.fori_loop(0,capacity,swap,values)
    return rng,(values,present)
'''
