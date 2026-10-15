"""Deferred, pointwise-maximum gradient convolution for rectangular grids."""
GRADIENT_SOURCE = '''
def gradient_convolution(values,proportion,width,height,radius,moore):
    # Native proportion zero requests the default proportion one.
    proportion=jnp.where(proportion==0.,jnp.asarray(1.,jnp.float64),proportion)
    original=values.reshape((height,width))
    source=jnp.where(original<0.,0.,original)
    padded=jnp.pad(source,((radius,radius),(radius,radius)),constant_values=0.)
    result=jnp.full((height,width),-jnp.finfo(jnp.float64).max,jnp.float64)
    neighbors=8 if moore else 4
    for y in range(2*radius+1):
        for x in range(2*radius+1):
            distance=max(abs(y-radius),abs(x-radius)) if moore else abs(y-radius)+abs(x-radius)
            coefficient=jax.lax.optimization_barrier(proportion/jnp.asarray(float(neighbors**distance),jnp.float64))
            product=jax.lax.optimization_barrier(padded[y:y+height,x:x+width]*coefficient)
            result=jnp.maximum(result,product)
    return jnp.where(result>source,result,original).reshape((-1,))
'''
