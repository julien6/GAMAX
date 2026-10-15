import jax
import jax.numpy as jnp
import numpy as np
from gamax.randomness import MERSENNE_SOURCE,MERSENNE_OPTIMIZED_SOURCE,mersenne_initial_state


def functions(source):
    namespace={'jax':jax,'jnp':jnp};exec(source,namespace);return namespace


def test_block_twist_preserves_in_place_dependencies_bit_for_bit():
    original=functions(MERSENNE_SOURCE);optimized=functions(MERSENNE_OPTIMIZED_SOURCE)
    rng=np.random.default_rng(20261014)
    words=[mersenne_initial_state(s) for s in range(1,11)]
    words += [np.zeros(624,np.uint32),np.full(624,0xffffffff,np.uint32),np.arange(624,dtype=np.uint32)]
    words += list(rng.integers(0,2**32,size=(32,624),dtype=np.uint32))
    baseline=jax.jit(jax.vmap(original['mersenne_twist']));fast=jax.jit(jax.vmap(optimized['mersenne_twist']))
    a=b=jnp.asarray(np.array(words,dtype=np.uint32))
    for _ in range(12):
        a=baseline(a);b=fast(b);np.testing.assert_array_equal(a,b)


def test_word_sequences_cross_twist_boundaries_exactly():
    original=functions(MERSENNE_SOURCE);optimized=functions(MERSENNE_OPTIMIZED_SOURCE)
    def sequence(function,state):
        return jax.lax.scan(lambda s,_:function(s),state,None,length=1300)
    baseline=jax.jit(lambda s:sequence(original['mersenne_word'],s));fast=jax.jit(lambda s:sequence(optimized['mersenne_word'],s))
    state=(jnp.asarray(mersenne_initial_state(2),jnp.uint32),jnp.asarray(624,jnp.int32))
    a,wa=baseline(state);b,wb=fast(state)
    np.testing.assert_array_equal(wa,wb);np.testing.assert_array_equal(a[0],b[0]);assert int(a[1])==int(b[1])


def test_cached_uniforms_and_discard_keep_native_stream_exact():
    from gamax.randomness import MERSENNE_CACHED_SOURCE
    jax.config.update('jax_enable_x64',True)
    original=functions(MERSENNE_SOURCE);fast=functions(MERSENNE_CACHED_SOURCE)
    def sequence(function,state):return jax.lax.scan(lambda s,_:function(s),state,None,length=1300)
    baseline=jax.jit(lambda s:sequence(original['mersenne_uniform'],s));cached=jax.jit(lambda s:sequence(fast['mersenne_uniform'],s))
    for seed in (1,2,10):
        raw=jnp.asarray(mersenne_initial_state(seed),jnp.uint32);index=jnp.asarray(624,jnp.int32)
        a,va=baseline((raw,index));b,vb=cached((raw,index,jnp.zeros((312,),jnp.float64)))
        np.testing.assert_array_equal(va,vb);np.testing.assert_array_equal(a[0],b[0]);assert int(a[1])==int(b[1])
    discard=jax.jit(fast['mersenne_discard_uniforms'])
    reference=jax.jit(lambda s,n:jax.lax.fori_loop(0,n,lambda _,v:original['mersenne_uniform'](v)[0],s))
    future_a=jax.jit(original['mersenne_uniform']);future_b=jax.jit(fast['mersenne_uniform'])
    raw=jnp.asarray(mersenne_initial_state(2),jnp.uint32)
    for position in (0,1,622,623,624):
        for count in (0,1,2,311,312,313,624,1300):
            index=jnp.asarray(position,jnp.int32);n=jnp.asarray(count,jnp.int32)
            a=reference((raw,index),n);b=discard((raw,index,fast['mersenne_cache'](raw)),n)
            np.testing.assert_array_equal(a[0],b[0]);assert int(a[1])==int(b[1])
            _,va=future_a(a);_,vb=future_b(b);np.testing.assert_array_equal(va,vb)
