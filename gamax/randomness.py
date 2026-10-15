"""Pinned GAMA 2025.06.4 MT19937 seed conversion and standalone JAX source.

The seed array is not NumPy's integer-seed convention: native GAMA repeats the
little-endian IEEE-754 seed bytes, then decodes four big-endian int32 words.
"""
import math
import struct


def mersenne_initial_state(seed):
    seed = abs(float(seed))
    if not math.isfinite(seed) or seed == 0:
        raise ValueError('A finite nonzero seed is required; native seed zero is nondeterministic')
    if seed < 1: seed *= 9223372036854775808.0
    data = struct.pack('<d', seed) * 2
    key = [int.from_bytes(data[i:i+4], 'big') for i in range(0, 16, 4)]
    mask = (1 << 32) - 1
    mt = [19650218] + [0]*623
    for i in range(1,624):
        mt[i] = (1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i) & mask
    i,j=1,0
    for _ in range(624):
        mt[i] = ((mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30))*1664525)) + key[j] + j) & mask
        i+=1;j+=1
        if i>=624: mt[0]=mt[623];i=1
        if j>=4:j=0
    for _ in range(623):
        mt[i] = ((mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30))*1566083941)) - i) & mask
        i+=1
        if i>=624:mt[0]=mt[623];i=1
    mt[0]=0x80000000
    return mt


MERSENNE_SOURCE = '''def mersenne_twist(mt):
    def update(i, words):
        y = (words[i] & jnp.uint32(0x80000000)) | (words[(i + 1) % 624] & jnp.uint32(0x7fffffff))
        value = words[(i + 397) % 624] ^ (y >> jnp.uint32(1)) ^ jnp.where((y & jnp.uint32(1)) != 0, jnp.uint32(0x9908b0df), jnp.uint32(0))
        return words.at[i].set(value)
    return jax.lax.fori_loop(0, 624, update, mt)

def mersenne_word(state):
    mt, index = state
    mt, index = jax.lax.cond(index >= 624, lambda args: (mersenne_twist(args[0]), jnp.asarray(0, jnp.int32)), lambda args: args, (mt, index))
    word = mt[index]
    word ^= word >> jnp.uint32(11)
    word ^= (word << jnp.uint32(7)) & jnp.uint32(0x9d2c5680)
    word ^= (word << jnp.uint32(15)) & jnp.uint32(0xefc60000)
    word ^= word >> jnp.uint32(18)
    return (mt, index + jnp.asarray(1, jnp.int32)), word

def mersenne_uniform(state):
    state, first = mersenne_word(state)
    state, second = mersenne_word(state)
    bits = ((first.astype(jnp.uint64) >> jnp.uint64(6)) << jnp.uint64(27)) + (second.astype(jnp.uint64) >> jnp.uint64(5))
    return state, bits.astype(jnp.float64) / jnp.asarray(1 << 53, jnp.float64)
'''


# Within each block the displaced word has already been updated in an earlier
# block, or still belongs to the old suffix. The final word reads the NEW word 0.
# This is the same in-place recurrence as mersenne_twist, not a simultaneous twist.
MERSENNE_BLOCK_TWIST = '''def mersenne_twist(mt):
    for lower, upper in ((0,227),(227,454),(454,623)):
        indices=jnp.arange(lower,upper,dtype=jnp.int32)
        y=(mt[indices] & jnp.uint32(0x80000000)) | (mt[indices+1] & jnp.uint32(0x7fffffff))
        values=mt[(indices+397)%624] ^ (y >> jnp.uint32(1)) ^ jnp.where((y & jnp.uint32(1))!=0,jnp.uint32(0x9908b0df),jnp.uint32(0))
        mt=mt.at[lower:upper].set(values)
    y=(mt[623] & jnp.uint32(0x80000000)) | (mt[0] & jnp.uint32(0x7fffffff))
    value=mt[396] ^ (y >> jnp.uint32(1)) ^ jnp.where((y & jnp.uint32(1))!=0,jnp.uint32(0x9908b0df),jnp.uint32(0))
    return mt.at[623].set(value)

'''
MERSENNE_OPTIMIZED_SOURCE = MERSENNE_BLOCK_TWIST + MERSENNE_SOURCE[MERSENNE_SOURCE.index('def mersenne_word'):]


MERSENNE_CACHED_SOURCE = MERSENNE_BLOCK_TWIST + '''def mersenne_cache(mt):
    words=mt ^ (mt >> jnp.uint32(11))
    words=words ^ ((words << jnp.uint32(7)) & jnp.uint32(0x9d2c5680))
    words=words ^ ((words << jnp.uint32(15)) & jnp.uint32(0xefc60000))
    words=words ^ (words >> jnp.uint32(18))
    bits=((words[::2].astype(jnp.uint64) >> jnp.uint64(6)) << jnp.uint64(27)) + (words[1::2].astype(jnp.uint64) >> jnp.uint64(5))
    return bits.astype(jnp.float64) / jnp.asarray(1 << 53,jnp.float64)

def mersenne_refill(args):
    mt,index,cache=args
    mt=mersenne_twist(mt)
    return mt,jnp.asarray(0,jnp.int32),mersenne_cache(mt)

def mersenne_word(state):
    mt,index,cache=jax.lax.cond(state[1]>=624,mersenne_refill,lambda s:s,state)
    word=mt[index]
    word ^= word >> jnp.uint32(11)
    word ^= (word << jnp.uint32(7)) & jnp.uint32(0x9d2c5680)
    word ^= (word << jnp.uint32(15)) & jnp.uint32(0xefc60000)
    word ^= word >> jnp.uint32(18)
    return (mt,index+jnp.asarray(1,jnp.int32),cache),word

def mersenne_uniform(state):
    def even(s):
        mt,index,cache=jax.lax.cond(s[1]>=624,mersenne_refill,lambda a:a,s)
        return (mt,index+jnp.asarray(2,jnp.int32),cache),cache[index//2]
    def odd(s):
        s,a=mersenne_word(s)
        s,b=mersenne_word(s)
        bits=((a.astype(jnp.uint64)>>jnp.uint64(6))<<jnp.uint64(27))+(b.astype(jnp.uint64)>>jnp.uint64(5))
        return s,bits.astype(jnp.float64)/jnp.asarray(1<<53,jnp.float64)
    return jax.lax.cond((state[1] & 1)==0,even,odd,state)

def mersenne_discard_uniforms(state,count):
    mt,index,cache=state
    total=index+jnp.asarray(2,jnp.int32)*count
    twists=jnp.where(count>0,(total-1)//624,0)
    mt=jax.lax.fori_loop(0,twists,lambda _,words:mersenne_twist(words),mt)
    cache=jax.lax.cond(twists>0,mersenne_cache,lambda _:cache,mt)
    return mt,(total-twists*624).astype(jnp.int32),cache
'''

MERSENNE_BATCH_SOURCE = '''
def mersenne_uniforms_bounded(state,requested,size):
    if size==0:return state,jnp.zeros((0,),jnp.float64)
    def even(state):
        result=jnp.zeros((size+312,),jnp.float64)
        def condition(carry):return carry[0]<requested
        def copy_chunk(carry):
            position,rng,result=carry
            rng=jax.lax.cond(rng[1]>=624,mersenne_refill,lambda value:value,rng)
            mt,index,cache=rng
            available=(624-index)//2
            used=jnp.minimum(requested-position,available).astype(jnp.int32)
            result=jax.lax.dynamic_update_slice(result,jnp.roll(cache,-(index//2)),(position,))
            return position+used,(mt,index+2*used,cache),result
        _,rng,result=jax.lax.while_loop(condition,copy_chunk,(jnp.asarray(0,jnp.int32),state,result))
        return rng,result[:size]
    def odd(state):
        def sample(index,carry):
            rng,result=carry;rng,value=mersenne_uniform(rng)
            return rng,result.at[index].set(value)
        return jax.lax.fori_loop(0,requested,sample,(state,jnp.zeros((size,),jnp.float64)))
    return jax.lax.cond((state[1]&1)==0,even,odd,state)

def mersenne_uniforms(state,size):
    return mersenne_uniforms_bounded(state,jnp.asarray(size,jnp.int32),size)

def mersenne_uniform_pair(state):
    def sample(index,carry):
        rng,values=carry
        rng,value=mersenne_uniform(rng)
        return rng,values.at[index].set(value)
    return jax.lax.fori_loop(0,2,sample,(state,jnp.zeros((2,),jnp.float64)))
'''
