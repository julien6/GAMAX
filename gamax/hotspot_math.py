"""Experimental reconstruction of one-turn HotSpot movement arithmetic.

Mathematical lookup data are generated independently at extended precision.
The range-reduction/compensated-sum scheme is documented by Intel/OpenJDK:
https://github.com/openjdk/jdk21u/blob/master/src/hotspot/cpu/x86/stubGenerator_x86_64_sin.cpp
This prototype requires JVM primitive and full native trajectory validation.
"""
from .movement_trig_table import TABLE_HEX
HOTSPOT_TRIG_SOURCE = 'MOVEMENT_TRIG_TABLE=jnp.asarray('+repr([[float.fromhex(v) for v in row] for row in TABLE_HEX])+',jnp.float64)\n'+'''
def native_hotspot_heading_sincos(heading):
    barrier=jax.lax.optimization_barrier
    angle=barrier(jnp.asarray(heading,jnp.float64)*jnp.asarray(0.017453292519943295,jnp.float64))
    return jax.lax.cond(jnp.abs(angle)<=2.*math.pi,_native_hotspot_one_turn_sincos,lambda value:jnp.stack([jnp.sin(value),jnp.cos(value)]),angle)

def _native_hotspot_one_turn_sincos(angle):
    barrier=jax.lax.optimization_barrier
    nearest=jnp.trunc(barrier(barrier(angle*float.fromhex('0x1.45f306dc9c883p3'))+jnp.copysign(.5,angle))).astype(jnp.int32)
    multiple=nearest.astype(jnp.float64)
    first=barrier(angle-barrier(multiple*float.fromhex('0x1.921fb54400000p-4')))
    second=barrier(multiple*float.fromhex('0x1.0b4611a600000p-38'))
    remainder=barrier(first-second)
    negative_tail=barrier(barrier(multiple*float.fromhex('0x1.3198a2e037073p-73'))-barrier(barrier(first-remainder)-second))
    square=barrier(remainder*remainder);fourth=barrier(square*square)
    sin_poly=barrier(barrier(-1./6.+barrier((1./120.)*square))+barrier(fourth*barrier(-1./5040.+barrier(barrier((1./362880.)*first)*remainder))))
    cos_poly=barrier(barrier(-.5+barrier((1./24.)*square))+barrier(fourth*barrier(-1./720.+barrier(barrier((1./40320.)*first)*remainder))))
    def evaluate(phase,cosine):
        correction,high,low,power=MOVEMENT_TRIG_TABLE[(nearest+phase)&63]
        complete_cosine=barrier(correction+power)
        high_part=barrier(high+barrier(power*remainder))
        rounding_high=barrier(barrier(high-high_part)+barrier(power*remainder))
        middle=barrier(correction*remainder)
        total=barrier(high_part+middle)
        rounding_middle=barrier(barrier(high_part-total)+middle)
        tail=barrier(barrier(negative_tail*barrier(barrier(high*remainder)-complete_cosine))+low)
        sine_term=barrier(sin_poly*barrier(barrier(complete_cosine*remainder)*square))
        cosine_term=barrier(cos_poly*barrier(high*square))
        residual=barrier(rounding_high+tail) if cosine else barrier(tail+rounding_high)
        residual=barrier(residual+rounding_middle)
        residual=barrier(residual+sine_term)
        residual=barrier(residual+cosine_term)
        return barrier(residual+total) if cosine else barrier(total+residual)
    sine=evaluate(0,False);cosine=evaluate(16,True)
    sine=jnp.where(jnp.abs(angle)<2.**-252,angle,sine)
    cosine=jnp.where(jnp.abs(angle)<2.**-252,1.-jnp.abs(angle),cosine)
    return jnp.stack([sine,cosine])
'''
