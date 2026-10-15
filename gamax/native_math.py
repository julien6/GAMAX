"""Standalone fdlibm arctangent, matching the pinned JVM's StrictMath.

Based on https://netlib.org/fdlibm/s_atan.c and e_atan2.c.
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
 granted, provided that this notice is preserved.
"""
ATAN_SOURCE = '''# fdlibm s_atan.c / e_atan2.c, https://netlib.org/fdlibm/
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunSoft, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.
def native_atan(value):
    r = jax.lax.optimization_barrier
    value = jnp.asarray(value,jnp.float64)
    ax = jnp.abs(value)
    ident = jnp.where(ax<0.4375,-1,jnp.where(ax<0.6875,0,jnp.where(ax<1.1875,1,jnp.where(ax<2.4375,2,3))))
    a = r(r(r(2.*ax)-1.)/r(2.+ax))
    b = r(r(ax-1.)/r(ax+1.))
    c = r(r(ax-1.5)/r(1.+r(1.5*ax)))
    d = r(-1./ax)
    x = jnp.where(ident<0,value,jnp.where(ident==0,a,jnp.where(ident==1,b,jnp.where(ident==2,c,d))))
    coeff = [float.fromhex(h) for h in ('0x1.555555555550dp-2','-0x1.999999998ebc4p-3','0x1.24924920083ffp-3','-0x1.c71c6fe231671p-4','0x1.745cdc54c206ep-4','-0x1.3b0f2af749a6dp-4','0x1.10d66a0d03d51p-4','-0x1.dde2d52defd9ap-5','0x1.97b4b24760debp-5','-0x1.2b4442c6a6c2fp-5','0x1.0ad3ae322da11p-6')]
    hi = jnp.asarray([float.fromhex(h) for h in ('0x1.dac670561bb4fp-2','0x1.921fb54442d18p-1','0x1.f730bd281f69bp-1','0x1.921fb54442d18p0')])
    lo = jnp.asarray([float.fromhex(h) for h in ('0x1.a2b7f222f65e2p-56','0x1.1a62633145c07p-55','0x1.007887af0cbbdp-56','0x1.1a62633145c07p-54')])
    z = r(x*x);w = r(z*z)
    odd = jnp.asarray(coeff[10]);even = jnp.asarray(coeff[9])
    for i in (8,6,4,2,0):odd = r(coeff[i]+r(w*odd))
    for i in (7,5,3,1):even = r(coeff[i]+r(w*even))
    s1 = r(z*odd);s2 = r(w*even)
    small = r(x-r(x*r(s1+s2)))
    index = jnp.maximum(0,ident)
    large = r(hi[index]-r(r(r(x*r(s1+s2))-lo[index])-x))
    result = jnp.where(ident<0,small,jnp.where(jnp.signbit(value),-large,large))
    result = jnp.where(ax<2.**-29,value,result)
    result = jnp.where(ax>=2.**66,jnp.copysign(hi[3]+lo[3],value),result)
    return jnp.where(jnp.isnan(value),value+value,result)

def native_atan2(y,x):
    r = jax.lax.optimization_barrier
    x,y = jnp.asarray(x,jnp.float64),jnp.asarray(y,jnp.float64)
    bits_x = jax.lax.bitcast_convert_type(x,jnp.uint64)
    bits_y = jax.lax.bitcast_convert_type(y,jnp.uint64)
    ix = ((bits_x>>32)&jnp.uint64(0x7fffffff)).astype(jnp.int64)
    iy = ((bits_y>>32)&jnp.uint64(0x7fffffff)).astype(jnp.int64)
    k = (iy-ix)>>20
    pi = jnp.asarray(math.pi,jnp.float64)
    lo = jnp.asarray(float.fromhex('0x1.1a62633145c07p-53'),jnp.float64)
    z = native_atan(jnp.abs(r(y/x)))
    z = jnp.where(k>60,pi/2.+0.5*lo,jnp.where(jnp.signbit(x)&(k<-60),0.,z))
    result = jnp.where(jnp.signbit(x),jnp.where(jnp.signbit(y),r(r(z-lo)-pi),r(pi-r(z-lo))),jnp.where(jnp.signbit(y),-z,z))
    result = jnp.where(jnp.isinf(y),jnp.copysign(pi/2.,y),result)
    infinite_x = jnp.where(jnp.isinf(y),jnp.copysign(jnp.where(jnp.signbit(x),3.*(pi/4.),pi/4.),y),jnp.copysign(jnp.where(jnp.signbit(x),pi,0.),y))
    result = jnp.where(jnp.isinf(x),infinite_x,result)
    result = jnp.where(x==0.,jnp.copysign(pi/2.,y),result)
    result = jnp.where(y==0.,jnp.where(jnp.signbit(x),jnp.copysign(pi,y),y),result)
    result = jnp.where(x==1.,native_atan(y),result)
    return jnp.where(jnp.isnan(x)|jnp.isnan(y),x+y,result)
'''

# Movement headings are normalized to one turn. Keep fdlibm's two-part
# reduction and evaluation order for that range; larger inputs keep the
# existing JAX implementation and require their own differential validation.
TRIG_SOURCE = '''
# Adapted from https://netlib.org/fdlibm/{e_rem_pio2,k_sin,k_cos}.c
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunSoft, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.
def native_heading_sincos(heading):
    r=jax.lax.optimization_barrier
    angle=r(jnp.asarray(heading,jnp.float64)*jnp.asarray(0.017453292519943295,jnp.float64))
    def reduced(angle):
        absolute=jnp.abs(angle)
        high=(jax.lax.bitcast_convert_type(absolute,jnp.uint64)>>jnp.uint64(32)).astype(jnp.int64)
        small=high<=0x3fe921fb
        special=high<0x4002d97c
        p1=float.fromhex('0x1.921fb54400000p0');p1t=float.fromhex('0x1.0b4611a626331p-34')
        p2=float.fromhex('0x1.0b4611a600000p-34');p2t=float.fromhex('0x1.3198a2e037073p-69')
        p3=float.fromhex('0x1.3198a2e000000p-69');p3t=float.fromhex('0x1.b839a252049c1p-104')
        n=jnp.trunc(r(r(absolute*float.fromhex('0x1.45f306dc9c883p-1'))+.5)).astype(jnp.int32)
        fn=n.astype(jnp.float64)
        remainder=r(absolute-r(fn*p1));tail=r(fn*p1t);head=r(remainder-tail)
        exponent=high>>20
        def exponent_of(value):
            return ((jax.lax.bitcast_convert_type(value,jnp.uint64)>>jnp.uint64(52)) & jnp.uint64(0x7ff)).astype(jnp.int64)
        refine=exponent-exponent_of(head)>16
        saved=remainder;correction=r(fn*p2);refined=r(saved-correction)
        corrected_tail=r(r(fn*p2t)-r(r(saved-refined)-correction))
        remainder=jnp.where(refine,refined,remainder);tail=jnp.where(refine,corrected_tail,tail);head=r(remainder-tail)
        refine3=refine & (exponent-exponent_of(head)>49)
        saved=remainder;correction=r(fn*p3);refined=r(saved-correction)
        corrected_tail=r(r(fn*p3t)-r(r(saved-refined)-correction))
        remainder=jnp.where(refine3,refined,remainder);tail=jnp.where(refine3,corrected_tail,tail);head=r(remainder-tail)
        low=r(r(remainder-head)-tail)
        z=r(absolute-p1);near=high==0x3ff921fb
        z=jnp.where(near,r(z-p2),z);special_tail=jnp.where(near,p2t,p1t)
        special_head=r(z-special_tail);special_low=r(r(z-special_head)-special_tail)
        head=jnp.where(special,special_head,head);low=jnp.where(special,special_low,low);n=jnp.where(special,1,n)
        head=jnp.where(small,absolute,head);low=jnp.where(small,0.,low);n=jnp.where(small,0,n)
        x=jnp.where(jnp.signbit(angle),-head,head);y=jnp.where(jnp.signbit(angle),-low,low);n=jnp.where(jnp.signbit(angle),-n,n)
        z=r(x*x);v=r(z*x)
        S=[float.fromhex(h) for h in ('-0x1.5555555555549p-3','0x1.111111110f8a6p-7','-0x1.a01a019c161d5p-13','0x1.71de357b1fe7dp-19','-0x1.ae5e68a2b9cebp-26','0x1.5d93a5acfd57cp-33')]
        polynomial=jnp.asarray(S[5],jnp.float64)
        for coefficient in S[4:0:-1]:polynomial=r(coefficient+r(z*polynomial))
        sine_plain=r(x+r(v*r(S[0]+r(z*polynomial))))
        sine_tail=r(x-r(r(r(z*r(r(.5*y)-r(v*polynomial)))-y)-r(v*S[0])))
        sine=jnp.where(small,sine_plain,sine_tail)
        C=[float.fromhex(h) for h in ('0x1.555555555554cp-5','-0x1.6c16c16c15177p-10','0x1.a01a019cb1590p-16','-0x1.27e4f809c52adp-22','0x1.1ee9ebdb4b1c4p-29','-0x1.8fae9be8838d4p-37')]
        polynomial=jnp.asarray(C[5],jnp.float64)
        for coefficient in C[4::-1]:polynomial=r(coefficient+r(z*polynomial))
        polynomial=r(z*polynomial)
        xhigh=(jax.lax.bitcast_convert_type(jnp.abs(x),jnp.uint64)>>jnp.uint64(32)).astype(jnp.int64)
        qbits=(xhigh-0x00200000).astype(jnp.uint64)<<jnp.uint64(32)
        q=jnp.where(xhigh>0x3fe90000,.28125,jax.lax.bitcast_convert_type(qbits,jnp.float64))
        correction=r(r(z*polynomial)-r(x*y))
        cosine=jnp.where(xhigh<0x3fd33333,r(1.-r(r(.5*z)-correction)),r(r(1.-q)-r(r(r(.5*z)-q)-correction)))
        sine=jnp.where(xhigh<0x3e400000,x,sine);cosine=jnp.where(xhigh<0x3e400000,1.,cosine)
        quadrant=n & 3
        sin_result=jnp.where(quadrant==0,sine,jnp.where(quadrant==1,cosine,jnp.where(quadrant==2,-sine,-cosine)))
        cos_result=jnp.where(quadrant==0,cosine,jnp.where(quadrant==1,-sine,jnp.where(quadrant==2,-cosine,sine)))
        return jnp.stack([sin_result,cos_result])
    return jax.lax.cond(jnp.abs(angle)<=2.*math.pi,reduced,lambda value:jnp.stack([jnp.sin(value),jnp.cos(value)]),angle)
'''
