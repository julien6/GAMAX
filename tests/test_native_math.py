"""Compare generated fdlibm arithmetic to pinned-JDK bit patterns."""
from pathlib import Path
import math
import numpy as np
import jax
import jax.numpy as jnp
from gamax.native_math import ATAN_SOURCE


def test_arctangent_matches_java_bits_and_tracks_subnormal_limits():
    jax.config.update('jax_enable_x64',True)
    namespace=dict(jax=jax,jnp=jnp,math=math)
    exec(ATAN_SOURCE,namespace)
    root=Path(__file__).resolve().parents[1]/'tests/fixtures/native/native-atan-bit-contract-2026-10-13'
    def bits(name):return np.asarray([[int(v,16) for v in line.split()] for line in (root/name).read_text().splitlines()],dtype=np.uint64)
    inputs=bits('inputs.txt').view(np.float64);expected=bits('java_bits.txt')
    options={'xla_cpu_enable_fast_math':False,'xla_disable_hlo_passes':'fusion,algsimp'}
    actual=np.stack([np.asarray(jax.jit(namespace['native_atan'],compiler_options=options)(inputs[:,0])),np.asarray(jax.jit(namespace['native_atan2'],compiler_options=options)(inputs[:,0],inputs[:,1]))],axis=1).view(np.uint64)
    # The first 10,000 pairs are ordinary finite coordinates. Remaining pairs
    # include signed zero, infinities, NaNs, boundaries and subnormal quotients.
    assert np.array_equal(actual[:10000],expected[:10000])
    valid=(actual==expected)|(np.isnan(actual.view(np.float64))&np.isnan(expected.view(np.float64)))
    mismatches=np.argwhere(~valid)
    assert len(mismatches)==2
    for i,j in mismatches:
        assert j==1 and abs(expected.view(np.float64)[i,j])<np.finfo(np.float64).tiny
    np.testing.assert_allclose(actual.view(np.float64),expected.view(np.float64),atol=1e-12,rtol=1e-10,equal_nan=True)
