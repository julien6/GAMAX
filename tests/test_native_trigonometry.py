"""Bitwise oracle for normalized movement headings, including cancellations."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from gamax.native_math import TRIG_SOURCE


def test_normalized_heading_trigonometry_matches_strictmath_bits(tmp_path):
    root=Path(__file__).resolve().parents[1];java=root/'.tools/gama/jdk/bin/java'
    if not java.exists():pytest.skip('Native evaluation JDK is unavailable')
    namespace={'math':__import__('math'),'jax':jax,'jnp':jnp};exec(TRIG_SOURCE,namespace)
    rng=np.random.default_rng(918)
    boundaries=np.asarray([0.,-0.,45.,90.,135.,180.,225.,270.,315.,360.,-360.,110.78591156875652])
    headings=np.concatenate([boundaries,np.nextafter(boundaries,-np.inf),np.nextafter(boundaries,np.inf),rng.uniform(-360,360,4000)])
    inputs=tmp_path/'inputs.txt';inputs.write_text('\n'.join(float(v).hex() for v in headings)+'\n')
    source=tmp_path/'TrigOracle.java'
    source.write_text('''import java.nio.file.*; class TrigOracle {static double sink; static void emit(double h){double a=h*0.017453292519943295;System.out.println(Long.toUnsignedString(Double.doubleToRawLongBits(StrictMath.sin(a)))+" "+Long.toUnsignedString(Double.doubleToRawLongBits(StrictMath.cos(a))));} public static void main(String[] args) throws Exception {for(int i=0;i<100000;i++)sink=Math.sin(i*.0001)+Math.cos(i*.0001);for(String line:Files.readAllLines(Path.of(args[0])))emit(Double.parseDouble(line));}}''')
    result=subprocess.run([str(java),str(source),str(inputs)],capture_output=True,text=True,check=True)
    expected=np.asarray([[int(v) for v in line.split()] for line in result.stdout.splitlines()],dtype=np.uint64)
    for options in ({'xla_cpu_enable_fast_math':False,'xla_disable_hlo_passes':'fusion,algsimp'},{'xla_cpu_enable_fast_math':False,'xla_disable_hlo_passes':'algsimp','xla_cpu_max_isa':'AVX'}):
        actual=np.asarray(jax.jit(jax.vmap(namespace['native_heading_sincos']),compiler_options=options)(jnp.asarray(headings)))
        np.testing.assert_array_equal(actual.view(np.uint64),expected)


def test_hotspot_heading_trigonometry_matches_compiled_math_bits(tmp_path):
    import math
    from gamax.hotspot_math import HOTSPOT_TRIG_SOURCE
    root=Path(__file__).resolve().parents[1];java=root/'.tools/gama/jdk/bin/java'
    if not java.exists():pytest.skip('Native evaluation JDK is unavailable')
    namespace={'math':math,'jax':jax,'jnp':jnp};exec(HOTSPOT_TRIG_SOURCE,namespace)
    boundaries=np.asarray([0.,-0.,45.,90.,135.,180.,225.,270.,315.,360.,-360.,110.78591156875652])
    headings=np.r_[boundaries,np.nextafter(boundaries,-np.inf),np.nextafter(boundaries,np.inf),np.random.default_rng(918).uniform(-360,360,40000)]
    inputs=tmp_path/'inputs.txt';inputs.write_text('\n'.join(float(v).hex() for v in headings)+'\n')
    source=tmp_path/'HotspotOracle.java'
    source.write_text('''import java.nio.file.*; class HotspotOracle {static double sink; static double[] compute(double h){double a=h*0.017453292519943295;return new double[]{Math.sin(a),Math.cos(a)};} public static void main(String[] args) throws Exception {for(int i=0;i<50000;i++)sink=compute((i*.173)%360)[0];for(String line:Files.readAllLines(Path.of(args[0]))){double[] v=compute(Double.parseDouble(line));System.out.println(Long.toUnsignedString(Double.doubleToRawLongBits(v[0]))+" "+Long.toUnsignedString(Double.doubleToRawLongBits(v[1])));}}}''')
    result=subprocess.run([str(java),str(source),str(inputs)],capture_output=True,text=True,check=True)
    expected=np.asarray([[int(v) for v in line.split()] for line in result.stdout.splitlines()],dtype=np.uint64)
    for options in ({'xla_cpu_enable_fast_math':False,'xla_disable_hlo_passes':'fusion,algsimp'},{'xla_cpu_enable_fast_math':False,'xla_disable_hlo_passes':'algsimp','xla_cpu_max_isa':'AVX'}):
        actual=np.asarray(jax.jit(jax.vmap(namespace['native_hotspot_heading_sincos']),compiler_options=options)(jnp.asarray(headings)))
        np.testing.assert_array_equal(actual.view(np.uint64),expected)
