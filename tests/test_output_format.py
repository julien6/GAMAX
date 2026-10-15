"""Check Java's float-to-string convention against the installed oracle JVM."""
from pathlib import Path
import random
import struct
import subprocess
import pytest
from gamax.output_format import OUTPUT_FORMAT_SOURCE


def test_native_scalar_format_matches_java_double_to_string(tmp_path):
    java=Path(__file__).resolve().parents[1]/'.tools/gama/jdk/bin/java'
    javac=java.with_name('javac')
    if not javac.exists():pytest.skip('Oracle JDK not installed')
    source=tmp_path/'ScalarText.java'
    source.write_text('import java.nio.file.*; public class ScalarText { public static void main(String[] a) throws Exception { for(String s:Files.readAllLines(Path.of(a[0]))) System.out.println(Double.toString(Double.longBitsToDouble(Long.parseUnsignedLong(s,16)))); }}')
    subprocess.run([str(javac),str(source)],check=True,capture_output=True)
    rng=random.Random(4)
    values=[0.,-0.,float('nan'),float('inf'),float('-inf'),2.8753457722419107e-4]
    values += [struct.unpack('>d',i.to_bytes(8,'big'))[0] for i in range(1,1001)]
    values += [10.**i for i in range(-323,309)]
    values += [struct.unpack('>d',rng.getrandbits(64).to_bytes(8,'big'))[0] for _ in range(10000)]
    bits=tmp_path/'bits.txt';bits.write_text('\n'.join(struct.pack('>d',v).hex() for v in values))
    expected=subprocess.run([str(java),'-cp',str(tmp_path),'ScalarText',str(bits)],check=True,capture_output=True,text=True).stdout.splitlines()
    namespace={};exec(OUTPUT_FORMAT_SOURCE,namespace)
    assert [namespace['native_scalar_text'](v) for v in values]==expected
    assert namespace['native_scalar_text'](True)=='true'
    assert namespace['native_scalar_text'](2)=='2'
