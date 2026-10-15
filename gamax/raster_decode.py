"""Compile-time raster resizing through standard Java ImageIO/AWT.

GAMA's sized image matrix uses BufferedImage TYPE_INT_RGB and drawImage with
its default interpolation. The emitted module stores immutable pixels and needs
only Python/JAX; this translator helper does not load GAMA classes.
"""
from pathlib import Path
import os
import hashlib
import shutil
import subprocess
import tempfile

JAVA_SOURCE = r'''import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import javax.imageio.ImageIO;
import java.io.File;
public class RasterDecode {
 public static void main(String[] args) throws Exception {
  BufferedImage input=ImageIO.read(new File(args[0]));
  if(input==null) throw new IllegalArgumentException("Unsupported image");
  int width=Integer.parseInt(args[1]),height=Integer.parseInt(args[2]);
  BufferedImage image=new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
  Graphics2D graphics=image.createGraphics();
  graphics.drawImage(input,0,0,width,height,null); graphics.dispose();
  for(int y=0;y<height;y++) for(int x=0;x<width;x++) System.out.println(image.getRGB(x,y));
 }
}'''


def resized_pixels(path,width,height):
    java=os.environ.get('GAMAX_JAVA') or shutil.which('java')
    if not java:raise ValueError('Sized raster compilation requires Java ImageIO (set GAMAX_JAVA)')
    with tempfile.TemporaryDirectory(prefix='gamax-raster-') as directory:
        source=Path(directory)/'RasterDecode.java';source.write_text(JAVA_SOURCE)
        result=subprocess.run([java,'-Djava.awt.headless=true',str(source),str(path),str(width),str(height)],capture_output=True,text=True,timeout=60)
    if result.returncode:raise ValueError('Raster decode failed: '+result.stderr[-2000:])
    pixels=[int(line) for line in result.stdout.splitlines()]
    if len(pixels)!=width*height:raise ValueError('Raster size mismatch')
    return pixels


def java_identity():
    java=os.environ.get('GAMAX_JAVA') or shutil.which('java')
    if not java:raise ValueError('Sized raster compilation requires Java ImageIO')
    executable=Path(java).resolve()
    result=subprocess.run([str(executable),'-version'],capture_output=True,text=True,timeout=10,check=True)
    return dict(executable=str(executable),sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),version=(result.stderr or result.stdout).strip(),contract='standard ImageIO/AWT TYPE_INT_RGB drawImage')
