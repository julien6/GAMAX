"""Resolve immutable image resources into checked, portable opaque file handles.

Packed pixel reads and bounded sized matrices use explicit native-compatible
decoding contracts. Handles preserve identity; their declared files are bundled
and verified when the standalone module resets.
"""
from pathlib import Path
import hashlib
import os
from .ir import Expr,CompileError


def resolve_assets(ir):
    records=[];known={}
    def walk(value):
        if isinstance(value,Expr):
            if value.op in ('call:image_file','call:file'):
                if len(value.args)!=1 or value.args[0].op!='literal' or value.args[0].type!='string':raise CompileError(value.token.source,value.token,'image_file currently requires an immutable literal relative path','GAMAX_ASSET')
                path=Path(value.args[0].args[0])
                if path.is_absolute():raise CompileError(value.token.source,value.token,'Portable assets require relative paths','GAMAX_ASSET')
                original=(Path(value.token.source).parent/path).resolve()
                if not original.is_file():raise CompileError(value.token.source,value.token,'Missing image resource: '+str(path),'GAMAX_ASSET')
                if original not in known:
                    known[original]=len(records);records.append(dict(path=str(original),sha256=hashlib.sha256(original.read_bytes()).hexdigest(),kind='image_file'))
                value.op='literal';value.type='image_file';value.args=[known[original]]
            for arg in value.args:walk(arg)
        elif isinstance(value,(list,tuple)):
            for item in value:walk(item)
        elif isinstance(value,dict):
            for item in value.values():walk(item)
        elif hasattr(value,'__dataclass_fields__'):
            for name in value.__dataclass_fields__:walk(getattr(value,name))
    walk(ir.scopes)
    for resource in ir.presentation_resources:
        for relative in resource['paths']:
            original=(Path(resource['source']).parent/relative).resolve()
            if original not in known:
                known[original]=len(records);records.append(dict(path=str(original),sha256=hashlib.sha256(original.read_bytes()).hexdigest(),kind='presentation_resource'))
    if records:
        root=Path(os.path.commonpath([*[str(Path(record['path']).parent) for record in records],*[str(Path(field.token.source).parent) for scope in ir.scopes for field in scope.fields if field.token.source]]))
        for record in records:
            record['key']=str(Path(record['path']).relative_to(root))
            record['asset_bundle_path']='assets/'+record['key']
    def raster_used(value):
        if isinstance(value,Expr):return value.op=='at' or any(raster_used(a) for a in value.args) or raster_used(value.metadata)
        if isinstance(value,(list,tuple)):return any(raster_used(v) for v in value)
        if isinstance(value,dict):return any(raster_used(v) for v in value.values())
        if hasattr(value,'__dataclass_fields__'):return any(raster_used(getattr(value,n)) for n in value.__dataclass_fields__)
        return False
    if records and raster_used(ir.scopes):
        from PIL import Image
        for record in records:
            if record['kind']!='image_file':continue
            with Image.open(record['path']) as image:
                rgba=image.convert('RGBA');record['raster_size']=list(rgba.size)
                pixels=[]
                for r,g,b,a in rgba.getdata():
                    packed=(a<<24)|(r<<16)|(g<<8)|b
                    pixels.append(packed if packed<2**31 else packed-2**32)
                record['raster_pixels']=pixels
    constants={f.name:f.value.args[0] for f in ir.scopes[0].fields if f.value.op=='literal' and f.value.type in ('int','float','image_file')}
    def scalar(expr):
        if expr.op=='literal':return expr.args[0]
        if expr.op=='name' and expr.args[0] in constants:return constants[expr.args[0]]
        if expr.op in ('+','-','*','/'):
            a,b=map(scalar,expr.args);return {'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a/b}[expr.op]()
        raise ValueError('Nonstatic raster dimension/handle')
    def matrices(value):
        if isinstance(value,Expr):
            if value.op=='call:as_matrix':
                from .raster_decode import resized_pixels,java_identity
                try:
                    if len(value.args)!=2 or value.args[1].op!='point':raise ValueError('Sized matrix requires a two-dimensional point')
                    handle=int(scalar(value.args[0]));width,height=[scalar(v) for v in value.args[1].args[:2]]
                    if any(type(n) not in (int,float) or int(n)!=n or not 1<=n<=4096 for n in (width,height)) or width*height>1000000:raise ValueError('Raster matrix dimension outside bounded profile')
                    width,height=int(width),int(height)
                    pixels=resized_pixels(records[handle]['path'],width,height)
                    if 'raster_decoder' not in records[handle]:records[handle]['raster_decoder']=java_identity()
                except Exception as error:raise CompileError(value.token.source,value.token,str(error),'GAMAX_ASSET') from error
                value.op='raster_matrix';value.args=[];value.type='matrix:int';value.metadata={'pixels':pixels,'width':width,'height':height,'handle':handle,'decode':'standard Java ImageIO/AWT TYPE_INT_RGB drawImage; no GAMA dependency at translation or execution'}
            for arg in value.args:matrices(arg)
            matrices(value.metadata)
        elif isinstance(value,(list,tuple)):
            for v in value:matrices(v)
        elif isinstance(value,dict):
            for v in value.values():matrices(v)
        elif hasattr(value,'__dataclass_fields__'):
            for key in value.__dataclass_fields__:matrices(getattr(value,key))
    matrices(ir.scopes)
    ir.assets=records
