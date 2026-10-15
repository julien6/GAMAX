"""Standalone Java-style scalar serialization for emitted GAML outputs."""

OUTPUT_FORMAT_SOURCE = '''
def native_scalar_text(value):
    if type(value) is bool:return str(value).lower()
    if type(value) is not float:return str(value)
    if value!=value:return 'NaN'
    if value==float('inf'):return 'Infinity'
    if value==float('-inf'):return '-Infinity'
    if value==0.:return '-0.0' if str(value).startswith('-') else '0.0'
    from decimal import Decimal
    decimal=Decimal(repr(value))
    # Java considers two significant digits when the shortest representation
    # has only one; choose the closest candidate (notably subnormal doubles).
    if len(decimal.normalize().as_tuple().digits)==1:
        decimal=Decimal(format(value,'.1e'))
    if 0.001<=abs(value)<10000000.:
        text=format(decimal.normalize(),'f')
        return text if '.' in text else text+'.0'
    coefficient,exponent=format(decimal.normalize(),'E').split('E')
    if '.' not in coefficient:coefficient+='.0'
    return coefficient+'E'+str(int(exponent))
'''
