"""Lexical tokens and operator precedence for the supported GAML profile."""
import re

TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|#[A-Za-z_]\w*|//[^\n]*|/\*.*?\*/|\s+|<-|<<|>>|>=|<=|!=|[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|\d+|[{}\[\]():;,+*/.\-<>=?!%^@]', re.S)
PRECEDENCE = {"where": 0, "collect": 0, "count": 0, "distance_to": 5, "neighbors_at": 5, "or": 1, "and": 2, "=": 3, "!=": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5, "*": 6, "/": 6, "mod": 6, "div": 6}
FUNCTIONS = {"abs", "sqrt", "sin", "cos", "exp", "int", "float", "min", "max", "length", "sum", "mean", "rnd", "flip", "square", "rgb", "one_of"}
FUNCTIONS.update(('shuffle','first','last','copy','circle','image_file','triangle','rectangle','empty','signum'))
COLORS={'yellow':[255,255,0,255],'red':[255,0,0,255],'blue':[0,0,255,255],
        'orange':[255,165,0,255],'green':[0,128,0,255],'pink':[255,192,203,255],
        'magenta':[255,0,255,255],'cyan':[0,255,255,255],'white':[255,255,255,255],
        'black':[0,0,0,255],'gray':[128,128,128,255],'lightgray':[211,211,211,255],
        'lightblue':[173,216,230,255]}
PRECEDENCE['as'] = 7
PRECEDENCE['of'] = 7
PRECEDENCE['accumulate'] = 0
PRECEDENCE['overlapping'] = 5
PRECEDENCE['at'] = 5
PRECEDENCE['at_distance']=5
PRECEDENCE.update(first_with=0,inside=5,overlaps=5)
TOKEN = re.compile(TOKEN.pattern.replace('|<-|','|->|<-|'),re.S)
RESERVED = {"index", "cycle", "empty", "self", "world", "true", "false", "first", "last", "length", "int", "float", "bool",
            "model", "global", "species", "experiment", "init", "reflex", "if", "else", "action", "do", "skills", "point", "rng", "seed",
            "create", "ask", "each", "myself", "and", "or", "not", "number", "when", "type", "gui", "loop", "times", "from", "to"} | FUNCTIONS



PRECEDENCE.update(min_of=0,max_of=0,with_max_of=0,with_min_of=0)

FUNCTIONS.update(('file','round','as_matrix'))

FUNCTIONS.update(('topology',))
PRECEDENCE['distance_between']=5

FUNCTIONS.add('point')
PRECEDENCE['sort_by']=0
