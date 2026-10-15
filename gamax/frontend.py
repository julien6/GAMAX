"""Fail-closed GAML frontend and action resolution."""
from __future__ import annotations
from pathlib import Path
import copy
import ast
import json
import math
import re
import struct
from .ir import Token, CompileError, Expr, Field, Statement, Behavior, Action, Scope, ModelIR
from .lexing import TOKEN, PRECEDENCE, FUNCTIONS, RESERVED,COLORS
from .analysis import CheckerMixin
from .inheritance import resolve_species,resolve_field_dependencies

class Parser(CheckerMixin):
    def __init__(self, source: str, path: str):
        self.path = path
        self.tokens = []
        pos = 0
        for m in TOKEN.finditer(source):
            if m.start() != pos:
                self.fail(Token(source[pos], source.count("\n", 0, pos)+1,
                                pos-source.rfind("\n", 0, pos)), "Unrecognized syntax")
            value = m.group()
            if not value.isspace() and not value.startswith(("//", "/*")):
                self.tokens.append(Token(value, source.count("\n", 0, m.start())+1,
                                         m.start()-source.rfind("\n", 0, m.start()), path))
            pos = m.end()
        if pos != len(source):
            self.fail(Token(source[pos], source.count("\n", 0, pos)+1,
                            pos-source.rfind("\n", 0, pos)), "Unrecognized syntax")
        self.tokens.append(Token("<EOF>", source.count("\n")+1, 1))
        self.declared_species = {self.tokens[i+1].text for i,t in enumerate(self.tokens[:-1]) if t.text in ("species","grid") and re.fullmatch(r'[A-Za-z_]\w*',self.tokens[i+1].text)}
        self.i = 0
        self.raw_scopes = []
        self.populations = {}
        self.scope = None
        self.symbols = {}
        self.collection_scope = None
        self.imports = []

    @property
    def token(self): return self.tokens[self.i]

    def fail(self, token, reason, code="GAMAX_UNSUPPORTED"):
        raise CompileError(self.path, token, reason, code)

    def take(self, expected=None):
        t = self.token
        if expected is not None and t.text != expected:
            self.fail(t, f"Expected {expected!r}; found {t.text!r}")
        self.i += 1
        return t

    def name(self):
        t = self.take()
        if not re.fullmatch(r"[A-Za-z_]\w*", t.text) or t.text in RESERVED:
            self.fail(t, "Expected an identifier without a reserved built-in name")
        return t

    def raw_block(self):
        self.take('{');depth=1;tokens=[]
        while depth:
            token=self.take()
            if token.text=='<EOF>':self.fail(token,'Unterminated block','GAMAX_OUTPUT')
            if token.text=='{':depth+=1
            elif token.text=='}':depth-=1
            if depth:tokens.append(token.text)
        return {'tokens':tokens,'source':self.path}

    def presentation_block(self):
        """Retain GUI declarations and explicitly classify render-only effects."""
        block=self.raw_block();tokens=block['tokens'];render_random=[];events=[]
        index=0
        while index<len(tokens):
            token=tokens[index]
            if token=='event':
                if index+2>=len(tokens) or tokens[index+1] not in ('#mouse_move','#mouse_down','#mouse_up','#mouse_click') or tokens[index+2]!='{':self.fail(self.token,'Unsupported GUI event classification','GAMAX_OUTPUT')
                event_name=tokens[index+1];depth=1;index+=3;body=[]
                while index<len(tokens) and depth:
                    if tokens[index]=='{':depth+=1
                    elif tokens[index]=='}':depth-=1
                    if depth:body.append(tokens[index])
                    index+=1
                events.append({'input':event_name,'tokens':body,'execution':'GUI callback excluded from headless simulation'})
                continue
            if token in ('<-','do','create','ask','die','write','action','reflex','init'):self.fail(self.token,'Effectful non-event presentation declarations require an output backend','GAMAX_OUTPUT')
            if token in ('rnd','flip','one_of'):render_random.append(token)
            index+=1
        return {**block,'render_random':render_random,'events':events,'execution':'declarative GUI retained; render RNG and input callbacks excluded from headless simulation; no native GUI equivalence'}

    def expression(self, minimum=0):
        t = self.take()
        if t.text in ("-", "not", "!"):
            lhs = Expr(("not" if t.text == "!" else t.text) + "_unary", [self.expression(7)], "", t)
        elif t.text == "{":
            args = [self.expression()]
            self.take(","); args.append(self.expression())
            if self.token.text == ",": self.take(); args.append(self.expression())
            self.take("}")
            if len(args) == 2: args.append(Expr("literal", [0.0], "float", t))
            lhs = Expr("point", args, "point", t)
        elif t.text=='[':
            args=[]
            if self.token.text!=']':
                args.append(self.expression())
                while self.token.text==',':self.take();args.append(self.expression())
            self.take(']');lhs=Expr('list_literal',args,'',t)
        elif t.text == "(":
            lhs = self.expression()
            self.take(")")
        elif t.text.startswith('#') and t.text[1:] in COLORS:
            lhs = Expr("literal",[COLORS[t.text[1:]]],"rgb",t)
        elif t.text=='#user_location':lhs=Expr('gui_input',[t.text],'point',t)
        elif t.text.startswith(('"',"'")):
            lhs = Expr("literal", [ast.literal_eval(t.text)], "string", t)
        elif t.text in ("true", "false"):
            lhs = Expr("literal", [t.text == "true"], "bool", t)
        elif t.text == 'nil':
            lhs = Expr('literal',[-1],'nil',t)
        elif re.fullmatch(r"\d+", t.text):
            if int(t.text) > 2147483647:
                self.fail(t, "Integer literal outside signed int32 profile")
            lhs = Expr("literal", [int(t.text)], "int", t)
        elif re.fullmatch(r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+", t.text):
            value = float(t.text)
            if not math.isfinite(value): self.fail(t, 'Non-finite floating literal outside numeric profile', 'GAMAX_TYPE')
            lhs = Expr("literal", [value], "float", t)
        elif t.text=='matrix' and self.token.text=='<':
            self.take('<');element=self.take();self.take('>');self.take('(');value=self.expression();self.take(')')
            if element.text not in ('float','int'):self.fail(element,'Unsupported matrix cast element','GAMAX_TYPE')
            lhs=Expr('matrix_cast:'+element.text,[value],'',t)
        elif re.fullmatch(r"[A-Za-z_]\w*",t.text) and self.token.text == "(" and t.text != "my":
            self.take("(")
            args = [] if self.token.text == ")" else [self.expression()]
            while self.token.text == ",":
                self.take(); args.append(self.expression())
            self.take(")")
            lhs = Expr(("grid_at:" if t.text in self.declared_species else "call:" if t.text in FUNCTIONS else "invoke:") + t.text, args, "", t)
            if t.text=='rgb' and len(args)==1 and args[0].op=='literal' and args[0].type=='string':
                if args[0].args[0] not in COLORS:self.fail(t,'Unknown/unverified named color','GAMAX_TYPE')
                lhs=Expr('literal',[COLORS[args[0].args[0]]],'rgb',t)
        elif t.text=='my':
            self.take('(');value=self.expression();self.take(')')
            if value.op!='name':self.fail(t,'my currently requires a named caller attribute','GAMAX_NAME')
            lhs=Expr('my',value.args,'',t)
        elif t.text in ("each", "myself"):
            if t.text == 'each' and self.token.text != '.':
                lhs = Expr('each_ref', [], '', t)
            else:
                self.take(".")
                attribute = self.take()
                if not re.fullmatch(r"[A-Za-z_]\w*", attribute.text): self.fail(attribute, "Expected attribute")
                lhs = Expr(t.text, [attribute.text], "", t)
        elif re.fullmatch(r"[A-Za-z_]\w*", t.text):
            lhs = Expr("name", [t.text], "", t)
        else:
            self.fail(t, "Expected a supported expression")
        while self.token.text in ('.','['):
            if self.token.text=='[':
                token=self.take();x=self.expression();self.take(',');y=self.expression();self.take(']')
                lhs=Expr('matrix_at',[lhs,x,y],'',token);continue
            self.take(); component = self.take()
            if not re.fullmatch(r'[A-Za-z_]\w*',component.text):self.fail(component,"Expected an attribute")
            lhs = Expr(("component:" if component.text in ("x","y","z") else "attribute:")+component.text,[lhs],"",component)
        while PRECEDENCE.get(self.token.text, -1) >= minimum:
            if self.token.text=='of' and self.tokens[self.i+1].text==':':break
            op = self.take()
            if op.text=='of':
                if lhs.op!='name':self.fail(op,'of requires a named agent attribute','GAMAX_TYPE')
                target=self.expression(PRECEDENCE['of']+1)
                lhs=Expr('attribute:'+lhs.args[0],[target],'',op)
                continue
            if op.text == 'as':
                cast=self.take()
                if cast.text=='list':lhs=Expr('as_list',[lhs],'',cast)
                elif cast.text in ('int','float'):lhs=Expr('call:'+cast.text,[lhs],'',cast)
                else:self.fail(cast,'Unsupported explicit cast','GAMAX_TYPE')
                continue
            rhs = self.expression(PRECEDENCE[op.text]+1)
            if op.text=='distance_between':
                if lhs.op!='call:topology' or len(lhs.args)!=1 or lhs.args[0].op!='name' or rhs.op!='list_literal' or len(rhs.args)!=2:self.fail(op,'Explicit grid distance requires topology(species) and two locations','GAMAX_SPATIAL')
                lhs=Expr('grid_distance',rhs.args,'',op,{'grid':lhs.args[0].args[0]})
            elif op.text in ('min_of','max_of'):
                lhs=Expr('call:'+op.text.split('_')[0],[Expr('collect',[lhs,rhs],'',op)],'',op)
            else:lhs = Expr(op.text, [lhs, rhs], "", op)
        if minimum == 0 and self.token.text == "?":
            question = self.take(); yes = self.expression(); self.take(":"); no = self.expression()
            lhs = Expr("conditional", [lhs, yes, no], "", question)
        return lhs

    def block(self, initialization=False):
        self.take("{")
        result = []
        while self.token.text != "}":
            t = self.take()
            if t.text == "return":
                value=self.expression();self.take(";")
                result.append(Statement("return",None,value,[],[],t))
            elif t.text == "if":
                result.append(self.conditional(t, initialization))
            elif t.text == "loop":
                name = None if self.token.text == "times" else self.name()
                facet = self.take()
                self.take(":")
                def constant(expr):
                    if expr.op == "literal" and expr.type == "int": return expr.args[0]
                    if expr.op == "-_unary" and expr.args[0].op == "literal" and expr.args[0].type == "int": return -expr.args[0].args[0]
                    return None
                if facet.text == 'over':
                    if name is None: self.fail(facet, 'Collection loops require an iteration variable', 'GAMAX_LOOP')
                    collection = self.expression()
                    result.append(Statement('collection_loop', name.text, collection, self.block(initialization), [], t))
                    continue
                if facet.text == "times":
                    if name is not None:
                        self.fail(name, "GAMA times loops do not declare an iteration variable", "GAMAX_LOOP")
                    count = self.expression()
                    bounds = [count]
                    literals = [constant(count)]
                    values = range(literals[0]) if literals[0] is not None else None
                    if literals[0] is not None and literals[0] < 0:
                        self.fail(facet, "Loop times must be nonnegative", "GAMAX_LOOP")
                elif facet.text == "from":
                    if name is None: self.fail(facet, "Range loops require an iteration variable", "GAMAX_LOOP")
                    start = self.expression(); self.take("to"); self.take(":"); stop = self.expression()
                    step = Expr("literal", [1], "int", t)
                    if self.token.text == "step":
                        self.take(); self.take(":"); step = self.expression()
                    bounds = [start, stop, step]
                    literals = [constant(value) for value in bounds]
                    if literals[2] == 0: self.fail(facet, "Loop step cannot be zero", "GAMAX_LOOP")
                    values = range(literals[0], literals[1] + (1 if literals[2] > 0 else -1), literals[2]) if all(v is not None for v in literals) else None
                else:
                    self.fail(facet, "Supported loops: times or from/to/step", "GAMAX_LOOP")
                if values is not None and len(values) > 1024:
                    self.fail(facet, "Static loops are limited to 1024 iterations", "GAMAX_LOOP")
                body = self.block(initialization)
                if values is not None:
                    result.append(Statement("static_loop", name.text if name else None,
                        Expr("iterations", list(values), "", t), body, [], t))
                else:
                    result.append(Statement("dynamic_loop", name.text if name else None,
                        Expr("loop_bounds", bounds, facet.text, t), body, [], t))
            elif t.text in ("int", "float", "bool", "point","list") or t.text in self.declared_species and self.token.text!='(':
                kind='agent:'+t.text if t.text in self.declared_species else t.text
                if t.text=='list':
                    self.take('<');element=self.take();self.take('>')
                    if element.text not in self.declared_species:self.fail(element,'Local lists require a declared agent element type','GAMAX_TYPE')
                    kind='agents:'+element.text
                name = self.take() if self.token.text in ("min","max","range") else self.name()
                self.take("<-"); value = self.expression(); self.take(";")
                result.append(Statement("local:" + kind, name.text, value, [], [], name))
            elif t.text == "do":
                name = self.name()
                arguments = []
                named = []
                if self.token.text == "(":
                    self.take()
                    if self.token.text != ")":
                        while True:
                            if self.tokens[self.i+1].text==':':
                                parameter=self.name();self.take(':');named.append(parameter.text)
                            elif named:self.fail(self.token,'Cannot mix positional and named arguments','GAMAX_TYPE')
                            arguments.append(self.expression())
                            if self.token.text!=',':break
                            self.take()
                    self.take(")")
                else:
                    while self.token.text != ";":
                        parameter = self.name(); self.take(":")
                        if parameter.text in named:
                            self.fail(parameter, "Duplicate named action argument", "GAMAX_TYPE")
                        named.append(parameter.text)
                        arguments.append(self.expression())
                self.take(";")
                result.append(Statement("do", name.text, Expr("arguments", arguments, ",".join(named), name), [], [], t))
            elif t.text=='transition':
                self.take('to');self.take(':');target=self.name();self.take('when');self.take(':');guard=self.expression();self.take(';')
                result.append(Statement('fsm_transition',target.text,guard,[],[],t))
            elif t.text=='diffuse':
                facets={}
                while self.token.text!=';':
                    facet=self.take();self.take(':')
                    if facet.text in facets:self.fail(facet,'Duplicate diffusion facet','GAMAX_EFFECT')
                    if facet.text in ('var','on','propagation','method'):facets[facet.text]=self.name().text
                    elif facet.text in ('proportion','radius'):facets[facet.text]=self.expression()
                    else:self.fail(facet,'Unsupported diffusion facet','GAMAX_EFFECT')
                self.take(';');result.append(Statement('diffuse',None,None,[],[],t,facets))
            elif t.text == "save":
                value=self.expression();self.take('to');self.take(':');destination=self.expression()
                rewrite=Expr('literal',[False],'bool',t)
                if self.token.text=='rewrite':self.take();self.take(':');rewrite=self.expression()
                self.take(';')
                result.append(Statement('save',None,value,[],[],t,{'destination':destination,'rewrite':rewrite}))
            elif t.text == "write":
                value = self.expression(); self.take(";")
                result.append(Statement("write",None,value,[],[],t))
            elif t.text == "create":
                dynamic_self=self.token.text=='species'
                if self.token.text=='species':
                    self.take();self.take('(');self.take('self');self.take(')')
                    if self.scope=='global':self.fail(t,'species(self) creation requires an agent caller','GAMAX_EFFECT')
                    species=self.scope
                else:species = self.name().text
                count = Expr("literal", [1], "int", t)
                if self.token.text == "number":
                    self.take(); self.take(":"); count = self.expression()
                initial_attributes={}
                if self.token.text=='with':
                    self.take();self.take(':');self.take('(')
                    while self.token.text!=')':
                        attribute=self.name();self.take(':')
                        if attribute.text in initial_attributes:self.fail(attribute,'Duplicate creation attribute','GAMAX_NAME')
                        initial_attributes[attribute.text]=self.expression()
                        if self.token.text!=',':break
                        self.take()
                    self.take(')')
                caller_scope=self.scope;self.scope=species
                body = self.block(True) if self.token.text == "{" else []
                self.scope=caller_scope
                if not body and self.token.text == ";": self.take()
                elif self.token.text == ";": self.take()
                elif not body and self.tokens[self.i-1].text != "}": self.fail(self.token, "Expected creation body or semicolon")
                self.populations.setdefault(species,0)
                result.append(Statement("create" if initialization and caller_scope=='global' else 'birth', species, count, body, [], t,{'dynamic_self_species':dynamic_self,**({'initial_attributes':initial_attributes} if initial_attributes else {})}))
            elif t.text == "ask":
                selection=self.expression()
                old_scope=self.scope
                hint=selection
                while hint.op in ('where','as_list','at_distance','overlapping'):hint=hint.args[0]
                species=hint.args[0] if hint.op=='name' and hint.args[0] in self.declared_species else old_scope
                self.scope=species
                body=self.block(initialization)
                self.scope=old_scope
                result.append(Statement('ask',species,selection,body,[],t))
            else:
                if not re.fullmatch(r"[A-Za-z_]\w*", t.text):
                    self.fail(t, "Only assignment and if statements are supported")
                self.i-=1;left=self.expression(8)
                operator=self.take()
                if operator.text not in ('<-','<<','>>'):self.fail(operator,'Expected assignment or ordered-list insertion/removal')
                value=self.expression();self.take(';')
                if left.op.startswith('attribute:') and operator.text=='<-':
                    result.append(Statement('attribute_assign',left.op.split(':')[1],value,[],[],t,{'reference':left.args[0]}))
                elif left.op=='matrix_at' and operator.text=='<-' and left.args[0].op=='name':
                    result.append(Statement('matrix_assign',left.args[0].args[0],value,[],[],t,{'indices':left.args[1:]}))
                elif left.op=='name':
                    if operator.text in ('<<','>>'):value=Expr('append_agent' if operator.text=='<<' else 'remove_agent',[left,value],'',operator)
                    result.append(Statement('assign',left.args[0],value,[],[],t))
                else:self.fail(t,'Unsupported assignment target','GAMAX_EFFECT')
        self.take("}")
        return result

    def conditional(self, token, initialization):
        condition = self.expression()
        body = self.block(initialization)
        alternative = []
        if self.token.text == "else":
            self.take()
            if self.token.text == "if":
                alternative = [self.conditional(self.take(), initialization)]
            else:
                alternative = self.block(initialization)
        return Statement("if", None, condition, body, alternative, token)

    def parse_scope(self, name):
        self.scope = name
        self.take("{")
        fields, initial, behaviors, actions = [], [], [], []
        aspects={};fsm_states=[]
        init_seen = False
        while self.token.text != "}":
            t = self.take()
            if t.text in ('obj_file','font') or t.text=='geometry' and any(token.text=='svg_file' for token in self.tokens[self.i:next((j for j in range(self.i,len(self.tokens)) if self.tokens[j].text in (';','{')),self.i)]):
                resource_name=self.name();tokens=[]
                while self.token.text!=';':
                    token=self.take()
                    if token.text=='<EOF>':self.fail(token,'Unterminated resource declaration','GAMAX_OUTPUT')
                    tokens.append(token.text)
                self.take(';')
                fields.append(Field(resource_name.text,'presentation_resource',Expr('presentation_resource',[],t.text,t,{'tokens':tokens}),resource_name))
                continue
            elif t.text in ("int", "bool", "float", "point", "geometry", "rgb", "list","image_file","file","matrix") or t.text in self.declared_species:
                declared_type = 'agent:'+t.text if t.text in self.declared_species else 'image_file' if t.text=='file' else t.text
                if t.text=='matrix':
                    self.take('<');element=self.take();self.take('>')
                    if element.text not in ('int','float'):self.fail(element,'Unsupported matrix element','GAMAX_TYPE')
                    declared_type='matrix:'+element.text
                if t.text == 'list':
                    declared_type='collection'
                    if self.token.text=='<':
                        self.take('<'); element = self.take(); self.take('>')
                        if element.text in self.declared_species or element.text=='agent':declared_type='agents:'+element.text
                        elif element.text in ('int','float','bool','point','rgb','image_file'):declared_type='values:'+element.text
                        else:self.fail(element,'Unsupported collection element type','GAMAX_TYPE')
                n = self.name()
                if self.token.text in ("{","("):
                    parameters=[]
                    if self.token.text=="(":
                        self.take()
                        while self.token.text!=")":
                            kind=self.take();parameter=self.name()
                            if kind.text not in ("int","float","bool","point") and kind.text not in self.declared_species:self.fail(kind,"Unsupported function parameter type","GAMAX_TYPE")
                            parameters.append((parameter.text,"agent:"+kind.text if kind.text in self.declared_species else kind.text))
                            if self.token.text!=",":break
                            self.take()
                        self.take(")")
                    if any(a.name==n.text for a in actions):self.fail(n,"Duplicate action","GAMAX_NAME")
                    actions.append(Action(n.text,parameters,self.block(),n,return_type=declared_type))
                    continue
                if any(f.name == n.text for f in fields):
                    self.fail(n, "Duplicate attribute", "GAMAX_NAME")
                value = None
                getter=None
                facets = {}
                while self.token.text != ";":
                    facet = self.take()
                    if facet.text=='->':
                        if getter is not None or value is not None:self.fail(facet,'Duplicate/composed computed initializer','GAMAX_FACET')
                        if self.token.text=='{':
                            self.take('{');getter=self.expression();self.take('}')
                        else:getter=self.expression()
                    elif facet.text == "<-":
                        if getter is not None:self.fail(facet,'Computed attribute cannot also have an initializer','GAMAX_FACET')
                        if value is not None: self.fail(facet, "Duplicate attribute initializer", "GAMAX_FACET")
                        value = self.expression()
                    else:
                        if facet.text=='of' and t.text=='list':
                            self.take(':');element=self.take()
                            if element.text not in ('int','float','bool','point','rgb'):self.fail(element,'Unsupported value-list element type','GAMAX_TYPE')
                            if declared_type!='collection':self.fail(facet,'Duplicate collection element declaration','GAMAX_TYPE')
                            declared_type='values:'+element.text
                            continue
                        if facet.text not in ("min", "max", "step","update","const") or facet.text not in ('update','const') and t.text not in ("int", "float"):
                            self.fail(facet, "Unsupported attribute declaration facet", "GAMAX_FACET")
                        if facet.text in facets: self.fail(facet, "Duplicate attribute declaration facet", "GAMAX_FACET")
                        self.take(":"); facets[facet.text] = self.expression()
                if value is None:
                    if declared_type.startswith('agents:'):
                        value=Expr('empty_agents',[],declared_type,n)
                    elif declared_type.startswith('agent:'):
                        value=Expr("literal",[-1],declared_type,n)
                    elif declared_type.startswith('values:'):
                        value=Expr('list_literal',[],declared_type,n)
                    elif t.text in ("int","float","bool"):
                        value = Expr("literal", [{"int": 0, "float": 0.0, "bool": False}[t.text]], t.text, n)
                    elif t.text=='image_file':value=Expr('literal',[-1],'image_file',n)
                    elif t.text=='rgb':value=Expr('literal',[[-1,-1,-1,-1]],'rgb',n)
                    elif getter is not None and t.text in ('point','rgb'):
                        value=Expr('literal',[[0.0,0.0,0.0] if t.text=='point' else [0,0,0,0]],t.text,n)
                    else:
                        self.fail(n, "Uninitialized non-scalar value remains unsupported", "GAMAX_TYPE")
                self.take(";")
                fields.append(Field(n.text, declared_type, value, n, facets,getter=getter))
            elif t.text == "init":
                if init_seen: self.fail(t, "Only one init block is supported")
                init_seen = True
                initial = self.block(True)
            elif t.text == "action":
                action_name = self.name(); parameters = []
                if self.token.text == "(":
                    self.take()
                    while self.token.text != ")":
                        type_ = self.take()
                        if type_.text not in ("int", "float", "bool", "point") and type_.text not in self.declared_species:
                            self.fail(type_, "Action parameters require scalar/point or declared agent types")
                        parameter = self.name()
                        if any(n == parameter.text for n, _ in parameters): self.fail(parameter, "Duplicate parameter")
                        parameters.append((parameter.text, 'agent:'+type_.text if type_.text in self.declared_species else type_.text))
                        if self.token.text != ",": break
                        self.take()
                    self.take(")")
                if any(a.name == action_name.text for a in actions): self.fail(action_name, "Duplicate action")
                virtual=False
                if self.token.text=='virtual':
                    self.take();self.take(':');self.take('true');virtual=True
                    self.take(';');body=[]
                else:body=self.block()
                actions.append(Action(action_name.text, parameters, body, action_name,virtual))
            elif t.text == 'aspect':
                aspect=self.take()
                if not re.fullmatch(r'[A-Za-z_]\w*',aspect.text):self.fail(aspect,'Expected aspect name','GAMAX_OUTPUT')
                if aspect.text in aspects:self.fail(aspect,'Duplicate aspect','GAMAX_OUTPUT')
                aspects[aspect.text]=self.presentation_block()
            elif t.text=='state':
                state_name=self.name();state_initial=False
                if self.token.text=='initial':self.take();self.take(':');self.take('true');state_initial=True
                if any(state.name==state_name.text for state in fsm_states):self.fail(state_name,'Duplicate FSM state','GAMAX_NAME')
                body=self.block()
                body=[item for item in body if item.kind!='fsm_transition']+[item for item in body if item.kind=='fsm_transition']
                fsm_states.append(Behavior(state_name.text,Expr('literal',[state_initial],'bool',state_name),body,state_name))
            elif t.text == "reflex":
                n = self.name()
                if any(b.name == n.text for b in behaviors): self.fail(n, "Duplicate reflex name")
                guard = Expr("literal", [True], "bool", n)
                if self.token.text == "when":
                    self.take(); self.take(":"); guard = self.expression()
                behaviors.append(Behavior(n.text, guard, self.block(), n))
            else:
                self.fail(t, "Supported scope members: initialized int/bool/float/point, init, reflex, action")
        self.take("}")
        scope=Scope(name, fields, initial, behaviors, actions)
        scope.aspects=aspects;scope.fsm_states=fsm_states
        return scope

    def parse(self, validate=True):
        if self.token.text=="@":
            annotation=self.take();self.take("no_warning")
        self.take("model"); model = self.name().text
        while self.token.text == "import":
            self.take(); token = self.take()
            if not token.text.startswith('"'): self.fail(token, "Import requires a quoted relative path")
            self.imports.append((json.loads(token.text), token))
        scopes = [Scope("global", [], [], [])]
        world_torus = False
        world_torus_explicit=False
        if self.token.text == "global":
            self.take()
            if self.token.text == "torus":
                world_torus_explicit=True
                self.take(); self.take(":"); flag = self.take()
                if flag.text not in ("true", "false"): self.fail(flag, "World torus requires a literal bool in this profile", "GAMAX_SPATIAL")
                world_torus = flag.text == "true"
            scopes = [self.parse_scope("global")]
        experiments = []
        presentation={}
        while self.token.text != "<EOF>":
            t = self.take()
            if t.text == "species":
                name = self.name()
                if name.text == 'agent': self.fail(name, "GAMA's built-in agent species cannot inherit from itself", 'GAMAX_NAME')
                if name.text == "global" or any(s.name == name.text for s in scopes):
                    self.fail(name, "Duplicate scope", "GAMAX_NAME")
                skills = [];parent=None;parent_token=None;control=None
                while self.token.text!='{':
                    facet=self.take();self.take(':')
                    if facet.text=='parent':
                        if parent is not None:self.fail(facet,'Duplicate parent','GAMAX_INHERITANCE')
                        parent_token=self.name();parent=parent_token.text
                    elif facet.text=='control':
                        self.take('fsm');control='fsm'
                    elif facet.text=='skills':
                        self.take('[');skill=self.take()
                        if skill.text!='moving' or skills:self.fail(skill,'Only one restricted moving skill is supported','GAMAX_SPATIAL')
                        self.take(']');skills=['moving']
                    else:self.fail(facet,'Unsupported species facet','GAMAX_FACET')
                scope = self.parse_scope(name.text)
                scope.skills = skills
                scope.parent=parent;scope.parent_token=parent_token
                if scope.fsm_states and control!='fsm':self.fail(name,'States require FSM control','GAMAX_EFFECT')
                if control=='fsm':
                    initial=[i for i,state in enumerate(scope.fsm_states) if state.guard.args==[True]]
                    if len(initial)!=1:self.fail(name,'FSM requires exactly one initial state','GAMAX_EFFECT')
                    if any(f.name=='state' for f in scope.fields):self.fail(name,'FSM state variable cannot be redeclared','GAMAX_NAME')
                    scope.fields.append(Field('state','fsm_state',Expr('literal',[initial[0]],'fsm_state',name),name))
                    body=[Statement('fsm_state',state.name,None,state.body,[],state.token) for state in scope.fsm_states]
                    for state in scope.fsm_states:state.body=[]
                    scope.behaviors.append(Behavior('gamax_fsm_dispatch',Expr('literal',[True],'bool',name),[Statement('fsm_dispatch',None,None,body,[],name)],name))
                if skills:
                    inherited=[Field("location","point",Expr("point",[Expr("literal",[0.0],"float",name) for _ in range(3)],"point",name,metadata={'native_moving_default':True}),name)]
                    inherited += [Field(attr,"float",Expr("literal",[1.0 if attr=='speed' else 0.0],"float",name,metadata={'native_moving_default':True}),name) for attr in ('speed','heading','real_speed')]
                    if any(f.name=='location' for f in scope.fields):self.fail(name,'Moving location redeclaration requires separate native initialization validation','GAMAX_SPATIAL')
                    own={f.name:f for f in scope.fields}
                    scope.fields=[own.get(f.name,f) for f in inherited]+[f for f in scope.fields if f.name not in {g.name for g in inherited}]
                def sets_shape(items):
                    return any(item.kind=='assign' and item.target=='shape' or sets_shape(item.body) or sets_shape(item.alternative) for item in items)
                if sets_shape(scope.init) or any(sets_shape(item.body) for item in [*scope.behaviors,*scope.actions]):
                    if not skills:self.fail(name,'Agent polygons require a verified moving location','GAMAX_SPATIAL')
                    if any(f.name=='shape' for f in scope.fields):self.fail(name,'Explicit agent shape declarations are unsupported','GAMAX_SPATIAL')
                    scope.fields.append(Field('shape','polygon',Expr('agent_shape',[],'polygon',name),name))
                scopes.append(scope)
            elif t.text == "grid":
                name = self.name()
                if any(s.name == name.text for s in scopes): self.fail(name, "Duplicate scope", "GAMAX_NAME")
                facets = {}
                allowed = {"width", "height", "neighbors", "frequency", "use_regular_agents", "use_individual_shapes", "use_neighbors_cache"}
                while self.token.text != "{":
                    facet = self.take()
                    if facet.text not in allowed: self.fail(facet, "Unsupported grid declaration facet", "GAMAX_SPATIAL")
                    if facet.text in facets: self.fail(facet, "Duplicate grid declaration facet", "GAMAX_SPATIAL")
                    self.take(":"); value = self.expression()
                    if value.op != "literal" and facet.text not in ('width','height','frequency'):self.fail(value.token,'Non-dimensional grid facets require literals','GAMAX_SPATIAL')
                    facets[facet.text] = value.args[0] if value.op=='literal' else value
                width, height = facets.get("width", 100), facets.get("height", 100)
                if any(not isinstance(n,Expr) and (type(n) is not int or n <= 0) for n in (width,height)):
                    self.fail(name, "Positive literal grid dimensions required", "GAMAX_SPATIAL")
                neighbors = facets.get("neighbors", 4)
                frequency = facets.get("frequency", 1)
                if neighbors not in (4,8) or type(neighbors) is not int or not isinstance(frequency,Expr) and (type(frequency) is not int or not 0<=frequency<=100):
                    self.fail(name, "Only rectangular 4/8-neighbor grids and frequency 0/1 are supported", "GAMAX_SPATIAL")
                for facet in ("use_regular_agents", "use_individual_shapes", "use_neighbors_cache"):
                    if facet in facets and type(facets[facet]) is not bool: self.fail(name, "Grid option requires bool", "GAMAX_SPATIAL")
                if not facets.get("use_neighbors_cache",True):
                    self.fail(name, "Current grid profile requires cached neighborhoods", "GAMAX_SPATIAL")
                scope = self.parse_scope(name.text)
                scope.grid = dict(width=width,height=height,neighbors=neighbors,frequency=frequency,torus=world_torus,facets=facets)
                def uses_color(value):
                    if isinstance(value,Expr):return value.op=='name' and value.args[0]=='color' or any(uses_color(arg) for arg in value.args)
                    if isinstance(value,Statement):return value.target=='color' or uses_color(value.value) or uses_color(value.body) or uses_color(value.alternative)
                    if isinstance(value,(list,tuple)):return any(uses_color(v) for v in value)
                    return False
                if not any(f.name=='color' for f in scope.fields) and (any(uses_color(f.value) for f in scope.fields) or uses_color(scope.init) or any(uses_color(b.body) for b in scope.behaviors)):
                    scope.fields.insert(0,Field('color','rgb',Expr('literal',[[255,255,255,255]],'rgb',name),name))
                self.populations[name.text] = width*height if not any(isinstance(n,Expr) for n in (width,height)) else 0
                scopes.append(scope)
            elif t.text == "experiment":
                token=self.take()
                n=ast.literal_eval(token.text) if token.text.startswith(('"',"'")) else token.text
                if not n:self.fail(token,'Experiment name must not be empty','GAMAX_EXPERIMENT')
                metadata={'virtual':False,'parent':None,'unsupported_effects':[],'pacing':None};seen=set()
                while self.token.text!='{':
                    facet=self.take();self.take(':')
                    if facet.text in seen:self.fail(facet,'Duplicate experiment facet','GAMAX_EXPERIMENT')
                    seen.add(facet.text)
                    if facet.text=='type':self.take('gui')
                    elif facet.text=='virtual':
                        flag=self.take()
                        if flag.text not in ('true','false'):self.fail(flag,'Virtual experiment needs a Boolean literal','GAMAX_EXPERIMENT')
                        metadata['virtual']=flag.text=='true'
                    elif facet.text=='parent':metadata['parent']=self.name().text
                    elif facet.text=='record':
                        self.take('every');self.take('(');interval=self.take();self.take(')')
                        if not re.fullmatch(r'\d+',interval.text) or int(interval.text)<=0:self.fail(interval,'GUI recording interval must be positive','GAMAX_OUTPUT')
                        metadata['record']={'every_cycles':int(interval.text),'execution':'GUI recording retained; excluded from headless simulation outputs'}
                    elif facet.text=='autorun':
                        self.take('true');metadata['autorun']=True
                    else:self.fail(facet,'Unsupported experiment facet','GAMAX_EXPERIMENT')
                if 'type' not in seen and not metadata['virtual'] and not metadata['parent']:self.fail(token,'Concrete experiment requires GUI type or parent','GAMAX_EXPERIMENT')
                self.take('{');parameters=[];outputs=[]
                while self.token.text!='}':
                    member=self.take()
                    if member.text=='parameter':
                        label=self.take()
                        if not label.text.startswith(('"',"'")):self.fail(label,'Experiment parameter needs a quoted label','GAMAX_EXPERIMENT')
                        record={'label':ast.literal_eval(label.text)};facets=set()
                        while self.token.text!=';':
                            facet=self.take();self.take(':')
                            if facet.text in facets:self.fail(facet,'Duplicate parameter facet','GAMAX_EXPERIMENT')
                            facets.add(facet.text)
                            if facet.text=='var':record['variable']=self.name().text
                            elif facet.text=='unit':
                                unit=self.take()
                                if not unit.text.startswith(('\"',"'")):self.fail(unit,'Quoted unit metadata required','GAMAX_EXPERIMENT')
                                record['unit']=ast.literal_eval(unit.text)
                            elif facet.text=='init':
                                initial=self.expression()
                                if initial.op!='literal' or initial.type not in ('int','float','bool'):self.fail(initial.token,'Unsupported parameter initializer','GAMAX_EXPERIMENT')
                                record['initial']=initial.args[0]
                            elif facet.text in ('min','max','step'):
                                value=self.expression()
                                if value.op!='literal' or value.type not in ('int','float'):self.fail(facet,'Parameter bounds require numeric literals','GAMAX_EXPERIMENT')
                                record[facet.text]=value.args[0]
                            elif facet.text=='category':
                                category=self.take()
                                if not category.text.startswith(('"',"'")):self.fail(category,'Quoted parameter category required','GAMAX_EXPERIMENT')
                                record['category']=ast.literal_eval(category.text)
                            else:self.fail(facet,'Unsupported parameter metadata','GAMAX_EXPERIMENT')
                        self.take(';')
                        if 'variable' not in record:self.fail(label,'Parameter variable required','GAMAX_EXPERIMENT')
                        parameters.append(record)
                    elif member.text=='user_command':
                        label=self.take()
                        if not label.text.startswith(('\"',"'")):self.fail(label,'User command label must be quoted','GAMAX_OUTPUT')
                        metadata.setdefault('user_commands',[]).append({'label':ast.literal_eval(label.text),'body':self.raw_block(),'execution':'manual GUI callback retained; not invoked in headless transitions'})
                    elif member.text=='permanent':
                        metadata.setdefault('permanent_outputs',[]).append(self.presentation_block())
                    elif member.text=='output':
                        facets={}
                        while self.token.text!='{':
                            facet=self.take();self.take(':')
                            if facet.text!='synchronized' or facet.text in facets:self.fail(facet,'Unsupported/duplicate output facet','GAMAX_OUTPUT')
                            flag=self.take()
                            if flag.text not in ('true','false'):self.fail(flag,'Output synchronization requires a literal Boolean','GAMAX_OUTPUT')
                            facets[facet.text]=flag.text=='true'
                        outputs.append({**self.presentation_block(),'facets':facets})
                    elif member.text in ('init','abort'):
                        metadata['unsupported_effects'].append({'kind':member.text,**self.raw_block()})
                    elif member.text in ('int','float','bool','point'):
                        start=self.i-1;variable=self.name();self.take('<-');value=self.expression();self.take(';')
                        if variable.text=='minimum_cycle_duration' and member.text=='float' and value.op=='literal' and value.type=='float' and math.isfinite(value.args[0]) and value.args[0]>=0:metadata['pacing']=value.args[0]
                        else:metadata['unsupported_effects'].append({'kind':'experiment_variable','tokens':[token.text for token in self.tokens[start:self.i]]})
                    else:self.fail(member,'Unsupported effectful/unknown experiment member','GAMAX_EXPERIMENT')
                self.take('}')
                if n in experiments:self.fail(t,'Duplicate experiment')
                experiments.append(n)
                presentation[n]={**metadata,'parameters':parameters,'outputs':outputs}
            else:
                self.fail(t, "Only species and empty GUI experiments follow global")
        names = {s.name for s in scopes[1:]}
        ir = ModelIR(model, scopes, experiments, self.populations)
        ir.presentation=presentation
        ir.world_torus=world_torus;ir.world_torus_explicit=world_torus_explicit
        if validate:
            ir.scopes=resolve_species(ir.scopes)
            resolve_field_dependencies(ir.scopes)
            self.expand_actions(ir)
            self.resolve_creation_counts(ir)
            parents={s.parent for s in ir.scopes if s.parent}
            for parent in parents:ir.populations.setdefault(parent,0)
            if set(ir.populations)!=names:
                self.fail(self.token,'Every non-grid species must be created in global init','GAMAX_CAPACITY')
            self.check(ir)
        return ir

    def resolve_ask_targets(self,ir):
        scopes={scope.name:scope for scope in ir.scopes}
        globals_={f.name:f.type for f in ir.scopes[0].fields}
        def type_of(expr,env,owner):
            if expr.type:return expr.type
            if expr.op.startswith('grid_at:'):return 'agent:'+expr.op.split(':')[1]
            if expr.op=='name':
                if expr.args[0]=='self':return 'agent:'+owner
                if expr.args[0] in scopes:return 'agents:'+expr.args[0]
                return env.get(expr.args[0],'')
            if expr.op in ('where','as_list','overlapping','at_distance','remove_agent','append_agent'):return type_of(expr.args[0],env,owner)
            if expr.op in ('call:one_of','call:first','call:last','at'):
                return type_of(expr.args[0],env,owner).replace('agents:','agent:',1)
            if expr.op.startswith('attribute:'):
                base=type_of(expr.args[0],env,owner)
                if base.startswith('agent:'):return next((f.type for f in scopes[base.split(':')[1]].fields if f.name==expr.op.split(':')[1]),'')
            return ''
        def walk(items,owner,env):
            env=dict(env)
            for item in items:
                if item.kind.startswith('local:'):env[item.target]=item.kind.split(':',1)[1]
                if item.kind=='ask':
                    selected=type_of(item.value,env,owner)
                    if not selected.startswith(('agent:','agents:')):self.fail(item.token,'Ask selection requires a statically typed agent or collection','GAMAX_TYPE')
                    target=selected.split(':')[1]
                    if target not in scopes:self.fail(item.token,'Unknown ask target species','GAMAX_NAME')
                    item.target=target
                    captured={name:kind for name,kind in env.items() if name not in globals_ and name not in {f.name for f in scopes[owner].fields} and name!='index'}
                    walk(item.body,target,globals_|{f.name:f.type for f in scopes[target].fields}|captured)
                else:
                    target=item.target if item.kind in ('create','birth') else owner
                    if item.metadata.get('dynamic_self_species'):item.target=owner;target=owner
                    walk(item.body,target,env if target==owner else globals_|{f.name:f.type for f in scopes[target].fields})
                walk(item.alternative,owner,env)
        for scope in ir.scopes:
            env=globals_|{f.name:f.type for f in scope.fields}
            walk(scope.init,scope.name,env)
            for member in scope.behaviors:walk(member.body,scope.name,env)
            for action in scope.actions:walk(action.body,scope.name,env|dict(action.parameters))

    def expand_actions(self, ir):
        """Inline calls with evaluated-once, lexically scoped parameters; reject recursion."""
        scopes = {s.name: s for s in ir.scopes}
        serial = 0
        def expand_expr(expr,scope,stack,phase):
            nonlocal serial
            if expr is None:return
            for arg in expr.args:
                if isinstance(arg,Expr):expand_expr(arg,scope,stack,phase)
            if not expr.op.startswith('invoke:'):return
            name=expr.op.split(':',1)[1]
            target=next((a for a in scopes[scope].actions if a.name==name),None)
            if target is None or target.return_type is None:self.fail(expr.token,'Unknown value-returning action '+name,'GAMAX_NAME')
            if target.virtual or target.presentation_only:self.fail(expr.token,'Unresolved/presentation action in simulation expression','GAMAX_EFFECT')
            signature=(scope,name)
            if signature in stack:self.fail(expr.token,'Recursive value-returning action','GAMAX_EFFECT')
            if len(expr.args)!=len(target.parameters):self.fail(expr.token,'Function argument count mismatch','GAMAX_TYPE')
            if phase and phase not in target.phases:target.phases.append(phase)
            serial+=1;prefix=f'gamax_function_{serial}_'
            names={n:prefix+n for n,_ in target.parameters}
            body=copy.deepcopy(target.body)
            def rename(value):
                if isinstance(value,Expr):
                    if value.op=='name' and value.args[0] in names:value.args[0]=names[value.args[0]]
                    for arg in value.args:rename(arg)
                elif isinstance(value,Statement):
                    if value.target in names:value.target=names[value.target]
                    rename(value.value);rename(value.body);rename(value.alternative)
                elif isinstance(value,list):
                    for item in value:rename(item)
            rename(body)
            params=[Statement('local:'+kind,names[n],arg,[],[],expr.token) for (n,kind),arg in zip(target.parameters,expr.args)]
            body=expand(body,scope,stack+(signature,),phase)
            expr.op='function_value';expr.args=[];expr.type=target.return_type
            expr.metadata={'body':params+body,'owner_scope':scope,'action_name':name,'return_type':target.return_type,'parameter_count':len(params)}
        def expand(items, scope, stack=(),phase=None):
            nonlocal serial
            result = []
            for item in items:
                expand_expr(item.value,scope,stack,phase)
                for facet in item.metadata.values():
                    if isinstance(facet,Expr):expand_expr(facet,scope,stack,phase)
                if item.kind != "do":
                    if phase and item.metadata.get('action_name'):
                        owner=scopes[item.metadata['action_scope']]
                        called=next(a for a in owner.actions if a.name==item.metadata['action_name'])
                        if phase not in called.phases:called.phases.append(phase)
                    if item.kind in ('ask','create','birth') and item.target not in scopes:
                        self.fail(item.token,'Unknown creation/ask species','GAMAX_NAME')
                    item.body = expand(item.body, item.target if item.kind in ('ask','create','birth') else scope, stack,phase)
                    item.alternative = expand(item.alternative, scope, stack,phase)
                    result.append(item); continue
                if item.target=='pause':
                    if item.value.args:self.fail(item.token,'pause takes no arguments','GAMAX_EFFECT')
                    result.append(Statement('pause',None,None,[],[],item.token));continue
                if item.target=='die':
                    if scope=='global' or item.value.args:self.fail(item.token,'die currently requires an agent caller without arguments','GAMAX_EFFECT')
                    result.append(Statement('die',scope,None,[],[],item.token));continue
                if item.target=='goto' and 'moving' in scopes[scope].skills:
                    names=item.value.type.split(',') if item.value.type else []
                    if not names or set(names)-{'target','speed'} or 'target' not in names:self.fail(item.token,'Point goto requires target and optional speed facets','GAMAX_SPATIAL')
                    facets=dict(zip(names,item.value.args))
                    result.append(Statement('skill_goto',None,facets['target'],[],[],item.token,metadata={'speed':facets.get('speed')}));continue
                if item.target in ('move','wander') and 'moving' in scopes[scope].skills:
                    names=item.value.type.split(',') if item.value.type else []
                    allowed={'speed','amplitude'} if item.target=='wander' else {'speed','heading'}
                    if item.value.args and not names or not set(names)<=allowed:self.fail(item.token,'Unsupported moving skill argument/facet','GAMAX_SPATIAL')
                    facets=dict(zip(names,item.value.args))
                    def random_argument(expr):
                        return expr.op in ('call:rnd','call:flip','call:one_of','call:shuffle','neighbors_at','overlapping') or any(random_argument(a) for a in expr.args if isinstance(a,Expr))
                    if any(random_argument(expr) for expr in facets.values()):self.fail(item.token,'Effectful moving facets require a separate native evaluation-order contract','GAMAX_EFFECT')
                    if item.target=='wander':
                        amplitude=facets.get('amplitude',Expr('literal',[359.0],'float',item.token))
                        half=Expr('/',[Expr('call:float',[amplitude],'',item.token),Expr('literal',[2.0],'float',item.token)],'',item.token)
                        random=Expr('call:rnd',[Expr('-_unary',[half],'',item.token),half],'',item.token)
                        heading=Expr('+',[Expr('name',['heading'],'',item.token),random],'',item.token)
                        result.append(Statement('assign','heading',heading,[],[],item.token))
                    elif 'heading' in facets:result.append(Statement('assign','heading',facets['heading'],[],[],item.token))
                    result.append(Statement('skill_move',None,facets.get('speed'),[],[],item.token,metadata={'wander':True} if item.target=='wander' else {}));continue
                target = next((a for a in scopes[scope].actions if a.name == item.target), None)
                # Global actions called from species run in caller scope in GAMA;
                # this profile only supports actions declared in the current scope.
                if target is None: self.fail(item.token, f"Unknown local action {item.target!r}; skill/global dispatch is unsupported", "GAMAX_NAME")
                if target.presentation_only:self.fail(item.token,'GUI-input action cannot execute in a headless simulation transition','GAMAX_OUTPUT')
                if target.virtual:self.fail(item.token,'Called virtual action has no concrete override','GAMAX_INHERITANCE')
                if phase and phase not in target.phases:target.phases.append(phase)
                signature = (scope, target.name)
                if signature in stack: self.fail(item.token, "Recursive action calls cannot be statically expanded", "GAMAX_EFFECT")
                if len(item.value.args) != len(target.parameters): self.fail(item.token, "Action argument count mismatch", "GAMAX_TYPE")
                if item.value.type:
                    named = item.value.type.split(",")
                    if set(named) != {name for name, _ in target.parameters}:
                        self.fail(item.token, "Named action arguments must match declared parameters", "GAMAX_TYPE")
                    # GAMA evaluates named arguments in declaration order; the
                    # native fixture deliberately reverses two random draws.
                    arguments_by_name = dict(zip(named, item.value.args))
                    bindings = [(name, type_, arguments_by_name[name]) for name, type_ in target.parameters]
                else:
                    bindings = [(name, type_, arg) for (name, type_), arg in zip(target.parameters, item.value.args)]
                serial += 1
                prefix = f"gamax_call_{serial}_"
                names = {name: prefix + name for name, _ in target.parameters}
                body = copy.deepcopy(target.body)
                def rename_expr(expr):
                    if expr.op == "name" and expr.args[0] in names: expr.args[0] = names[expr.args[0]]
                    elif expr.op not in ("name", "literal", "each", "myself", "iterations"):
                        for arg in expr.args: rename_expr(arg)
                def rename(items):
                    for stmt in items:
                        if stmt.target in names: stmt.target = names[stmt.target]
                        if stmt.value: rename_expr(stmt.value)
                        rename(stmt.body); rename(stmt.alternative)
                rename(body)
                parameters = [Statement("local:" + type_, names[name], arg, [], [], item.token)
                              for name, type_, arg in bindings]
                result.append(Statement("block", None, None, parameters + expand(body, scope, stack + (signature,),phase), [], item.token,metadata={'action_scope':scope,'action_name':target.name}))
            return result
        for scope in ir.scopes:
            # Check every definition, including unused actions, to keep failures explicit.
            for action in scope.actions:
                if not action.presentation_only:action.body = expand(copy.deepcopy(action.body), scope.name, ((scope.name, action.name),))
            for field in scope.fields:
                for expr in [field.value,field.getter,*field.facets.values()]:expand_expr(expr,scope.name,(),'init')
            scope.init = expand(scope.init, scope.name,phase='init')
            for behavior in scope.behaviors:
                expand_expr(behavior.guard,scope.name,(),'runtime')
                behavior.body = expand(behavior.body, scope.name,phase='runtime')







