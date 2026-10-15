"""Small fail-closed frontend, typed ordered IR, and standalone JAX code generator.

No GAML is evaluated as Python and no model-specific transition is embedded here.
Unsupported syntax is rejected, including effectful experiment/display blocks.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import re
import struct


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    column: int
    source: str | None = None


class CompileError(ValueError):
    def __init__(self, path, token, reason, code="GAMAX_UNSUPPORTED"):
        self.diagnostic = dict(code=code, severity="error", file=token.source or str(path),
                               line=token.line, column=token.column, reason=reason)
        super().__init__(f"{path}:{token.line}:{token.column}: {code}: {reason}")


@dataclass
class Expr:
    op: str
    args: list
    type: str
    token: Token


@dataclass
class Field:
    name: str
    type: str
    value: Expr
    token: Token


@dataclass
class Statement:
    kind: str
    target: str | None
    value: Expr | None
    body: list
    alternative: list
    token: Token


@dataclass
class Behavior:
    name: str
    guard: Expr
    body: list[Statement]
    token: Token
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)


@dataclass
class Action:
    name: str
    parameters: list[tuple[str, str]]
    body: list[Statement]
    token: Token


@dataclass
class Scope:
    name: str
    fields: list[Field]
    init: list[Statement]
    behaviors: list[Behavior]
    actions: list[Action] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)


@dataclass
class ModelIR:
    name: str
    scopes: list[Scope]
    experiments: list[str]
    populations: dict[str, int]
    random_seed: float | None = None


TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|#[A-Za-z_]\w*|//[^\n]*|/\*.*?\*/|\s+|<-|>=|<=|!=|[A-Za-z_]\w*|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|\d+|[{}\[\]():;,+*/.\-<>=?!%^@]', re.S)
PRECEDENCE = {"where": 0, "collect": 0, "distance_to": 5, "or": 1, "and": 2, "=": 3, "!=": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "+": 5, "-": 5, "*": 6, "/": 6, "mod": 6, "div": 6}
FUNCTIONS = {"abs", "sqrt", "sin", "cos", "exp", "int", "float", "min", "max", "length", "sum", "mean", "rnd", "flip"}
RESERVED = {"index", "cycle", "self", "world", "true", "false", "first", "last", "length", "int", "float", "bool",
            "model", "global", "species", "experiment", "init", "reflex", "if", "else", "action", "do", "skills", "point", "rng", "seed",
            "create", "ask", "each", "myself", "and", "or", "not", "number", "when", "type", "gui"} | FUNCTIONS


class Parser:
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

    def expression(self, minimum=0):
        t = self.take()
        if t.text in ("-", "not"):
            lhs = Expr(t.text + "_unary", [self.expression(7)], "", t)
        elif t.text == "{":
            args = [self.expression()]
            self.take(","); args.append(self.expression())
            if self.token.text == ",": self.take(); args.append(self.expression())
            self.take("}")
            if len(args) == 2: args.append(Expr("literal", [0.0], "float", t))
            lhs = Expr("point", args, "point", t)
        elif t.text == "(":
            lhs = self.expression()
            self.take(")")
        elif t.text.startswith('"'):
            lhs = Expr("literal", [json.loads(t.text)], "string", t)
        elif t.text in ("true", "false"):
            lhs = Expr("literal", [t.text == "true"], "bool", t)
        elif re.fullmatch(r"\d+", t.text):
            if int(t.text) > 2147483647:
                self.fail(t, "Integer literal outside signed int32 profile")
            lhs = Expr("literal", [int(t.text)], "int", t)
        elif re.fullmatch(r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+", t.text):
            value = float(t.text)
            if not math.isfinite(value): self.fail(t, 'Non-finite floating literal outside numeric profile', 'GAMAX_TYPE')
            lhs = Expr("literal", [value], "float", t)
        elif t.text in FUNCTIONS and self.token.text == "(":
            self.take("(")
            args = [self.expression()]
            while self.token.text == ",":
                self.take(); args.append(self.expression())
            self.take(")")
            lhs = Expr("call:" + t.text, args, "", t)
        elif t.text in ("each", "myself"):
            self.take(".")
            attribute = self.take()
            if not re.fullmatch(r"[A-Za-z_]\w*", attribute.text): self.fail(attribute, "Expected attribute")
            lhs = Expr(t.text, [attribute.text], "", t)
        elif re.fullmatch(r"[A-Za-z_]\w*", t.text):
            lhs = Expr("name", [t.text], "", t)
        else:
            self.fail(t, "Expected a supported expression")
        while self.token.text == ".":
            self.take(); component = self.take()
            if component.text not in ("x", "y", "z"): self.fail(component, "Only point component access is supported")
            lhs = Expr("component:" + component.text, [lhs], "float", component)
        while PRECEDENCE.get(self.token.text, -1) >= minimum:
            op = self.take()
            rhs = self.expression(PRECEDENCE[op.text]+1)
            lhs = Expr(op.text, [lhs, rhs], "", op)
        return lhs

    def block(self, initialization=False):
        self.take("{")
        result = []
        while self.token.text != "}":
            t = self.take()
            if t.text == "if":
                condition = self.expression()
                body = self.block(initialization)
                alternative = []
                if self.token.text == "else":
                    self.take()
                    alternative = self.block(initialization)
                result.append(Statement("if", None, condition, body, alternative, t))
            elif t.text in ("int", "float", "bool", "point"):
                name = self.name()
                self.take("<-"); value = self.expression(); self.take(";")
                result.append(Statement("local:" + t.text, name.text, value, [], [], name))
            elif t.text == "do":
                name = self.name()
                arguments = []
                if self.token.text == "(":
                    self.take()
                    if self.token.text != ")":
                        arguments.append(self.expression())
                        while self.token.text == ",":
                            self.take(); arguments.append(self.expression())
                    self.take(")")
                self.take(";")
                result.append(Statement("do", name.text, Expr("arguments", arguments, "", name), [], [], t))
            elif t.text == "create":
                if not initialization or self.scope != "global":
                    self.fail(t, "Creation is restricted to unconditional global init")
                species = self.name().text
                self.take("number"); self.take(":")
                count = self.take()
                if not count.text.isdigit() or not 1 <= int(count.text) <= 1_000_000:
                    self.fail(count, "Creation requires a literal count in [1, 1000000]")
                self.take(";")
                if species in self.populations:
                    self.fail(t, "Exactly one creation statement per species is required")
                self.populations[species] = int(count.text)
                result.append(Statement("create", species, None, [], [], t))
            elif t.text == "ask":
                if initialization or self.scope != "global":
                    self.fail(t, "Only global reflex ask over a fixed species is supported")
                species = self.name().text
                # No nested asks or captured outer locals in this profile.
                old_scope = self.scope
                self.scope = species
                body = self.block()
                self.scope = old_scope
                result.append(Statement("ask", species, None, body, [], t))
            else:
                if not re.fullmatch(r"[A-Za-z_]\w*", t.text):
                    self.fail(t, "Only assignment and if statements are supported")
                self.take("<-")
                value = self.expression(); self.take(";")
                result.append(Statement("assign", t.text, value, [], [], t))
        self.take("}")
        return result

    def parse_scope(self, name):
        self.scope = name
        self.take("{")
        fields, initial, behaviors, actions = [], [], [], []
        init_seen = False
        while self.token.text != "}":
            t = self.take()
            if t.text in ("int", "bool", "float", "point"):
                n = self.name()
                if any(f.name == n.text for f in fields):
                    self.fail(n, "Duplicate attribute", "GAMAX_NAME")
                self.take("<-"); value = self.expression(); self.take(";")
                fields.append(Field(n.text, t.text, value, n))
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
                        if type_.text not in ("int", "float", "bool", "point"):
                            self.fail(type_, "Action parameters require scalar types")
                        parameter = self.name()
                        if any(n == parameter.text for n, _ in parameters): self.fail(parameter, "Duplicate parameter")
                        parameters.append((parameter.text, type_.text))
                        if self.token.text != ",": break
                        self.take()
                    self.take(")")
                if any(a.name == action_name.text for a in actions): self.fail(action_name, "Duplicate action")
                actions.append(Action(action_name.text, parameters, self.block(), action_name))
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
        return Scope(name, fields, initial, behaviors, actions)

    def parse(self, validate=True):
        self.take("model"); model = self.name().text
        while self.token.text == "import":
            self.take(); token = self.take()
            if not token.text.startswith('"'): self.fail(token, "Import requires a quoted relative path")
            self.imports.append((json.loads(token.text), token))
        scopes = [Scope("global", [], [], [])]
        if self.token.text == "global":
            self.take(); scopes = [self.parse_scope("global")]
        experiments = []
        while self.token.text != "<EOF>":
            t = self.take()
            if t.text == "species":
                name = self.name()
                if name.text == 'agent': self.fail(name, "GAMA's built-in agent species cannot inherit from itself", 'GAMAX_NAME')
                if name.text == "global" or any(s.name == name.text for s in scopes):
                    self.fail(name, "Duplicate scope", "GAMAX_NAME")
                skills = []
                if self.token.text == "skills":
                    self.take(); self.take(":"); self.take("["); skill = self.take()
                    if skill.text != "moving": self.fail(skill, "Only the restricted 2D moving skill is supported")
                    self.take("]"); skills = ["moving"]
                scope = self.parse_scope(name.text)
                scope.skills = skills
                if skills:
                    inherited = [Field("location", "point", Expr("point", [Expr("literal", [0.0], "float", name) for _ in range(3)], "point", name), name)]
                    inherited += [Field(attr, "float", Expr("literal", [0.0], "float", name), name) for attr in ("speed", "heading", "real_speed")]
                    if any(f.name in {g.name for g in inherited} for f in scope.fields): self.fail(name, "Redeclaring moving attributes is unsupported")
                    assigned = {stmt.target for stmt in scope.init if stmt.kind == "assign"}
                    if not {"location", "speed", "heading"} <= assigned:
                        self.fail(name, "Moving requires unconditional explicit init of location, speed and heading", "GAMAX_SPATIAL")
                    builtins = {f.name for f in inherited}
                    def names_in(expr):
                        if expr.op == "name": return {expr.args[0]}
                        if expr.op in ("literal", "each", "myself"): return set()
                        return set().union(*(names_in(a) for a in expr.args))
                    for member in scope.fields:
                        if names_in(member.value) & builtins: self.fail(member.token, "Moving defaults must be overwritten before use; attribute initialization cannot depend on them", "GAMAX_SPATIAL")
                    initialized = {"real_speed"}
                    def inspect(items, known):
                        for stmt in items:
                            if stmt.value and names_in(stmt.value) & (builtins - known):
                                self.fail(stmt.token, "Moving init reads an implicit random default before overwriting it", "GAMAX_SPATIAL")
                            inspect(stmt.body, set(known)); inspect(stmt.alternative, set(known))
                            if stmt.kind == "assign" and stmt.target in builtins: known.add(stmt.target)
                    inspect(scope.init, initialized)
                    last_location = max(i for i, stmt in enumerate(scope.init) if stmt.target == "location")
                    last_heading = max(i for i, stmt in enumerate(scope.init) if stmt.target == "heading")
                    if last_heading < last_location: self.fail(name, "Explicit init heading must follow location (native setter changes heading)", "GAMAX_SPATIAL")
                    scope.fields = inherited + scope.fields
                scopes.append(scope)
            elif t.text == "experiment":
                n = self.name().text
                self.take("type"); self.take(":"); self.take("gui")
                self.take("{"); self.take("}")
                if n in experiments: self.fail(t, "Duplicate experiment")
                experiments.append(n)
            else:
                self.fail(t, "Only species and empty GUI experiments follow global")
        names = {s.name for s in scopes[1:]}
        if validate and set(self.populations) != names:
            self.fail(self.token, "Each declared species must be created exactly once in global init")
        ir = ModelIR(model, scopes, experiments, self.populations)
        if validate:
            self.expand_actions(ir)
            self.check(ir)
        return ir

    def expand_actions(self, ir):
        """Inline calls with evaluated-once, lexically scoped parameters; reject recursion."""
        scopes = {s.name: s for s in ir.scopes}
        serial = 0
        def expand(items, scope, stack=()):
            nonlocal serial
            result = []
            for item in items:
                if item.kind != "do":
                    item.body = expand(item.body, item.target if item.kind == "ask" else scope, stack)
                    item.alternative = expand(item.alternative, scope, stack)
                    result.append(item); continue
                if item.target == "move" and "moving" in scopes[scope].skills:
                    if item.value.args: self.fail(item.token, "move facets/arguments are outside the restricted spatial profile", "GAMAX_SPATIAL")
                    result.append(Statement("skill_move", None, None, [], [], item.token)); continue
                target = next((a for a in scopes[scope].actions if a.name == item.target), None)
                # Global actions called from species run in caller scope in GAMA;
                # this profile only supports actions declared in the current scope.
                if target is None: self.fail(item.token, f"Unknown local action {item.target!r}; skill/global dispatch is unsupported", "GAMAX_NAME")
                signature = (scope, target.name)
                if signature in stack: self.fail(item.token, "Recursive action calls cannot be statically expanded", "GAMAX_EFFECT")
                if len(item.value.args) != len(target.parameters): self.fail(item.token, "Action argument count mismatch", "GAMAX_TYPE")
                serial += 1
                prefix = f"gamax_call_{serial}_"
                names = {name: prefix + name for name, _ in target.parameters}
                body = copy.deepcopy(target.body)
                def rename_expr(expr):
                    if expr.op == "name" and expr.args[0] in names: expr.args[0] = names[expr.args[0]]
                    elif expr.op not in ("name", "literal", "each", "myself"):
                        for arg in expr.args: rename_expr(arg)
                def rename(items):
                    for stmt in items:
                        if stmt.target in names: stmt.target = names[stmt.target]
                        if stmt.value: rename_expr(stmt.value)
                        rename(stmt.body); rename(stmt.alternative)
                rename(body)
                parameters = [Statement("local:" + type_, names[name], arg, [], [], item.token)
                              for (name, type_), arg in zip(target.parameters, item.value.args)]
                result.append(Statement("block", None, None, parameters + expand(body, scope, stack + (signature,)), [], item.token))
            return result
        for scope in ir.scopes:
            # Check every definition, including unused actions, to keep failures explicit.
            for action in scope.actions: action.body = expand(copy.deepcopy(action.body), scope.name, ((scope.name, action.name),))
            scope.init = expand(scope.init, scope.name)
            for behavior in scope.behaviors: behavior.body = expand(behavior.body, scope.name)

    def check(self, ir):
        global_init = ir.scopes[0].init
        configuration = [i for i, stmt in enumerate(global_init) if stmt.kind == "assign" and stmt.target in ("rng", "seed")]
        if configuration:
            expected = global_init[-2:]
            if configuration != list(range(len(global_init)-2, len(global_init))) or [stmt.target for stmt in expected] != ["rng", "seed"]:
                self.fail(global_init[configuration[0]].token, "RNG configuration must be the final rng/seed pair in global init", "GAMAX_RANDOM")
            rng, seed = [stmt.value for stmt in expected]
            if rng.op != "literal" or rng.args != ["java"] or seed.op != "literal" or seed.type not in ("int", "float"):
                self.fail(expected[0].token, "Random profile requires literal rng = java and finite positive literal seed", "GAMAX_RANDOM")
            if seed.args[0] == 0: self.fail(seed.token, "GAMA seed zero requests a fresh random seed; a positive seed is required", "GAMAX_RANDOM")
            ir.random_seed = float(seed.args[0])
            del global_init[-2:]
        self.random_seed = ir.random_seed
        if any(scope.skills for scope in ir.scopes) and any(f.name in ("step", "shape", "torus") for f in ir.scopes[0].fields):
            self.fail(ir.scopes[0].fields[0].token, "moving supports only the default world and one-second clock", "GAMAX_SPATIAL")
        self.scopes = {scope.name: scope for scope in ir.scopes}
        globals_ = {f.name: f.type for f in ir.scopes[0].fields}
        for scope in ir.scopes:
            self.checked_scope = scope.name
            local = {f.name: f.type for f in scope.fields}
            if scope.name != "global" and set(local) & set(globals_):
                self.fail(scope.fields[0].token, "Local/global attribute shadowing is outside this profile")
            initialized = dict(globals_) if scope.name != "global" else {}
            if scope.name != "global": initialized["index"] = "int"
            self.initializing = True
            for field in scope.fields:
                self.check_expr(field.value, initialized)
                self.compatible(field.type, field.value)
                initialized[field.name] = field.type
            env = globals_ | local
            if scope.name != "global": env["index"] = "int"
            self.check_statements(scope.init, env, local, scope.name, unconditional=True)
            self.initializing = False
            for action in scope.actions:
                action_env = dict(env); action_local = dict(local)
                for name, type_ in action.parameters:
                    if name in action_env: self.fail(action.token, "Action parameter shadowing is unsupported", "GAMAX_NAME")
                    action_env[name] = action_local[name] = type_
                self.check_statements(action.body, action_env, action_local, scope.name, allow_global_writes=True)
            for behavior in scope.behaviors:
                self.check_expr(behavior.guard, env)
                self.compatible("bool", behavior.guard)
                self.check_statements(behavior.body, env, local, scope.name, allow_global_writes=True)
                behavior.reads, behavior.writes = self.effects(behavior, scope.name, local)

    def effects(self, behavior, scope, local):
        reads, writes = set(), set()
        def qualified(name):
            return scope + "." + name if name in local else "global." + name
        def expression(expr):
            if expr.op == "name":
                if expr.args[0] in local or any(f.name == expr.args[0] for f in self.scopes["global"].fields): reads.add(qualified(expr.args[0]))
            elif expr.op == "each":
                # Conservative whole-population dependencies for every candidate scope.
                for name, target in self.scopes.items():
                    if name != 'global' and any(f.name == expr.args[0] for f in target.fields): reads.add(name+'.'+expr.args[0])
            elif expr.op == "myself": reads.add(qualified(expr.args[0]))
            elif expr.op != "literal":
                for arg in expr.args: expression(arg)
        def statements(items):
            for item in items:
                if item.value: expression(item.value)
                if item.kind == "ask":
                    target = self.scopes[item.target]
                    nested = Behavior('ask', Expr('literal', [True], 'bool', item.token), item.body, item.token)
                    nr, nw = self.effects(nested, target.name, {f.name: f.type for f in target.fields})
                    reads.update(nr); writes.update(nw)
                else:
                    if item.kind == "skill_move":
                        reads.update(scope + "." + name for name in ("location", "heading", "speed"))
                        writes.update(scope + "." + name for name in ("location", "heading", "real_speed"))
                    if item.kind == "assign" and item.target == "location" and "moving" in self.scopes[scope].skills:
                        reads.add(scope + ".location"); writes.add(scope + ".heading")
                    if item.kind == "assign" and (item.target in local or any(f.name == item.target for f in self.scopes["global"].fields)): writes.add(qualified(item.target))
                    statements(item.body); statements(item.alternative)
        expression(behavior.guard); statements(behavior.body)
        return sorted(reads), sorted(writes)

    def compatible(self, type_, expr):
        if type_ != expr.type and not (type_ == "float" and expr.type == "int"):
            self.fail(expr.token, f"Cannot assign {expr.type} to {type_}", "GAMAX_TYPE")

    def check_statements(self, statements, env, local, scope, unconditional=False, allow_global_writes=False):
        env, local = dict(env), dict(local)
        for statement in statements:
            if statement.kind.startswith("local:"):
                if statement.target in env: self.fail(statement.token, "Local shadowing is unsupported", "GAMAX_NAME")
                self.check_expr(statement.value, env)
                type_ = statement.kind.split(":")[1]
                self.compatible(type_, statement.value)
                env[statement.target] = local[statement.target] = type_
            elif statement.kind == "skill_move":
                if self.initializing or scope == "global": self.fail(statement.token, "move requires a species transition", "GAMAX_SPATIAL")
            elif statement.kind == "block":
                self.check_statements(statement.body, env, local, scope, unconditional=unconditional, allow_global_writes=allow_global_writes)
            elif statement.kind == "assign":
                if statement.target not in local and not (allow_global_writes and statement.target in env and statement.target != "index"):
                    self.fail(statement.token, "Init writes must be local; reflex writes may also target declared globals", "GAMAX_EFFECT")
                self.check_expr(statement.value, env)
                self.compatible(env[statement.target], statement.value)
            elif statement.kind == "if":
                self.check_expr(statement.value, env); self.compatible("bool", statement.value)
                self.check_statements(statement.body, env, local, scope, allow_global_writes=allow_global_writes)
                self.check_statements(statement.alternative, env, local, scope, allow_global_writes=allow_global_writes)
            elif statement.kind == "create":
                if not unconditional:
                    self.fail(statement.token, "Conditional creation is not supported", "GAMAX_EFFECT")
            elif statement.kind == "ask":
                if statement.target not in self.populations:
                    self.fail(statement.token, "Ask requires a declared fixed species", "GAMAX_NAME")
                target = self.scopes[statement.target]
                target_local = {f.name: f.type for f in target.fields}
                globals_ = {f.name: f.type for f in self.scopes['global'].fields}
                saved_scope = self.checked_scope
                self.checked_scope = target.name
                self.check_statements(statement.body, globals_ | target_local | {'index': 'int'}, target_local, target.name, allow_global_writes=True)
                self.checked_scope = saved_scope

    def check_expr(self, expr, env):
        if expr.op == "literal":
            if expr.type == "string": self.fail(expr.token, "String dynamics are outside this profile", "GAMAX_TYPE")
            return
        if expr.op in ('each', 'myself'):
            if expr.op == 'myself': self.fail(expr.token, 'myself refers to the calling scope in GAMA; caller-scope aliases are not lowered in this profile')
            if self.initializing: self.fail(expr.token, 'Agent-context aliases during initialization are outside this profile')
            owner = self.collection_scope if expr.op == 'each' else self.checked_scope
            if owner is None or owner == 'global' or expr.args[0] == 'index' and expr.op == 'myself':
                self.fail(expr.token, 'Agent expression outside its supported context', 'GAMAX_NAME')
            fields = {f.name: f.type for f in self.scopes[owner].fields} | {'index': 'int'}
            if expr.args[0] not in fields: self.fail(expr.token, 'Unknown agent attribute', 'GAMAX_NAME')
            expr.type = fields[expr.args[0]]; return
        if expr.op in ('where', 'collect'):
            self.check_expr(expr.args[0], env)
            if not expr.args[0].type.startswith('agents:'): self.fail(expr.token, 'Collection operator requires fixed agents', 'GAMAX_TYPE')
            owner = expr.args[0].type.split(':')[1]
            saved = self.collection_scope
            if saved is not None: self.fail(expr.token, 'Nested each bindings are outside this profile')
            self.collection_scope = owner
            self.check_expr(expr.args[1], env)
            self.collection_scope = saved
            if expr.op == 'where':
                self.compatible('bool', expr.args[1]); expr.type = expr.args[0].type
            else:
                if expr.args[1].type not in ('int','float','bool'): self.fail(expr.token, 'Collect requires scalar values', 'GAMAX_TYPE')
                expr.type = 'values:'+expr.args[1].type
            return
        if expr.op == "point":
            for arg in expr.args:
                self.check_expr(arg, env)
                if arg.type not in ("int", "float"): self.fail(arg.token, "Point coordinates must be numeric", "GAMAX_TYPE")
            expr.type = "point"; return
        if expr.op.startswith("component:"):
            self.check_expr(expr.args[0], env)
            if expr.args[0].type != "point": self.fail(expr.token, "Coordinate access requires a point", "GAMAX_TYPE")
            expr.type = "float"; return
        if expr.op == "name":
            if expr.args[0] in self.populations:
                if self.initializing: self.fail(expr.token, 'Agent collection reads during initialization are outside this profile')
                expr.type = 'agents:'+expr.args[0]; return
            if expr.args[0] not in env:
                self.fail(expr.token, f"Unknown or unsupported name {expr.args[0]!r}", "GAMAX_NAME")
            expr.type = env[expr.args[0]]; return
        for arg in expr.args: self.check_expr(arg, env)
        types = [arg.type for arg in expr.args]
        if expr.op.startswith('call:'):
            function = expr.op.split(':')[1]
            if function in ("rnd", "flip"):
                if self.random_seed is None: self.fail(expr.token, "Runtime randomness requires explicit final global-init rng/seed configuration", "GAMAX_RANDOM")
                if self.initializing or self.collection_scope is not None:
                    self.fail(expr.token, "Random draws in initialization/collection expressions are not supported", "GAMAX_RANDOM")
                if function == "flip":
                    if len(types) != 1 or types[0] not in ("int", "float"): self.fail(expr.token, "flip requires one numeric probability", "GAMAX_TYPE")
                    expr.type = "bool"
                else:
                    if len(types) not in (1, 2) or len(set(types)) != 1 or types[0] not in ("int", "float"):
                        self.fail(expr.token, "rnd requires one or two same-type numeric bounds", "GAMAX_TYPE")
                    if not all(arg.op == "literal" for arg in expr.args): self.fail(expr.token, "rnd bounds must be literals in this profile", "GAMAX_RANDOM")
                    bounds = [0, expr.args[0].args[0]] if len(types) == 1 else [a.args[0] for a in expr.args]
                    if bounds[0] > bounds[1]: self.fail(expr.token, "rnd bounds must be ordered", "GAMAX_RANDOM")
                    expr.type = types[0]
                return
            arity = 2 if function in ('min','max') else 1
            if function in ('min','max') and len(types) == 1 and types[0].startswith('values:'): arity = 1
            if len(types) != arity: self.fail(expr.token, f'{function} requires {arity} operands', 'GAMAX_TYPE')
            if function in ('length','sum','mean') or function in ('min','max') and arity == 1:
                if not types[0].startswith(('agents:', 'values:')): self.fail(expr.token, 'Reducer requires a fixed collection', 'GAMAX_TYPE')
                if function != 'length' and types[0] not in ('values:int','values:float'): self.fail(expr.token, 'Numeric collection required', 'GAMAX_TYPE')
                expr.type = 'int' if function == 'length' else ('float' if function == 'mean' else types[0].split(':')[1])
            else:
                if any(t not in ('int','float') for t in types): self.fail(expr.token, 'Numeric scalar required', 'GAMAX_TYPE')
                expr.type = 'int' if function == 'int' else ('float' if function in ('float','sqrt','sin','cos','exp') else ('float' if 'float' in types else 'int'))
            return
        if "point" in types:
            if expr.op == "distance_to" and types == ["point", "point"]: expr.type = "float"; return
            if expr.op in ("+", "-") and types == ["point", "point"]: expr.type = "point"; return
            if expr.op == "*" and len(types) == 2 and all(t in ("point", "int", "float") for t in types) and types.count("point") == 1: expr.type = "point"; return
            if expr.op == "/" and types[0] == "point" and types[1] in ("int", "float"):
                if expr.args[1].op != "literal" or expr.args[1].args[0] == 0: self.fail(expr.token, "Point division requires a nonzero literal")
                expr.type = "point"; return
            self.fail(expr.token, "Unsupported point operation", "GAMAX_TYPE")
        if expr.op == "distance_to": self.fail(expr.token, "distance_to requires points", "GAMAX_TYPE")
        if any(t not in ('int','float','bool') for t in types): self.fail(expr.token, 'Collections must be reduced before arithmetic', 'GAMAX_TYPE')
        if expr.op in ("and", "or", "not_unary"):
            if any(t != "bool" for t in types): self.fail(expr.token, "Boolean operands required", "GAMAX_TYPE")
            expr.type = "bool"
        elif expr.op in ("=", "!="):
            if "float" in types:
                self.fail(expr.token, "GAMA floating equality has special rounding semantics; not lowered in this profile", "GAMAX_UNSUPPORTED")
            if "bool" in types and types[0] != types[1]: self.fail(expr.token, "Incompatible equality operands", "GAMAX_TYPE")
            expr.type = "bool"
        else:
            if "bool" in types: self.fail(expr.token, "Numeric operands required", "GAMAX_TYPE")
            if expr.op in ('div','mod') and types != ['int','int']: self.fail(expr.token, 'Integer div/mod profile requires int operands', 'GAMAX_TYPE')
            if expr.op in ('/','div','mod'):
                denominator = expr.args[1]
                safe = denominator.op == 'literal' and denominator.args[0] != 0
                if denominator.op == 'call:max':
                    safe = any(a.op == 'literal' and a.args[0] > 0 for a in denominator.args)
                if not safe: self.fail(expr.token, 'Denominator must be a nonzero literal or max with a positive literal; zero-divisor semantics are not approximated')
            expr.type = "bool" if expr.op in ("<", ">", "<=", ">=") else ("float" if "float" in types or expr.op == '/' else "int")


class Generator:
    def __init__(self, ir, reverse_agents=False):
        self.ir = ir
        self.reverse_agents = reverse_agents
        self.agent_index = None
        self.locals = {}
        self.eval_indent = "    "
        self.active_mask = "jnp.asarray(True)"
        self.each_scope = None
        self.lines = ["# Automatically generated by GAMAX. Do not edit model dynamics.",
                      "import jax", "jax.config.update('jax_enable_x64', True)",
                      "import jax.numpy as jnp", "DTYPES = {'int': jnp.int32, 'bool': jnp.bool_, 'float': jnp.float64, 'point': jnp.float64}", ""]
        self.lines += [
            "JIT_OPTIONS = " + repr({'xla_cpu_enable_fast_math': False, **({'xla_disable_hlo_passes': 'fusion'} if any(f.type in ('float', 'point') for s in ir.scopes for f in s.fields) else {})}),
            '',
            'def ordered_sum(values, selected, dtype):',
            '    values = values.astype(dtype)',
            '    def accumulate(i, total):',
            '        return jax.lax.cond(selected[i], lambda t: jax.lax.optimization_barrier(t + values[i]), lambda t: t, total)',
            '    return jax.lax.fori_loop(0, values.shape[0], accumulate, jnp.asarray(0, dtype=dtype))',
            '',
        ]
        if ir.random_seed is not None:
            self.lines += [
                "def java_uniform(state):",
                "    mask = jnp.uint64((1 << 48) - 1)",
                "    first = (state * jnp.uint64(0x5DEECE66D) + jnp.uint64(11)) & mask",
                "    second = (first * jnp.uint64(0x5DEECE66D) + jnp.uint64(11)) & mask",
                "    bits = ((first >> jnp.uint64(22)) << jnp.uint64(27)) + (second >> jnp.uint64(21))",
                "    return second, bits.astype(jnp.float64) / jnp.asarray(1 << 53, jnp.float64)", "",
            ]
        self.map = []
        self.counter = 0
        self.scopes = {s.name: s for s in ir.scopes}

    def emit(self, text, token=None):
        self.lines.append(text)
        if token: self.map.append(dict(generated_line=len(self.lines), source_line=token.line, source_column=token.column, source_file=token.source))

    def expression(self, expr, scope):
        op = expr.op
        if op == "point":
            values = [self.expression(arg, scope) for arg in expr.args]
            return f"jnp.stack(jnp.broadcast_arrays({', '.join(values)}), axis=-1).astype(jnp.float64)"
        if op.startswith("component:"):
            value = self.expression(expr.args[0], scope)
            return f"({value})[..., {('x','y','z').index(op.split(':')[1])}]"
        if op == "literal": return f"jnp.asarray({expr.args[0]!r}, dtype=DTYPES[{expr.type!r}])"
        if op == "name":
            name = expr.args[0]
            if name in self.locals: return f"s[{self.locals[name][0]!r}]"
            if name in self.ir.populations:
                count = self.ir.populations[name]
                return f'(jnp.arange({count}, dtype=jnp.int32), jnp.ones(({count},), dtype=jnp.bool_))'
            if name == "index":
                return self.agent_index if self.agent_index is not None else f"jnp.arange({self.ir.populations[scope]}, dtype=jnp.int32)"
            owner = scope if any(f.name == name for f in self.scopes[scope].fields) else "global"
            value = f"s[{(owner+'.'+name)!r}]"
            return f"{value}[{self.agent_index}]" if owner != "global" and self.agent_index is not None else value
        if op in ('each','myself'):
            owner = self.each_scope if op == 'each' else scope
            name = expr.args[0]
            if name == 'index': return f'jnp.arange({self.ir.populations[owner]}, dtype=jnp.int32)'
            value = f's[{(owner+"."+name)!r}]'
            return value if op == 'each' else f'{value}[{self.agent_index}]'
        if op in ('where','collect'):
            container = self.expression(expr.args[0], scope)
            saved = self.each_scope
            self.each_scope = expr.args[0].type.split(':')[1]
            values = self.expression(expr.args[1], scope)
            self.each_scope = saved
            if op == 'where': return f'({container}[0], {container}[1] & ({values}))'
            return f'(jnp.broadcast_to({values}, {container}[1].shape), {container}[1])'
        if op in ("and", "or"):
            left = self.expression(expr.args[0], scope)
            self.counter += 1; name = f"boolean_{self.counter}"
            self.emit(f"{self.eval_indent}{name} = {left}")
            saved = self.active_mask
            self.active_mask = f"({saved}) & ({name})" if op == "and" else f"({saved}) & ~({name})"
            right = self.expression(expr.args[1], scope)
            self.active_mask = saved
            return f"({name} {'&' if op == 'and' else '|'} ({right}))"
        values = [self.expression(arg, scope) for arg in expr.args]
        if op.startswith('call:'):
            function = op.split(':')[1]
            if function in ("rnd", "flip"):
                self.counter += 1
                uniform = f"random_{self.counter}"
                self.emit(f"{self.eval_indent}s['$rng'], {uniform} = jax.lax.cond({self.active_mask}, java_uniform, lambda r: (r, jnp.asarray(0.0, jnp.float64)), s['$rng'])", expr.token)
                if function == "flip": return f"({uniform} < {values[0]})"
                low, high = ("jnp.asarray(0, jnp.float64)", values[0]) if len(values) == 1 else values
                if expr.type == "int": return f"(({low}).astype(jnp.int64) + jnp.trunc((({high}).astype(jnp.int64) + 1 - ({low}).astype(jnp.int64)).astype(jnp.float64) * {uniform}).astype(jnp.int64)).astype(jnp.int32)"
                return f"jax.lax.optimization_barrier(({low}) + jax.lax.optimization_barrier((({high}) - ({low})) * {uniform}))"
            if function == 'length': return f'jnp.sum({values[0]}[1], dtype=jnp.int32)'
            if function in ('sum','mean'):
                dtype = 'float' if function == 'mean' else expr.args[0].type.split(':')[1]
                total = f'ordered_sum({values[0]}[0], {values[0]}[1], DTYPES[{dtype!r}])'
                if function == 'sum': return total
                # GAMA numeric mean of an empty container is zero (checked natively).
                return f'(({total}).astype(jnp.float64) / jnp.maximum(1, jnp.sum({values[0]}[1], dtype=jnp.int32)).astype(jnp.float64))'
            if function in ('min','max') and len(values) == 1:
                dtype = expr.args[0].type.split(':')[1]
                bound = ('2147483647' if function == 'min' else '-2147483648') if dtype == 'int' else ("jnp.inf" if function == 'min' else "-jnp.inf")
                reduction = f'jnp.{function}(jnp.where({values[0]}[1], {values[0]}[0], {bound}))'
                return f'jnp.where(jnp.any({values[0]}[1]), {reduction}, 0).astype(DTYPES[{dtype!r}])'
            if function == 'int':
                # Java's narrowing conversion saturates overflow and maps NaN to zero.
                return f'jnp.nan_to_num(jnp.clip({values[0]}, -2147483648, 2147483647), nan=0).astype(jnp.int32)'
            if function == 'float': return f'({values[0]}).astype(jnp.float64)'
            if function in ('sin','cos'): return f'jnp.{function}(jnp.deg2rad(({values[0]}).astype(jnp.float64)))'
            if function in ('sqrt','exp'): return f'jnp.{function}(({values[0]}).astype(jnp.float64))'
            function = {'min':'minimum','max':'maximum'}.get(function,function)
            return f'jnp.{function}({", ".join(values)})'
        if op == "distance_to":
            difference = f"(({values[0]}) - ({values[1]}))"
            return f"jnp.sqrt(jnp.sum(({difference}) * ({difference}), axis=-1))"
        if expr.type == "point":
            a, b = values
            if expr.args[0].type != "point": a = f"jnp.expand_dims({a}, -1)"
            if expr.args[1].type != "point": b = f"jnp.expand_dims({b}, -1)"
            return f"jax.lax.optimization_barrier(({a}) {op} ({b}))"
        if op in ('mod','div'):
            # Widen before abs/division to handle Java MIN_VALUE safely.
            a, b = [f'({v}).astype(jnp.int64)' for v in values]
            quotient = f'((jnp.abs({a}) // jnp.abs({b})) * jnp.where(({a} < 0) != ({b} < 0), -1, 1))'
            return f'({quotient}).astype(jnp.int32)' if op == 'div' else f'({a} - {quotient} * {b}).astype(jnp.int32)'
        if op in ("not_unary", "-_unary"): return f"({'~' if op == 'not_unary' else '-'}{values[0]})"
        if op == '/': return f'jax.lax.optimization_barrier(({values[0]}).astype(jnp.float64) / ({values[1]}).astype(jnp.float64))'
        operator = {"and": "&", "or": "|", "=": "=="}.get(op, op)
        operation = f"({values[0]} {operator} {values[1]})"
        return f'jax.lax.optimization_barrier({operation})' if expr.type == 'float' else operation

    def mask(self, expression, indent):
        self.counter += 1
        name = f"mask_{self.counter}"
        self.emit(f"{indent}{name} = {expression}")
        return name

    def statements(self, statements, scope, mask, indent="    "):
        saved_locals = dict(self.locals)
        for statement in statements:
            self.eval_indent, self.active_mask = indent, mask
            if statement.kind.startswith("local:"):
                self.counter += 1
                key = f"$local{self.counter}"
                type_ = statement.kind.split(":")[1]
                value = self.expression(statement.value, scope)
                self.emit(f"{indent}s[{key!r}] = ({value}).astype(DTYPES[{type_!r}])", statement.token)
                self.locals[statement.target] = (key, type_)
            elif statement.kind == "block":
                self.statements(statement.body, scope, mask, indent)
            elif statement.kind == "assign" and statement.target in self.locals:
                key, type_ = self.locals[statement.target]
                value = self.expression(statement.value, scope)
                selected = f"jnp.expand_dims({mask}, -1)" if type_ == "point" else mask
                self.emit(f"{indent}s[{key!r}] = jnp.where({selected}, {value}, s[{key!r}]).astype(DTYPES[{type_!r}])", statement.token)
            elif statement.kind == "assign":
                owner = scope if any(f.name == statement.target for f in self.scopes[scope].fields) else "global"
                field = next(f for f in self.scopes[owner].fields if f.name == statement.target)
                key = f"{owner}.{field.name}"
                value = self.expression(statement.value, scope)
                if field.name == "location" and "moving" in self.scopes[scope].skills and self.agent_index is not None:
                    self.counter += 1; delta = f"location_delta_{self.counter}"; angle = f"location_angle_{self.counter}"
                    evaluated = f"location_value_{self.counter}"
                    self.emit(f"{indent}{evaluated} = {value}")
                    value = evaluated
                    self.emit(f"{indent}{delta} = ({value}) - s[{key!r}][{self.agent_index}]")
                    self.emit(f"{indent}{angle} = jnp.arctan2({delta}[1], {delta}[0]) * jnp.asarray(57.29577951308232, jnp.float64)")
                    self.emit(f"{indent}{angle} = jnp.mod(jnp.where({angle} < 0.0, {angle} + 360.0, {angle}), 360.0)")
                    heading_key = scope + ".heading"
                    self.emit(f"{indent}s[{heading_key!r}] = s[{heading_key!r}].at[{self.agent_index}].set(jnp.where(({mask}) & ~jnp.all(jnp.abs({delta}) <= 0.01), {angle}, s[{heading_key!r}][{self.agent_index}]))")
                old = f"s[{key!r}]"
                if owner != "global" and self.agent_index is not None:
                    old += f"[{self.agent_index}]"
                selected = f"jnp.expand_dims({mask}, -1)" if field.type == "point" else mask
                if field.name == "heading" and "moving" in self.scopes[scope].skills: value = f"jnp.mod({value}, 360.0)"
                update = f"jnp.where({selected}, {value}, {old}).astype(DTYPES[{field.type!r}])"
                if owner != "global" and self.agent_index is not None:
                    update = f"s[{key!r}].at[{self.agent_index}].set({update})"
                self.emit(f"{indent}s[{key!r}] = {update}", statement.token)
            elif statement.kind == "skill_move":
                self.counter += 1; suffix = str(self.counter)
                def attr(name): return self.expression(Expr("name", [name], "point" if name == "location" else "float", statement.token), scope)
                position, heading, speed = attr("location"), attr("heading"), attr("speed")
                destination, valid = "destination_" + suffix, "valid_" + suffix
                self.emit(f"{indent}{destination} = jax.lax.optimization_barrier({position} + jnp.stack([jax.lax.optimization_barrier({speed} * jnp.cos(jnp.deg2rad({heading}))), jax.lax.optimization_barrier({speed} * jnp.sin(jnp.deg2rad({heading}))), jnp.asarray(0.0)]))", statement.token)
                self.emit(f"{indent}{valid} = jnp.all(({destination}[:2] >= 0.0) & ({destination}[:2] <= 100.0))")
                self.emit(f"{indent}s[{(scope+'.real_speed')!r}] = s[{(scope+'.real_speed')!r}].at[{self.agent_index}].set(jnp.where({mask}, jnp.where({valid}, jnp.sqrt(jnp.sum(({destination} - {position}) ** 2)), 0.0), {attr('real_speed')}))")
                delta = "delta_" + suffix
                orientation = "orientation_" + suffix
                self.emit(f"{indent}{delta} = {destination} - {position}")
                self.emit(f"{indent}{orientation} = jnp.arctan2({delta}[1], {delta}[0]) * jnp.asarray(57.29577951308232, jnp.float64)")
                self.emit(f"{indent}{orientation} = jnp.mod(jnp.where({orientation} < 0.0, {orientation} + 360.0, {orientation}), 360.0)")
                self.emit(f"{indent}{orientation} = jnp.where(jnp.all(jnp.abs({delta}) <= 0.01), {heading}, {orientation})")
                self.emit(f"{indent}s[{(scope+'.heading')!r}] = s[{(scope+'.heading')!r}].at[{self.agent_index}].set(jnp.where({mask}, jnp.where({valid}, {orientation}, jnp.mod({heading} - 180.0, 360.0)), {heading}))")
                self.emit(f"{indent}s[{(scope+'.location')!r}] = s[{(scope+'.location')!r}].at[{self.agent_index}].set(jnp.where(({mask}) & {valid}, {destination}, {position}))")
            elif statement.kind == "if":
                condition = self.mask(self.expression(statement.value, scope), indent)
                yes = self.mask(f"({mask}) & {condition}", indent)
                no = self.mask(f"({mask}) & ~{condition}", indent)
                self.statements(statement.body, scope, yes, indent)
                self.statements(statement.alternative, scope, no, indent)
            elif statement.kind == "create":
                self.initialize(self.scopes[statement.target])
            elif statement.kind == 'ask':
                self.counter += 1
                function = f'ask_{self.counter}'
                self.emit(f'{indent}def {function}(iteration, state):', statement.token)
                self.emit(f'{indent}    s = dict(state)')
                saved = self.agent_index
                self.agent_index = 'ask_index'
                count = self.ir.populations[statement.target]
                self.emit(f'{indent}    ask_index = {count}-1-iteration' if self.reverse_agents else f'{indent}    ask_index = iteration')
                self.statements(statement.body, statement.target, mask, indent+'    ')
                self.emit(f'{indent}    return s')
                self.agent_index = saved
                self.emit(f'{indent}s = jax.lax.fori_loop(0, {count}, {function}, s)')

        for name in self.locals:
            if name in saved_locals: continue
            self.emit(f"{indent}s.pop({self.locals[name][0]!r})")
        self.locals = saved_locals

    def initialize(self, scope):
        for field in scope.fields:
            self.eval_indent, self.active_mask = "    ", "jnp.asarray(True)"
            value = self.expression(field.value, scope.name)
            if scope.name != "global": value = f"jnp.broadcast_to({value}, ({self.ir.populations[scope.name]}, 3))" if field.type == "point" else f"jnp.broadcast_to({value}, ({self.ir.populations[scope.name]},))"
            self.emit(f"    s[{(scope.name+'.'+field.name)!r}] = ({value}).astype(DTYPES[{field.type!r}])", field.token)
        self.statements(scope.init, scope.name, "jnp.asarray(True)")

    def generate(self):
        self.emit("def reset():"); self.emit("    s = {}")
        self.initialize(self.ir.scopes[0])
        if self.ir.random_seed is not None:
            # GAMA generateSeed(8) uses little-endian IEEE bits, GamaRNG reads big-endian.
            seed = abs(self.ir.random_seed)
            if seed < 1: seed *= 9223372036854775808.0
            native_seed = int.from_bytes(struct.pack("<d", seed), "big")
            state = (native_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
            self.emit(f"    s['$rng'] = jnp.asarray({state}, jnp.uint64)")
        self.emit("    return s"); self.emit("")
        self.emit("def step(state):"); self.emit("    s = dict(state)")
        for scope in self.ir.scopes:
            if scope.name == "global":
                for behavior in scope.behaviors:
                    self.eval_indent, self.active_mask = "    ", "jnp.asarray(True)"
                    guard = self.mask(self.expression(behavior.guard, scope.name), "    ")
                    self.statements(behavior.body, scope.name, guard)
                continue
            if not scope.behaviors: continue
            function = f"advance_{scope.name}"
            self.emit(f"    def {function}(iteration, state):")
            self.emit("        s = dict(state)")
            self.agent_index = "agent_index"
            count = self.ir.populations[scope.name]
            self.emit(f"        agent_index = {count}-1-iteration" if self.reverse_agents else "        agent_index = iteration")
            for behavior in scope.behaviors:
                self.eval_indent, self.active_mask = "        ", "jnp.asarray(True)"
                guard = self.mask(self.expression(behavior.guard, scope.name), "        ")
                self.statements(behavior.body, scope.name, guard, "        ")
            self.emit("        return s")
            self.agent_index = None
            self.emit(f"    s = jax.lax.fori_loop(0, {count}, {function}, s)")
        self.emit("    return s"); self.emit(""); self.emit("step_jit = jax.jit(step, compiler_options=JIT_OPTIONS)")
        return "\n".join(self.lines)+"\n"


def compile_model(source_path, output_dir):
    path = Path(source_path).resolve()
    source = path.read_text()
    ir, dependencies = load_model(path)
    generator = Generator(ir)
    code = generator.generate()
    # All validation precedes filesystem mutation.
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    module = output / "model_jax.py"
    module.write_text(code)
    manifest = dict(schema_version=1, compiler_version="0.3.0", compiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model=ir.name,
                    source=str(path), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    generated_sha256=hashlib.sha256(code.encode()).hexdigest(),
                    profile="fixed-population-actions-java-rng-moving-v0.3", experiments=ir.experiments,
                    populations=ir.populations, fields={f"{s.name}.{f.name}": f.type for s in ir.scopes for f in s.fields},
                    schedule=[f"{s.name}.{b.name}" for s in ir.scopes for b in s.behaviors],
                    effects={f"{s.name}.{b.name}": dict(reads=b.reads, writes=b.writes) for s in ir.scopes for b in s.behaviors},
                    numeric_profile="Java int32 wraparound and truncating div/mod; float64 division; source-order numeric reductions and floating arithmetic barriers; GAMA degree trigonometry; nonzero-divisor checks",
                    capabilities=['fixed species where/collect and scalar reductions', 'global reflex ask over fixed species in index order',
                                  'explicit coordinate/adjacency attributes; restricted moving profile; no built-in graph skills'],
                    jit_contract=dict(cpu_fast_math=False, floating_models_disable_hlo_fusion=any(f.type in ('float', 'point') for s in ir.scopes for f in s.fields),
                                      entrypoint='step_jit or outer jit with generated JIT_OPTIONS; arbitrary re-jitting may violate the numeric contract'),
                    limitations=["no native RNG, nested asks, runtime creation/death, imports, actions, spatial operations beyond the restricted moving profile, graph operations, floating equality or effectful experiments",
                                 "no collection reads during init, nested each bindings, variable-sized lists or unchecked division",
                                 "agent transitions read/write self attributes and declared globals; init writes are local",
                                 "species declaration order; ascending agent index; all source-order reflexes of one agent before the next; tested against pinned GAMA, not a full-language proof"])
    manifest['random_profile'] = None if ir.random_seed is None else dict(generator='java', seed=ir.random_seed,
        reseed='final global init; runtime only', draws='java.util.Random nextDouble, scalar rnd/flip, lazy guard/branch/and/or, source and agent order')
    if ir.random_seed is not None: manifest['capabilities'].append('native-compatible Java runtime rnd/flip with explicit reseeding')
    manifest['dependencies'] = dependencies
    common = Path(os.path.commonpath([str(Path(item['path']).parent) for item in dependencies]))
    manifest['source_bundle_entry'] = str(Path('sources') / path.relative_to(common))
    for item in dependencies:
        original = Path(item['path'])
        relative = Path('sources') / original.relative_to(common)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes())
        item['bundle_path'] = str(relative)
    manifest['spatial_profile'] = dict(world='default non-toroidal square [0,100]²', step_seconds=1, skills=['moving: do move without facets'], unsupported=['custom shapes/topologies', 'wander/goto', 'geometry neighborhoods', 'graphs/grids']) if any(scope.skills for scope in ir.scopes) else None
    manifest['capabilities'] += ['point coordinates/arithmetic/distance_to; restricted 2D moving']
    manifest['capabilities'] += ['recursive relative imports without overrides/inheritance', 'nonrecursive same-scope scalar actions and block locals']
    manifest['limitations'][0] = "no default Mersenne RNG or random initialization/collections, nested asks, runtime creation/death, inheritance/import overrides, spatial operations beyond the restricted moving profile, graph operations, floating equality or effectful experiments"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (output / "source_map.json").write_text(json.dumps(generator.map, indent=2)+"\n")
    (output / "source.gaml").write_text(source)
    (output / "ir.json").write_text(json.dumps(asdict(ir), indent=2)+"\n")
    return module, manifest


def load_model(path):
    """Resolve a dependency DAG once; preserve original per-file token locations."""
    path = Path(path).resolve()
    ordered, visiting, seen = [], [], set()
    def visit(current, via=None):
        if current in visiting:
            raise CompileError(current, via or Token("import", 1, 1), "Cyclic import dependency", "GAMAX_IMPORT")
        if current in seen: return
        if not current.is_file():
            raise CompileError(current, via or Token("import", 1, 1), "Imported source does not exist", "GAMAX_IMPORT")
        visiting.append(current)
        parser = Parser(current.read_text(), str(current))
        ir = parser.parse(validate=False)
        for relative, token in parser.imports:
            if Path(relative).is_absolute(): parser.fail(token, "Imports must be relative", "GAMAX_IMPORT")
            visit((current.parent / relative).resolve(), token)
        visiting.pop(); seen.add(current); ordered.append((current, parser, ir))
    visit(path)
    entry_parser = ordered[-1][1]
    merged = ModelIR(ordered[-1][2].name, [Scope("global", [], [], [])], [], {})
    for current, parser, ir in ordered:
        for scope in ir.scopes:
            if scope.name == "global":
                target = merged.scopes[0]
                if target.init and scope.init: parser.fail(Token("init", 1, 1, str(current)), "Multiple imported init blocks need explicit override semantics", "GAMAX_IMPORT")
                for attribute in ("fields", "behaviors", "actions"):
                    existing = {member.name for member in getattr(target, attribute)}
                    for member in getattr(scope, attribute):
                        if member.name in existing: parser.fail(member.token, "Imported member override is unsupported", "GAMAX_IMPORT")
                        getattr(target, attribute).append(member); existing.add(member.name)
                target.init += scope.init
            else:
                if any(s.name == scope.name for s in merged.scopes): parser.fail(Token(scope.name, 1, 1, str(current)), "Species override/inheritance is unsupported", "GAMAX_IMPORT")
                merged.scopes.append(scope)
        for species, count in ir.populations.items():
            if species in merged.populations: parser.fail(Token(species, 1, 1, str(current)), "Duplicate imported creation", "GAMAX_IMPORT")
            merged.populations[species] = count
        for experiment in ir.experiments:
            if experiment in merged.experiments: parser.fail(Token(experiment, 1, 1, str(current)), "Experiment override is unsupported", "GAMAX_IMPORT")
            merged.experiments.append(experiment)
    entry_parser.populations = merged.populations
    if set(merged.populations) != {s.name for s in merged.scopes[1:]}:
        entry_parser.fail(entry_parser.token, "Each declared species must be created exactly once in global init")
    entry_parser.expand_actions(merged); entry_parser.check(merged)
    dependencies = [dict(path=str(current), sha256=hashlib.sha256(current.read_bytes()).hexdigest()) for current, _, _ in ordered]
    return merged, dependencies
