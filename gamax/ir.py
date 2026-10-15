"""Typed, source-located ordered intermediate representation."""
from __future__ import annotations
from dataclasses import dataclass, field

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
    metadata: dict = field(default_factory=dict)


@dataclass
class Field:
    name: str
    type: str
    value: Expr
    token: Token
    facets: dict[str, Expr] = field(default_factory=dict)
    capacity: int | None = None
    list_mutable: bool = False
    getter: Expr | None = None
    getter_random: bool = False


@dataclass
class Statement:
    kind: str
    target: str | None
    value: Expr | None
    body: list
    alternative: list
    token: Token
    metadata: dict = field(default_factory=dict)


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
    virtual: bool = False
    phases: list[str] = field(default_factory=list)
    presentation_only: bool = False
    return_type: str | None = None


@dataclass
class Scope:
    name: str
    fields: list[Field]
    init: list[Statement]
    behaviors: list[Behavior]
    actions: list[Action] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    grid: dict | None = None
    aspects: dict = field(default_factory=dict)
    parent: str | None = None
    parent_token: Token | None = None
    fsm_states: list[Behavior] = field(default_factory=list)


@dataclass
class ModelIR:
    name: str
    scopes: list[Scope]
    experiments: list[str]
    populations: dict[str, int]
    random_seed: float | None = None
    random_initialization: bool = False
    random_generator: str = "java"
    random_implicit: bool = False
    capacities: dict[str, int] = field(default_factory=dict)
    presentation: dict = field(default_factory=dict)
    fixed_grid_parameters: dict = field(default_factory=dict)
    world_torus: bool = False
    world_torus_explicit: bool = False
    reference_aliases: dict = field(default_factory=dict)
    assets: list = field(default_factory=list)
    presentation_resources: list = field(default_factory=list)


