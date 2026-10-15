"""Freeze sampled initial data in GAML; contains no Python transition dynamics."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ('forest_fire', 'schelling', 'sir_grid', 'sir_network', 'granovetter',
         'traffic', 'boids', 'social_force', 'predator_prey', 'ants')


def small_world(rng, n=32, probability=0.2):
    """Watts-Strogatz ring (k=4): rewire outgoing edges, avoiding duplicates/self loops."""
    endpoints = [[(i + 1) % n, (i + 2) % n] for i in range(n)]
    edges = {tuple(sorted((i, j))) for i, row in enumerate(endpoints) for j in row}
    for i, row in enumerate(endpoints):
        for slot, previous in enumerate(list(row)):
            if rng.random() >= probability: continue
            available = [j for j in range(n) if j != i and tuple(sorted((i, j))) not in edges]
            if not available: continue
            target = int(rng.choice(available))
            edges.remove(tuple(sorted((i, previous))))
            edges.add(tuple(sorted((i, target))))
            row[slot] = target
    return endpoints


def initialization(rows):
    def literal(value):
        if isinstance(value, (bool, np.bool_)): return 'true' if value else 'false'
        if isinstance(value, (int, np.integer)): return str(int(value))
        return format(float(value), '.17f')
    lines = ['init {']
    for index, row in enumerate(rows):
        assignments = ' '.join(f'{key} <- {literal(value)};' for key, value in row.items())
        lines.append(f'    if index = {index} {{ {assignments} }}')
    return '\n'.join([*lines, '}'])


def materialize(case, seed):
    if case not in CASES: raise ValueError(f'Unknown case {case}')
    # Every engine receives the identical frozen GAML, not matching integer RNG seeds.
    rng = np.random.default_rng(seed)
    source = (ROOT / 'models' / f'{case}.gaml').read_text()
    source = source.replace('int scenario <- 0;', f'int scenario <- {seed};')
    rows = []
    if case == 'forest_fire':
        states = np.where(rng.random(64) < 0.75, 1, 0)
        states[rng.choice(np.flatnonzero(states == 1), size=2, replace=False)] = 2
        rows = [dict(status=int(v), previous=int(v)) for v in states]
    elif case == 'schelling':
        sites = rng.choice(64, size=32, replace=False)
        groups = rng.permutation([0] * 16 + [1] * 16)
        rows = [dict(px=int(p % 8), py=int(p // 8), group_id=int(g)) for p, g in zip(sites, groups)]
    elif case in ('sir_grid', 'sir_network', 'granovetter'):
        count = 64 if case == 'sir_grid' else 32
        infected = set(rng.choice(count, size=3, replace=False).tolist())
        graph = small_world(rng) if count == 32 else None
        for i in range(count):
            row = {} if graph is None else dict(link_a=graph[i][0], link_b=graph[i][1])
            if case == 'granovetter':
                row.update(threshold=float(rng.uniform(0.1, 0.8)), adopted=i in infected, previous=i in infected)
            else: row.update(status=int(i in infected), previous=int(i in infected))
            rows.append(row)
    elif case == 'traffic':
        sites = sorted(rng.choice(128, size=24, replace=False).tolist())
        rows = [dict(position=p, old_position=p, velocity=int(rng.integers(0, 6))) for p in sites]
    elif case in ('boids', 'social_force'):
        for _ in range(24):
            px, py = rng.uniform(5, 15, 2) if case == 'boids' else (rng.uniform(10, 17), rng.uniform(6, 14))
            vx, vy = rng.uniform(-0.5, 0.5, 2) if case == 'boids' else (0.0, 0.0)
            row = dict(px=float(px), py=float(py), ox=float(px), oy=float(py), vx=float(vx), vy=float(vy))
            if case == 'boids': row.update(ovx=float(vx), ovy=float(vy))
            rows.append(row)
    elif case == 'ants':
        rows = [dict(px=int(rng.integers(0, 8)), py=int(rng.integers(0, 8))) for _ in range(24)]
    elif case == 'predator_prey':
        # Two separate initializations, still the same generic species-init syntax.
        for count, active_count in ((32, 16), (12, 4)):
            active = set(rng.choice(count, size=active_count, replace=False).tolist())
            population = []
            for i in range(count):
                x, y, energy = int(rng.integers(0, 8)), int(rng.integers(0, 8)), int(rng.integers(4, 10))
                population.append(dict(px=x, py=y, ox=x, oy=y, energy=energy, old_energy=energy,
                                       active=i in active, old_active=i in active))
            source = source.replace('// @initialization', initialization(population), 1)
        return source
    return source.replace('// @initialization', initialization(rows))
