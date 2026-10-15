# GAMAX

GAMAX automatically compiles a supported subset of GAML into standalone
Python/JAX simulations. It preserves ordered effects and native-compatible
randomness within its documented language contract. JaxMARL integration and
GPU validation are outside the evaluated scope of this artifact.

## Reproduce the experiments

Start with the **[step-by-step reproduction tutorial](docs/reproduction.md)**.
It explains installation of the pinned official GAMA 2025.06.4 distribution,
fresh coupled validation, CPU comparisons with and without common rendering,
and analysis of the results. No previous local experiment archives are required
for this workflow.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/install_gama.py

# Plan only: no simulations or output files.
.venv/bin/python scripts/reproduce.py \
  --cases traffic --seeds 1 --repeats 1 --regime cold \
  -o results/first-run

# Add --execute to actually run the displayed experiment.
```

The six development models are spatial SIR, Schelling, Boids, one-road traffic,
predator–prey and ant foraging. The default full reproduction uses seeds 1–10,
100 cycles and three repetitions; these are exploratory configurations, not an
exhaustive validation of GAML. Reserved variants are not compiled by this workflow.
A common offscreen renderer is used, rather than the native GAMA GUI.

## Documentation

- [Reproduction tutorial](docs/reproduction.md)
- [Language contract and restrictions](docs/gaml_language_contract.md)
- [Control-flow semantics](docs/compiler_control_flow.md)
- [Optimizer](docs/optimizer.md)
- [Artifact contents and historical-data limits](docs/anonymous_artifact.md)
- [Checks performed on the reproduction workflow](docs/reproduction_preparation_checks.md)

Curated native regression fixtures are included under `tests/fixtures/native/`.
Selected historical scalar measurements are under `benchmarks/published_results/`.
They do not replace the complete raw historical archives.

```bash
.venv/bin/python -m pytest -q tests/test_reproduction.py
python3 scripts/check_published_results.py
python3 scripts/audit_anonymous_repo.py
```

The manuscript, author notes, Grid’5000 handbook/profiles, local installations
and raw output archives are intentionally excluded from publication. Git history
and commit identities require separate review before publishing anonymously.
