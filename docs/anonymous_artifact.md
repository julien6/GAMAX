# Anonymous repository scope

The publication tree contains compiler/runtime sources, tests and GAML fixtures,
benchmark source corpora and assets, reproduction/validation scripts, locked
Python dependencies and selected technical/reproduction documentation.
Third-party credits, citations and license notices remain intact.

## Local material excluded from Git

The complete `paper/` tree, `grid5000_llm_handbook.md`, Grid’5000 profiles,
internal roadmaps, historical narrative reports and paper-only export scripts
are excluded. The anonymous repository is an executable research artifact,
not a copy of the authors’ working directory.

Downloaded GAMA/JDK, virtual environments, emitted modules, raw `results/`,
deployment/import/control directories and editor settings remain local.
Nothing is deleted by these ignore rules. Raw historical archives are not
provided; the included scalar measurements allow checking ratios, and the
fresh reproduction workflow creates new independent output directories.
Do not describe this source repository as a complete raw v0 evidence archive.

## Review before committing

Run `python3 scripts/audit_anonymous_repo.py` for a read-only inspection of current
Git publication candidates, tracked ignored files, personal absolute paths and
LaTeX author declarations. It performs no simulation, remote connection or staging.
Review `git status --short` and the eventual staged diff. The audit is not a complete
anonymity guarantee: inspect binary metadata, links, acknowledgements and Git
identity/history too. Legitimate third-party attribution should not be removed.

The current `.git` history is local metadata and is not covered by `.gitignore`.
An anonymous remote name does not anonymize commit authors or earlier revisions.
Choose the commit identity and history that will be published before pushing;
no Git identity, commit, history or remote is changed by this cleanup.

The official ant SVG contains an upstream Openclipart author directory in its
metadata. Its reviewed, SHA-256-bound original bytes are retained in each corpus;
the audit reports this explicitly as upstream metadata rather than a project
author leak. Official source/asset digests are not changed.

## Fresh-clone reproduction

Use [the reproduction tutorial](reproduction.md) and `scripts/reproduce.py`.
Fresh coupled validation replaces the historical acceptance prerequisite for this
path. Curated native regression fixtures are published in `tests/fixtures/native/`
with SHA-256 provenance; selected infrastructure metadata is omitted explicitly.
Compact historical scalar measurements in `benchmarks/published_results/` permit
checking published batch ratios. Neither subset replaces all raw v0 archives.
