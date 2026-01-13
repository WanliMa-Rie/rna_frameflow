# AGENTS.md — rna-backbone-design

This repo is a Python 3.11 project (SE(3) flow matching for RNA backbone design) using `uv` for env/deps, Hydra configs for experiments, and PyTorch/Lightning for training.

This file is guidance for agentic coding tools working in this repository.

## Quick start (environment)

- Create env + sync deps:
  - `pip install uv`
  - `uv venv`
  - `uv pip install hatchling`
  - `uv sync`
- Optional (per README) extra install for FlashAttention:
  - `uv add flash-attn --no-build-isolation`

Notes:
- `pyproject.toml` sets `requires-python = ">= 3.11,<3.12"`.
- `pyproject.toml` configures `uv` indexes/sources (PyTorch CUDA 12.1 index; `flash_ipa` via git).
- If dependency downloads time out, increase `UV_HTTP_TIMEOUT` and retry.

## Common commands

### Run scripts

Scripts are typically executed via `uv run` from repo root:
- Training:
  - `uv run train_se3_flows.py`
- Inference (sampling):
  - `uv run inference_se3_flows.py`
- Data preprocessing:
  - `uv run process_rna_pdb_files.py --pdb_dir data/rnasolo/ --write_dir data/rnasolo_proc/`

Hydra configs live in `configs/`:
- Main training config: `configs/config.yaml`
- Inference config: `configs/inference.yaml`

### Tests

Pytest settings are in `pyproject.toml:[tool.pytest.ini_options]`.

Run the full test suite:
- `uv run pytest`

Run a single test file:
- `uv run pytest tests/test_something.py`

Run a single test by node id (recommended):
- `uv run pytest tests/test_something.py::test_name`

Run tests matching a substring:
- `uv run pytest -k "substring"`

Run only “slow” tests (if any are marked):
- `uv run pytest -m slow`

Run with extra output / stop early:
- `uv run pytest -x -vv`

### Build / packaging

This project uses Hatchling (`pyproject.toml:[build-system]`).

Build wheel/sdist:
- `uv run python -m build` (if `build` is installed)
- Or via hatch (if installed): `uv run hatch build`

Editable install for local development:
- `uv pip install -e .`

### Lint / formatting

No formatter/linter configuration is currently checked into the repo (no `ruff.toml`, `pyproject.toml:[tool.ruff]`, `pyproject.toml:[tool.black]`, or `pre-commit`).

Guidance:
- If you want to introduce a linter/formatter, ask first and keep it minimal.
- If running in CI elsewhere, follow that configuration.

Pragmatic local checks (optional):
- Syntax/type sanity: `uv run python -m compileall rna_backbone_design`
- Import sanity: `uv run python -c "import rna_backbone_design"`

## Repository layout

- `rna_backbone_design/` — main package
  - `models/` — Lightning module + flow model components
  - `data/` — parsing, constants, atom representations, transforms
  - `analysis/` — metrics + `EvalSuite`
  - `tools/` — vendored/embedded third-party tools (RhoFold + gRNAde API wrappers)
- `configs/` — Hydra YAML configs
- `tests/` — (currently absent in this checkout)

## Code style guidelines (repo-specific)

### Imports

- Prefer standard grouping:
  1. Python stdlib
  2. Third-party libs (`torch`, `numpy`, `pandas`, `pytorch_lightning`, etc.)
  3. Local imports from `rna_backbone_design...`

- Keep imports explicit; avoid deep wildcard imports.
- If you alias local modules, use meaningful aliases (existing pattern: `utils as au`, `utils as du`, `utils as mu`).
- Avoid duplicate imports in the same file (there is at least one duplicate import in `rna_backbone_design/models/flow_module.py`; when touching files, clean duplicates locally).

### Formatting

- Follow PEP 8 with a pragmatic scientific-code bent:
  - 4-space indentation.
  - Keep lines ~88–100 chars when reasonable.
  - Use trailing commas in multi-line literals.

- This repo contains some non-PEP8 patterns (e.g., `import os, gc, json, torch`).
  - When editing such files, prefer to improve readability if changes are already being made, but avoid large unrelated reformatting.

### Types and tensors

- Use type hints for public functions and non-trivial utilities.
  - Common types used here: `Optional`, `Dict[str, Any]`, `List[...]`, `torch.Tensor`, `np.ndarray`.

- Tensor shape conventions:
  - Batch-first for ML: `[B, N, ...]`.
  - Use explicit variables like `num_batch`, `num_res` already common in the code.

- Conversions:
  - Convert tensors to numpy using `.detach().cpu().numpy()` (pattern exists as `to_numpy`).
  - Prefer `@torch.no_grad()` for evaluation utilities.

### Naming

- Modules/classes: `PascalCase` (`FlowModule`, `EvalSuite`).
- Functions/vars: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.

- Prefer semantically meaningful names over single-letter variables, except for very local math contexts.
- Config objects are commonly named `cfg` and are Hydra/OmegaConf objects.

### Configuration (Hydra/OmegaConf)

- Config entry points use YAML under `configs/`.
- Keep config keys consistent with existing structure:
  - `data_cfg`, `interpolant`, `model`, `experiment`, `inference`.

- Avoid “magic globals”; pass values through config where possible.
- Any new config option must have:
  - Sensible default in YAML.
  - Clear naming (no abbreviations unless well-established).

### Error handling and validation

- Prefer explicit validation early:
  - Raise `ValueError` for invalid user inputs/shapes.
  - Use `assert` only for internal invariants that should never fail (shape checks, assumptions).

- Avoid printing in library code.
  - Use logging where possible (see `rna_backbone_design/utils.py:get_pylogger`).
  - Within Lightning modules, use `self.log(...)` for metrics.

- NaN/inf handling:
  - Existing code sometimes prints and zeros NaN losses in training.
  - If you modify loss code, keep behavior stable unless explicitly asked.

### Logging

- For multi-GPU friendliness, use `get_pylogger()` when adding new loggers.
- In Lightning modules, prefer `self.log` and avoid repeated file I/O.

### I/O and paths

- Use `os.path.join` and `os.makedirs(..., exist_ok=True)`.
- Avoid writing large artifacts during unit tests.
- Keep paths configurable; avoid hardcoding absolute paths.

### Performance and GPU safety

- Use `.to(device)` once and re-use tensors on the same device.
- Prefer vectorized operations; avoid Python loops over residues unless unavoidable.
- Wrap eval-only code in `torch.no_grad()`.

## Vendored tools / third-party code

- `rna_backbone_design/tools/` contains embedded external code (e.g., RhoFold, gRNAde wrappers).
- When editing vendored code:
  - Prefer minimal patches.
  - Avoid wholesale refactors/reformatting.
  - Keep original API surface stable for downstream callers.

## Adding tests (if needed)

This checkout has `pytest` configuration but no `tests/` directory.

If you add tests:
- Put them under `tests/`.
- Name files `test_*.py`.
- Keep tests lightweight (no GPU requirement; avoid downloading models/data).
- Use small synthetic tensors and seed RNGs.

## Editor/agent rules

- Cursor rules: none found (`.cursor/rules/` and `.cursorrules` not present).
- Copilot instructions: none found (`.github/copilot-instructions.md` not present).

## Known gotchas

- Dependencies are heavy (PyTorch, torch-geometric, RDKit, flash-attn). Expect long `uv sync` times.
- `uv` uses custom indexes (`tool.uv.index`) and git sources (`tool.uv.sources`). Don’t replace those casually.
- Inference/eval may require external binaries (USalign/qTMclust/TMscore) and additional model checkpoints (RhoFold) as described in `README.md`.
