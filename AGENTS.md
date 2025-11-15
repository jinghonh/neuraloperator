# Repository Guidelines

## Project Structure & Module Organization
- `neuralop/` is the primary package; submodules (`data`, `layers`, `models`, `training`, `losses`) each ship their own `tests/` folder.
- `config/` holds YAML presets and `test_config.py` for training flows; keep custom configs versioned alongside them.
- `examples/` contains runnable notebooks, while `scripts/` hosts utilities such as `scripts/test_from_config.py`.
- Documentation sources live in `doc/` (Sphinx) and the disposable build output in `doc/build/html`.

## Build, Test, and Development Commands
- `python -m venv venv && source venv/bin/activate` — isolate dependencies before hacking.
- `pip install -e .[dev]` — install NeuralOperator in editable mode plus dev extras.
- `pytest neuralop -v` — run the full unit suite; prefer `pytest neuralop/models/tests/test_fno.py` when iterating on a specific module.
- `python scripts/test_from_config.py --config config/test_config.py` — verify configs still instantiate models, datasets, and trainers end-to-end.
- `black .` — apply the canonical formatting pass; rerun before every commit.
- `(cd doc && make html)` — build local docs when you touch anything under `doc/` or public APIs.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, 88-char lines (black default), and NumPy-style docstrings describing shapes and units. Modules, functions, and files use `snake_case`; classes and torch modules use `PascalCase`. Add type hints where they clarify tensor dimensions or configuration contracts.

## Testing Guidelines
Pytest is the only framework in use. New features must land alongside `test_*.py` cases placed in the same domain directory (e.g., `neuralop/layers/tests`). Assert both tensor shapes and numerical tolerances, cover failure modes (bad configs, device mismatches), and rerun `pytest neuralop` plus any dataset-specific smoke tests before opening a PR.

## Commit & Pull Request Guidelines
History favors short, imperative subjects with optional PR references (e.g., `Fix ICLoss (#683)`); keep commits atomic and feel free to add `feat:`/`fix:` prefixes for clarity. Every PR should describe motivation, list major changes, link related issues, and paste key command outputs (`pytest`, `black`, `make html`). Rebase on `main`, resolve conflicts locally, and wait for CI before requesting review.

## Configuration & Experiment Tracking
Configuration files use `zencfg`; extend existing YAMLs instead of cloning defaults. Store private Weights & Biases credentials in `neuraloperator/config/wandb_api_key.txt`, for example `echo "$WANDB_API_KEY" > neuraloperator/config/wandb_api_key.txt && chmod 600 neuraloperator/config/wandb_api_key.txt`. Treat dataset paths and artifact URIs as parameters so scripts remain portable across machines.
