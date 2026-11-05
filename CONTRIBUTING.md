# Contributing

The `aucreshape` library is finished: bug reports and documentation fixes are welcome, but new
features are unlikely to be added. Its evaluation is being redone; the code is in
`benchmarks/`.

## Development setup

You need [uv](https://docs.astral.sh/uv/). Then:

```bash
git clone https://github.com/ibouftini/AUCReshaping && cd aucreshape
uv sync --all-groups            # creates .venv with the package and all dev tools
uv run pre-commit install       # optional: run format and lint on every commit
```

Development uses CPU-only PyTorch wheels to keep installs small; this does not affect the
package's own dependency on `torch>=2.1`.

## Checks (all must pass before a PR is merged)

```bash
uv run ruff format .            # format
uv run ruff check . --fix       # lint
uv run mypy src                 # strict type check
uv run pytest -m "not slow"     # fast tests
uv run pytest                   # full suite with the 90% coverage gate
uv build                        # wheel and sdist
```

Test markers: `slow` (torch.compile, long runs), `gpu` (skipped without CUDA), `distributed`
(2-process gloo on CPU) and `examples` (runs `examples/`).

## Conventions

- **Tests first.** Add a failing test, then the code. Never weaken a test to make it pass.
- **Public API** needs full type hints and a NumPy-style docstring that states tensor shapes.
  `tests/unit/test_public_api.py` pins the public API; change it only on purpose.
- **Losses** must never produce NaN/inf for finite logits and must keep gradients flowing.
  No host syncs (`.item()`, `.cpu()`, Python `if` on tensors) in a forward pass.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`). One topic branch per change;
  PRs are squash-merged into `main`.
- Update `CHANGELOG.md` under *Unreleased* when behaviour changes.
