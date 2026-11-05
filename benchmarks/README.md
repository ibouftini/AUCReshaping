# Evaluation code

The code behind the evaluation of `aucreshape`. The library in `src/` does not depend on it, and it is
not part of the installed package. The evaluation is being redone under a stricter protocol, fixed
before the new runs: [`PROTOCOL.md`](PROTOCOL.md).

```bash
uv sync --group test --group bench
uv run pytest benchmarks/tests              # the tests
uv run python -m benchmarks.review.smoke    # one tiny run of each pipeline, on generated data
uv run python -m benchmarks.review.rebuild  # every table from saved results, without training
```

Results are not stored here. `rebuild` skips any step whose saved results are absent, and says so. The
datasets are public and are not copied here.

The runs use private Kaggle kernels. To rerun them, set `KAGGLE_USER` to your own account.

| Folder or file | What it holds |
|---|---|
| `PROTOCOL.md` | The protocol of the chest X-ray study, fixed before its runs |
| `review/xray*.py` | The chest X-ray study: data preparation, training, coded runs and the analysis |
| `review/` | The synthetic laboratory, statistics, direct contrasts and the credit-card baseline |
| `replication.py` | The first real-data pipeline |
| `kaggle/` | Kernel generation and collection |
| `tests/` | Tests, including `test_protocol_pins.py`: the promises made to a sceptical reader |
| Other files | Earlier benchmarks |
