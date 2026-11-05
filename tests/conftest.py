from __future__ import annotations

import pytest
import torch


@pytest.fixture
def seed() -> int:
    torch.manual_seed(0)
    return 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA is not available on this machine")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
