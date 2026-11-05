from __future__ import annotations

import aucreshape


def test_import_and_version():
    assert isinstance(aucreshape.__version__, str)
    assert aucreshape.__version__ != ""


def test_all_is_defined():
    assert isinstance(aucreshape.__all__, list)


OPTIONAL = {"aucreshape.integrations.lightning": "lightning"}


def test_every_module_imports():
    import importlib
    import importlib.util
    import pkgutil

    for module in pkgutil.walk_packages(aucreshape.__path__, prefix="aucreshape."):
        framework = OPTIONAL.get(module.name)
        if framework and importlib.util.find_spec(framework) is None:
            continue  # optional integration; its import error is tested separately
        importlib.import_module(module.name)


def test_core_import_needs_only_torch():
    import subprocess
    import sys

    code = (
        "import sys; import aucreshape; "
        "bad = [m for m in ('lightning', 'transformers', 'sklearn') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
