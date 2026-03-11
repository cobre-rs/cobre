"""Smoke tests for the cobre Python extension module foundation.

These tests verify that the PyO3 extension module loads correctly and that
the top-level module and its empty sub-modules are importable. They are
intended to be run after `maturin develop --uv` installs the extension.

Run with:
    pytest crates/cobre-python/tests/
"""


def test_import_cobre() -> None:
    """Importing cobre must succeed without errors."""
    import cobre  # noqa: F401, PLC0415


def test_version() -> None:
    """cobre.__version__ must be a non-empty string."""
    import cobre  # noqa: PLC0415

    assert isinstance(cobre.__version__, str)
    assert len(cobre.__version__) > 0


def test_submodules_exist() -> None:
    """cobre.model, cobre.io, cobre.run, and cobre.results must be importable."""
    import cobre.io  # noqa: F401, PLC0415
    import cobre.model  # noqa: F401, PLC0415
    import cobre.results  # noqa: F401, PLC0415
    import cobre.run  # noqa: F401, PLC0415
