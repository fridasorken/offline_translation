from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))


@pytest.fixture(autouse=True)
def reset_app_state() -> Iterator[None]:
    try:
        from app.main import app
    except Exception:
        yield
        return

    for attr in ("acronyms", "input_language", "translator"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)

    yield

    for attr in ("acronyms", "input_language", "translator"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
