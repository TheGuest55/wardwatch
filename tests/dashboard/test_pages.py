"""AppTest smoke: every page renders without an exception, with data and with an empty data root."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = sorted(str(p) for p in [Path("src/dashboard/app.py"), *Path("src/dashboard/pages").glob("*.py")])


def run(page, root, monkeypatch):
    monkeypatch.setenv("WARDWATCH_DATA_ROOT", str(root))
    at = AppTest.from_file(page, default_timeout=60).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_data(page, preds_root, monkeypatch):
    at = run(page, preds_root, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_empty_root(page, tmp_path, monkeypatch):
    at = run(page, tmp_path, monkeypatch)
    assert not at.exception, [e.value for e in at.exception]


def test_overview_headline_values(preds_root, monkeypatch):
    at = run("src/dashboard/app.py", preds_root, monkeypatch)
    heads = [d.value for d in at.dataframe]
    flat = " ".join(str(h.to_dict()) for h in heads)
    assert "0.920" in flat and "xgboost" in flat and "not available" in flat
