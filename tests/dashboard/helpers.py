from streamlit.testing.v1 import AppTest


def run(page, root, monkeypatch):
    monkeypatch.setenv("WARDWATCH_DATA_ROOT", str(root))
    return AppTest.from_file(page, default_timeout=60).run()
