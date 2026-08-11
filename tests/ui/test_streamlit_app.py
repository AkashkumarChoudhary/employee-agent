from pathlib import Path

from streamlit.testing.v1 import AppTest

# Streamlit >=1.50 resolves relative AppTest paths against the *calling* file,
# so the app path is derived from the repo root rather than the cwd.
APP = Path(__file__).resolve().parents[2] / "employee_agent" / "ui" / "streamlit_app.py"


def test_initial_screen_renders_without_network():
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception
    assert any("Employee Agent" in t.value for t in at.title)
