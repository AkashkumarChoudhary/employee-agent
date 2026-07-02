from streamlit.testing.v1 import AppTest


def test_initial_screen_renders_without_network():
    at = AppTest.from_file("employee_agent/ui/streamlit_app.py").run()
    assert not at.exception
    assert any("Employee Agent" in t.value for t in at.title)
