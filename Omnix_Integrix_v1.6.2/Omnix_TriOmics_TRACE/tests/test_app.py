"""Smoke tests of the Streamlit app in every data-source mode (no files uploaded yet, demos)."""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_modes_load_without_errors():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    for mode in ("Upload my data", "Demo: partially overlapping", "Demo: paired tri-omics"):
        at.sidebar.radio[0].set_value(mode).run()
        assert not at.exception, (mode, [e.message for e in at.exception])
