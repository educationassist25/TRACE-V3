"""compat.py - run the Integrix app on any Streamlit >= 1.40.

Newer Streamlit releases added keyword arguments the app uses (expander key/on_change, button width,
popover width, text_input icon, ...). On an older Streamlit those raise TypeError and the page does not load.
`patch_streamlit()` wraps the affected functions so unsupported keyword arguments are dropped (and
width="stretch" becomes use_container_width=True where only that exists). Behaviour degrades gracefully: for
example the one-step-open accordion in the setup panel needs a recent Streamlit, the page still works without it.
"""
from __future__ import annotations

import functools
import inspect

WRAPPED = ("expander", "popover", "button", "download_button", "text_input", "toggle", "plotly_chart", "image",
           "container", "selectbox", "multiselect", "radio", "slider", "select_slider", "number_input",
           "file_uploader", "checkbox", "color_picker", "segmented_control", "form_submit_button")


def _params(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    # Only named parameters count. A catch-all **kwargs (e.g. st.plotly_chart on Streamlit <1.51 forwards
    # them to Plotly and warns "keyword arguments have been deprecated") must not receive our new options.
    return {n for n, p in sig.parameters.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def _wrap(fn):
    if getattr(fn, "_trace_compat", False):
        return fn
    allowed = _params(fn)
    if allowed is None:
        return fn

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if "width" in kwargs and "width" not in allowed:
            w = kwargs.pop("width")
            if w == "stretch" and "use_container_width" in allowed:
                kwargs["use_container_width"] = True
        for k in [k for k in kwargs if k not in allowed]:
            kwargs.pop(k)
        return fn(*args, **kwargs)

    inner._trace_compat = True
    return inner


def patch_streamlit(st) -> dict:
    """Wrap st.<name> and DeltaGenerator.<name>. Returns which newer features the installed Streamlit has."""
    from streamlit.delta_generator import DeltaGenerator
    for name in WRAPPED:
        if hasattr(DeltaGenerator, name):
            setattr(DeltaGenerator, name, _wrap(getattr(DeltaGenerator, name)))
        if hasattr(st, name):
            setattr(st, name, _wrap(getattr(st, name)))
    exp = _params(getattr(st.expander, "__wrapped__", st.expander)) or set()
    tabs = _params(st.tabs) or set()
    return {"expander_state": "key" in exp and "on_change" in exp, "lazy_tabs": "on_change" in tabs,
            "segmented_control": hasattr(st, "segmented_control")}
