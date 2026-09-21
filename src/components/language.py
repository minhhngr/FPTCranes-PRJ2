"""Streamlit session integration for the pure resource-based translator."""

from __future__ import annotations

from functools import lru_cache

from ai_job_market.i18n import RESOURCE_DIR, Translator, load_catalog

LANGUAGE_KEY = "app_language"


@lru_cache(maxsize=4)
def _catalog(signature):
    # The signature invalidates cached resources when translations change on disk.
    return load_catalog(RESOURCE_DIR, strict=False)


def get_translator(st) -> Translator:
    """Read validated session language without global per-user state."""
    signature = tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(RESOURCE_DIR.glob("*.json"))
    )
    catalog = _catalog(signature)
    return Translator(catalog, st.session_state.get(LANGUAGE_KEY, "EN"))


def render_language_selector(st) -> Translator:
    """Render once in the entrypoint, including before the login gate."""
    translator = get_translator(st)
    # Normalize before constructing the widget, never after it is instantiated.
    st.session_state[LANGUAGE_KEY] = translator.language
    st.selectbox(
        translator.text("settings.language"),
        options=list(translator.catalog.languages),
        format_func=translator.catalog.languages.__getitem__,
        key=LANGUAGE_KEY,
    )
    return get_translator(st)
