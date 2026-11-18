"""Shared Streamlit helpers: cached loading, the sidebar view switch, 'not available' notices and the
provenance expander. Pages stay thin; every number comes from loaders/metrics_view."""
import streamlit as st

from . import config, loaders, metrics_view as mv

VIEWS = {"All rows": "all", "Attacker host excluded": "noatk"}


@st.cache_data(show_spinner=False)
def _records(root):
    r = loaders.load_records(root)
    return r.data, r.reason


def page(title):
    st.set_page_config(page_title=f"WardWatch · {title}", layout="wide")
    st.title(title)
    root = str(config.data_root())
    with st.sidebar:
        st.markdown("**WardWatch** results dashboard (read-only)")
        view = VIEWS[st.radio("Rows scored", list(VIEWS), key="view",
                              help="Every metric is shown on all rows and with the attacker host's rows removed.")]
        st.caption(f"data root: `{root}`")
    df, reason = _records(root)
    return root, view, df, reason


def na(reason, what=None):
    st.info(f"{what + ': ' if what else ''}not available. {reason}")


def provenance(df, label="Provenance: records behind this page", only=None):
    if df is None:
        return
    d = df if only is None else df[df["model"].isin(only)]
    with st.expander(label):
        st.caption("`eligible` = counted in headline tables (for_report, an A/B/C/D experiment, not smoke, "
                   "not a sensitivity analysis, not a latency timing, has a seed). Excluded records are listed with the reason.")
        st.dataframe(mv.provenance_table(d), use_container_width=True, hide_index=True)
        for p in df.attrs.get("problems", []):
            st.warning(p)


def derived_note(src=None):
    st.caption("Derived from this run's saved predictions (not a logged metric)"
               + (f": `{src}`" if src else "") + ".")
