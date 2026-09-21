from __future__ import annotations

import os

from components.language import get_translator


def require_login(st):
    t = get_translator(st).text
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None
    if st.session_state.auth_role:
        with st.sidebar:
            st.success(t("auth.signed_in", role=t(f"roles.{st.session_state.auth_role}")))
            if st.button(t("auth.log_out"), width="stretch", key="logout"):
                st.session_state.auth_role = None
                st.rerun()
        return st.session_state.auth_role
    st.title(t("auth.title"))
    st.caption(t("auth.caption"))
    with st.form("login"):
        user = st.text_input(t("auth.username"), key="login_username")
        pwd = st.text_input(t("auth.password"), type="password", key="login_password")
        submit = st.form_submit_button(t("auth.sign_in"), width="stretch")
    admin_user = os.getenv("AIJOB_ADMIN_USER", "admin")
    admin_pwd = os.getenv("AIJOB_ADMIN_PASSWORD", "AIJob2026!")
    if submit:
        if user == admin_user and pwd == admin_pwd:
            st.session_state.auth_role = "admin"
            st.rerun()
        elif user == "user" and pwd == "user123":
            st.session_state.auth_role = "user"
            st.rerun()
        else:
            st.error(t("auth.invalid"))
    st.stop()
