from __future__ import annotations

import os


def require_login(st):
    if "auth_role" not in st.session_state:
        st.session_state.auth_role = None
    if st.session_state.auth_role:
        with st.sidebar:
            st.success(f"Signed in as {st.session_state.auth_role}")
            if st.button("Log out", use_container_width=True):
                st.session_state.auth_role = None
                st.rerun()
        return st.session_state.auth_role
    st.title("AI Job Market Analytics")
    st.caption(
        "Academic demonstration login. Replace demo credentials before any shared deployment."
    )
    with st.form("login"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in", use_container_width=True)
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
            st.error("Invalid demonstration credential.")
    st.stop()
