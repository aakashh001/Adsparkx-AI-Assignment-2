"""
Bonus Streamlit chat UI.

Run with:
    streamlit run streamlit_app.py
"""
import uuid
import json
import streamlit as st

from app.agent import run_turn

st.set_page_config(page_title="Adsparkx Support Agent", page_icon="💬", layout="centered")
st.title("💬 Adsparkx Persona-Adaptive Support Agent")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### Session")
    st.code(st.session_state.session_id, language=None)
    if st.button("Start new session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

prompt = st.chat_input("Describe your issue...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = run_turn(st.session_state.session_id, prompt)

    meta_lines = [f"**Persona:** {result.persona}"]
    if result.retrieved_sources:
        srcs = ", ".join(f"{s['source']} ({s['section']})" for s in result.retrieved_sources)
        meta_lines.append(f"**Sources:** {srcs}")
    if result.escalated:
        meta_lines.append(f"🔴 **Escalated** — {result.escalation_reason}")
    else:
        meta_lines.append("🟢 Handled automatically")

    meta = "  \n".join(meta_lines)

    with st.chat_message("assistant"):
        st.markdown(result.response)
        st.caption(meta)
        if result.handoff_summary:
            with st.expander("Human handoff summary"):
                st.json(result.handoff_summary)

    st.session_state.messages.append({"role": "assistant", "content": result.response, "meta": meta})
