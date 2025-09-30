import streamlit as st
from rag_client import ask_rag

# Page config
st.set_page_config(
    page_title="GrokGPT",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title
st.markdown(
    "<h1 style='text-align: center;'>GrokGPT</h1>",
    unsafe_allow_html=True
)

# Initialize conversation history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# Input area
def handle_ask():
    query = st.session_state.get("query", "")
    if query.strip():
        st.session_state["history"].append({"role": "user", "content": query})
        answer = ask_rag(query, st.session_state["history"])
        if isinstance(answer, dict):
            st.session_state["history"].append({"role": "assistant", "content": answer.get("answer", "")})
        else:
            st.session_state["history"].append({"role": "assistant", "content": answer})
    else:
        st.warning("Please enter a query first!")

# Input area
query = st.text_input(
    "Ask anything about your application!",
    key="query",
    on_change=handle_ask
)

# Buttons row: always center the "Clear Chat" button
with st.container():
    _, center_col, _ = st.columns([4, 1, 4])
    with center_col:
        clear_clicked = st.button("Clear Chat", key="clear_button")

# Handle button actions
if clear_clicked:
    st.session_state["history"] = []

# Display conversation history: latest turn at top, user first then assistant
for i in range(len(st.session_state["history"]) - 1, -1, -2):
    if i - 1 >= 0:
        user_msg = st.session_state["history"][i - 1]
        assistant_msg = st.session_state["history"][i]
        st.markdown(
            f"<span style='color:#1f77b4;'><b>You:</b> {user_msg['content']}</span>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<span style='color:#9467bd;'><b>Assistant:</b> {assistant_msg['content']}</span>",
            unsafe_allow_html=True
        )
    else:
        msg = st.session_state["history"][i]
        if msg["role"] == "user":
            st.markdown(
                f"<span style='color:#1f77b4;'><b>You:</b> {msg['content']}</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<span style='color: green; font-weight: bold;'>Assistant:</span> {answer}",
                unsafe_allow_html=True
            )
