"""Entry point: Run loop (evaluation) and the Loop Debug Console.

    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="Self-healing Text2SQL", page_icon=":material/autorenew:", layout="wide")

page = st.navigation([
    st.Page("app_pages/run_loop.py", title="运行 Loop", icon=":material/play_circle:", default=True),
    st.Page("dashboard.py", title="Loop Debug Console", icon=":material/monitoring:"),
], position="top")
page.run()
