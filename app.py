import streamlit as st

pages = {
    "CapitalHumain Agents": [
        st.Page("main_app.py", title="Langgraph : Multi Agents"),
        st.Page("single_app.py", title="Single Agent"),

    ],
    }

pg = st.navigation(pages)
pg.run()