import asyncio
import uuid
from functools import wraps

import streamlit as st
APP_TITLE = "Capital Humain"
APP_ICON_PATH = "images/deer.png"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON_PATH,
)


from langchain_core.messages import AIMessage, HumanMessage
from streamlit.runtime.scriptrunner import get_script_run_ctx

from utils.schema import show_questions_in_sidebar, show_schema_in_sidebar
from utils.sidebar import enable_login
from utils.sidebar_search import search_documents, search_questions


# Cache page configuration and context data
def get_context_data():
    ctx = get_script_run_ctx()

    pages_dict = ctx.pages_manager.get_pages()
    current_page = ctx.pages_manager.intended_page_name

    # Find page hashes
    main_app_hash = None
    single_app_hash = None
    for key, value in pages_dict.items():
        if value["page_name"] == "Langgraph : Multi Agents":
            main_app_hash = value["page_script_hash"]
        elif value["page_name"] == "Single Agent":
            single_app_hash = value["page_script_hash"]

    return {
        "ctx": ctx,
        "current_page": current_page,
        "main_app_hash": main_app_hash,
        "single_app_hash": single_app_hash,
        "current_hash": ctx.page_script_hash,
    }


# Constants (kept outside of functions for cleaner code)
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
USER_AVATAR_PATH = "images/avataruser.png"
DEBUGGING = st.secrets.get("DEBUGGING", False)


# Initialize session state variables
@st.cache_resource
def initialize_session_state():
    if "initialized" not in st.session_state:
        # Initialize message arrays if they don't exist
        if "single_messages" not in st.session_state:
            st.session_state["single_messages"] = [AIMessage(content=WELCOME_MESSAGE)]

        if "messages" not in st.session_state:
            st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]

        if "thread_id" not in st.session_state:
            st.session_state["thread_id"] = str(uuid.uuid4())

        st.session_state["initialized"] = True


# Reset chat - implemented as a fragment for efficiency
@st.fragment
def reset_chat_history():
    st.session_state["single_messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["thread_id"] = str(uuid.uuid4())

    # Clean up specific states
    for key in ["in_human_feedback_state", "schema_template"]:
        if key in st.session_state:
            del st.session_state[key]


# Toggle debug panel - implemented as a fragment
@st.fragment
def toggle_debug_panel():
    """Toggle the visibility of the debug panel."""
    st.session_state["show_debug_panel"] = not st.session_state.get(
        "show_debug_panel", False
    )


# Async wrapper for streamlit
def to_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


# Handle example questions as a fragment for better reactivity
@st.fragment
def process_example_question():
    if "_example_question" in st.session_state:
        ctx_data = get_context_data()
        example_q = st.session_state["_example_question"]

        if (
            ctx_data["ctx"].page_script_hash == ctx_data["single_app_hash"]
            or ctx_data["current_page"] == "single_app"
        ):
            st.session_state["single_messages"].append(HumanMessage(content=example_q))
        else:
            st.session_state["messages"].append(HumanMessage(content=example_q))

        del st.session_state["_example_question"]


# Cached sidebar info based on current page
def get_sidebar_info(page_hash, single_app_hash, current_page):
    print(f"Page Hash: {page_hash}, Single App Hash: {single_app_hash}, Current Page: {current_page}")
    if page_hash == single_app_hash or single_app_hash == None or current_page == "single_app":
        return "Chatbot utilisant un single agent pour l'interrogation du dataset 'Capital Humain' et la réponse aux questions posées."
    else:
        return "Chatbot utilisant une architecture multi-agent LangGraph avec human-in-the-loop pour l'interrogation du dataset 'Capital Humain' et la visualisation des workflows sous forme de graphs."


# Main function to run the app
def main():
    # Initialize session state
    initialize_session_state()

    # Get context data (cached)
    ctx_data = get_context_data()

    # Display model selector (assuming this is lightweight or already optimized)

    # Process example questions if they exist
    process_example_question()

    # Create sidebar
    with st.sidebar:
        if not DEBUGGING:
            if (
                ctx_data["ctx"].page_script_hash == ctx_data["single_app_hash"]
                or ctx_data["current_page"] == "single_app"
            ):
                enable_login("single_messages")
            else:
                enable_login()

        # Get and display the appropriate sidebar info
        sidebar_info = get_sidebar_info(
            ctx_data["current_hash"],
            ctx_data["single_app_hash"],
            ctx_data["current_page"],
        )
        st.info(sidebar_info)
        # Reset chat button
        st.button(
            "Nouveau chat",
            on_click=reset_chat_history,
            icon=":material/edit_square:",
            use_container_width=True,
        )
        # Debug panel toggle
        if DEBUGGING:
            st.button(
                "Show Session State",
                on_click=toggle_debug_panel,
                key="debug_button",
                use_container_width=True,
            )

            # Show debug panel if enabled
            if st.session_state.get("show_debug_panel", False):
                st.write("### Session State Contents")
                for key in sorted(st.session_state.keys()):
                    with st.expander(f"Key: {key}"):
                        st.write(st.session_state[key])

        # Search questions
        st.header("Questions répondables")
        search_questions()
        search_documents()

        # Show schema in sidebar
    if DEBUGGING:
        show_schema_in_sidebar()
    show_questions_in_sidebar()

    # Page navigation
    pages = {
        "CapitalHumain Agents": [
            st.Page("single_app.py", title="Single Agent"),
            st.Page("main_app.py", title="Langgraph : Multi Agents"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()


# Run the app
if __name__ == "__main__":
    main()
