import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client

from utils.graph import invoke_our_graph
from utils.show_db import show_schema_in_sidebar
from utils.st_callable_util import get_streamlit_cb

# Load environment variables
load_dotenv()

# LangSmith client configuration
# TODO
langsmith_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
)
langchain_tracer = LangChainTracer(
    client=langsmith_client,
)

# Application constants
APP_TITLE = "Capital Humain"
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
APP_ICON_PATH = "images/deer.png"
USER_AVATAR_PATH = "images/avataruser.png"
SYSTEM_PROMPT_PATH = "prompt_templates/full_prompt_no_answer_n_ask.txt"

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON_PATH,
)

st.logo(
    APP_ICON_PATH,
    icon_image=APP_ICON_PATH,
    size="large",
)


def initialize_session():
    """Initialize session state variables for chat history and configuration"""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]

    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())

    if "system_message" not in st.session_state:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as file:
            system_prompt_template = file.read()
        st.session_state.system_message = system_prompt_template


def reset_chat_history():
    """Reset chat history to initial state"""
    st.session_state.messages = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["thread_id"] = str(uuid.uuid4())


def setup_user_authentication():
    """Handle user authentication in sidebar"""
    with st.sidebar:
        if st.experimental_user.get("is_logged_in", True):
            if st.button("Log out"):
                st.logout()
            st.write(f"Hello, {st.experimental_user.get('email')}")
            st.write(f"Hello, {st.experimental_user.to_dict()}")
        else:
            if st.button("Log in with Google"):
                st.login()


def setup_sidebar_controls():
    """Configure sidebar elements and controls"""
    with st.sidebar:
        st.button(
            "Nouveau chat",
            on_click=reset_chat_history,
            icon=":material/edit_square:",
            use_container_width=True,
        )

        show_schema_in_sidebar()


# Initialize the application
initialize_session()
setup_user_authentication()
setup_sidebar_controls()

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        st.chat_message("assistant", avatar=APP_ICON_PATH).write(message.content)
    if isinstance(message, HumanMessage):
        st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)

# Handle new user input
user_message = st.chat_input("Message ChatCapitalHumain...")
if user_message:
    # Display user message
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(user_message)
    st.session_state.messages.append(HumanMessage(content=user_message))

    # Generate and display assistant response
    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        response_placeholder = st.empty()
        streamlit_callback = get_streamlit_cb(st.empty())

        # Process user input through graph
        graph_response = invoke_our_graph(
            user_message, [streamlit_callback], st.session_state["thread_id"]
        )

        # Update chat history and display
        final_response = graph_response["messages"][-1].content
        st.session_state.messages.append(AIMessage(content=final_response))
        response_placeholder.write(final_response)
