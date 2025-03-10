import os
import uuid

import streamlit as st

APP_TITLE = "Capital Humain"
APP_ICON_PATH = "images/deer.png"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON_PATH,
)

from dotenv import load_dotenv

# Import your LangChain and LangSmith classes as before
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client

# Import your existing utilities (adjust if needed)
from single_agent_app import invoke_our_graph
from utils.database import save_chat_logs

# from utils.sidebar import setup_sidebar_controls
from utils.st_callable_util import get_streamlit_cb

# Load environment variables
load_dotenv()

# LangSmith client configuration
langsmith_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
)
langchain_tracer = LangChainTracer(client=langsmith_client)

# Application constants
USER_AVATAR_PATH = "images/avataruser.png"
SYSTEM_PROMPT_PATH = "prompt_templates/full_prompt_no_answer_n_ask.txt"
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"

# Page configuration

st.logo(
    APP_ICON_PATH,
    icon_image=APP_ICON_PATH,
    size="large",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())


def reset_chat_history():
    """Reset chat history to initial state and create a new thread_id."""
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["thread_id"] = str(uuid.uuid4())


with st.sidebar:
    st.button(
        "Nouveau chat",
        on_click=reset_chat_history,
        icon=":material/edit_square:",
        use_container_width=True,
    )

# setup_sidebar_controls()

# Display chat history
for message in st.session_state["messages"]:
    if isinstance(message, AIMessage):
        st.chat_message("assistant", avatar=APP_ICON_PATH).write(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)

# Handle new user input
user_message = st.chat_input("Message ChatCapitalHumain...")
if user_message:
    # Display and record the user's message
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(user_message)
    st.session_state["messages"].append(HumanMessage(content=user_message))

    # Generate assistant response via your graph
    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        response_placeholder = st.empty()
        streamlit_callback = get_streamlit_cb(st.empty())
        graph_response = invoke_our_graph(
            user_message, [streamlit_callback], st.session_state["thread_id"]
        )
        final_response = graph_response["messages"][-1].content
        st.session_state["messages"].append(AIMessage(content=final_response))
        response_placeholder.write(final_response)

    # Save the conversation only after the assistant has responded,
    # and only if the user is logged in (i.e. email exists)
    # if st.experimental_user.get("email"):
    #     save_chat_logs()
