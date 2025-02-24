import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from utils.connection import collection
from utils.schema import show_schema_in_sidebar
from utils.utils import get_env_variable, logger

WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"


def reset_chat_history():
    """Reset chat history to initial state and create a new thread_id."""
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["thread_id"] = str(uuid.uuid4())


def dict_to_message(d):
    """Convert a dict (with role and content) back into a message object."""
    if d.get("role") == "assistant":
        return AIMessage(content=d.get("content", ""))
    elif d.get("role") == "human":
        return HumanMessage(content=d.get("content", ""))
    return None


def load_user_conversations(email):
    """
    Retrieve all conversation logs for the logged in user.
    Returns a list of conversation documents sorted by timestamp descending.
    """
    try:
        client = MongoClient(get_env_variable("MONGO_URI"), server_api=ServerApi("1"))
        db = client["capitalhumain_db"]
        collection = db["capitalhumain_convs"]
        convs = list(collection.find({"email": email}))
        convs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return convs
    except Exception as e:
        logger.error("Error loading conversations: %s", str(e))
        return []


def load_conversation(thread_id):
    """
    Load a specific conversation by thread_id.
    """
    try:
        client = MongoClient(get_env_variable("MONGO_URI"), server_api=ServerApi("1"))
        db = client["capitalhumain_db"]
        collection = db["capitalhumain_convs"]
        conv = collection.find_one({"_id": thread_id})
        return conv
    except Exception as e:
        logger.error("Error loading conversation %s: %s", thread_id, str(e))
        return None


def setup_sidebar_controls():
    """Configure sidebar elements including new chat and conversation loading."""
    with st.sidebar:
        if st.experimental_user.get("is_logged_in", True):
            if st.button("Log out"):
                st.logout()
            user_email = st.experimental_user.get("email")
        else:
            if st.button("Log in with Google"):
                st.login()

        # If user is logged in, show a selectbox to load saved conversations

        if user_email:
            st.write(f"Hello, {user_email}")
            convs = load_user_conversations(user_email)
            if convs:
                # Prepare a mapping of display label to thread_id
                conv_options = {
                    f"{conv.get('timestamp', 'Unknown')} - {conv.get('_id')}": conv.get(
                        "_id"
                    )
                    for conv in convs
                }
                selected_conv_display = st.selectbox(
                    "Load Conversation", list(conv_options.keys())
                )
                if st.button("Load Selected Conversation"):
                    selected_thread_id = conv_options[selected_conv_display]
                    conv_data = load_conversation(selected_thread_id)
                    if conv_data and "messages" in conv_data:
                        st.session_state["messages"] = [
                            dict_to_message(m) for m in conv_data["messages"]
                        ]
                        st.session_state["thread_id"] = selected_thread_id
                        st.rerun()  # Rerun to update the chat display

        show_schema_in_sidebar()
