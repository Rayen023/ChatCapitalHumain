import os
import uuid
from typing import Any, Dict

import streamlit as st

APP_TITLE = "Capital Humain"
APP_ICON_PATH = "images/deer.png"


from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client

# Import utilities
from utils.database import save_chat_logs
from utils.graph_classes import graph
from utils.st_callable_util import get_streamlit_cb
from utils.utils import add_visualization_buttons_to_message, format_response

# Application constants
USER_AVATAR_PATH = "images/avataruser.png"
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
SCHEMA_TEMPLATE_PATH = os.path.join("utils", "schema_template.txt")
DEBUGGING = st.secrets.get("DEBUGGING", False)

# Load environment variables
load_dotenv()

# Set up page configuration


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "schema_template" not in st.session_state:
        with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
            st.session_state.schema_template = file.read()


def display_chat_history():
    """Display the chat history."""

    for message in st.session_state["messages"]:
        if isinstance(message, AIMessage):
            add_visualization_buttons_to_message(st.empty(), message.content)
        elif isinstance(message, HumanMessage):
            st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)
    total_words = sum(
        len(message.content.split()) for message in st.session_state["messages"]
    )
    if total_words > 2000:
        st.warning(
            "⚠️ Attention: La conversation devient trop longue. Le contexte des modèles LLM est limité et plus la conversation s'allonge et aborde des sujets divers, plus la qualité des réponses risque de se dégrader. Veuillez envisager de démarrer une nouvelle conversation. Vous pourrez recharger celle-ci à tout moment depuis la barre latérale une fois connecté."
        )


def process_graph_event(event, response_placeholder):
    """Process events from the graph stream and update UI accordingly."""
    if event.get("analyze_request"):
        analysis_result = event["analyze_request"]["analysis_result"]
        if not analysis_result.is_db_related_and_answerable:
            response = analysis_result.response
            st.session_state["messages"].append(AIMessage(content=response))
            add_visualization_buttons_to_message(response_placeholder, response)

    elif event.get("check_schema_formulate_instructions"):
        query_proposal = event["check_schema_formulate_instructions"]["query_proposal"]
        if not query_proposal.is_accepted_by_human_analyst:

            response = format_response(query_proposal)
            st.session_state["messages"].append(AIMessage(content=response))
            add_visualization_buttons_to_message(response_placeholder, response)

            # Display feedback buttons
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "Oui/Correct",
                    on_click=add_button_response,
                    args=("Oui, c'est correct.",),
                    key="yes_button",
                    use_container_width=True,
                )
            with col2:
                st.button(
                    "Rechercher des questions similaires",
                    on_click=add_button_response,
                    args=(
                        "Rechercher des questions similaires qui pourraient également répondre à ma demande.",
                    ),
                    key="search_similar",
                    use_container_width=True,
                )

            st.session_state["in_human_feedback_state"] = True
        else:
            st.session_state["in_human_feedback_state"] = False

    elif event.get("finalize_query"):

        final_answer = event["finalize_query"]["final_answer"]
        st.session_state["messages"].append(AIMessage(content=final_answer))
        add_visualization_buttons_to_message(response_placeholder, final_answer)
        config = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
        }

        graph.update_state(
            config,
            {
                "query_proposal": None,
                "human_analyst_feedback": None,
                "query_results": None,
                "final_answer": None,
                "analysis_result": None,
                "user_request": None,
                "message_history": None,
            },
        )


def add_button_response(button_str):
    """Add the button response as a user message"""
    st.session_state["messages"].append(HumanMessage(content=button_str))
    # st.chat_message("user", avatar=USER_AVATAR_PATH).write(button_str)
    # st.rerun()  # Rerun to process the new message


# Initialize app
init_session_state()

# Display logo (assuming st.logo is custom component or use st.image instead)
st.logo(
    APP_ICON_PATH,
    icon_image=APP_ICON_PATH,
    size="large",
)

langsmith_client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    api_url=os.getenv("LANGSMITH_ENDPOINT"),
)
langchain_tracer = LangChainTracer(client=langsmith_client)

display_chat_history()

# Handle chat input
prompt = st.chat_input("Message ChatCapitalHumain...")
if prompt:
    st.session_state["messages"].append(HumanMessage(content=prompt))
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(prompt)

# Process latest message if it's from the user
if st.session_state["messages"] and isinstance(
    st.session_state.messages[-1], HumanMessage
):

    user_message = st.session_state.messages[-1].content
    response_placeholder = st.empty()
    streamlit_callback = get_streamlit_cb(st.empty())

    config = {
        "callbacks": [streamlit_callback],
        "configurable": {"thread_id": st.session_state["thread_id"]},
    }

    # Handle initial user message or human feedback
    if not st.session_state.get("in_human_feedback_state", False):
        # Process initial user request
        input_data = {
            "user_request": user_message,
            "message_history": st.session_state["messages"],
        }
        if not DEBUGGING:
            try:
                with st.spinner("Réflexion en cours...", show_time=True):
                    for event in graph.stream(
                        input_data,
                        config=config.pop("configurable"),
                        stream_mode="updates",
                    ):
                        process_graph_event(event, response_placeholder)
            except Exception as e:
                # st.session_state["selected_model"] = "anthropic/claude-3.7-sonnet" # This return errorss
                # update_model()
                st.error(
                    "Une erreur temporaire s'est produite. Veuillez rafraîchir la page ou commencer un nouveau chat. Si l'erreur persiste, n'hésitez pas à nous contacter."
                )
        else:
            for event in graph.stream(input_data, config=config, stream_mode="updates"):
                process_graph_event(event, response_placeholder)

    elif user_message:
        # Process human feedback
        graph.update_state(
            config,
            {"human_analyst_feedback": user_message},
            as_node="human_feedback",
        )
        if not DEBUGGING:
            try:
                with st.spinner("Traitement en cours...", show_time=True):
                    for event in graph.stream(
                        None, config=config.pop("configurable"), stream_mode="updates"
                    ):
                        process_graph_event(event, response_placeholder)
            except Exception as e:
                st.error(
                    "Une erreur temporaire s'est produite. Veuillez rafraîchir la page ou commencer un nouveau chat. Si l'erreur persiste, n'hésitez pas à nous contacter."
                )
        else:
            for event in graph.stream(None, config=config, stream_mode="updates"):
                process_graph_event(event, response_placeholder)
    if not DEBUGGING and st.user.get("email"):
        save_chat_logs("messages")
