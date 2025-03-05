import os
import uuid
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import Client

# Import utilities
from utils.database import save_chat_logs
from utils.graph_classes import display_model_selector, graph
from utils.schema import show_schema_in_sidebar
from utils.sidebar import enable_login
from utils.st_callable_util import get_streamlit_cb
from utils.utils import add_visualization_buttons_to_message, format_response

# Application constants
APP_TITLE = "Capital Humain"
APP_ICON_PATH = "images/deer.png"
USER_AVATAR_PATH = "images/avataruser.png"
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
SCHEMA_TEMPLATE_PATH = os.path.join("utils", "schema_template.txt")

# Load environment variables
load_dotenv()

# Set up page configuration
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON_PATH)


def init_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = str(uuid.uuid4())
    if "schema_template" not in st.session_state:
        with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
            st.session_state.schema_template = file.read()


def reset_chat_history():
    """Reset chat history to initial state and create a new thread_id."""
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["thread_id"] = str(uuid.uuid4())
    if "in_human_feedback_state" in st.session_state:
        del st.session_state["in_human_feedback_state"]
    if "schema_template" in st.session_state:
        del st.session_state["schema_template"]


def toggle_debug_panel():
    """Toggle the visibility of the debug panel."""
    st.session_state["show_debug_panel"] = not st.session_state.get(
        "show_debug_panel", False
    )


def ask_example_question(question: str):
    """Set an example question to be processed."""
    st.session_state["_example_question"] = question


def display_chat_history():
    """Display the chat history."""
    for message in st.session_state["messages"]:
        if isinstance(message, AIMessage):
            add_visualization_buttons_to_message(st.empty(), message.content)
            print(len(message.content.split(" ")))
        elif isinstance(message, HumanMessage):
            st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)


def rerun_last_question():
    st.rerun()

    # config = {
    #     "configurable": {"thread_id": st.session_state["thread_id"]},
    # }
    # st.warning(f" state : {graph.get_state(config)}")
    # print(graph.get_state(config).tasks)
    # print(graph.get_state(config).values)
    # if "final_answer" in graph.get_state(config).values:
    #     graph.update_state(
    #         config,
    #         {
    #             "query_results": None,
    #             "final_answer": None,
    #         },
    #         as_node="run_query",
    #     )
    # st.session_state["messages"].pop(-1)


def setup_sidebar():
    """Configure and display the sidebar."""
    with st.sidebar:
        st.markdown("---")

        st.button(
            "Nouveau chat",
            on_click=reset_chat_history,
            icon=":material/edit_square:",
            use_container_width=True,
        )

        # st.button(
        #     ":material/autorenew:",
        #     on_click=rerun_last_question,
        #     key="rerun_last_question",
        # )

        st.markdown("---")
        # enable_login()
        st.markdown("---")

        # Example questions section
        st.write("### Example Questions")

        example_questions = [
            "Comment les sources de financement des élèves se distribuent selon l’école en 2018 ?",
            "Quels étaient les différents moyens de transport utilisés par les élèves pour aller à l'école en 2018, ventilés par genre ?",
            "Quelle est la corrélation entre les résultats scolaires en 10e, 11e et 12e année et les projets des élèves pour septembre, selon leur genre ?",
            "Comment les résultats scolaires en 10e, 11e et 12e année sont-ils liés aux raisons de ne pas poursuivre des études postsecondaires ?",
            "Comment l’intention de s’établir dans la Péninsule acadienne varie-t-elle selon le niveau d’éducation et l’occupation actuelle ?",
        ]

        for question in example_questions:
            st.button(
                question,
                on_click=ask_example_question,
                args=(question,),
                use_container_width=True,
            )
        st.markdown("---")

        st.button(
            "Show Session State",
            on_click=toggle_debug_panel,
            key="debug_button",
            use_container_width=True,
        )
        st.markdown("---")

        # Debug panel
        if st.session_state.get("show_debug_panel", False):
            st.write("### Session State Contents")
            for key in sorted(st.session_state.keys()):
                with st.expander(f"Key: {key}"):
                    st.write(st.session_state[key])

        show_schema_in_sidebar()


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


def main():
    """Main application function."""
    # Initialize app
    init_session_state()

    # Display logo (assuming st.logo is custom component or use st.image instead)
    st.logo(
        APP_ICON_PATH,
        icon_image=APP_ICON_PATH,
        size="large",
    )

    # Initialize model selection and LangSmith
    display_model_selector()

    langsmith_client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY"),
        api_url=os.getenv("LANGSMITH_ENDPOINT"),
    )
    langchain_tracer = LangChainTracer(client=langsmith_client)

    # Display UI components
    setup_sidebar()
    display_chat_history()

    # Handle chat input
    prompt = st.chat_input("Message ChatCapitalHumain...")
    if prompt:
        st.session_state["messages"].append(HumanMessage(content=prompt))
        st.chat_message("user", avatar=USER_AVATAR_PATH).write(prompt)

    # Handle example questions
    if "_example_question" in st.session_state:
        example_q = st.session_state["_example_question"]
        st.session_state["messages"].append(HumanMessage(content=example_q))
        st.chat_message("user", avatar=USER_AVATAR_PATH).write(example_q)
        del st.session_state["_example_question"]

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

            try:
                for event in graph.stream(
                    input_data, config=config, stream_mode="updates"
                ):
                    process_graph_event(event, response_placeholder)
            except Exception as e:
                # st.session_state["selected_model"] = "anthropic/claude-3.7-sonnet" # This return errorss
                # update_model()

                st.error(
                    "Une erreur temporaire s'est produite. Vous pouvez résoudre ce problème en sélectionnant un autre modèle dans le coin supérieur droit de la page et réessayer."
                )

        elif user_message:
            # Process human feedback
            graph.update_state(
                config,
                {"human_analyst_feedback": user_message},
                as_node="human_feedback",
            )
            try:
                for event in graph.stream(None, config=config, stream_mode="updates"):
                    process_graph_event(event, response_placeholder)
            except Exception as e:
                st.error(
                    "Une erreur temporaire s'est produite. Vous pouvez résoudre ce problème en sélectionnant un autre modèle dans le coin supérieur droit de la page et réessayer."
                )
        # Optional: Save chat logs
        # if st.experimental_user.get("email"):
        # save_chat_logs()


if __name__ == "__main__":
    main()
