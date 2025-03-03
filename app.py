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

from utils.database import save_chat_logs

# Import your existing utilities (adjust if needed)
from utils.graph_classes import display_model_selector, graph
from utils.sidebar import setup_sidebar_controls
from utils.st_callable_util import get_streamlit_cb

display_model_selector()
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
    if "in_human_feedback_state" in st.session_state:
        del st.session_state["in_human_feedback_state"]
    if "schema_template" in st.session_state:
        del st.session_state["schema_template"]


# setup_sidebar_controls()

import io
import re
from contextlib import redirect_stderr, redirect_stdout


@st.dialog("School Data Visualization")
def plot_school_data(code_content):
    """
    Executes the code in a safe manner and displays any outputs or plots in Streamlit.

    Args:
        code_content (str): The Python code to execute
    """
    # Create string buffers to capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Add imports for data visualization to the global namespace
    globals_dict = {"st": st, "plt": None, "pd": None, "np": None}

    # Try to import common data visualization libraries
    try:
        # import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # globals_dict["plt"] = plt
        globals_dict["pd"] = pd
        globals_dict["np"] = np
    except ImportError as e:
        st.warning(f"Some visualization libraries couldn't be imported: {e}")

    # Execute the code with redirected output
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Execute in a controlled environment
            exec(code_content, globals_dict)

        # Display standard output if any
        stdout_content = stdout_buffer.getvalue()
        if stdout_content.strip():
            st.text("Output:")
            st.code(stdout_content)

        # Display any matplotlib plots
        # if globals_dict["plt"] is not None and plt.get_fignums():
        #     st.pyplot(plt.gcf())
        #     plt.close("all")  # Clean up plots

    except Exception as e:
        st.error(f"Error executing code: {e}")

    # Display errors if any
    stderr_content = stderr_buffer.getvalue()
    if stderr_content.strip():
        st.error("Errors:")
        st.code(stderr_content)


# Find Python code blocks in markdown text
def find_code_blocks(markdown_text):
    """
    Find Python code blocks in markdown text.

    Args:
        markdown_text (str): The markdown text to process

    Returns:
        list: List of tuples with (index, code_content)
    """
    code_block_pattern = r"```python\s*(.*?)```"
    return list(re.finditer(code_block_pattern, markdown_text, re.DOTALL))


# Display chat history
for message in st.session_state["messages"]:
    # Display the message first
    if isinstance(message, AIMessage):
        chat_message = st.chat_message("assistant", avatar=APP_ICON_PATH)
        chat_message.write(message.content)

        # After displaying the message, find code blocks and add buttons for AI messages only
        matches = find_code_blocks(message.content)
        if matches:
            # Add a small visual separator
            chat_message.markdown("---")
            # Add buttons at the bottom of the message
            for i, match in enumerate(matches):
                code_content = match.group(1)
                chat_message.button(
                    f"📊 Visualize Code Block {i+1}",
                    on_click=plot_school_data,
                    use_container_width=True,
                    args=(code_content,),
                    key=f"viz_{uuid.uuid4()}",
                )
    elif isinstance(message, HumanMessage):
        st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)


# Add this function to your app
def toggle_debug_panel():
    """Toggle the visibility of the debug panel."""
    if "show_debug_panel" not in st.session_state:
        st.session_state["show_debug_panel"] = True
    else:
        st.session_state["show_debug_panel"] = not st.session_state["show_debug_panel"]


# Update your sidebar section
with st.sidebar:
    st.button(
        "Nouveau chat",
        on_click=reset_chat_history,
        icon=":material/edit_square:",
        use_container_width=True,
    )
    # Debug button to show session state
    st.button(
        "Show Session State",
        on_click=toggle_debug_panel,
        key="debug_button",
        use_container_width=True,
    )

    # Add example question section
    st.write("### Example Questions")

    # Function to send example question as user message
    def ask_example_question(question):
        st.session_state["_example_question"] = question

    # Example question buttons
    st.button(
        "Students by transport mode",
        on_click=ask_example_question,
        args=(
            "How many students went to school on foot in 2018, aggregate by school and gender?",
        ),
        use_container_width=True,
    )

    st.button(
        "Random data visualization",
        on_click=ask_example_question,
        args=(
            "write me a python code block that displays random data using a streamlit componenet like barchart, i am testing a new feature",
        ),
        use_container_width=True,
    )

    st.button(
        "Financial aid comparison",
        on_click=ask_example_question,
        args=(
            "Compare the percentage of students receiving financial aid between 2017 and 2019 by department.",
        ),
        use_container_width=True,
    )

    # Display session state when debug panel is open
    if "show_debug_panel" in st.session_state and st.session_state["show_debug_panel"]:
        st.write("### Session State Contents")
        for key in sorted(st.session_state.keys()):
            with st.expander(f"Key: {key}"):
                st.write(st.session_state[key])

import os

from utils.schema import show_schema_in_sidebar

show_schema_in_sidebar()


def add_visualization_buttons_to_message(chat_message, message_content):
    """
    Adds visualization buttons for Python code blocks in an AI message.

    Args:
        chat_message: The Streamlit chat message container
        message_content (str): The content of the message to process
    """
    matches = find_code_blocks(message_content)
    if matches:
        # Add a small visual separator
        chat_message.markdown("---")
        # Add buttons at the bottom of the message
        for i, match in enumerate(matches):
            code_content = match.group(1)
            chat_message.button(
                f"📊 Visualize Code Block {i+1}",
                on_click=plot_school_data,
                use_container_width=True,
                args=(code_content,),
                key=f"viz_{uuid.uuid4()}",
            )


SCHEMA_TEMPLATE_PATH = os.path.join("utils", "schema_template.txt")
if "schema_template" not in st.session_state:
    with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
        prompt_temp = file.read()
    st.session_state.schema_template = prompt_temp

prompt = st.chat_input("Message ChatCapitalHumain...")
if prompt:
    st.session_state["messages"].append(HumanMessage(content=prompt))
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(prompt)

# Add this near the end of the file, before any other conditional blocks
if "_example_question" in st.session_state:
    st.session_state["messages"].append(
        HumanMessage(content=st.session_state["_example_question"])
    )
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(
        st.session_state["_example_question"]
    )
    # Display the message in the chat
    # st.chat_message("user", avatar=USER_AVATAR_PATH).write(user_message)

    # Remove the temporary state to prevent reprocessing
    del st.session_state["_example_question"]

    # Force a rerun to process the message
    # st.rerun()


if isinstance(st.session_state.messages[-1], HumanMessage):
    user_message = st.session_state.messages[-1].content

    # Generate assistant response via your graph
    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        response_placeholder = st.empty()
        streamlit_callback = get_streamlit_cb(st.empty())
        config = {
            "callbacks": [streamlit_callback],
            "configurable": {"thread_id": st.session_state["thread_id"]},
        }
        if (
            st.session_state.get("in_human_feedback_state", False) is False
        ):  # ss.get return False if key not found else returns its value, which nust be false or doesn't exist for the condition to be true
            for event in graph.stream(
                {
                    "user_request": user_message,
                    "message_history": st.session_state["messages"],
                },
                config=config,
                stream_mode="updates",
            ):
                # st.info(event)

                if event.get("analyze_request", None):
                    if (
                        event["analyze_request"][
                            "analysis_result"
                        ].is_db_related_and_answerable
                        == False
                    ):
                        st.session_state["messages"].append(
                            AIMessage(
                                content=event["analyze_request"][
                                    "analysis_result"
                                ].response
                            )
                        )
                        response_placeholder.write(
                            event["analyze_request"]["analysis_result"].response
                        )
                if event.get("check_schema_formulate_instructions", None):
                    if (
                        event["check_schema_formulate_instructions"][
                            "query_proposal"
                        ].is_accepted_by_human_analyst
                        == False
                    ):
                        response_explanation = event[
                            "check_schema_formulate_instructions"
                        ]["query_proposal"].explanation

                        response_explanation = (
                            response_explanation
                            + """\n\n 💡 NOTE : Veuillez valider si les étapes suggérées sont correctes en répondant par **OUI** ou **CORRECT**, sinon, veuillez indiquer les **modifications/suggestions** pour les étapes alternatives."""
                        )

                        st.session_state["messages"].append(
                            AIMessage(content=response_explanation)
                        )
                        response_placeholder.write(response_explanation)
                        st.session_state["in_human_feedback_state"] = True
                        user_message = None
                    else:
                        st.session_state["in_human_feedback_state"] = False

        if (
            st.session_state.get("in_human_feedback_state", False) and user_message
        ):  # ss.get return False if key not found else returns its value
            # streamlit_callback = get_streamlit_cb(st.empty())
            # config = {
            #     "callbacks": [streamlit_callback],
            #     "configurable": {"thread_id": st.session_state["thread_id"]},
            # }
            graph.update_state(
                config,
                {"human_analyst_feedback": user_message},
                as_node="human_feedback",
            )
            # st.session_state["in_human_feedback_state"] = False

            for event in graph.stream(None, config=config, stream_mode="updates"):
                # st.info(event)
                if event.get("check_schema_formulate_instructions", None):
                    if (
                        event["check_schema_formulate_instructions"][
                            "query_proposal"
                        ].is_accepted_by_human_analyst
                        == False
                    ):

                        response_explanation = event[
                            "check_schema_formulate_instructions"
                        ]["query_proposal"].explanation

                        response_explanation = (
                            response_explanation
                            + """\n\n 💡 NOTE : Veuillez valider si les étapes suggérées sont correctes en répondant par **OUI** ou **CORRECT**, sinon, veuillez indiquer les **modifications/suggestions** pour les étapes alternatives."""
                        )

                        st.session_state["messages"].append(
                            AIMessage(content=response_explanation)
                        )
                        response_placeholder.write(response_explanation)

                        st.session_state["in_human_feedback_state"] = True
                    else:
                        st.session_state["in_human_feedback_state"] = False
                        user_message = None
                if event.get("finalize_query", None):
                    st.session_state["messages"].append(
                        AIMessage(content=event["finalize_query"]["final_answer"])
                    )
                    response_placeholder.write(event["finalize_query"]["final_answer"])
                    st.warning(event["finalize_query"]["final_answer"])
                    st.info(event["finalize_query"]["final_answer"][-1])
            # st.rerun()

        # if st.experimental_user.get("email"):
        # save_chat_logs()
