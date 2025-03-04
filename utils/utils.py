import io
import logging
import os
import re
import uuid
from contextlib import redirect_stderr, redirect_stdout

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

APP_ICON_PATH = "images/deer.png"


def get_llm(MODEL_CONFIG):
    model = MODEL_CONFIG.get("model_name")
    if model is None:
        raise ValueError("No model specified in MODEL_CONFIG")

    config = MODEL_CONFIG.copy()
    config.pop("model_name", None)

    if model.startswith("google/"):
        model = model[len("google/") :]
        return ChatGoogleGenerativeAI(model=model, **config)
    elif model.startswith("anthropic/"):
        model = model[len("anthropic/") :]
        return ChatAnthropic(model="claude-3-7-sonnet-20250219", **config)
    else:
        return ChatOpenAI(
            model_name=model,
            openai_api_key=st.secrets["OPENROUTER_API_KEY"],
            openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
            **config,
        )


logging.basicConfig(
    filename="logs.log",
    encoding="UTF-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Helper to get environment variables
def get_env_variable(var_name):
    try:
        if var_name in os.environ:
            return os.environ[var_name]
        if var_name in st.secrets:
            return st.secrets[var_name]
    except Exception as e:
        logger.error(
            "An error occurred retrieving the environment variable %s: %s",
            var_name,
            str(e),
        )
    return None


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


def add_visualization_buttons_to_message(chat_message_container, message_content):
    """
    Adds visualization buttons for Python code blocks in an AI message.

    Args:
        chat_message_container: The Streamlit chat message container
        message_content (str): The content of the message to process
    """
    chat_message_container = st.chat_message("assistant", avatar=APP_ICON_PATH)
    chat_message_container.write(message_content)
    matches = find_code_blocks(message_content)
    if matches:
        # Add a small visual separator
        chat_message_container.markdown("---")
        # Add buttons at the bottom of the message
        for i, match in enumerate(matches):
            code_content = match.group(1)
            chat_message_container.button(
                f"📊 Visualize Code Block {i+1}",
                on_click=plot_school_data,
                use_container_width=True,
                args=(code_content,),
                key=f"viz_{uuid.uuid4()}",
            )
