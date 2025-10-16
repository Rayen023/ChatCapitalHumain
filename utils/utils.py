import io
import json
import logging
import os
import re
import uuid
from contextlib import redirect_stderr, redirect_stdout

import streamlit as st
from config import Config

APP_ICON_PATH = "images/deer.png"

logging.basicConfig(
    filename="logs.log",
    encoding="UTF-8",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Helper to get environment variables
def get_env_variable(var_name):
    """
    Get environment variable using centralized config.
    This function is kept for backward compatibility.
    """
    return Config.get_env_variable(var_name)


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


def remove_code_blocks(markdown_text):
    """
    Remove Python code blocks from markdown text.

    Args:
        markdown_text (str): The markdown text to process

    Returns:
        str: Text with code blocks removed
    """
    code_block_pattern = r"```python\s*.*?```"
    return re.sub(code_block_pattern, "", markdown_text, flags=re.DOTALL).strip()


def execute_code_inline(code_content):
    """
    Executes the code and returns whether it was successful.

    Args:
        code_content (str): The Python code to execute

    Returns:
        bool: True if execution was successful, False otherwise
    """
    # Create string buffers to capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Add imports for data visualization to the global namespace
    globals_dict = {"st": st, "plt": None, "pd": None, "np": None, "px": None, "go": None}

    # Try to import common data visualization libraries
    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        globals_dict["pd"] = pd
        globals_dict["np"] = np
        globals_dict["px"] = px
        globals_dict["go"] = go
    except ImportError as e:
        st.warning(f"Some visualization libraries couldn't be imported: {e}")
        return False

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

        return True

    except Exception as e:
        st.error(f"Error executing code: {e}")
        # Display errors if any
        stderr_content = stderr_buffer.getvalue()
        if stderr_content.strip():
            st.error("Details:")
            st.code(stderr_content)
        return False


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
    globals_dict = {"st": st, "plt": None, "pd": None, "np": None, "px": None, "go": None}

    # Try to import common data visualization libraries
    try:
        # import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        # globals_dict["plt"] = plt
        globals_dict["pd"] = pd
        globals_dict["np"] = np
        globals_dict["px"] = px
        globals_dict["go"] = go
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


import json
import re
from typing import Any, Dict, List, Optional, Union


def format_response(query_proposal):
    """
    Format response from query_proposal object with improved JSON parsing.

    Args:
        query_proposal: Object containing questions_responses, user_request_after_feedback, and explanation

    Returns:
        str: Formatted response
    """
    questions_responses = query_proposal.questions_responses
    user_request_after_feedback = query_proposal.user_request_after_feedback
    explanation = query_proposal.explanation

    # Parse the questions_responses into a list of dictionaries
    parsed_questions = parse_questions(questions_responses)

    # Format the questions for display
    if parsed_questions:
        formatted_questions = format_questions_list(parsed_questions)
        questions_str = "\n".join(formatted_questions)
    else:
        questions_str = (
            questions_responses  # Fallback to original string if parsing fails
        )

    # Check if any question has empty response_options
    include_explanation = not parsed_questions or any(
        not item.get("response_options", []) for item in parsed_questions
    )

    # Build the response with conditional parts
    response_parts = []
    if include_explanation:
        response_parts.append(explanation)

    response_parts.append(f"User_request : {user_request_after_feedback}")
    response_parts.append(f"Questions : \n{questions_str}")
    response_parts.append(
        "\n\n 💡 NOTE : Veuillez valider si les étapes suggérées sont correctes en répondant par **OUI** ou **CORRECT**, sinon, veuillez indiquer les **modifications/suggestions** pour les étapes alternatives."
    )

    return "\n\n".join(response_parts)


def parse_questions(questions_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse questions string into a list of dictionaries with robust handling of apostrophes.

    Args:
        questions_str: String representation of questions list

    Returns:
        List of question dictionaries or None if parsing fails
    """
    # Try to determine if the input is already a list (string representation of an existing object)
    if not isinstance(questions_str, str):
        return None

    # Method 1: Try ast.literal_eval which is safer than eval
    try:
        import ast

        parsed = ast.literal_eval(questions_str)
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
    except (SyntaxError, ValueError):
        pass

    # Method 2: Try JSON parsing after replacing problematic characters
    try:
        # Create a temporary string for JSON processing
        json_str = questions_str

        # Handle special case of French apostrophes (l'intention)
        # First, temporarily replace valid apostrophes with a marker
        json_str = re.sub(r"([a-zA-Z])'([a-zA-Z])", r"\1§\2", json_str)

        # Now replace all remaining single quotes with double quotes
        json_str = json_str.replace("'", '"')

        # Restore the apostrophes we marked
        json_str = json_str.replace("§", "'")

        parsed = json.loads(json_str)
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Method 3: Regular expressions as a last resort
    try:
        questions_list = []
        # Extract each dictionary pattern
        for match in re.finditer(r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", questions_str):
            dict_str = match.group(0)
            # More robust key-value extraction
            item = {}
            for key_match in re.finditer(
                r"'([^']+)'\s*:\s*((?:\[[^\]]*\])|(?:'[^']*')|(?:\d+))", dict_str
            ):
                key = key_match.group(1)
                value_str = key_match.group(2)

                # Process the value based on its type
                if value_str.startswith("[") and value_str.endswith("]"):
                    # List value
                    try:
                        value = json.loads(value_str.replace("'", '"'))
                    except json.JSONDecodeError:
                        value = []
                elif value_str.startswith("'") and value_str.endswith("'"):
                    # String value
                    value = value_str[1:-1]
                else:
                    # Number value
                    try:
                        value = int(value_str)
                    except ValueError:
                        try:
                            value = float(value_str)
                        except ValueError:
                            value = value_str

                item[key] = value

            if item:
                questions_list.append(item)

        if questions_list:
            return questions_list
    except Exception:
        pass

    # If all parsing methods fail
    return None


def format_questions_list(questions_list: List[Dict[str, Any]]) -> List[str]:
    """
    Format a list of question dictionaries for display.

    Args:
        questions_list: List of dictionaries containing question data

    Returns:
        List of formatted question strings
    """
    formatted_questions = []
    for i, question in enumerate(questions_list, 1):
        lines = [f"Question {i}:"]
        for key, value in question.items():
            # Skip any field containing "id" if needed
            if "id" not in key.lower():
                lines.append(f"  - {key}: {value}")
        formatted_questions.append("\n".join(lines))
    return formatted_questions
