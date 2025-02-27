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
from utils.graph_classes import invoke_our_graph
from utils.sidebar import setup_sidebar_controls
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


# setup_sidebar_controls()

# Display chat history
for message in st.session_state["messages"]:
    if isinstance(message, AIMessage):
        st.chat_message("assistant", avatar=APP_ICON_PATH).write(message.content)
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

    # Display session state when debug panel is open
    if "show_debug_panel" in st.session_state and st.session_state["show_debug_panel"]:
        st.write("### Session State Contents")
        for key in sorted(st.session_state.keys()):
            with st.expander(f"Key: {key}"):
                st.write(st.session_state[key])


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
        graph_response, graph = invoke_our_graph(
            user_message, [streamlit_callback], st.session_state["thread_id"]
        )
        # final_response = graph_response["messages"][-1].content
        # st.session_state["messages"].append(AIMessage(content=final_response))
        # response_placeholder.write(final_response)

        # With multi_agents

        state = graph.get_state(
            {"configurable": {"thread_id": st.session_state["thread_id"]}}
        )
        print("state", state)
        st.warning(graph_response)
        st.info(graph_response["analysis_result"].response)
        st.session_state["messages"].append(
            AIMessage(content=graph_response["analysis_result"].response)
        )
        response_placeholder.write(graph_response)
        if graph_response.get("final_answer"):
            st.session_state["messages"].append(
                AIMessage(content=graph_response["final_answer"])
            )
            response_placeholder.write(graph_response["final_answer"])

        # TODO check state of the graph_response, if feedback in its name :

        # # if graph_response["analysis_result"].is_answerable == True:
        # further_feedack = st.text_input(
        #     "Veuillez donner plus de détails pour une meilleure réponse"
        # )
        # # if further_feedack:
        # print("further feedback", further_feedack)
        # graph.update_state(
        #     {"configurable": {"thread_id": st.session_state["thread_id"]}},
        #     {
        #         "human_analyst_feedback": "inclut aussi comme bons resultats tout ayant a partir de plus de 55%"
        #     },
        #     as_node="human_feedback",
        # )

        # for event in graph.stream(
        #     None,
        #     {"configurable": {"thread_id": st.session_state["thread_id"]}},
        #     stream_mode="updates",
        # ):
        #     print("--Node--")
        #     node_name = next(iter(event.keys()))
        #     print(node_name)
        # final_state = graph.get_state(
        #     {"configurable": {"thread_id": st.session_state["thread_id"]}}
        # )
        # print(final_state.values.get("final_query_instructions"))

        # st.session_state["messages"].append(AIMessage(content=str(graph_response)))

    # Save the conversation only after the assistant has responded,
    # and only if the user is logged in (i.e. email exists)
    # if st.experimental_user.get("email"):
    # save_chat_logs()
