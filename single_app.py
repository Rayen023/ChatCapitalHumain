import uuid
from typing import Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy import create_engine
from typing_extensions import TypedDict

from utils.st_callable_util import get_streamlit_cb

APP_TITLE = "Capital Humain"
APP_ICON_PATH = "images/deer.png"
USER_AVATAR_PATH = "images/avataruser.png"
SYSTEM_PROMPT_PATH = "prompt_templates/full_prompt_no_answer_n_ask.txt"
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON_PATH,
)

load_dotenv()

MODEL_CONFIG = {
    "model_name": "google/gemini-2.0-flash-001",
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}


class State(TypedDict):
    messages: Annotated[list, add_messages]


if "db" not in st.session_state:
    engine = create_engine(st.secrets["db_url"])
    st.session_state.db = SQLDatabase(engine)

if "messages" not in st.session_state:
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
    **MODEL_CONFIG,
)

toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=llm)
tools = toolkit.get_tools()
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph_runnable = graph_builder.compile(checkpointer=memory)


def invoke_our_graph(user_input, callables, thread_id):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": callables, "configurable": {"thread_id": thread_id}},
    )


def reset_chat_history():
    st.session_state["messages"] = [AIMessage(content=WELCOME_MESSAGE)]
    st.session_state["thread_id"] = str(uuid.uuid4())


with st.sidebar:
    st.button(
        "Nouveau chat",
        on_click=reset_chat_history,
        icon=":material/edit_square:",
        use_container_width=True,
    )

for message in st.session_state["messages"]:
    if isinstance(message, AIMessage):
        st.chat_message("assistant", avatar=APP_ICON_PATH).write(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)

user_message = st.chat_input("Message ChatCapitalHumain...")
if user_message:
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(user_message)
    st.session_state["messages"].append(HumanMessage(content=user_message))

    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        response_placeholder = st.empty()
        streamlit_callback = get_streamlit_cb(st.empty())
        graph_response = invoke_our_graph(
            user_message, [streamlit_callback], st.session_state["thread_id"]
        )
        final_response = graph_response["messages"][-1].content
        st.session_state["messages"].append(AIMessage(content=final_response))
        response_placeholder.write(final_response)
