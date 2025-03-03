from typing import Annotated

import streamlit as st
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy import create_engine
from typing_extensions import TypedDict

MODEL_CONFIG = {
    # "model_name": "anthropic/claude-3.5-sonnet:beta",
    # "model_name": "google/gemini-2.0-pro-exp-02-05:free",
    "model_name": "google/gemini-2.0-flash-001",
    # "model_name": "anthropic/claude-3.7-sonnet",
    # "model_name": "openai/o3-mini",
    # "model_name": "openai/o3-mini-high",
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

if "db" not in st.session_state:
    engine = create_engine(st.secrets["db_url"])
    st.session_state.db = SQLDatabase(engine)

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
    **MODEL_CONFIG,
)

toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=llm)

python_repl = PythonREPL()


repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. Use to plot charts using only streamlit chart elements, matplotlib is not support in the interface and to print tables use st.dataframe.",
    func=python_repl.run,
)


tools = toolkit.get_tools()
tools.append(repl_tool)

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")


memory = MemorySaver()

graph_runnable = graph_builder.compile(checkpointer=memory)


def invoke_our_graph(user_input, callables, thread_id):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return graph_runnable.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"callbacks": callables, "configurable": {"thread_id": thread_id}},
    )
