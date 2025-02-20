import asyncio

import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks.tracers import LangChainTracer
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

tracer = LangChainTracer()
import os

from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client

from show_db import show_schema_in_sidebar

# You can create a client instance with an api key and api url
client = Client(
    api_key=os.getenv(
        "LANGSMITH_API_KEY"
    ),  # Update appropriately for self-hosted installations
    api_url=os.getenv(
        "LANGSMITH_ENDPOINT"
    ),  # Update appropriately for self-hosted installations
)
tracer = LangChainTracer(
    client=client,
)

# Constants
DEBUGGING = True
PAGE_TITLE = "Capital Humain"
INITIAL_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
MODEL_CONFIG = {
    # "model_name": "anthropic/claude-3.5-sonnet:beta",
    # "model_name": "google/gemini-2.0-pro-exp-02-05:free",
    "model_name": "google/gemini-2.0-flash-001",
    # "model_name": "openai/o3-mini",
    # "model_name": "openai/o3-mini-high",
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}

# Page setup
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="deer.png",
)

st.logo(
    "deer.png",
    icon_image="deer.png",
    size="large",
)


def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]
    if "db" not in st.session_state:
        engine = create_engine(st.secrets["db_url"])
        st.session_state.db = SQLDatabase(engine)
        # save db to csv

    if "system_message" not in st.session_state:
        with open("full_prompt_no_answer_n_ask.txt", "r", encoding="utf-8") as file:
            prompt_temp = file.read()
        st.session_state.system_message = prompt_temp


def clear_chat_history():
    """Reset chat history"""
    st.session_state.messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]


def setup_sidebar():
    """Setup sidebar elements"""
    with st.sidebar:
        # st.title(PAGE_TITLE)
        st.button(
            "Nouveau chat",
            on_click=clear_chat_history,
            icon=":material/edit_square:",
            use_container_width=True,
        )

        show_schema_in_sidebar()


def setup_chat_interface():
    """Setup chat interface and message history"""
    history = ChatMessageHistory()
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="deer.png"):
                st.write(message["content"])
                history.add_user_message("AI : " + message["content"])
        else:
            with st.chat_message(message["role"], avatar="avataruser.png"):
                st.write(message["content"])
                history.add_user_message(message["content"])
    return history


@st.cache_resource
def get_prompt_template():
    """Get chat prompt template"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def create_agent(db, memory):
    """Create LangChain agent"""
    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
        **MODEL_CONFIG
    )
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_tool_calling_agent(llm, toolkit.get_tools(), get_prompt_template())

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=toolkit.get_tools(),
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )


async def process_events(agent_executor, message_placeholder):
    """Process agent events and update UI"""
    accumulated_text = ""
    async for event in agent_executor.astream_events(
        {"input": st.session_state.messages[-1]["content"]},
        {"callbacks": [tracer]},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            # print(content), print(event)
            if content:
                accumulated_text += content
                message_placeholder.empty()
                message_placeholder.write(accumulated_text)
                st.session_state["accumulated_text"] = accumulated_text
        if event["event"] == "on_tool_end":
            st.session_state["accumulated_text"] = ""
            accumulated_text = ""
            message_placeholder.empty()


def main():
    init_session_state()
    setup_sidebar()

    prompt = st.chat_input("Message ChatCapitalHumain...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

    history = setup_chat_interface()
    memory = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history", chat_memory=history
    )

    agent_executor = create_agent(st.session_state.db, memory)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant", avatar="deer.png"):
            message_placeholder = st.empty()
            if DEBUGGING:
                st_callback = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=True,
                    collapse_completed_thoughts=True,
                )
                with tracing_v2_enabled(
                    client=client, project_name=os.getenv("LANGSMITH_PROJECT")
                ):
                    response = agent_executor.invoke(
                        {"input": st.session_state.messages[-1]["content"]},
                        {"callbacks": [st_callback, tracer]},
                    )
                st.write(response["output"])
                message = {"role": "assistant", "content": response["output"]}
            else:
                with tracing_v2_enabled(
                    client=client, project_name=os.getenv("LANGSMITH_PROJECT")
                ):
                    asyncio.run(process_events(agent_executor, message_placeholder))
                message = {
                    "role": "assistant",
                    "content": st.session_state["accumulated_text"],
                }

            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
