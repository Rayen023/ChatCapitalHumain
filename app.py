import asyncio
#import os
#from typing import Dict, List

import streamlit as st
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer()
from langchain_core.tracers.context import tracing_v2_enabled

# Constants
DEBUGGING = True
PAGE_TITLE = "Capital Humain"
INITIAL_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
MODEL_CONFIG = {
    "model_name": "anthropic/claude-3.5-sonnet:beta",
    #"model_name": "openai/gpt-4o-mini",
    "temperature":0,
    "max_tokens":8096,
    "timeout":None,
    "max_retries":2,
    "streaming":True,
}

# Page setup
st.set_page_config(page_title=PAGE_TITLE)

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]
    if "db" not in st.session_state:
        engine = create_engine(st.secrets["db_url"])
        st.session_state.db = SQLDatabase(engine)
    if "system_message" not in st.session_state:
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        with open("prompt_info.txt", "r", encoding="utf-8") as file:
            prompt_temp = file.read()
        system_message = prompt_template.format(dialect="PostgreSQL", top_k=5)
        st.session_state.system_message = system_message + "\n" + prompt_temp

def clear_chat_history():
    """Reset chat history"""
    st.session_state.messages = [{"role": "assistant", "content": INITIAL_MESSAGE}]

def setup_sidebar():
    """Setup sidebar elements"""
    with st.sidebar:
        st.title(PAGE_TITLE)
        st.button("Nouveau chat", on_click=clear_chat_history, icon=":material/edit_square:")

def setup_chat_interface():
    """Setup chat interface and message history"""
    history = ChatMessageHistory()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            prefix = "AI : " if message["role"] == "assistant" else ""
            history.add_user_message(prefix + message["content"])
    return history

@st.cache_resource
def get_prompt_template():
    """Get chat prompt template"""
    return ChatPromptTemplate.from_messages([
        ("system", st.session_state.system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


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
        {"input": st.session_state.messages[-1]["content"]}, version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            #print(content), print(event)
            if content :
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
        return_messages=True, 
        memory_key="chat_history", 
        chat_memory=history
    )
    
    agent_executor = create_agent(st.session_state.db, memory)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            if DEBUGGING:
                st_callback = StreamlitCallbackHandler(
                    st.container(),
                    expand_new_thoughts=True,
                    collapse_completed_thoughts=True,
                )
                with tracing_v2_enabled():
                    response = agent_executor.invoke(
                        {"input": st.session_state.messages[-1]["content"]},
                        {"callbacks": [st_callback]},
                    )
                st.write(response["output"])
                message = {"role": "assistant", "content": response["output"]}
            else:
                with tracing_v2_enabled():
                    asyncio.run(process_events(agent_executor, message_placeholder))
                message = {"role": "assistant", "content": st.session_state["accumulated_text"]}
            
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()

    