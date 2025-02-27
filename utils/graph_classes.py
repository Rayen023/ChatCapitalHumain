from typing import List, Optional

import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from typing_extensions import TypedDict

from utils.utils import get_llm

MODEL_CONFIG = {
    # "model_name": "anthropic/claude-3.5-sonnet:beta",
    # "model_name": "openai/gpt-4o-mini",
    # "model_name": "google/gemini-2.0-flash-001",
    "model_name": "anthropic/claude-3.7-sonnet",
    # "model_name": "openai/o3-mini",
    # "model_name": "openai/o3-mini-high",
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}

llm = get_llm(MODEL_CONFIG)


# Define our structured data models
class QueryProposal(BaseModel):
    tables: List[str] = Field(
        description="The tables in the database that would be used to answer the query",
    )
    columns: List[str] = Field(
        description="The columns from the tables that would be relevant for the query",
    )
    fields: List[str] = Field(
        description="The specific fields or values to filter or select in the query",
    )
    explanation: str = Field(
        description="Explanation of how these tables, columns, and fields would be used to answer the query with instructions in SQL and what SQL functions can be used",
    )
    is_accepted_by_human_analyst: bool = Field(
        description="Whether the proposal was accepted by the human analyst, Must be set to False by default",
    )


class AnalysisResult(BaseModel):
    is_db_related_and_answerable: bool = Field(
        description="Whether the query can be answered with the available database",
    )
    response: str = Field(
        description="Conversational Response or Explanation of why the query is not answerable",
    )


# class CheckSchemaFormulateInstructions(BaseModel):
#     query_proposal: Optional[QueryProposal] = Field(
#         description="Proposal for query structure if answerable", default=None
#     )


class DatabaseQueryState(TypedDict):
    user_request: str  # The original user query
    message_history: List[HumanMessage | AIMessage]  # History of messages
    analysis_result: Optional[AnalysisResult]  # Result of query analysis
    query_proposal: Optional[
        QueryProposal
    ]  # Proposal for query structure if answerable
    human_analyst_feedback: Optional[str]  # Feedback from human expert
    query_results: Optional[str]  # Results of the query
    final_answer: Optional[str]  # Final instructions for SQL agent


SCHEMA_TEMPLATE_PATH = "schema_template.txt"
if "schema_template" not in st.session_state:
    with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
        prompt_temp = file.read()
    st.session_state.schema_template = prompt_temp


# Function to analyze the user request
def analyze_request(state: DatabaseQueryState):
    """Analyze whether the user request can be answered with the database"""
    query_analysis_instructions = """You are a database query assistant with access to questionnaire response data.
    Note : As a conversational chatbot you have access to previous messages history and you have memory of past conversations.

    Database Overview:
    - The database contains data for each year and questionnaire.
    - For every year, each questionnaire includes multiple questions, each offering several answer options.
    - For each answer option, the total number of male and female respondents is recorded.
    - Important: Data is aggregated at the school level only, not individual students. Thus : It is IMPOSSIBLE to answer queries that attempt to link responses across multiple questions.

    Examples:
    - Answerable: "What is the total number of male and female responses for question X in school Y for the year 2015?"
    - Answerable: "What is the total number of students for question X ? aggregate by year, school and gender"
    - Answerable: "What is the number of students that go to school on foot in 2018 per school ?"
    - Answerable: "Nombre d'etudiant qui ont de bons resultats scolaires en 2018?"
    - NOT answerable: "How many students who answered question A also answered question B?"
    - NOT answerable: "What is the total number of students that go to school on foot and have good grades ? aggregate by school and gender"
    ---
    if db related and answerable:
    set is_db_related_and_answerable to True
    and in the response field reformulate the user request to a full well formulated instruction in natural language do not try to convert to SQL.

    Note : For greetings, general non related questions on different subjects, also set to not answerable but in the response field answer it normally in a conversational polite manner as a helpful general chatbot.
    """
    # Format system message
    user_request = state["user_request"]
    history = state["message_history"]

    # Convert message history to the correct format
    messages_history = [
        ("ai", msg.content) if isinstance(msg, AIMessage) else ("human", msg.content)
        for msg in history
        if isinstance(msg, AIMessage) or isinstance(msg, HumanMessage)
    ]

    # Create structured LLM call
    structured_llm = llm.with_structured_output(AnalysisResult)
    messages = [
        ("system", query_analysis_instructions + "Messages History: "),
        *messages_history,  # Unpacking the list of tuples
        ("human", "User_request: " + user_request),
    ]
    print(messages)

    # Generate analysis
    analysis_result = structured_llm.invoke(messages)

    # Return updated state
    return {"analysis_result": analysis_result}


# Conditional edge function to route based on answerability
def route_after_analysis(state: DatabaseQueryState):
    """Determine next step based on whether query is answerable"""

    analysis_result = state["analysis_result"]

    if analysis_result.is_db_related_and_answerable:
        return "check_schema_formulate_instructions"
    else:
        return END


def check_schema_formulate_instructions(state: DatabaseQueryState):
    """Create final query instructions based on the analysis and human feedback"""

    reformulated_request = state["analysis_result"].response
    human_analyst_feedback = state.get("human_analyst_feedback", "Not checked yet")
    previous_query_proposal = state.get("query_proposal", None)

    # System message for finalizing query
    #     system_message = f"""Based on the user's request, the DB schema, and human expert feedback, create the final SQL query instructions.
    #     You are a database query assistant with access to questionnaire response data.
    #     Note : As a conversational chatbot you have access to previous messages history and you have memory of past conversations.
    #     Always set is_accepted_by_human_analyst to False

    #     Database Overview:
    #     - The database contains data for each year and questionnaire.
    #     - For every year, each questionnaire includes multiple questions, each offering several answer options.
    #     - For each answer option, the total number of male and female respondents is recorded.
    #     - Important: Data is aggregated at the school level only, not individual students.

    #     User request: {reformulated_request}
    #     Previous Query Proposal: {previous_query_proposal}
    #     Human Expert Feedback: {human_analyst_feedback}
    #     DB Schema: {st.session_state.schema_template}

    #     Task:
    #     1. Identify the tables, columns, and fields needed to answer the User request.
    #     2. Add an explanation of how these tables, columns, and fields would be used to answer the User request.
    #     If human expert feedback is not provided, set the 'is_accepted_by_human_analyst' field to False and continue.
    #     else :
    #         3. Depending on the human expert feedback, you may need to adjust the query proposal.
    #         4. If the human expert feedback accepts the proposals : Human Expert Feedback: contains words such as coorect or yes : keep the rest of fields as is and set the 'is_accepted_by_human_analyst' field to True.
    # """
    system_message = f"""Based on the user's request, the DB schema, and human expert feedback, create the final SQL query instructions.
    You are a database query assistant with access to questionnaire response data.
    Note : As a conversational chatbot you have access to previous messages history and you have memory of past conversations.
    Always set is_accepted_by_human_analyst to True


    Database Overview:
    - The database contains data for each year and questionnaire.
    - For every year, each questionnaire includes multiple questions, each offering several answer options.
    - For each answer option, the total number of male and female respondents is recorded.
    - Important: Data is aggregated at the school level only, not individual students.

    User request: {reformulated_request}
    Previous Query Proposal: {previous_query_proposal}
    Human Expert Feedback: {human_analyst_feedback}
    DB Schema: {st.session_state.schema_template}

    Task:
    1. Identify the tables, columns, and fields needed to answer the User request.
    2. Add an explanation of how these tables, columns, and fields would be used to answer the User request.
        3. Depending on the human expert feedback, you may need to adjust the query proposal.
"""

    structured_llm = llm.with_structured_output(QueryProposal)
    query_proposal = structured_llm.invoke(system_message)

    return {"query_proposal": query_proposal}


def route_after_feedback(state: DatabaseQueryState):
    """Route based on presence of human feedback"""

    query_proposal = state["query_proposal"]

    # Return updated state
    if query_proposal.is_accepted_by_human_analyst == True:
        return "run_query"
    return "human_feedback"


# Function for human feedback step
def human_feedback(state: DatabaseQueryState):
    """No-op node that should be interrupted for human feedback"""
    pass


def run_query(state: DatabaseQueryState):
    """Run the query"""
    reformulated_request = state["analysis_result"].response
    query_proposal = state["query_proposal"]
    tables = query_proposal.tables
    columns = query_proposal.columns
    fields = query_proposal.fields
    explanation = query_proposal.explanation

    run_query_template = f"""System: You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer.
        You can order the results by a relevant column to return the most interesting examples in the database.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. !!
        To start you should ALWAYS look at the tables in the database to see what you can query. 
        Answer the user request: {reformulated_request} using the following tables, columns, and fields:
        Tables: {tables}
        Columns: {columns}
        Fields: {fields}
        And follow the instructions: {explanation}
        
        Using the tools provided, Excecute the necessary SQL query to answer the user request using all information provided.
        """

    llm = get_llm(MODEL_CONFIG)
    engine = create_engine(st.secrets["db_url"])
    db = SQLDatabase(engine)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    llm_with_tools = llm.bind_tools(tools)
    print("*" * 100)
    print(run_query_template)
    print("*" * 100)

    # response = llm_with_tools.invoke(run_query_template)

    agent_executor = create_react_agent(llm_with_tools, tools)
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": run_query_template}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)

    print("response", step["messages"][-1])
    print("*" * 100)

    return {"query_results": step["messages"][-1].content}


def finalize_query(state: DatabaseQueryState):

    reformulated_request = state["analysis_result"].response
    query_results = state["query_results"]

    finalize_query_template = f"""System: You are an agent responsible for finalizing the query results.
    Given the user request: {reformulated_request}
    The query results are: {query_results}
    Provide the final answer to the user in a clear and well-formulated manner.
    And plot the results in a chart if possible.
        
        """

    python_repl = PythonREPL()

    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. Use to plot charts using only streamlit chart elements, matplotlib is not support in the interface and to print tables use st.dataframe.",
        func=python_repl.run,
    )

    llm = get_llm(MODEL_CONFIG)
    tools = [repl_tool]

    llm_with_tools = llm.bind_tools(tools)

    agent_executor = create_react_agent(llm_with_tools, tools)
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": finalize_query_template}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)
    print("*" * 100)

    print("response", step["messages"][-1])
    print("*" * 100)

    return {"final_answer": step["messages"][-1].content}


# Build the graph
builder = StateGraph(DatabaseQueryState)

# Add nodes
builder.add_node("analyze_request", analyze_request)
builder.add_node(
    "check_schema_formulate_instructions", check_schema_formulate_instructions
)
builder.add_node("human_feedback", human_feedback)
builder.add_node("run_query", run_query)
builder.add_node("finalize_query", finalize_query)

# Add edges
builder.add_edge(START, "analyze_request")
builder.add_conditional_edges(
    "analyze_request",
    route_after_analysis,
    ["check_schema_formulate_instructions", END],
)
builder.add_conditional_edges(
    "check_schema_formulate_instructions",
    route_after_feedback,
    ["human_feedback", "run_query"],
)
builder.add_edge("human_feedback", "check_schema_formulate_instructions")
builder.add_edge("run_query", "finalize_query")
builder.add_edge("finalize_query", END)

# Compile the graph
memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


# Example usage
# config = {"configurable": {"thread_id": "123"}}
# result = graph.invoke({"user_request": "What is the total number of male students who answered 'Yes' to Question 3 in 2018?"}, config)

# Display the graph
# import os
# import subprocess
# import sys

# # Get the PNG data from your graph
# png_data = graph.get_graph(xray=1).draw_mermaid_png()

# # Write the PNG data to a file
# filename = "graph.png"
# with open(filename, "wb") as f:
#     f.write(png_data)

# # Open the file using the default image viewer based on your OS
# if sys.platform.startswith("win"):
#     os.startfile(filename)
# elif sys.platform == "darwin":  # macOS
#     subprocess.call(["open", filename])
# else:  # Linux and others
#     subprocess.call(["xdg-open", filename])

# # Input
# #user_request = "Nombre d'etudiant qui ont de bons resultats scolaires et vont sur pieds a l'ecole"
# user_request = "Nombre d'etudiant qui ont de bons resultats scolaires en 2018"
# #user_request = "Bonjour"

# thread = {"configurable": {"thread_id": "1"}}

# # Run the graph until the first interruption
# for event in graph.stream({"user_request":user_request}, thread, stream_mode="values"):
#     print(event)

# state = graph.get_state(thread)
# print(state.next)

# # If we are satisfied, then we simply supply no feedback
# further_feedack = "seems correct"
# graph.update_state(thread, {"human_analyst_feedback":
#                             further_feedack}, as_node="human_feedback")

# for event in graph.stream(None, thread, stream_mode="updates"):
#     print("--Node--")
#     node_name = next(iter(event.keys()))
#     print(node_name)


# final_state = graph.get_state(thread)
# print(final_state.values.get('final_query_instructions'))
def invoke_our_graph(user_input, callables, thread_id):
    # Ensure the callables parameter is a list as you can have multiple callbacks
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    # Invoke the graph with the current messages and callback configuration
    return (
        graph.invoke(
            {
                "user_request": user_input,
                "message_history": st.session_state["messages"],
            },
            config={"callbacks": callables, "configurable": {"thread_id": thread_id}},
        ),
        graph,
    )
