from typing import List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from utils.utils import get_llm

MODEL_CONFIG = {
    # "model_name": "anthropic/claude-3.5-sonnet:beta",
    # "model_name": "openai/gpt-4o-mini",
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
        description="Explanation of how these tables, columns, and fields would be used to answer the query",
    )


class AnalysisResult(BaseModel):
    is_answerable: bool = Field(
        description="Whether the query can be answered with the available database",
    )
    explanation: str = Field(
        description="Explanation of why the query is or is not answerable",
    )
    query_proposal: Optional[QueryProposal] = Field(
        description="Proposal for query structure if answerable", default=None
    )


class DatabaseQueryState(TypedDict):
    user_request: str  # The original user query
    analysis_result: Optional[AnalysisResult]  # Result of query analysis
    human_analyst_feedback: Optional[str]  # Feedback from human expert
    final_query_instructions: Optional[str]  # Final instructions for SQL agent


SCHEMA_TEMPLATE_PATH = "schema_template.txt"
if "schema_template" not in st.session_state:
    with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
        prompt_temp = file.read()
    st.session_state.schema_template = prompt_temp


# System prompt for query analysis
query_analysis_instructions = """You are a database query assistant with access to questionnaire response data.

Database Overview:
- The database contains data for each year and questionnaire.
- For every year, each questionnaire includes multiple questions, each offering several answer options.
- For each answer option, the total number of male and female respondents is recorded.
- Important: Data is aggregated at the school level only, not individual students.

Your task is to analyze the user's request and determine:
1. Is the request answerable with the available data?
2. If NOT answerable, explain why (especially if it tries to link multiple questions). 
3. If answerable, suggest the tables, columns, and fields needed for the query.

Remember: It is IMPOSSIBLE to answer queries that attempt to link responses across multiple questions.

Examples:
- Answerable: "What is the total number of male and female responses for question X in school Y for the year 2015?"
- NOT answerable: "How many students who answered question A also answered question B?"
---

Note : For greetings, general non related questions on different subjects, also set to not answerable but in the explanation field answer it normally in a conversational polite manner as a helpful general chatbot.

{schema_template}

""".format(
    schema_template=st.session_state.schema_template
)


# Function to analyze the user request
def analyze_request(state: DatabaseQueryState):
    """Analyze whether the user request can be answered with the database"""

    # Format system message
    user_request = state["user_request"]

    # Create structured LLM call
    structured_llm = llm.with_structured_output(AnalysisResult)
    messages = [
        (
            "system",
            query_analysis_instructions,
        ),
        ("human", "User_request: " + user_request),
    ]

    # Generate analysis
    analysis_result = structured_llm.invoke(messages)

    # Return updated state
    return {"analysis_result": analysis_result}


# Function for human feedback step
def human_feedback(state: DatabaseQueryState):
    """No-op node that should be interrupted for human feedback"""
    pass


# Function to finalize query instructions
def finalize_query(state: DatabaseQueryState):
    """Create final query instructions based on the analysis and human feedback"""

    user_request = state["user_request"]
    analysis_result = state["analysis_result"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # System message for finalizing query
    system_message = f"""Based on the user's request, the analysis of whether it's answerable, and human expert feedback, create the final SQL query instructions.

User Request: {user_request}

Analysis: {analysis_result.explanation}

Proposed Tables: {', '.join(analysis_result.query_proposal.tables) if analysis_result.query_proposal else 'None'}
Proposed Columns: {', '.join(analysis_result.query_proposal.columns) if analysis_result.query_proposal else 'None'}
Proposed Fields: {', '.join(analysis_result.query_proposal.fields) if analysis_result.query_proposal else 'None'}

Human Expert Feedback: {human_analyst_feedback}

Create a clear and well-formatted set of instructions for the SQL agent."""

    # Generate final instructions
    response = llm.invoke([SystemMessage(content=system_message)])

    # Return updated state
    return {"final_query_instructions": response.content}


# Conditional edge function to route based on answerability
def route_after_analysis(state: DatabaseQueryState):
    """Determine next step based on whether query is answerable"""

    analysis_result = state["analysis_result"]

    if analysis_result.is_answerable:
        return "human_feedback"
    else:
        return END


# Conditional edge function after human feedback
def should_continue(state: DatabaseQueryState):
    """Route based on presence of human feedback"""

    human_analyst_feedback = state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "finalize_query"

    # If no human feedback, end
    return END


# Build the graph
builder = StateGraph(DatabaseQueryState)

# Add nodes
builder.add_node("analyze_request", analyze_request)
builder.add_node("human_feedback", human_feedback)
builder.add_node("finalize_query", finalize_query)

# Add edges
builder.add_edge(START, "analyze_request")
builder.add_conditional_edges(
    "analyze_request", route_after_analysis, ["human_feedback", END]
)
builder.add_conditional_edges(
    "human_feedback", should_continue, ["finalize_query", END]
)
builder.add_edge("finalize_query", END)

# Compile the graph
memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


# Example usage
# config = {"configurable": {"thread_id": "123"}}
# result = graph.invoke({"user_request": "What is the total number of male students who answered 'Yes' to Question 3 in 2018?"}, config)

# Display the graph
import os
import subprocess
import sys

# Get the PNG data from your graph
png_data = graph.get_graph(xray=1).draw_mermaid_png()

# Write the PNG data to a file
filename = "graph.png"
with open(filename, "wb") as f:
    f.write(png_data)

# Open the file using the default image viewer based on your OS
if sys.platform.startswith("win"):
    os.startfile(filename)
elif sys.platform == "darwin":  # macOS
    subprocess.call(["open", filename])
else:  # Linux and others
    subprocess.call(["xdg-open", filename])

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
            {"user_request": user_input},
            config={"callbacks": callables, "configurable": {"thread_id": thread_id}},
        ),
        graph,
    )
