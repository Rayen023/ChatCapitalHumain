from typing import List, Optional

import streamlit as st
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
    "model_name": "google/gemini-2.0-flash-001",
    #"model_name": "anthropic/claude-3.7-sonnet",
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
    questions_text: List[str] = Field(
        description="The questions exact text whose answers are most relevant for the query",
    )
    fields: List[str] = Field(
        description="The specific fields or values to filter or select in the query",
    )
    explanation: str = Field(
        description="Explanation of how the tables, fields would be used to answer the query with instructions in SQL and what SQL functions can be used",
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




# Function to analyze the user request
def analyze_request(state: DatabaseQueryState):
    """Analyze whether the user request can be answered with the database"""
    query_analysis_instructions = """

### **Role and Context**

You are the **first agent** in a chain of agents. Your task:
1. Interact seamlessly with the user. YOu have access to previous messages history and you have memory of past conversations.
2. Determine whether their query requires data from the **Capital Humain** database (spanning 2004–2019) or if it is a general/non-database question.
3. Forward the user’s query in the correct format, indicating whether it is (a) database-related and answerable, or (b) not database-related / not answerable.

---

### **Output Format**

You must provide **two fields**:
1. **is_db_related_and_answerable**: a boolean (`True` or `False`)
2. **response**: a string with your answer or explanation

---

### **Decision Logic**

1. **If the user’s query is NOT related to the database**  
   - Example: simple greetings, general knowledge, conversation, etc.  
   - **Set** `is_db_related_and_answerable = False`.  
   - **In the `response` field**, answer the user as a normal conversational chatbot would.

2. **If the user’s query IS related to the database**  
   The **Capital Humain** database covers:
   - **Schools** (2004–2019):
     1. Aux quatre vents  
     2. Centre La Fontaine  
     3. Secondaire Népisiguit  
     4. Louis-Mailloux  
     5. Marie-Esther  
     6. Roland-Pépin  
     7. W.-A.-Losier  
   - **Questionnaires** (each with multiple questions and response breakdowns by gender):  
     1. Questions Générales  
     2. SD – Renseignements Socio-Démographiques  
     3. ED – Éducation PostSecondaire  
     4. MT – Marché du travail  
     5. RE – Attente d'emploi / Recherche d'emploi / Sans emploi

   **Important constraints**:  
   - Data is aggregated **only** by:
     - School  
     - Year  
     - Questionnaire  
     - Gender  
   - There is **no way to correlate** individual student answers **across different questions**.  

   Therefore:
   1. **If the user’s request attempts to correlate or link answers from multiple distinct questions**:  
      - **Set** `is_db_related_and_answerable = False`.  
      - **In the `response` field**, explain why this answer is impossible (cross-question correlation is not supported by the database).

   2. **If the user’s request is about one question (or multiple sub-answers of the same question)** and does not require cross-question correlation:  
      - **Set** `is_db_related_and_answerable = True`.  
      - **In the `response` field**, restate the user’s query clearly and concisely so it can be passed on to the database query agents.

---

### **Examples**

**Answerable (set `is_db_related_and_answerable = True` and return the query):**  
- “What is the total number of male and female responses for question X in school Y for 2015?”  
- “How many students chose to go to school by foot or by bus in 2018, by school?”  
  - (*These both concern a single question or its various answer choices, without linking separate questions.*)

**Not Answerable (set `is_db_related_and_answerable = False` and explain):**  
- “How many students who answered question A also answered question B?” (This is cross-question correlation.)  
- “How many students go to school on foot **and** have good grades, by school and gender?” (Again, involves two separate questions: transportation method and grades.)

---

Use this decision tree and output format in every interaction:
- If **DB-related & answerable** → `True` + restate query  
- Otherwise → `False` + normal conversational answer or explanation  

This ensures a consistent, structured approach to every user query.
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
    human_analyst_feedback = state.get("human_analyst_feedback", "Not checked by human expert yet")
    previous_query_proposal = state.get("query_proposal", None)

    system_message = f"""
    You are an agent in a multi-agent workflow. Your role is to analyze database queries by:
    1. Examining the provided database schema
    2. Understanding the user's request
    3. Identifying necessary questions and corresponding fields to answer the query

    WORKFLOW:
    - First, you will suggest query components to a human expert for review
    - Only after receiving positive feedback will you forward instructions to the SQL agent

    INPUT:
    - User request: {reformulated_request}
    - Previous field suggestions: {previous_query_proposal}
    - Human expert feedback: {human_analyst_feedback}
    - Database schema: {st.session_state.schema_template}

    TASK:
    1. Identify the specific questions_text and database fields needed to answer the user request
    - Use exact field names from the schema for accurate processing
    - Be precise and comprehensive in your selections

    2. Provide a clear explanation of how these tables and fields would be used to fulfill the request

    3. Set approval status:
    - If no human feedback is provided yet: set 'is_accepted_by_human_analyst' to False
    - If human feedback contains approval words (e.g., "correct" or "yes"): set 'is_accepted_by_human_analyst' to True
    - If human feedback suggests changes: adjust your proposal accordingly before submitting
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
    questions_text = query_proposal.questions_text
    fields = query_proposal.fields
    explanation = query_proposal.explanation

    run_query_template = f"""
        You are a PostgreSQL query agent. Your task is to generate and execute SQL queries based on user requests, following guidelines and return a final response with results found.

        GUIDELINES:
        - Always start by examining the database tables
        - Create syntactically correct PostgreSQL queries
        - If you encounter errors, rewrite and retry the query
        - NEVER execute DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
        - Only use the tools provided to interact with the database
        - Only use information returned by these tools in your final answer

        USER REQUEST:
        {reformulated_request}

        DATABASE ELEMENTS TO USE:
        - Questions: {questions_text}
        - Response options: {fields}
        - Usage instructions: {explanation}

        IMPORTANT: Use these exact strings in your query as they match the database structure.

        ACTION:
        Execute the appropriate SQL query that addresses the user request using all provided information.
        """

    llm = get_llm(MODEL_CONFIG)
    engine = create_engine(st.secrets["db_url"])
    db = SQLDatabase(engine)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    llm_with_tools = llm.bind_tools(tools)

    agent_executor = create_react_agent(llm_with_tools, tools)
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": run_query_template}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    return {"query_results": step["messages"][-1].content}


def finalize_query(state: DatabaseQueryState):

    reformulated_request = state["analysis_result"].response
    query_results = state["query_results"]

    finalize_query_template = f"""
    System: You are a result visualization and formatting agent.

    INPUT:
    - User request: {reformulated_request}
    - Query results: {query_results}

    TASK:
    1. Provide a clear, well-formatted answer based on the query results
    2. Create visualization using ONLY Streamlit components

    VISUALIZATION REQUIREMENTS:
    - ONLY use Streamlit chart elements (st.line_chart, st.bar_chart, st.area_chart, etc.)
    - NEVER use matplotlib, pyplot, seaborn or other external plotting libraries
    - For data display, use st.dataframe() exclusively

    in your final formaated response include:
    1. First provide the textual answer to the user query
    2. Then include a code block with visualization code
    """

    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell for executing code. Use this to create exactly ONE Streamlit visualization. NEVER use matplotlib or other external plotting libraries. For data display, use ONLY st.dataframe().",
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
