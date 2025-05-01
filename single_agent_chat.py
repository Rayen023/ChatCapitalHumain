import asyncio
import os
import uuid
from typing import Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine

from utils.database import save_chat_logs

APP_ICON_PATH = "images/deer.png"
USER_AVATAR_PATH = "images/avataruser.png"
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
SCHEMA_TEMPLATE_PATH = os.path.join("utils", "schema_template.txt")
with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
    schema_template = file.read()
DEBUGGING = st.secrets.get("DEBUGGING", False)

load_dotenv()

# Define available models
available_models = [
    "google/gemini-2.5-pro-preview-03-25",
    "google/gemini-2.0-flash-001",
    "openai/o3-mini",
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4.1",
]

# Initialize the selected model in session state if it doesn't exist
if "sa_selected_model" not in st.session_state:
    st.session_state.sa_selected_model = "anthropic/claude-3.7-sonnet"  # Default model


# Define callback for model selection changes
def on_model_change():
    st.session_state.sa_selected_model = st.session_state.model_selector


# Restore the sidebar elements to ensure they display properly
st.sidebar.title("ChatCapitalHumain")

# Add a selectbox to the sidebar for model selection
st.sidebar.selectbox(
    "Select Model",
    options=available_models,
    index=available_models.index(st.session_state.sa_selected_model),
    key="model_selector",
    on_change=on_model_change,
)

# Add any other sidebar elements that were previously there

MODEL_CONFIG = {
    "model_name": st.session_state.sa_selected_model,
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENROUTER_API_KEY"],
    openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
    **MODEL_CONFIG,
)


class State(MessagesState):
    """Represents the state of our graph."""

    # Use the built-in message state handler
    messages: Annotated[list[BaseMessage], add_messages]


if "db" not in st.session_state:
    engine = create_engine(st.secrets["db_url"])
    st.session_state.db = SQLDatabase(engine)

if "single_messages" not in st.session_state:
    st.session_state["single_messages"] = [AIMessage(content=WELCOME_MESSAGE)]
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())


def chatbot(state: State) -> dict:
    """Process user input and generate a response using the SQL toolkit."""

    system_message_content = f"""
    Vous êtes un agent conversationnel expert de la base de données Capital Humain. Votre rôle est de répondre aux questions des utilisateurs en utilisant les données disponibles, en respectant scrupuleusement les contraintes de la base.

    **Contexte et Limitations:**

    *   **Données Agrégées:** La base de données contient des données *agrégées* par école, année, questionnaire et sexe.  Vous n'avez *PAS* accès aux réponses individuelles des élèves.
    - summed_students_responses represente le nombre d'etudiants ayant choisi cette option.
    *   **Schéma de la Base de Données:** {schema_template}

    **Instructions et Comportement Attendu :**

    1.  **Analyser Attentivement la Requête:** Avant de tenter d'y répondre, déterminez si la requête est faisable *avec les données agrégées disponibles*.
    2.  **Identifier les Tentatives de Corrélation Implicites:** Soyez particulièrement vigilant face aux questions qui *semblent* simples mais qui impliquent une corrélation entre les réponses à différentes questions.
    3.  **Valider la Requête:** La requête est-elle conforme aux règles suivantes ?

        *   **Requêtes Acceptables:**
            *   Se concentrent sur *une seule question* et ses options de réponse.
            *   Peuvent être agrégées par école, année et/ou sexe.
            *   Autorisent des agrégations de *plusieurs options de réponse d'une même question*.  (Exemple: "Combien d'élèves vont à l'école à pied OU en bus en 2018, par école?")

        *   **Requêtes Inacceptables:**
            *   Tentent de lier les réponses de *deux questions différentes*.
            *   Tentent d'établir une corrélation au niveau individuel des élèves. (Exemple: "Combien d'élèves qui vont à l'école à pied ont également des bonnes notes?")
            *   Requêtes du type "les élèves qui ont répondu X à la question 1 *ET* ont répondu Y à la question 2".

    4.  **Répondre Clairement :**

        *   **Si la requête est ACCEPTABLE:**  Utilisez les outils à votre disposition pour formuler la requête SQL et obtenir la réponse.
        *   **Si la requête est INACCEPTABLE:** Expliquez clairement à l'utilisateur pourquoi vous ne pouvez pas répondre à sa question.  Mettez l'accent sur la limitation des données agrégées.  Proposez une alternative si possible.  (Exemple: "Je ne peux pas répondre à cette question car elle nécessite de lier les réponses de deux questions différentes. Cependant, je peux vous donner le nombre d'élèves qui vont à l'école à pied et le nombre d'élèves qui ont des bonnes notes séparément.")

    **Exemple d'Explication d'une Requête Refusée :**

    "Je suis désolé, je ne peux pas répondre à cette question. La base de données ne conserve pas les réponses individuelles des élèves. Je peux uniquement vous fournir des statistiques agrégées par école, année, questionnaire et sexe. Tenter de déterminer combien d'élèves qui utilisent un certain moyen de transport ont de bonnes notes nécessiterait de connaître les réponses individuelles, ce qui n'est pas possible."
    
    Si la reponse la plus frequente pour une question est "Non répondu", continuer la recherche à l'option suivante et informer l'utilisateur.
    """

    # Create a chat template with system message, history, and user input
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(
                variable_name="messages"
            ),  # Use the state messages directly
        ]
    )

    # Setup SQL tools
    if "db" not in st.session_state:
        engine = create_engine(st.secrets["db_url"])
        st.session_state.db = SQLDatabase(engine)

    toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=llm)
    tools = toolkit.get_tools()
    print(f"Tools: {tools}")
    print(f"type: {type(tools)}")
    for index, tool in enumerate(tools):
        print(f"Tool {index}: {tool.name}")
    # remove index 1 2 and 3
    tools = tools[:1]

    # Create the ReAct agent with the tools and prompt
    agent_executor = create_react_agent(llm, tools=tools, prompt=prompt)

    # Create a runnable that includes the agent
    runnable = agent_executor.with_config({"run_name": "agent"})

    # Get the response
    response = runnable.invoke(
        {
            "messages": state["messages"],
        }
    )

    # Return the updated state with the agent's response added
    return {"messages": response["messages"]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")


@st.cache_resource
def cache_memory():
    return MemorySaver()


memory = cache_memory()
graph_runnable = graph_builder.compile(checkpointer=memory)


@st.fragment
def display_message_history():
    for message in st.session_state["single_messages"]:
        if isinstance(message, AIMessage):
            st.chat_message("assistant", avatar=APP_ICON_PATH).write(message.content)
        elif isinstance(message, HumanMessage):
            st.chat_message("user", avatar=USER_AVATAR_PATH).write(message.content)


# Display current messages
display_message_history()


async def invoke_our_graph(user_input, callables, thread_id):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    # Properly format the input for the graph
    input_message = HumanMessage(content=user_input)

    # Create a message container for streaming that will be cleaned up after
    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        message_placeholder = st.empty()
        accumulated_content = ""

        async for event in graph_runnable.astream(
            {"messages": [input_message]},
            config={
                "callbacks": callables,
                "configurable": {"thread_id": thread_id},
            },
            stream_mode="messages",
        ):
            if (
                not isinstance(event[0], ToolMessage)
                and event[0].content
                and event[1]["langgraph_node"] == "agent"
            ):
                content = event[0].content
                accumulated_content += content
                message_placeholder.write(accumulated_content)

            if event[1]["langgraph_node"] != "agent":
                accumulated_content = ""

    # After streaming completes, add the final message to session state if it has content
    if accumulated_content:
        ai_message = AIMessage(content=accumulated_content)
        st.session_state["single_messages"].append(ai_message)

        # Force display of messages to update with the new message
        st.rerun()

    return {"messages": st.session_state["single_messages"]}


# User input handling
user_message = st.chat_input("Message ChatCapitalHumain...")
if user_message:
    # Display user message
    st.chat_message("user", avatar=USER_AVATAR_PATH).write(user_message)
    # Add to session state
    st.session_state["single_messages"].append(HumanMessage(content=user_message))

    # Process the message and display response
if st.session_state["single_messages"] and isinstance(
    st.session_state["single_messages"][-1], HumanMessage
):
    user_message = st.session_state["single_messages"][-1].content
    with st.spinner("Thinking...", show_time=True):
        # Run the async function using asyncio
        asyncio.run(invoke_our_graph(user_message, [], st.session_state["thread_id"]))

    if not DEBUGGING and st.user.get("email"):
        save_chat_logs("single_messages")
