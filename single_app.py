import os
import uuid
from typing import Annotated

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from sqlalchemy import create_engine
from typing_extensions import TypedDict

from utils.st_callable_util import get_streamlit_cb

APP_TITLE = "Capital Humain"
APP_ICON_PATH = "images/deer.png"
USER_AVATAR_PATH = "images/avataruser.png"
SCHEMA_TEMPLATE_PATH = os.path.join("utils", "schema_template.txt")
WELCOME_MESSAGE = "Comment puis-je vous aider ? | How can I help you ?"
with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
    schema_template = file.read()

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON_PATH,
)

load_dotenv()

MODEL_CONFIG = {
    # "model_name": "google/gemini-2.0-flash-001",
    "model_name": "anthropic/claude-3.7-sonnet",
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": True,
}


class State(MessagesState):
    """Represents the state of our graph."""

    # Use the built-in message state handler
    messages: Annotated[list[BaseMessage], add_messages]


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
# llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", **MODEL_CONFIG)


def chatbot(state: State) -> dict:
    """Process user input and generate a response using the SQL toolkit."""

    system_message_content = f"""
    Vous êtes un agent conversationnel expert de la base de données Capital Humain. Votre rôle est de répondre aux questions des utilisateurs en utilisant les données disponibles, en respectant scrupuleusement les contraintes de la base.

    **Contexte et Limitations:**

    *   **Données Agrégées:** La base de données contient des données *agrégées* par école, année, questionnaire et sexe.  Vous n'avez *PAS* accès aux réponses individuelles des élèves.
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
    toolkit = SQLDatabaseToolkit(db=st.session_state.db, llm=llm)
    tools = toolkit.get_tools()

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


def invoke_our_graph(user_input, callables, thread_id):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    # Properly format the input for the graph
    input_message = HumanMessage(content=user_input)

    return graph_runnable.invoke(
        {"messages": [input_message]},
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

if st.session_state["messages"] and isinstance(
    st.session_state.messages[-1], HumanMessage
):
    user_message = st.session_state.messages[-1].content

    with st.chat_message("assistant", avatar=APP_ICON_PATH):
        # Create a container for tool calls that will persist
        tool_calls_container = st.container(border=True)  # st empty overwrites
        response_placeholder = st.empty()
        # Get the callback with the tool calls container
        streamlit_callback = get_streamlit_cb(tool_calls_container)
        graph_response = invoke_our_graph(
            user_message, [streamlit_callback], st.session_state["thread_id"]
        )
        final_response = graph_response["messages"][-1].content
        st.session_state["messages"].append(AIMessage(content=final_response))
        response_placeholder.write(final_response)

with st.sidebar:
    if "thread_id" in st.session_state:
        st.write(st.session_state["thread_id"])
