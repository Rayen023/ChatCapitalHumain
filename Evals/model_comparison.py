import json
import os
import time
import uuid
from typing import Annotated, Any, Dict, List

import streamlit as st

# Import the questions from answerable_questions.py
from answerable_questions import questions as data
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine

data = data[2:20].copy()
# data = data.copy()
# data = [
#     "Pour l'école W.-A.-Losier en 2014, quel était les 4 domaines d'études postsecondaires (question 20) le plus fréquemment choisi ? ",
# ] 
# Load schema template
SCHEMA_TEMPLATE_PATH = os.path.join("utils", "schema_template.txt")
with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as file:
    schema_template = file.read()

load_dotenv()

# Define the models to test
MODELS = [
    # "google/gemini-2.5-pro-preview",
    # "google/gemini-2.5-flash-preview-05-20",
    # "google/gemini-2.5-flash-preview-05-20:thinking",
    # "anthropic/claude-sonnet-4",
    "openai/gpt-4.1",
]

# Base model configuration
BASE_MODEL_CONFIG = {
    "temperature": 0,
    "max_tokens": 8096,
    "timeout": None,
    "max_retries": 2,
    "streaming": False,  # Set to False for testing
}


class State(MessagesState):
    """Represents the state of our graph."""

    messages: Annotated[list[BaseMessage], add_messages]


def setup_database():
    """Set up the database connection."""
    engine = create_engine(st.secrets["db_url"])
    db = SQLDatabase(engine)
    return db


def create_agent(model_name, db):
    """Create an agent with the specified model."""
    # Configure the model
    model_config = BASE_MODEL_CONFIG.copy()
    model_config["model_name"] = model_name

    # Initialize the LLM
    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENROUTER_API_KEY"],
        openai_api_base=st.secrets["OPENROUTER_BASE_URL"],
        **model_config,
    )

    # System message content (same as in the original app)
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

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message_content),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Setup SQL tools
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    tools = tools[:1]

    # Create the ReAct agent
    agent_executor = create_react_agent(llm, tools=tools, prompt=prompt)

    # Create a runnable
    runnable = agent_executor.with_config({"run_name": "agent"})

    # Create and compile the graph
    graph_builder = StateGraph(State)

    def chatbot(state: State) -> dict:
        response = runnable.invoke({"messages": state["messages"]})
        return {"messages": response["messages"]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()
    graph_runnable = graph_builder.compile(checkpointer=memory)

    return graph_runnable


def process_question(question, model_name, db):
    """Process a question with the specified model and measure time."""
    start_time = time.time()

    try:
        # Create an agent with the model
        graph_runnable = create_agent(model_name, db)

        # Format the input for the graph
        input_message = HumanMessage(content=question)

        # Invoke the graph runnable
        thread_id = str(uuid.uuid4())
        response = graph_runnable.invoke(
            {"messages": [input_message]},
            config={"configurable": {"thread_id": thread_id}},
        )

        # Get the response content
        answer = response["messages"][-1].content

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        return {"answer": answer, "time_seconds": elapsed_time}
    except Exception as e:
        # Handle errors
        elapsed_time = time.time() - start_time
        return {"answer": f"Error: {str(e)}", "time_seconds": elapsed_time}


def main():
    """Main function to run the comparison."""
    print(
        f"Starting model comparison with {len(MODELS)} models and {len(data)} questions."
    )

    # Set up database connection
    db = setup_database()

    results = []

    # Process each question with each model
    for i, question in enumerate(data):
        question_result = {"question": question, "models": {}}

        print(f"Processing question {i+1}/{len(data)}: {question[:50]}...")

        for model_name in MODELS:
            print(f"  Testing model: {model_name}")
            result = process_question(question, model_name, db)
            question_result["models"][model_name] = result
            print(f"  Completed in {result['time_seconds']:.2f} seconds")

        results.append(question_result)

    # Save results to JSON file
    output_filename = f"model_comparison_results_{int(time.time())}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    main()
