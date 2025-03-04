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


def update_model():
    # Update the model config with the selected model
    st.session_state["MODEL_CONFIG"]["model_name"] = st.session_state["selected_model"]
    # Reinitialize the LLM with the new configuration
    st.session_state["llm"] = get_llm(st.session_state["MODEL_CONFIG"])


def display_model_selector():
    with st.sidebar:
        # Model selection dropdown
        DEFAULT_MODEL_CONFIG = {
            "model_name": "anthropic/claude-3.7-sonnet",
            "temperature": 0,
            "max_tokens": 8096,
            "timeout": None,
            "max_retries": 2,
            "streaming": True,
        }
        if "MODEL_CONFIG" not in st.session_state:
            st.session_state["MODEL_CONFIG"] = DEFAULT_MODEL_CONFIG.copy()
        if "llm" not in st.session_state:
            st.session_state["llm"] = get_llm(st.session_state["MODEL_CONFIG"])

        model_options = [
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-sonnet",
            "anthropic/claude-3.7-sonnet",
            # "openai/o3-mini",
            "openai/o3-mini-high",
        ]

        # Get the index of the currently selected model
        current_model = st.session_state["MODEL_CONFIG"]["model_name"]
        try:
            default_index = model_options.index(current_model)
        except ValueError:
            default_index = (
                2  # Default to claude-3.7-sonnet if current model not in list
            )

        # Create the selectbox without using key="selected_model"
        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=default_index,
            key="selected_model",
            on_change=update_model,
        )


# Define our structured data models
class QueryProposal(BaseModel):
    questions_text: List[str] = Field(
        description="The questions exact text whose answers are most relevant for the query",
    )
    response_options: List[str] = Field(
        description="The specific response options values under questions_text that can be used to answer the query",
    )
    explanation: str = Field(
        description="Explanation of how the tables, fields, question and responses exact texts would be used to answer the query with instructions in SQL and what SQL functions can be used",
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

### **Rôle et Contexte**

Vous êtes le **premier agent** dans une chaîne d'agents. Votre tâche :
1. Interagir de manière fluide avec l'utilisateur. Vous avez accès à l'historique des messages précédents et vous avez une mémoire des conversations passées.
2. Déterminer si sa requête nécessite des données de la base de données **Capital Humain** (couvrant 2004-2019) ou s'il s'agit d'une question générale/non liée à la base de données.
3. Transmettre la requête de l'utilisateur dans le format correct, en indiquant si elle est (a) liée à la base de données et répondable, ou (b) non liée à la base de données / non répondable.

Vous pouvez répondre à des questions générales ou à des tâches qui ne nécessitent pas d'accès à la base de données comme le ferait un chatbot conversationnel normal, mais définir is_db_related_and_answerable à False.

---

### **Format de Sortie**

Vous devez fournir **deux champs** :
1. **is_db_related_and_answerable** : un booléen (`True` ou `False`)
2. **response** : une chaîne avec votre réponse ou explication

---

### **Logique de Décision**

1. **Si la requête de l'utilisateur n'est PAS liée à la base de données**  
   - Exemple : simples salutations, connaissances générales, conversation, etc.  
   - **Définir** `is_db_related_and_answerable = False`.  
   - **Dans le champ `response`**, répondez à l'utilisateur comme le ferait un chatbot conversationnel normal.

2. **Si la requête de l'utilisateur EST liée à la base de données**  
   La base de données **Capital Humain** couvre :
   - **Écoles** (2004-2019) :
     1. Aux quatre vents  
     2. Centre La Fontaine  
     3. Secondaire Népisiguit  
     4. Louis-Mailloux  
     5. Marie-Esther  
     6. Roland-Pépin  
     7. W.-A.-Losier  
   - **Questionnaires** (chacun avec plusieurs questions et répartition des réponses par sexe) :  
     1. Questions Générales  
     2. SD – Renseignements Socio-Démographiques  
     3. ED – Éducation PostSecondaire  
     4. MT – Marché du travail  
     5. RE – Attente d'emploi / Recherche d'emploi / Sans emploi

   **Contraintes importantes** :  
   - Les données sont agrégées **uniquement** par :
     - École  
     - Année  
     - Questionnaire  
     - Sexe  
   - Il n'y a **aucun moyen de corréler** les réponses individuelles des élèves **à travers différentes questions**.  

   Par conséquent :
   1. **Si la demande de l'utilisateur tente de corréler ou de lier des réponses de plusieurs questions distinctes** :  
      - **Définir** `is_db_related_and_answerable = False`.  
      - **Dans le champ `response`**, expliquez pourquoi cette réponse est impossible (la corrélation entre questions n'est pas prise en charge par la base de données).

   2. **Si la demande de l'utilisateur concerne une question (ou plusieurs sous-réponses de la même question)** et ne nécessite pas de corrélation entre questions :  
      - **Définir** `is_db_related_and_answerable = True`.  
      - **Dans le champ `response`**, reformulez la requête de l'utilisateur de manière claire et concise afin qu'elle puisse être transmise aux agents de requête de base de données.

---

### **Exemples**

**Répondable (définir `is_db_related_and_answerable = True` et retourner la requête) :**  
- "Quel est le nombre total de réponses masculines et féminines pour la question X dans l'école Y pour 2015 ?"  
- "Combien d'élèves ont choisi d'aller à l'école à pied ou en bus en 2018, par école ?"  
  - (*Ces deux questions concernent une seule question ou ses différents choix de réponse, sans lier des questions séparées.*)

**Non Répondable (définir `is_db_related_and_answerable = False` et expliquer) :**  
- "Combien d'élèves qui ont répondu à la question A ont également répondu à la question B ?" (Il s'agit d'une corrélation entre questions.)  
- "Combien d'élèves vont à l'école à pied **et** ont de bonnes notes, par école et par sexe ?" (Encore une fois, cela implique deux questions distinctes : le mode de transport et les notes.)

---

Utilisez cet arbre de décision et ce format de sortie dans chaque interaction :
- Si **liée à la BD & répondable** → `True` + reformuler la requête et ta response passe à l'agent suivant dans le chaîne
- Sinon → `False` + réponse conversationnelle normale ou explication et ta response à l'utilisateur directement

Cela garantit une approche cohérente et structurée pour chaque requête utilisateur.
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
    structured_llm = st.session_state["llm"].with_structured_output(AnalysisResult)
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
    human_analyst_feedback = state.get(
        "human_analyst_feedback", "Not checked by human expert yet"
    )
    previous_query_proposal = state.get("query_proposal", None)

    system_message = f"""
    Vous êtes un agent dans un flux de travail multi-agents. Votre rôle est d'analyser les requêtes de base de données en :
    1. Examinant le schéma de base de données fourni
    2. Comprenant la demande de l'utilisateur
    3. Identifiant les questions nécessaires et les champs correspondants pour répondre à la requête

    FLUX DE TRAVAIL :
    - D'abord, vous suggérerez des composants de requête à un expert humain pour révision
    - Seulement après avoir reçu un retour positif, vous transmettrez les instructions à l'agent SQL

    ENTRÉE :
    - Demande utilisateur : {reformulated_request}
    - Suggestions précédentes : {previous_query_proposal}
    - Retour d'expert humain : {human_analyst_feedback}
    - Schéma de base de données : {st.session_state.schema_template}

    TÂCHE :
    1. Identifiez les questions_text spécifiques et les champs de base de données nécessaires pour répondre à la demande de l'utilisateur
    - Utilisez les noms exacts des champs du schéma pour un traitement précis
    - Soyez précis et exhaustif dans vos sélections

    2. Fournissez une explication claire de la façon dont ces tables et champs seraient utilisés pour répondre à la demande

    3. Définissez le statut d'approbation :
    - Si aucun retour humain n'est encore fourni : définissez 'is_accepted_by_human_analyst' à False
    - Si le retour humain contient des mots d'approbation (par ex., "correct" ou "oui") : définissez 'is_accepted_by_human_analyst' à True
    - Si le retour humain suggère des changements : ajustez votre proposition en conséquence avant de la soumettre
    """
    schema_llm_model_config = st.session_state["MODEL_CONFIG"].copy()
    schema_llm_model_config["model_name"] = "google/gemini-2.0-flash-001"
    schema_llm = get_llm(schema_llm_model_config)
    structured_llm = schema_llm.with_structured_output(QueryProposal)

    # structured_llm = st.session_state["llm"].with_structured_output(QueryProposal)
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
    response_options = query_proposal.response_options
    explanation = query_proposal.explanation

    run_query_template = f"""
        Vous êtes un agent de requête PostgreSQL. Votre tâche est de générer et d'exécuter des requêtes SQL basées sur les demandes des utilisateurs, en suivant les directives et en retournant une réponse finale avec les résultats trouvés.

        DIRECTIVES :
        - Commencez toujours par examiner les tables de la base de données
        - Créez des requêtes PostgreSQL syntaxiquement correctes
        - Si vous rencontrez des erreurs, réécrivez et réessayez la requête
        - N'exécutez JAMAIS d'instructions DML (INSERT, UPDATE, DELETE, DROP, etc.)
        - Utilisez uniquement les outils fournis pour interagir avec la base de données
        - Utilisez uniquement les informations retournées par ces outils dans votre réponse finale

        DEMANDE UTILISATEUR :
        {reformulated_request}

        ÉLÉMENTS DE BASE DE DONNÉES À UTILISER :
        - Questions : {questions_text}
        - Options de réponse : {response_options}
        - Instructions d'utilisation : {explanation}

        IMPORTANT : Utilisez ces chaînes exactes dans votre requête car elles correspondent à la structure de la base de données.

        ACTION :
        Exécutez la requête SQL appropriée qui répond à la demande de l'utilisateur en utilisant toutes les informations fournies.
        """

    # llm = get_llm(st.session_state["MODEL_CONFIG"])

    engine = create_engine(st.secrets["db_url"])
    db = SQLDatabase(engine)

    toolkit = SQLDatabaseToolkit(db=db, llm=st.session_state["llm"])
    tools = toolkit.get_tools()

    llm_with_tools = st.session_state["llm"].bind_tools(tools)

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
    Système : Vous êtes un agent de visualisation et de formatage de résultats.

    ENTRÉE :
    - Demande utilisateur : {reformulated_request}
    - Résultats de requête : {query_results}

    TÂCHE :
    Compte tenu des résultats fournis par l'agent précédent et de la demande de l'utilisateur, votre tâche est de :
    1. Fournir une réponse claire et bien formatée basée sur les résultats de la requête
    2. Retourner un script python pour la visualisation des résultats en utilisant UNIQUEMENT les composants Streamlit
    Votre réponse doit inclure la réponse finale à la requête de l'utilisateur et un bloc de code pour la visualisation lorsque c'est possible.

    EXIGENCES DE VISUALISATION :
    - Utilisez UNIQUEMENT les éléments graphiques Streamlit (st.line_chart, st.bar_chart, st.area_chart, etc.)
    - N'utilisez JAMAIS pyplot, seaborn, pillow ou d'autres bibliothèques de tracé externes
    - Pour l'affichage des données, utilisez exclusivement st.dataframe()

    Dans votre réponse formatée finale, incluez :
    1. D'abord, fournissez la réponse textuelle à la requête de l'utilisateur
    2. Ensuite, incluez un bloc de code avec le code de visualisation, il doit commencer par ```python et se terminer par ```
    3. Le bloc de code doit contenir le code python pour visualiser les résultats de la requête
    
    NOTE : Si vous utilisez st.bar_chart() avec des DataFrames pandas, assurez-vous de ne passer qu'un index à colonne unique plutôt qu'un index multi-niveaux. Les fonctions de graphiques intégrées de Streamlit attendent des structures de données simples et ne peuvent pas interpréter les indices hiérarchiques créés avec df.set_index([multiple_columns]). Pour éviter l'erreur "not in index", utilisez soit une seule colonne comme index, remodelez vos données avec pivot(), ou passez à des bibliothèques de visualisation plus flexibles comme Matplotlib lorsque vous devez représenter des données sur plusieurs dimensions catégorielles simultanément. Pour les visualisations complexes avec des données groupées, st.altair_chart() offre un meilleur support pour les structures de données hiérarchiques.
    """

    final_llm_model_config = st.session_state["MODEL_CONFIG"].copy()
    final_llm_model_config["model_name"] = "google/gemini-2.0-flash-001"
    final_llm = get_llm(final_llm_model_config)
    # st.session_state["MODEL_CONFIG"]["model_name"] = st.session_state["selected_model"]
    # st.session_state["llm"] = get_llm(st.session_state["MODEL_CONFIG"])
    final_result = final_llm.invoke(finalize_query_template)
    # 2. Then include a code block with visualization code

    # python_repl = PythonREPL()
    # repl_tool = Tool(
    #     name="python_repl",
    #     description="A Python shell for executing code. Use this to create exactly ONE Streamlit visualization. NEVER use matplotlib or other external plotting libraries. For data display, use ONLY st.dataframe().",
    #     func=python_repl.run,
    # )

    # # llm = get_llm(st.session_state["MODEL_CONFIG"])
    # tools = [repl_tool]

    # llm_with_tools = st.session_state["llm"].bind_tools(tools)

    # agent_executor = create_react_agent(llm_with_tools, tools)
    # for step in agent_executor.stream(
    #     {"messages": [{"role": "user", "content": finalize_query_template}]},
    #     stream_mode="values",
    # ):
    #     step["messages"][-1].pretty_print()

    return {"final_answer": final_result.content}


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
