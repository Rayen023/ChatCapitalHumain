from typing import Dict, List, Optional

import streamlit as st
from config import Config
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from typing_extensions import TypedDict


default_llm = ChatOpenAI(
    model_name="anthropic/claude-sonnet-4",
    **Config.get_openrouter_config(),
    temperature=0,
    max_tokens=8096,
    streaming=True,
)

flash_llm = ChatOpenAI(
    model_name="google/gemini-2.5-flash-preview-09-2025",
    **Config.get_openrouter_config(),
    temperature=0,
    max_tokens=8096,
    streaming=True,
)
analysis_llm = default_llm

# LLM for schema checking and query formulation
schema_llm = flash_llm

# LLM for running queries
query_llm = default_llm

# LLM for finalizing results
final_llm = default_llm


# Define our structured data models
class QueryProposal(BaseModel):
    user_request_after_feedback: str = Field(
        description="La requête de l'utilisateur après feedback du analyste humain",
    )
    questions_responses: str = Field(
        description="Liste de dictionnaires où chaque dictionnaire contient les informations d'une question pertinente et ses réponses possibles. Chaque dictionnaire doit inclure: 'question_id' (identifiant de la question), 'type_id' (identifiant du type de question), 'question_text' (texte exact de la question), et 'response_options' (options de réponses associées à cette question qui répondent à la requête de l'utilisateur)",
    )
    explanation: str = Field(
        description="Courte description en une phrase en français sur quelles fonctions SQL peuvent être utilisées.",
    )
    is_accepted_by_human_analyst: bool = Field(
        description="Indique si la proposition a été acceptée par l'analyste humain. Doit être défini à False par défaut",
    )


class AnalysisResult(BaseModel):
    is_db_related_and_answerable: bool = Field(
        description="Indique si la requête est liée à la base de données Capital Humain (2004-2019) et peut être répondue. Doit être True uniquement si la requête concerne une seule question ou différentes options d'une même question, sans corrélation entre questions distinctes. Les données sont agrégées par école, année, questionnaire et sexe, sans accès aux réponses individuelles des élèves.",
    )
    response: str = Field(
        description="Si is_db_related_and_answerable=True, contient la requête originale de l'utilisateur sans modification pour transmission à l'agent suivant. Si is_db_related_and_answerable=False, contient une réponse conversationnelle ou une explication détaillée de pourquoi la requête ne peut pas être répondue (par exemple, si elle tente de corréler des réponses de questions différentes).",
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
    query_analysis_instructions = """Rôle et Contexte :
IMPORTANT : Your output MUST ALWAYS be in the JSON format {"is_db_related_and_answerable": true/false, "response": "your_response"}. Avoid using SPECIAL CHARACTERS in your response. In order to avoid errors of type "ValueError: control character (\u0000-\u001f) found while parsing a string at line 4 column 0" make sure to follow the requested structure and NEVER use special charachters.

• Vous êtes le premier agent d'une chaîne chargé d'interagir avec l'utilisateur, en vous appuyant sur l'historique et la mémoire des conversations.
• Votre mission est de déterminer si la requête est liée à la base de données Capital Humain (période 2004–2019) ou relève d'une conversation générale.
• Vous devez transmettre la requête suivant ce format deux champs :
  – is_db_related_and_answerable : booléen (True/False)
  – response : chaîne de texte contenant soit la requête originale de l'utilisateur (sans ajouts ni reformulation) pour l'agent suivant, soit une réponse/exposé à l'utilisateur.

────────────────────────────── Décision et Processus :

Si la requête n'est PAS liée à la base de données :
 – (Exemples : salutations, infos générales, conversation au sens large)
  • Définissez is_db_related_and_answerable = False.
  • Dans response, répondez à l'utilisateur comme un chatbot classique.

Si la requête est liée à la base de données Capital Humain :
 – La base couvre :
  a. Écoles (2004–2019) : Aux quatre vents, Centre La Fontaine, Secondaire Népisiguit, Louis-Mailloux, Marie-Esther, Roland-Pépin, W.-A.-Losier.
  b. Questionnaires (avec des questions réparties par sexe) :
   1. Questions Générales
   2. SD – Renseignements Socio-Démographiques
   3. ED – Éducation PostSecondaire
   4. MT – Marché du travail
   5. RE – Attente/recherche/sans emploi

• Contraintes : Principe fondamental
Les données sont agrégées par école, année, questionnaire et sexe. Nous disposons des totaux par groupe démographique, mais pas des réponses individuelles des élèves.

Requêtes répondables (✅)

Toute requête portant sur une seule question et ses options de réponse

Les agrégations par école, année et/ou sexe sont toujours possibles

Les requêtes concernant plusieurs options de réponse d'une même question sont acceptables

Exemple : "Combien d'élèves vont à l'école à pied OU en bus en 2018, par école ?"
→ Répondable car "à pied" et "en bus" sont des options de la même question sur le transport

Requêtes non répondables (❌)

Toute requête essayant de lier les réponses de deux questions différentes

Toute tentative de corrélation au niveau individuel

Exemple : "Combien d'élèves qui vont à l'école à pied ont également des bonnes notes ?"
→ Non répondable car cela tente de corréler deux questions distinctes (transport et performance scolaire)

Point clé à retenir
Si la requête tente d'établir une relation du type "les élèves qui ont répondu X à la question 1 ET ont répondu Y à la question 2", elle est impossible à satisfaire car nos données ne permettent pas de suivre les réponses individuelles.

────────────────────────────── En résumé :

Procédez ainsi :
  a. Si la requête implique la corrélation entre deux ou plusieurs questions distinctes :
   – Définissez is_db_related_and_answerable = False.
   – Expliquez dans response pourquoi la corrélation n'est pas supportée.
  b. Si la requête concerne une seule question (ou différentes sous-parties d'une même question) sans lien inter-question :
   – Définissez is_db_related_and_answerable = True.
   – Dans response, transmettez la requête exacte de l'utilisateur sans reformulation ni explications
  


• Si la requête est liée à la BD et répondable → is_db_related_and_answerable = True, et response contient la requête originale de l'utilisateur sans modification.
• Sinon → is_db_related_and_answerable = False, et response doit être une réponse conversationnelle ou une explication.
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
    structured_llm = analysis_llm.with_structured_output(AnalysisResult)
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

    human_analyst_feedback = state.get(
        "human_analyst_feedback", "Not checked by human expert yet"
    )
    previous_query_proposal = state.get("query_proposal", None)
    if previous_query_proposal:
        reformulated_request = state.get(
            state["query_proposal"].user_request_after_feedback
        )
    else:
        reformulated_request = state["analysis_result"].response

    disclaimer = 'IMPORTANT : Your output MUST ALWAYS be in the JSON format {"user_request_after_feedback": "str without special chars", "questions_responses": "List of Dicts", "explanation": "str without special chars" , "is_accepted_by_human_analyst": true/false}. Avoid using SPECIAL CHARACTERS in your response. In order to avoid errors of type "ValueError: control character (\u0000-\u001f) found while parsing a string at line 4 column 0" make sure to follow the requested structure and NEVER use special charachters.'
    system_message = f"""
    {disclaimer}

    Vous êtes un agent dans un flux de travail multi-agents.  Votre rôle est d'analyser une requête utilisateur relative à une base de données et de proposer une solution, qui sera ensuite validée par un expert humain avant d'être exécutée.
    IMPORTANT : Assurez-vous de toujours suivre la structure de sortie demandée et de ne pas utiliser de caractères spéciaux.
IMPORTANT : Make sure to follow requested structure and NEVER use special charachters

    **OBJECTIF :** Analyser la requête utilisateur, identifier les éléments de base de données pertinents, et préparer une proposition de requête.

    **ENTRÉES :**

    *   **Demande utilisateur :** {reformulated_request} (La requête exprimée par l'utilisateur.)
    *   **Suggestions précédentes :** {previous_query_proposal} (Votre proposition précédente, si elle existe.)
    *   **Retour d'expert humain :** {human_analyst_feedback} (Le retour de l'expert humain sur votre proposition précédente. Peut être vide.)
    *   **Schéma de la base de données :** {st.session_state.schema_template} (La structure de la base de données.)

    **PROCESSUS :**

    1.  **Analyse de la requête utilisateur :** Comprendre précisément ce que l'utilisateur demande.

    2.  **Identification des composants de la requête :**  Déterminer :
        *   Les *questions_text* spécifiques pertinentes pour répondre à la demande. Inclure les question_id et type_id.
        *   Les options de reponses a cette question *exacts* (nommés tels qu'ils apparaissent dans le schéma) nécessaires pour répondre à la demande.
        Note : S'il s'agit de questions dans le type, liste tout les questions, ou liste les ecoles, questionnaire etc, donc une user_request qui n'a pas besoin d'une question et reponses specifiques, laisse vide.

    3.  **Explication de la proposition :** Fournir une definition *courte et concise* en une seule phrase des fonctions SQL à utiliser sur quelle colonne de quelle table seulement. 
    Note :     *   **Informations importantes sur les colonnes:** Column: summed_students_responses | Type: INTEGER | Description: Sum of students responses for a question thus the Function SUM not COUNT should be mostly used with this column, make sure to always include this explanation in your responses

    4.  **Détermination du statut d'approbation :** Déterminer la valeur du champ `is_accepted_by_human_analyst` :
        *   Si `human_analyst_feedback` est vide (première proposition) : `is_accepted_by_human_analyst = False`.
        *   Si `human_analyst_feedback` contient des mots d'approbation (par exemple, "correct", "oui", "approuvé") :  `is_accepted_by_human_analyst = True`.
        *   Si `human_analyst_feedback` contient des suggestions de modifications :
            *   **Modifier** votre proposition en tenant compte du feedback.
            *   `is_accepted_by_human_analyst = False`.

    5.  **Mise à jour de la requête utilisateur :** Remplir le champ `user_request_after_feedback`.
        *   Si le feedback de l'expert humain a conduit à une modification de votre proposition, ajustez la valeur du champ `user_request_after_feedback` en conséquence.
        *   S'il n'y a pas eu de changements suite au feedback ou il s'agit de l'expert n'a pas donné encore son feedback, conservez la valeur originale de `{reformulated_request}` dans le champ `user_request_after_feedback`.
        *   **Ne jamais laisser le champ `user_request_after_feedback` vide.**
        
    """
    structured_llm = schema_llm.with_structured_output(QueryProposal)
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
    query_proposal = state["query_proposal"].model_copy()
    reformulated_request = query_proposal.user_request_after_feedback
    questions_responses = query_proposal.questions_responses
    explanation = query_proposal.explanation

    run_query_template = f"""
    Système : Vous êtes un agent de requête PostgreSQL spécialisé dans l'extraction de données.

    ENTRÉE :
    - Demande utilisateur : {reformulated_request}

    TÂCHE :
    Générez et exécutez des requêtes SQL qui répondent précisément à la demande de l'utilisateur.

    DIRECTIVES ESSENTIELLES :
    1. Utilisez UNIQUEMENT les outils fournis pour interagir avec la base de données
    2. Créez des requêtes PostgreSQL syntaxiquement correctes
    3. En cas d'erreur, réécrire puis réessayer la requête.
    4. N'exécutez JAMAIS d'instructions DML (INSERT, UPDATE, DELETE, DROP, etc.)
    5. Utilisez ces chaînes de texte EXACTES pour accélérer vos requêtes : {questions_responses}, {explanation}
    6. Transmettez les résultats COMPLETS de votre dernière requête SQL dans votre réponse finale

    SORTIE ATTENDUE :
    - Retournez les résultats bruts et complets du dernier appel d'outil dans un format de liste en texte brut.
    - Ne formatez pas et n'analysez pas les résultats
    - Ne résumez pas et n'échantillonnez pas les données
    - Expect output to be only the full results values of last successful SQL query execution that answers user response, in a list format en texte brut and must include all values.
    - Format de sortie : {{"query_results": "votre_résultat"}} # liste des valeurs de résultats bruts
        """

    engine = create_engine(Config.get_database_url())
    db = SQLDatabase(engine)

    toolkit = SQLDatabaseToolkit(db=db, llm=query_llm)
    tools = toolkit.get_tools()

    llm_with_tools = query_llm.bind_tools(tools)

    agent_executor = create_react_agent(llm_with_tools, tools)
    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": run_query_template}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    return {"query_results": step["messages"][-1].content}


def finalize_query(state: DatabaseQueryState):

    reformulated_request = state["query_proposal"].user_request_after_feedback
    query_results = state["query_results"]

    print("-" * 50)
    print("Query results: ", query_results)
    print("-" * 50)

    finalize_query_template = f"""     
    Système : Vous êtes un agent de visualisation et de formatage de résultats.

    ENTRÉE :     
    - Demande utilisateur : {reformulated_request}     
    - Résultats de requête : {query_results}      

    TÂCHE :     
    Formatez une réponse complète à partir des résultats fournis par l'agent précédent, en incluant :     
    1. Une réponse textuelle claire qui intègre TOUTES les données et informations des résultats     
    2. Un script Python pour visualiser ces données avec streamlit UNIQUEMENT lorsque c'est pertinent et approprié

    EXIGENCES :     
    - Présentation des données : Incluez TOUTES les données des résultats, sans sélection partielle, dans votre texte explicatif     
    - Visualisation :
    • Créez des visualisations UNIQUEMENT lorsque:
        - Les données sont numériques ou catégoriques et se prêtent à une représentation graphique
        - Il y a plusieurs points de données à comparer (plus d'une valeur)
        - Un graphique apporterait une réelle valeur ajoutée à la compréhension des résultats
    • Ne créez PAS de visualisations pour:
        - Des réponses textuelles simples
        - Des résultats uniques ou des données non structurées
        - Des cas où le texte explicatif suffit à comprendre clairement les résultats
    • Adaptez le type de visualisation au type de données (histogrammes, barres, lignes, etc.)
    • Soyez dynamique et utilisez votre jugement pour déterminer si une visualisation est pertinente

    • Lorsqu'une visualisation est pertinente:
        • Utilisez EXCLUSIVEMENT des composants Streamlit pour l'affichage
        • Pour les données tabulaires, utilisez OBLIGATOIREMENT st.dataframe() ou st.table()
        • Pour les graphiques, utilisez OBLIGATOIREMENT st.plotly_chart() avec Plotly (plotly.express ou plotly.graph_objects) et un paramètre key unique (ex: key='chart1')
        • Pour le texte et les métriques, utilisez st.write(), st.metric(), st.columns(), etc.
        • N'utilisez JAMAIS matplotlib ou seaborn ou plt.show() ou fig.show() ou autres bibliothèques graphiques. Utilisez UNIQUEMENT les composants Streamlit pour tous les affichages
        • N'utilisez JAMAIS print() - utilisez uniquement les composants Streamlit pour l'affichage
        • Tous les outputs doivent être rendus via les composants Streamlit (st.*)

    FORMAT DE RÉPONSE :     
    1. Texte explicatif complet répondant à la demande utilisateur avec tous les résultats     
    2. Bloc de code Python complet entre ```python et ``` contenant le code de visualisation UNIQUEMENT si pertinent
    
    IMPORTANT - CONTRAINTES DE CODE :
    • Le code Python doit utiliser UNIQUEMENT les composants Streamlit pour tous les affichages
    • Remplacez TOUS les print() par st.write() ou st.text()
    • Utilisez st.plotly_chart() pour tous les graphiques (pas plt.show() ou fig.show())
    • Utilisez st.dataframe() ou st.table() pour toutes les données tabulaires
    • Assurez-vous que le code est directement exécutable dans un environnement Streamlit
    """

    final_result = final_llm.invoke(finalize_query_template)

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
@st.cache_resource
def cache_memory():
    return MemorySaver()


memory = cache_memory()
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
