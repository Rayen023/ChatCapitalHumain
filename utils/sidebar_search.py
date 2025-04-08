import streamlit as st
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain_cohere import CohereEmbeddings
import argparse


@st.fragment
@st.cache_data(show_spinner = False)
def query_vector_store(text_input):
    """Load the vector store for question answering."""
    #embeddings = VoyageAIEmbeddings(model="voyage-3-large")
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

    vector_store = FAISS.load_local(
        "answerable-questions-index-cohere",
        embeddings,
        allow_dangerous_deserialization=True,
    )
    compression_retriever = ContextualCompressionRetriever(
        #base_compressor=VoyageAIRerank(model="rerank-2", top_k=10),
        base_retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
    )
    results = compression_retriever.invoke(text_input)

    return results

@st.fragment
def search_questions_callback():
    """Callback function for search questions text input."""
    text_input = st.session_state.search_questions
    if text_input:
        results = query_vector_store(text_input)
        st.session_state.search_questions_results = results

@st.fragment
def select_subject_callback():
    """Callback function for subject dropdown selection."""
    selected_subject = st.session_state.question_selector
    if selected_subject:
        results = query_vector_store(selected_subject)
        st.session_state.subject_selection_results = results


def ask_example_question(question: str):
    """Set an example question to be processed."""
    st.session_state["_example_question"] = question

#@st.fragment
def search_questions():
    with st.sidebar:
        # Add example questions the user can select from
        example_questions = [ 
        "",
        "Relations sociales à l'école",
        "Projets post-secondaires",
        "Choix études postsecondaires",
        "Financement études",
        "Services orientation scolaire",
        "Importance cours secondaires",
        "Rendement académique",
        "Emploi pendant études",
        "Profil post-secondaire",
        "Parcours après secondaire",
        "Situation professionnelle",
        "Raisons changements d'emploi",
        "Mobilité géographique",
        "Retour aux études",
        "Emploi Péninsule acadienne"
        ]

        st.subheader("Recherche par sujet")
        selected_question = st.selectbox(
            "Choisir sujet:",
            example_questions,
            key="question_selector",
            on_change=select_subject_callback,
        )

        # Display results from subject selection if they exist
        if "subject_selection_results" in st.session_state and selected_question:
            results = st.session_state.subject_selection_results
            if not results:
                st.info("Aucun résultat trouvé pour ce sujet.")
            else:
                for i, doc in enumerate(results):
                    st.progress(doc.metadata.get("relevance_score", 0))
                    st.button(
                        doc.page_content,
                        on_click=ask_example_question,
                        args=(doc.page_content,),
                        use_container_width=True,
                        key=f"subject_result_{i}"
                    )

        st.text_input(
            "Rechercher des questions répondables",
            key="search_questions",
            on_change=search_questions_callback,
        )

        # Display results from text search if they exist
        if "search_questions_results" in st.session_state:
            st.subheader("Résultats de recherche")
            results = st.session_state.search_questions_results
            if not results:
                st.info("Aucun résultat trouvé.")
            else:
                for i, doc in enumerate(results):
                    st.progress(doc.metadata.get("relevance_score", 0))
                    st.button(
                        doc.page_content,
                        on_click=ask_example_question,
                        args=(doc.page_content,),
                        use_container_width=True,
                        key=f"search_result_{i}"
                    )


#@st.fragment
def search_documents_callback():
    """Callback function for document search text inputs."""
    text_input = st.session_state.search_query
    type_id = st.session_state.type_id
    question_id = st.session_state.question_id

    # Only proceed if at least one field has content
    if text_input or type_id or question_id:
        # Initialize embeddings
        embeddings = VoyageAIEmbeddings(model="voyage-3-large")

        # Load the vector store
        new_vector_store = FAISS.load_local(
            "faiss_index_db_questions",
            embeddings,
            allow_dangerous_deserialization=True,
        )

        # Set up the retriever with compression
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=VoyageAIRerank(model="rerank-2", top_k=6),
            base_retriever=new_vector_store.as_retriever(search_kwargs={"k": 8}),
        )

        # Build filter dictionary based on provided inputs
        filter_dict = {}
        if type_id:
            try:
                filter_dict["type_id"] = int(type_id)
            except ValueError:
                st.warning("Type ID must be a number")
                return

        if question_id:
            try:
                filter_dict["question_id"] = int(question_id)
            except ValueError:
                st.warning("Question ID must be a number")
                return

        # Always use a non-empty string for the query to avoid errors
        # Use " " (space) instead of empty string to ensure it works properly
        query = text_input if text_input else " "

        # Perform search with filters if they exist
        if filter_dict:
            results = compression_retriever.invoke(query, filter=filter_dict)
        else:
            results = compression_retriever.invoke(query)

        # Store results in session state
        st.session_state.search_documents_results = results

#@st.fragment
def search_documents():
    with st.sidebar:
        st.header("Search DB Questions")

        # Add on_change parameter to each input field
        st.text_input(
            "Search Query (optional)",
            key="search_query",
            on_change=search_documents_callback,
        )

        st.text_input(
            "Type ID (optional)", key="type_id", on_change=search_documents_callback
        )

        st.text_input(
            "Question ID (optional)",
            key="question_id",
            on_change=search_documents_callback,
        )

        # Display results if they exist in session state
        if "search_documents_results" in st.session_state:
            results = st.session_state.search_documents_results

            # Display results summary in sidebar
            if not results:
                st.info("No results found.")
            else:
                for i, doc in enumerate(results):
                    # Extract the first line of page_content as the question (or use the first 50 chars)
                    question_text = (
                        doc.page_content.split("\n")[0]
                        if "\n" in doc.page_content
                        else doc.page_content[:50]
                    )

                    with st.expander(f"{question_text}"):
                        # Display document content
                        st.markdown("### Content")
                        st.write(doc.page_content)

                        # Display metadata
                        st.markdown("### Metadata")
                        for key, value in doc.metadata.items():
                            st.write(f"**{key}**: {value}")

                        # Show relevance score in a more visual way
                        relevance = doc.metadata.get("relevance_score", 0)
                        st.progress(relevance)
