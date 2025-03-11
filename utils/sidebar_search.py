import streamlit as st
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank


def search_documents():
    with st.sidebar:
        st.header("Search Questions")
        text_input = st.text_input("Search Query", key="search_query")
        type_id = st.text_input("Type ID (optional)", key="type_id")
        question_id = st.text_input("Question ID (optional)", key="question_id")
        search_button = st.button("Search")

        # Only perform search when button is clicked and query is provided
        if search_button and text_input:
            # Initialize embeddings - adjust this based on your actual embeddings
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

            if question_id:
                try:
                    filter_dict["question_id"] = int(question_id)
                except ValueError:
                    st.warning("Question ID must be a number")

            # Perform search with filters if they exist
            if filter_dict:
                results = compression_retriever.invoke(text_input, filter=filter_dict)
            else:
                results = compression_retriever.invoke(text_input)

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
