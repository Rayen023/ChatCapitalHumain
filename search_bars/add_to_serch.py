import os
from uuid import uuid4

import faiss
from langchain.docstore.document import Document
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

VOYAGE_API_KEY = "pa-_GyP9a57BCZa2sX0mRJtQRfK49fTEmTRB2vg23Bgygs"
os.environ["VOYAGE_API_KEY"] = VOYAGE_API_KEY

from dotenv import load_dotenv

load_dotenv()


embeddings = VoyageAIEmbeddings(model="voyage-3-large")


# Read questions from the answerable_questions.txt file
questions_file_path = os.path.join(
    os.path.dirname(__file__), "answerable_questions.txt"
)

# data = []
# with open(questions_file_path, "r", encoding="utf-8") as file:
#     current_question = ""
#     for line in file:
#         line = line.strip()
#         if line:  # Non-empty line
#             if not current_question:
#                 current_question = line
#             else:
#                 # If we're still processing the same question (for multi-line questions)
#                 current_question += " " + line
#         else:  # Empty line indicates end of a question
#             if current_question:
#                 data.append(current_question)
#                 current_question = ""

#     # Don't forget the last question if file doesn't end with an empty line
#     if current_question:
#         data.append(current_question)

from answerable_questions import questions as data

documents = []
for question in data:
    document = Document(page_content=question)
    documents.append(document)

print(f"Loaded {len(documents)} questions")
index = faiss.IndexFlatL2(
    len(embeddings.embed_query("hello world"))
)  # identifies the dimension of the embeddings

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)

vector_store.save_local("faiss_index_answerable_db_questions")

new_vector_store = FAISS.load_local(
    "faiss_index_answerable_db_questions",
    embeddings,
    allow_dangerous_deserialization=True,
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=VoyageAIRerank(model="rerank-2", top_k=2),
    base_retriever=new_vector_store.as_retriever(search_kwargs={"k": 6}),
)
# kwargs (Any) – Additional arguments to pass to the retriever. #TODO add two filter boxes aswell
results = compression_retriever.invoke(
    "modes de transport",
)
for doc in results:
    print(f"\n\n [SIM={doc.metadata['relevance_score']:3f}] {doc.page_content}")
