import streamlit as st
import os
from haystack import Pipeline
from haystack.components.converters.csv import CSVToDocument
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

if "document_store" not in st.session_state:
    document_store = QdrantDocumentStore(
        url=os.getenv("QDRANT_HOST", "127.0.0.1"),
        port=os.getenv("QDRANT_PORT", "6333"),
        index="document",
        recreate_index=True,
        embedding_dim=768,  # Set appropriate dimension for your model
    )
    st.session_state.document_store = document_store

if "pipeline_csv_store" not in st.session_state:
    pipeline_csv_store = Pipeline()
    pipeline_csv_store.add_component("embedder", SentenceTransformersDocumentEmbedder())
    pipeline_csv_store.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    # pipeline_csv_store.connect("converter.documents", "embedder")
    pipeline_csv_store.connect("embedder", "writer")
    st.session_state.pipeline_csv_store = pipeline_csv_store

if "pipeline_pdf_store" not in st.session_state:
    pipeline_pdf_store = Pipeline()
    pipeline_pdf_store.add_component("converter", PyPDFToDocument())
    pipeline_pdf_store.add_component("cleaner", DocumentCleaner())
    pipeline_pdf_store.add_component("embedder", SentenceTransformersDocumentEmbedder())
    pipeline_pdf_store.add_component(
        "writer", DocumentWriter(document_store=document_store)
    )
    pipeline_pdf_store.connect("converter", "cleaner")
    pipeline_pdf_store.connect("cleaner", "embedder")
    pipeline_pdf_store.connect("embedder", "writer")
    st.session_state.pipeline_pdf_store = pipeline_pdf_store

if "pipeline_retrieve" not in st.session_state:
    template = """
    given these documents, answer the question based on these documents. 
    Documents:
    {% for document in documents %}
        {{ document.content }}
    {% endfor %}
    If the information is not available in the provided documents, just say that you don't have information about the requested topic."
    Question: {{ query }}?
    Answer:
    """
    pipeline_retrieve = Pipeline()
    pipeline_retrieve.add_component("embedder", SentenceTransformersTextEmbedder())
    pipeline_retrieve.add_component(
        "retriever", QdrantEmbeddingRetriever(document_store=document_store)
    )
    pipeline_retrieve.add_component("builder", PromptBuilder(template=template))
    ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    pipeline_retrieve.add_component(
        "generator",
        OllamaGenerator(
            model="llama3.2",
            url=f"http://{ollama_host}:{ollama_port}",
        ),
    )
    pipeline_retrieve.connect("embedder", "retriever.query_embedding")
    pipeline_retrieve.connect("retriever", "builder.documents")
    pipeline_retrieve.connect("builder", "generator")
    st.session_state.pipeline_retrieve = pipeline_retrieve

# Retrieve from session state
document_store = st.session_state.document_store
pipeline_csv_store = st.session_state.pipeline_csv_store
pipeline_pdf_store = st.session_state.pipeline_pdf_store
pipeline_retrieve = st.session_state.pipeline_retrieve

st.title("Local RAG Demo")
st.header("Preprocess", divider="rainbow")
st.subheader("Upload a CSV file")

with st.form("upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Choose csv or pdf file", type=["csv", "pdf"], accept_multiple_files=False
    )
    submitted = st.form_submit_button("submit")

    if uploaded_file is not None and submitted:
        if not os.path.exists("docs"):
            os.makedirs("docs")
        with open(os.path.join("docs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write("File Uploaded Successfully")

st.subheader("Update Document Store")
st.write("This will update the document store with the uploaded file")

if st.button("Update Document Store"):
    files = [f"./docs/{file}" for file in os.listdir("docs") if file.endswith(".csv")]
    converter = CSVToDocument()
    results = converter.run(sources=files)
    documents = results["documents"]
    pipeline_csv_store.run({"embedder": {"documents": documents}})
    files = [f"./docs/{file}" for file in os.listdir("docs") if file.endswith(".pdf")]
    pipeline_pdf_store.run({"converter": {"sources": files}})
    st.write("Document Store Updated Successfully")

st.subheader("Show Document Store")
st.write("This will show all documents in the document store")

if st.button("Show Document Store"):
    st.write(document_store.filter_documents())

st.header("RAG from Document Store", divider="rainbow")
st.subheader("Write your question")
st.write(
    "This will retrieve documents from the document store and use it to augment the answer generation"
)

query = st.text_input("Ask a question: ")

if st.button("Generate Answer"):
    result = pipeline_retrieve.run(
        {"embedder": {"text": query}, "builder": {"query": query}}
    )
    st.write(result["generator"]["replies"][0])
