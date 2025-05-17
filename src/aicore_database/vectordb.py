import os
import shutil
import tempfile
from typing import List

import fitz  # PyMuPDF
from haystack.dataclasses import Document as HaystackDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_vector_db(file_upload, logger, doc_embedder):
    """
    Create a vector database from an uploaded PDF file.
    - Only text is split and embedded.
    - Images/tables are NOT summarized or embedded.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, file_upload.name)

    with open(pdf_path, "wb") as f:
        f.write(file_upload.getvalue())
    logger.info(f"File saved to temporary path: {pdf_path}")

    haystack_documents: List[HaystackDocument] = []

    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                haystack_documents.append(
                    HaystackDocument(
                        content=text.strip(),
                        meta={
                            "source": f"{file_upload.name}_page_{page_num + 1}",
                            "type": "text"
                        }
                    )
                )

    # Split only text documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    split_haystack_documents = []
    for doc in haystack_documents:
        chunks = text_splitter.split_text(doc.content)
        for i, chunk in enumerate(chunks):
            split_haystack_documents.append(
                HaystackDocument(
                    content=chunk.strip(),
                    meta={**doc.meta, "chunk_id": i}
                )
            )

    logger.info(
        f"Total processed chunks (text only): {len(split_haystack_documents)}"
    )

    docs_with_embeddings = doc_embedder.run(documents=split_haystack_documents)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs_with_embeddings["documents"])

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")

    return document_store