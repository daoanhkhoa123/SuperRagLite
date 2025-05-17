
from aicore_database import vectordb
from aicore import (
    init_pipe,
    run_pipe,
    prompt_caller,
)
import aicore.airesource.config as config
import os
import sys
import logging
from pathlib import Path
from typing import List, Any

BASE_DIR = Path(__file__).resolve().parent
SRC_PARENT = BASE_DIR.parent  # This is the directory containing 'src'
if str(SRC_PARENT) not in sys.path:
    sys.path.insert(0, str(SRC_PARENT))

from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder
from haystack_integrations.components.generators.ollama import OllamaGenerator
import pdfplumber
import streamlit as st


# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="Adaptive Academics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize user info in session state if not already present
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {
        "Tên": "B",
        "Chuyên ngành": "Kinh tế",
        "Kinh tế vi mô": 8.7,
        "Kinh tế vĩ mô": 7.3,
        "Nguyên lý kế toán": 8.6,
        "Marketing cơ bản": 8.4,
        "Quản trị": 8.1,
    }

chat_history = {
    "user": None,
    "assistant": None,
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LLM_generate = OllamaGenerator(model=config.LLMNAME_GENERATE)
LLM_router = OllamaGenerator(model=config.LLMNAME_ROUTE)
LLM_embedd = OllamaDocumentEmbedder(model=config.LLMNAME_EMBEDDER)
Context = prompt_caller.context_init(LLM_generate, st.session_state["user_info"])


def process_question(question: str, document_store) -> str:
    global Context, chat_history
    user_info = st.session_state.get("user_info", {})

    # Standard question processing
    rewritten_question = prompt_caller.rewrite_query(
        LLM_generate, question, Context)
    run_dict = {
        "text_embedder": {"text": rewritten_question},
        "prompt_builder": {"query": rewritten_question},
        "router": {"query": rewritten_question},
        "hallu_router": {"query": rewritten_question},
        "prompt_builder_after_documents": {"context": Context},
        "prompt_builder_after_websearch": {"context": Context},
        "prompt_builder_after_user_info": {"context": Context},
        "hallu_prompt": {"context": Context},
    }
    Pipe = init_pipe.build_full_pipeline(user_info, document_store)
    result = run_pipe.run_single(Pipe, run_dict)
    chat_history["user"] = rewritten_question
    chat_history["assistant"] = result
    Context = prompt_caller.context_combine(
        LLM_generate, Context, chat_history, user_info
    )

    return result


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(
        f"Extracting all pages as images from file: {file_upload.name}")
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db, document_store=None) -> None:
    if vector_db is not None:
        logger.info("Deleting Chroma vector DB")
        try:
            vector_db.delete_collection()
            st.success("Các tệp tạm thời đã được xóa thành công.")
            logger.info("Chroma Vector DB and related session state cleared")
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi xóa tài liệu: {str(e)}")
            logger.error(f"Error deleting Chroma collection: {e}")
    elif document_store is not None:
        logger.info("Clearing InMemoryDocumentStore")
        try:
            document_store.clear_all_documents()
            st.success("InMemoryDocumentStore đã được xóa thành công.")
            logger.info("InMemoryDocumentStore cleared")
            st.session_state.pop("document_store", None)
            st.rerun()
        except Exception as e:
            st.error(f"Lỗi khi xóa InMemoryDocumentStore: {str(e)}")
            logger.error(f"Error clearing InMemoryDocumentStore: {e}")
    else:
        st.error("Không tìm thấy cơ sở dữ liệu vector hoặc document store để xóa.")
        logger.warning(
            "Attempted to delete vector DB or document store, but none was found")


def main() -> None:
    st.subheader("📚 Adaptive Academics", divider="gray", anchor=False)
    col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # --- UI Part ---
    file_upload = col1.file_uploader(
        "Tải lên tệp PDF hoặc MP4 ↓",
        type=["pdf", "mp4"],
        accept_multiple_files=False,
        key="file_uploader",
    )

    if file_upload:
        file_type = Path(file_upload.name).suffix.lower()

        if st.session_state.get("vector_db") is None:
            with st.spinner("Đang xử lý tệp đã tải lên..."):
                if file_type == ".pdf":
                    st.session_state["vector_db"] = vectordb.create_vector_db(
                        file_upload, logger, LLM_embedd
                    )
                    st.session_state["file_upload"] = file_upload

                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [
                            page.to_image().original for page in pdf.pages
                        ]
                        

        # --- Only for PDFs ---
        if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
            zoom_level = col1.slider(
                "Mức thu phóng tài liệu",
                min_value=100,
                max_value=2000,
                value=700,
                step=50,
                key="zoom_slider",
            )
            with col1:
                with st.container(height=410, border=True):
                    for page_image in st.session_state["pdf_pages"]:
                        st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "🔄 Loại bỏ tất cả các tài liệu đã upload hiện tại",
        type="secondary",
        key="delete_button",
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        message_container = st.container(height=500, border=True)

        # Display chat history
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "🎓"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input("Nhập câu hỏi vào đây...", key="chat_input"):
            try:
                st.session_state["messages"].append(
                    {"role": "user", "content": prompt}
                )
                with message_container.chat_message("user", avatar="🎓"):
                    st.markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[Đang xử lý...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"]
                            )
                            st.markdown(response)
                        else:
                            st.warning(
                                "Vui lòng tải lên tệp PDF hoặc video (.mp4) trước khi đặt câu hỏi."
                            )

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )
            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning(
                    "Tải lên tệp PDF hoặc video (.mp4) để bắt đầu trò chuyện..."
                )


if __name__ == "__main__":
    main()
