from typing import List

from aicore.airesource.prompt import (
    prompt_template,
    prompt_template_after_documents,
    prompt_template_after_websearch,
    prompt_template_after_user_info,
    propmt_hallu_grader,
)

from aicore.airesource.route import (
    routes,
    hallu_route,
)

from aicore.airesource.config import LLMNAME_GENERATE, LLMNAME_ROUTE, LLMNAME_EMBEDDER, VECTOR_TOPK


from haystack_integrations.components.generators.ollama import OllamaGenerator
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from haystack.components.routers import ConditionalRouter
from haystack.components.joiners import BranchJoiner
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack.dataclasses import Document
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack import Document, Pipeline
from aicore.usercore.usercore import UserInfo


def init_pipeline(user_dict=dict()):
    raise NotImplementedError("Not used anymore")

    pipe = Pipeline()
    pipe.add_component(
        "text_embedder", OllamaTextEmbedder(model=LLMNAME_EMBEDDER))
    # Note: retriever will be added later with the document_store
    pipe.add_component("prompt_builder", PromptBuilder(
        template=prompt_template, required_variables=["query", "documents"]))
    pipe.add_component("router_llm", OllamaGenerator(model=LLMNAME_ROUTE))
    pipe.add_component("router", ConditionalRouter(routes))
    pipe.add_component("prompt_builder_after_documents", PromptBuilder(
        template=prompt_template_after_documents, required_variables=["query", "documents", "context"]))
    pipe.add_component(
        "websearch", DuckduckgoApiWebSearch(top_k=5, backend="auto"))
    pipe.add_component("prompt_builder_after_websearch", PromptBuilder(
        template=prompt_template_after_websearch, required_variables=["user_info", "documents", "web_urls", "query", "context"]))
    pipe.add_component("prompt_builder_after_user_info", PromptBuilder(
        template=prompt_template_after_user_info, required_variables=["user_info", "query", "context"]))
    pipe.add_component("prompt_joiner", BranchJoiner(str))
    pipe.add_component("llm", OllamaGenerator(model=LLMNAME_GENERATE))
    pipe.add_component("hallu_llm", OllamaGenerator(model=LLMNAME_GENERATE))
    pipe.add_component("hallu_prompt", PromptBuilder(template=propmt_hallu_grader,
                                                     required_variables=["user_info", "documents", "llm_replies", "context"]))
    pipe.add_component("hallu_router", ConditionalRouter(hallu_route))
    pipe.add_component("document_joiner", BranchJoiner(
        List[Document]))  # Joiner for documents
    pipe.add_component("user_info", UserInfo(userdict=user_dict))
    return pipe

# Function to add the retriever and connect all the components


def setup_pipeline_with_document_store(pipe, document_store):
    raise NotImplementedError("Not used anymore")
    # Add retriever with the new document_store
    pipe.add_component(
        "retriever", InMemoryEmbeddingRetriever(document_store, top_k=VECTOR_TOPK))
    # Connect components as before
    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("user_info.user_info",
                 "prompt_builder_after_user_info.user_info")
    pipe.connect("user_info.user_info",
                 "prompt_builder_after_websearch.user_info")
    pipe.connect("user_info.user_info", "hallu_prompt.user_info")
    pipe.connect("prompt_builder", "router_llm")
    pipe.connect("router_llm.replies", "router.replies")
    pipe.connect("router.go_to_documents",
                 "prompt_builder_after_documents.query")
    pipe.connect("retriever", "prompt_builder_after_documents.documents")
    pipe.connect("prompt_builder_after_documents", "prompt_joiner")
    pipe.connect("router.go_to_websearch", "websearch.query")
    pipe.connect("router.go_to_websearch",
                 "prompt_builder_after_websearch.query")
    pipe.connect("websearch.documents",
                 "prompt_builder_after_websearch.documents")
    pipe.connect("websearch.documents",
                 "prompt_builder_after_websearch.web_urls")
    pipe.connect("prompt_builder_after_websearch", "prompt_joiner")
    pipe.connect("router.go_to_user_info",
                 "prompt_builder_after_user_info.query")
    pipe.connect("prompt_builder_after_user_info", "prompt_joiner")
    pipe.connect("prompt_joiner", "llm")
    pipe.connect("retriever", "document_joiner")
    pipe.connect("websearch.documents", "document_joiner")
    pipe.connect("document_joiner", "hallu_prompt.documents")
    pipe.connect("llm.replies", "hallu_router.llm_replies")
    pipe.connect("llm.replies", "hallu_prompt.llm_replies")
    pipe.connect("hallu_prompt", "hallu_llm")
    pipe.connect("hallu_llm.replies", "hallu_router.replies")
    return pipe


def build_full_pipeline(user_dict=dict(), document_store=None):
    pipe = Pipeline()
    pipe.add_component(
        "text_embedder", OllamaTextEmbedder(model=LLMNAME_EMBEDDER))
    pipe.add_component("prompt_builder", PromptBuilder(
        template=prompt_template, required_variables=["query", "documents"]))
    pipe.add_component("router_llm", OllamaGenerator(model=LLMNAME_ROUTE))
    pipe.add_component("router", ConditionalRouter(routes))
    pipe.add_component("prompt_builder_after_documents", PromptBuilder(
        template=prompt_template_after_documents, required_variables=["query", "documents", "context"]))
    pipe.add_component(
        "websearch", DuckduckgoApiWebSearch(top_k=5, backend="auto"))
    pipe.add_component("prompt_builder_after_websearch", PromptBuilder(
        template=prompt_template_after_websearch, required_variables=["user_info", "documents", "web_urls", "query", "context"]))
    pipe.add_component("prompt_builder_after_user_info", PromptBuilder(
        template=prompt_template_after_user_info, required_variables=["user_info", "query", "context"]))
    pipe.add_component("prompt_joiner", BranchJoiner(str))
    pipe.add_component("llm", OllamaGenerator(model=LLMNAME_GENERATE))
    pipe.add_component("hallu_llm", OllamaGenerator(model=LLMNAME_GENERATE))
    pipe.add_component("hallu_prompt", PromptBuilder(template=propmt_hallu_grader,
                                                     required_variables=["user_info", "documents", "llm_replies", "context"]))
    pipe.add_component("hallu_router", ConditionalRouter(hallu_route))
    pipe.add_component("document_joiner", BranchJoiner(List[Document]))
    pipe.add_component("user_info", UserInfo(userdict=user_dict))
    if document_store is not None:
        pipe.add_component(
            "retriever", InMemoryEmbeddingRetriever(document_store, top_k=VECTOR_TOPK))
        pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever", "prompt_builder.documents")
        pipe.connect("user_info.user_info",
                     "prompt_builder_after_user_info.user_info")
        pipe.connect("user_info.user_info",
                     "prompt_builder_after_websearch.user_info")
        pipe.connect("user_info.user_info", "hallu_prompt.user_info")
        pipe.connect("prompt_builder", "router_llm")
        pipe.connect("router_llm.replies", "router.replies")
        pipe.connect("router.go_to_documents",
                     "prompt_builder_after_documents.query")
        pipe.connect("retriever", "prompt_builder_after_documents.documents")
        pipe.connect("prompt_builder_after_documents", "prompt_joiner")
        pipe.connect("router.go_to_websearch", "websearch.query")
        pipe.connect("router.go_to_websearch",
                     "prompt_builder_after_websearch.query")
        pipe.connect("websearch.documents",
                     "prompt_builder_after_websearch.documents")
        pipe.connect("websearch.documents",
                     "prompt_builder_after_websearch.web_urls")
        pipe.connect("prompt_builder_after_websearch", "prompt_joiner")
        pipe.connect("router.go_to_user_info",
                     "prompt_builder_after_user_info.query")
        pipe.connect("prompt_builder_after_user_info", "prompt_joiner")
        pipe.connect("prompt_joiner", "llm")
        pipe.connect("retriever", "document_joiner")
        pipe.connect("websearch.documents", "document_joiner")
        pipe.connect("document_joiner", "hallu_prompt.documents")
        pipe.connect("llm.replies", "hallu_router.llm_replies")
        pipe.connect("llm.replies", "hallu_prompt.llm_replies")
        pipe.connect("hallu_prompt", "hallu_llm")
        pipe.connect("hallu_llm.replies", "hallu_router.replies")
    return pipe
