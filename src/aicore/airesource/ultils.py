def format_llm(s):
    return s.split("</think>")[-1]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
