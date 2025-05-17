from aicore.airesource.prompt import prompt_context_init, prompt_context_combine, propmt_chathist_summarize, prompt_rewritequery

def context_init(model) -> str:
    return model.run(prompt_context_init)["replies"][0]


def context_combine(model, curr, chat_hist, user_info) -> str:
    chat_hist = summarize_chathist(model, chat_hist)
    return model.run(prompt_context_combine.format(chat_history=chat_hist, current_thoughts=curr, user_info=user_info))["replies"][0]


def summarize_chathist(model, chat_hist) -> str:
    return model.run(propmt_chathist_summarize.format(chat_history=chat_hist))["replies"][0]


def rewrite_query(model, question, context) -> str:
    return model.run(prompt_rewritequery.format(query=question, context=context))["replies"][0]
