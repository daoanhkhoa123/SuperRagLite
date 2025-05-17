
routes = [
    {
        "condition": "{{'yes_answer' in replies[0]}}",
        "output": "{{query}}",
        "output_name": "go_to_documents",
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' in replies[0] and 'user_info_requested' not in replies[0]}}",
        "output": "{{query}}",
        "output_name": "go_to_websearch",
        "output_type": str,
    },
    {
        "condition": "{{'user_info_requested' in replies[0]}}",
        "output": "{{query}}",
        "output_name": "go_to_user_info",
        "output_type": str,
    },
]


hallu_route = [
    {
        "condition": "{{'yes_answer' in replies[0]}}",  # Kiểm tra 'yes' (không ảo giác)
        "output": "{{llm_replies[0]}}",  # Lấy câu trả lời từ node "llm" (replies[1] vì replies[0] là kết quả của grader)
        "output_name": "pass_answer",  # Tên output để chuyển tiếp câu trả lời
        "output_type": str,
    },
    {
        "condition": "{{'no_answer' in replies[0]}}",  # Kiểm tra 'no' (ảo giác)
        "output": "{{llm_replies[0]}}",  # Sử dụng lại câu hỏi ban đầu
        "output_name": "pass_answer",  # Tên output để kích hoạt lại node "llm"
        "output_type": str,
    },
          {
        "condition": "True",  # Kiểm tra 'no' (ảo giác) Changed to False
        "output": "{{llm_replies[0]}}",  # Sử dụng lại câu hỏi ban đầu
        "output_name": "pass_answer",  # Tên output để kích hoạt lại node "llm"
        "output_type": str,
    },
      {
        "condition": "False",  # Kiểm tra 'no' (ảo giác) Changed to False
        "output": "{{query}}",  # Sử dụng lại câu hỏi ban đầu
        "output_name": "not_use",  # Tên output để kích hoạt lại node "llm"
        "output_type": str,
    },
]
