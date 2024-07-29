from transformers import AutoTokenizer

model_id = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def token_length(user_prompt_str: str) -> int:
    """
    (Hopefully) calculates the token length of a given prompt in the chat template format
    :param user_prompt_str: A string you want to measure token length for
    :return: the token length
    """
    input_to_model = [{
        "role": "user",
        "content": user_prompt_str
    }
    ]
    tokenized_qa = tokenizer.apply_chat_template(input_to_model, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return tokenized_qa.size(1)
