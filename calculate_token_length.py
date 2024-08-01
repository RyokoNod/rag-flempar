from transformers import AutoTokenizer

model_id = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


def token_length_of_str(user_prompt_str: str) -> int:
    """
    Calculates the approximate token length of a given string.
    The LLM receiving the tokens as input is assumed to do chat completion.
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

def token_length_of_prompt(user_prompt: list) -> int:
    """
    Calculates the approximate token length of a given prompt.
    The LLM receiving the tokens as input is assumed to do chat completion.
    :param user_prompt_str: A string you want to measure token length for
    :return: the token length
    """
    tokenized_qa = tokenizer.apply_chat_template(user_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    return tokenized_qa.size(1)