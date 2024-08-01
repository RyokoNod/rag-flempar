from transformers import AutoTokenizer

class TokenLengthCalculator:
    def __init__(self, model_id: str):
        """
        Initializes the TokenLengthCalculator with a specified model ID.
        :param model_id: The ID of the model to use for the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def token_length_of_str(self, user_prompt_str: str) -> int:
        """
        Calculates the approximate token length of a given string.
        The LLM receiving the tokens as input is assumed to do chat completion.
        :param user_prompt_str: A string you want to measure token length for
        :return: the token length
        """
        input_to_model = [{
            "role": "user",
            "content": user_prompt_str
        }]
        tokenized_qa = self.tokenizer.apply_chat_template(input_to_model, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        return tokenized_qa.size(1)

    def token_length_of_prompt(self, user_prompt: list) -> int:
        """
        Calculates the approximate token length of a given prompt.
        The LLM receiving the tokens as input is assumed to do chat completion.
        :param user_prompt: A list of dicts representing the conversation
                            structure, with roles and contents.
        :return: the token length
        """
        tokenized_qa = self.tokenizer.apply_chat_template(user_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        return tokenized_qa.size(1)
