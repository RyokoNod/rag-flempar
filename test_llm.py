from llama_cpp import Llama
from transformers import AutoTokenizer

# In https://huggingface.co/bartowski/aya-23-8B-GGUF, it says aya has 8192 context limit
# Is that true? What does its tokenizer say?
# (If no value is set, it defaults to int(1e30))

# Looks like CohereAI did not set a context size in their tokenizer,
# so it's safer to go with the 8192 written on its model card.
# FYI going beyond the context window of the model is a bad idea,
# it does not increase the amount of tokens a model can actually take.
model_id = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.model_max_length == int(1e30):
    print("CohereAI did not set a maximum token number in its tokenizer.")
    print("Better believe it when the model card says 8192.")
else:
    print(f"The actual maximum token number seems to be {tokenizer.model_max_length}")

# What is the chat format? How should you interact with this model?
# This will be given in Jinja2
# Basically it says if a system message isn't given, act as
# "You are Command-R, a brilliant, sophisticated, AI-assistant
# trained to assist human users by providing thorough responses. You are trained by Cohere."
# The interaction should follow the user-assistant format aka the chat completion format
# https://huggingface.co/docs/transformers/en/chat_templating
print(tokenizer.chat_template)

# How to use aya-23-8B-GGUF
# References:
# https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file
# https://llama-cpp-python.readthedocs.io/en/latest/


# First download a quantized model of choice from https://huggingface.co/bartowski/aya-23-8B-GGUF
# I downloaded aya-23-8B-Q6_K.gguf in the model directory

# Load the model
llm = Llama(
      model_path="./model/aya-23-8B-Q6_K.gguf",
      n_ctx=8192 # max number of tokens for aya-23-8B
)

# Let's start with a basic question
response = llm.create_chat_completion(
            messages=[
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]
            )








def create_prompt(user_question, context):
    """
    Create a prompt for the aya-23-8B model.

    Args:
    - user_question (str): The user's question or input.
    - context (str): The context for the question.

    Returns:
    - dict: The formatted prompt for the model.
    """
    prompt = [{
        "role": "user",
        "content": f"Answer the user question or input using the following context.\n"
                   f"If the answer you are looking for cannot be found in the context, "
                   f"say you don't know instead of making up an answer.\n"
                   f"The user question or input can be given in English or Flemish, "
                   f"but the context will always be Flemish.\n"
                   f"Make sure you answer in the same language as the user question.\n"
                   f"User question (answer in this language): {user_question}\n"
                   f"Context: {context}"
    }]
    return prompt

# Example usage:
user_question = "What is the capital of France?"
context = "De hoofdstad van Frankrijk is Parijs."
prompt = create_prompt(user_question, context)
print(prompt)

