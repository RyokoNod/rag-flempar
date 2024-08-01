from llama_cpp import Llama
from transformers import AutoTokenizer
from calculate_token_length import TokenLengthCalculator

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
    n_ctx=8192  # max number of tokens for aya-23-8B
)

# Let's start with a basic question
response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Where are the Olympic games being held in 2024?"
        }
    ]
)

# Let's do a simplified, make-believe RAG with fake information
fake_response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "Where are the Olympic games being held in 2024?"
                       "Context: Due to a global outbreak of COVID-20, "
                       "the next epidemic after the 2020 COVID-19 pandemic, the 2024 Olympic games have been canceled."
                       "The next summer Olympics will be held in Los Angeles in 2028."
        }
    ]
)

# Can you do that in Dutch?
response_nl = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content":  "You are Command-R, a brilliant, sophisticated, AI-assistant "
                        "trained to assist human users by providing thorough responses."
                        "Your users are from Flanders, so make sure you answer using Flemish."
        },
        {
            "role": "user",
            "content": "Where are the Olympic games being held in 2024?"
        }
    ]
)

# Can you do the fake thing in Dutch?
fake_response_nl = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content":  "You are Command-R, a brilliant, sophisticated, AI-assistant "
                        "trained to assist human users by providing thorough responses."
                        "Your users are from Flanders, so make sure you answer using Flemish."
        },
        {
            "role": "user",
            "content": "Where are the Olympic games being held in 2024?"
                       "Context: Due to a global outbreak of COVID-20, "
                       "the next epidemic after the 2020 COVID-19 pandemic, the 2024 Olympic games have been canceled."
                       "The next summer Olympics will be held in Los Angeles in 2028."
        }
    ]
)

test_prompt = [
        {
            "role": "system",
            "content":  "You are Command-R, a brilliant, sophisticated, AI-assistant "
                        "trained to assist human users by providing thorough responses."
                        "Your users are from Flanders, so make sure you answer using Flemish."
        },
        {
            "role": "user",
            "content": "Where are the Olympic games being held in 2024?"
                       "Context: Due to a global outbreak of COVID-20, "
                       "the next epidemic after the 2020 COVID-19 pandemic, the 2024 Olympic games have been canceled."
                       "The next summer Olympics will be held in Los Angeles in 2028."
        }
    ]

token_counter = TokenLengthCalculator(model_id)
token_length_full = token_counter.token_length_of_prompt(test_prompt)
token_length_system = token_counter.token_length_of_prompt([test_prompt[0]])
token_length_user = token_counter.token_length_of_prompt([test_prompt[1]])

token_length_system + token_length_user