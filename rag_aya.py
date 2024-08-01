import os
import argparse
from calculate_token_length import TokenLengthCalculator
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
import json
import datetime

# the embedding model used to create the FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={'device': 'cpu'})

# the LLM used for user interface and where the quantized version is located.
model_id = "CohereForAI/aya-23-8B"
quantized_model_path = os.path.join('model', 'aya-23-8B-Q6_K.gguf')

# token counter, how many tokens the context should be at maximum, and how many tokens the LLM can take
token_counter = TokenLengthCalculator(model_id)
context_length_cutoff = 7000
llm_max_tokens = 8192

# the default system message that instructs the LLM's behavior
default_system_message = {
    "role": "system",
    "content": "You are an AI-assistant that will answer a user question about the Flemish Parliament."
               "The language of your answer should match the user question, not the context."
               "For each user question, you will be given some data, which are snippets of"
               "dictations from the Flemish Parliament. Try to answer the user question"
               "using the data. If the answer is not found in the data, answer"
               "'I am sorry, that information is not found in my data.'"

}


def search_faiss(user_question: str, vector_store_index: str) -> list:
    """
    Searches a FAISS index for the specified content.
    :param user_question: The user query to use to search the FAISS index.
    :param vector_store_index: The name of the FAISS index.
    :return: A list of documents from FAISS
    """
    vector_store = FAISS.load_local(vector_store_index, embeddings=embedding_model,
                                    allow_dangerous_deserialization=True)
    search_results = vector_store.similarity_search(user_question)
    return search_results


def create_context(list_of_search_results: list, context_length_cutoff: int) -> str:
    """
    Creates the context for RAG using a list of context information.
    If the context exceeds the cutoff token count, all context info beyond that will not be included in the context
    given to the LLM later.
    :param list_of_search_results: List of matched items from the vector store.
    :param context_length_cutoff: How many tokens the context should be at maximum.
    :return: The context for RAG. Each context will be preceded by "Data n" and end with a new line.
    """
    context_length = 0
    context = ""
    for i, search_result in enumerate(list_of_search_results):
        context_length += token_counter.token_length_of_str(search_result.page_content)
        if context_length >= context_length_cutoff:
            return context
        context = (
                context +
                f"Data {i + 1}: " +
                search_result.page_content +
                "\n"
        )
    return context


def create_prompt(system_settings: dict, user_question: str, context: str) -> list:
    """
    Creates the prompt for RAG.
    :param system_settings: The system message for the LLM.
    :param user_question: The user question.
    :param context: The context to be provided with the user_question.
    :return: The complete RAG prompt.
    """
    if system_settings is None:
        system_settings = default_system_message
    prompt = [
        system_settings,
        {
            "role": "user",
            "content": f"User question: {user_question}\n"
                       f"{context}"
        }
    ]
    return prompt


def log_response(prompt, response):
    """Logs the prompt and response to a file with a timestamp"""
    log_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "response": list(response)
    }
    with open("llm_logs.jsonl", "a") as log_file:
        log_file.write(json.dumps(log_data) + "\n")


def main(faiss_index: str, log: bool) -> None:
    # Get the user's question from input
    user_question = input("Please enter your question: ")

    # search FAISS for excerpts related to the user question
    related_documents = search_faiss(user_question, faiss_index)

    # create a context out of the search results
    context = create_context(related_documents, context_length_cutoff=context_length_cutoff)

    # create a prompt from the user question and context
    prompt = create_prompt(default_system_message, user_question, context)

    # Load the model
    llm = Llama(
        model_path=quantized_model_path,
        n_ctx=llm_max_tokens,
        verbose=False
    )

    # get the response to the user question
    response = llm.create_chat_completion(prompt, temperature=0, stream=True)

    req = ""
    for chunk in response:
        delta = chunk["choices"][0]["delta"]
        if "content" not in delta:
            continue
        tokens = delta['content'].split()
        for token in tokens:
            print(token, end=" ", flush=True)

    # Log the prompt and response if logging is enabled
    if log:
        log_response(prompt, response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Try RAG using the quantized version of CohereForAI/aya-23-8B')
    parser.add_argument('faiss_index', type=str, help='The FAISS index to search supporting information')
    parser.add_argument('--log', action='store_true', help='(Optional) Whether you want to log the results')
    args = parser.parse_args()

    main(args.faiss_index, args.log)

