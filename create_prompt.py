from calculate_token_length import token_length_of_prompt, token_length_of_str
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from llama_cpp import Llama

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={'device': 'cpu'})
model_id = "CohereForAI/aya-23-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

context_length_cutoff = 7000

default_system_message = {
    "role": "system",
    "content": "You are an AI-assistant that will answer a user question about the Flemish Parliament."
               "The language of your answer should match the user question, not the context."
               "For each user question, you will be given some data, which are snippets of"
               "dictations from the Flemish Parliament. Try to answer the user question"
               "using the data. If the answer is not found in the data, answer"
               "'I am sorry, that information is not found in my data.'"

}

token_length_of_prompt([default_system_message])


def create_context(list_of_search_results, context_length_cutoff):
    context_length = 0
    context = ""
    for i, search_result in enumerate(list_of_search_results):
        context_length += token_length_of_str(search_result.page_content)
        if context_length >= context_length_cutoff:
            return context
        context = (
                context +
                f"Data {i + 1}: " +
                search_result.page_content +
                "\n"
        )
    return context


def create_prompt(system_settings, user_question, context):
    if system_settings is None:
        system_settings = default_system_message
    prompt = [
        system_settings,
        {
            "role": "user",
            "content": f"User question: {user_question}\n"
                       f"{context}"
        }]
    return prompt


user_question = "Welke voordelen krijgen parlementsleden?"

new_db = FAISS.load_local("faiss_index_written_questions_202401_202406", embeddings=embedding_model,
                          allow_dangerous_deserialization=True)
docs = new_db.similarity_search(user_question)

test_context = create_context(docs, context_length_cutoff=context_length_cutoff)

test_prompt = create_prompt(default_system_message, user_question, test_context)

print(test_prompt[1]["content"])

# Load the model
llm = Llama(
    model_path="./model/aya-23-8B-Q6_K.gguf",
    n_ctx=8192  # max number of tokens for aya-23-8B

)

test_response = llm.create_chat_completion(test_prompt, temperature=0)
