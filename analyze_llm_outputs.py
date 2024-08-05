import json
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd

filename = r"./aya-23-8B-logs.jsonl"

with open(filename, 'r') as json_file:
    log_list = list(json_file)

jsonl_row_n = 1 # the jsonl row number I want to analyze

# load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small",
                                        model_kwargs={'device': 'cpu'})
db = FAISS.load_local("faiss_index_written_questions_202401_202406", embeddings=embedding_model,
                      allow_dangerous_deserialization=True)

# load original text
df = pd.read_csv("./data/written_questions_202401_202406.csv")

def extract_fields(text):
    """
    Extract user question and context from a prompt
    :param text: Prompt (user role)
    :return: Dictionary of user question and context
    """
    # Extract the user question
    user_question_match = re.search(r"User question:\s*(.*?)(?=(Data \d+:|$))", text, re.DOTALL)
    user_question = user_question_match.group(1).strip() if user_question_match else None

    # Extract all data fields
    data_matches = re.findall(r"(Data \d+):\s*(.*?)(?=(Data \d+:|$))", text, re.DOTALL)

    # Create a dictionary with the extracted fields
    result = {
        "user_question": user_question,
    }

    for match in data_matches:
        key = match[0].replace(" ", "").lower()
        value = match[1].strip()
        result[key] = value

    return result

def print_prompt_components(prompt_dict, context_n):
    """Print the uer question and contexts"""
    print("User question:\n", prompt_dict["user_question"])
    for i in range(context_n):
        context_key = "data" + str(i + 1)
        print(f"{context_key}:\n{prompt_dict[context_key]}")

def get_full_texts(prompt_dict, context_n):
    """Get the full texts of the prompt content"""
    full_texts = {}
    for i in range(context_n):
        context_key = "data" + str(i + 1)
        # the first match in the similarity search should be the original text, since they are exact matches
        docs = db.similarity_search(prompt_dict[context_key])
        # get the full text of the matched snippets
        full_text_id = int(docs[0].metadata["id_fact"])
        full_text = df.loc[df["id_fact"] == full_text_id, "text"].values[0]
        full_texts[full_text_id] = full_text
    return full_texts

def list_intersections(list1, list2):
    # Find the number of common elements in both lists
    common_elements = set(list1).intersection(set(list2))
    num_common_elements = len(common_elements)

    # Find the total number of unique elements in both lists
    total_elements = set(list1).union(set(list2))
    num_total_elements = len(total_elements)

    # Calculate the percentage similarity
    percentage_similarity = (num_common_elements / num_total_elements) * 100

    return percentage_similarity

def print_dictionary(dictionary):
    for k, v in dictionary.items():
        print(f"{k}:\n {v}")

log = json.loads(log_list[jsonl_row_n])
prompt = extract_fields(log["prompt"][1]["content"])
response = log["response"]["content"]

# By default, FAISS returns the top 4 matches
# Since I didn't override the default, there should always be 4 context excerpts
faiss_match_n = 4

# print the user question and contexts
print_prompt_components(prompt, faiss_match_n)

# get the full version of the contexts
full_text_context = get_full_texts(prompt, faiss_match_n)


# from here I compare the context and response I get for the same question but in a different language
jsonl_row_n_difflang = 2 # if I have the same question as jsonl_row_n in another language, that row number

log_difflang = json.loads(log_list[jsonl_row_n_difflang])

prompt_difflang = extract_fields(log_difflang["prompt"][1]["content"])
response_difflang = log_difflang["response"]["content"]

# print the user question and contexts
print_prompt_components(prompt_difflang, faiss_match_n)

# get the full version of the contexts
full_text_context_difflang = get_full_texts(prompt_difflang, faiss_match_n)

# how similar is the response when you ask the same question in a different language?
print(f"Response for language 1: \n {response}")
print(f"Response for language 2: \n {response_difflang}")

# Percentage similarity of context list
# using "|" operator + "&" operator + set()
context_ids = list(full_text_context.keys())
context_ids_difflang = list(full_text_context_difflang.keys())
list_similarity = list_intersections(context_ids, context_ids_difflang)
print("Context id match between languages: {:.2f}%".format(list_similarity))


# print contexts
print_dictionary(full_text_context)
print_dictionary(full_text_context_difflang)