# Creating a RAG pipeline from Flemish parliament discussions

This is a repository linked to Datamarinier's [blog series about creating a RAG pipeline](https://medium.com/@Datamarinier/making-an-api-out-of-a-hugging-face-model-introduction-a0c4b2408f52).
By using the files here, you will be able to follow and recreate the proof of concept described in the posts.

## A brief documentation about the files and folders

For a full description, please see the blog series.

### 1. download_written_questions.R

A script to download the written questions from the Flemish parliament.
These will be used as a context to feed to the LLM with the user input.

### 2. data

The download results from `download_written_questions.R`. 
This contains written questions from 1 January 2024 to 30 June 2024.

### 3. create_vector_store.py

The script that transforms the raw download in the `data` directory to a FAISS index.
Run like below from the command line (the command is for Linux and MacOS). 

```commandline
python3 path_to_flempar_download.csv
```

If you are using the exact same input as us, the command would be

```commandline
python3 create_vector_store.py data/written_questions_202401_202406.csv 
```

### 4. faiss_index_written_questions_202401_202406

The FAISS index created after we run `create_vector_store.py` using `data/written_questions_202401_202406.csv`
as input.

### 5. calculate_token_length.py

A helper module that calculates the approximate number of tokens included in a text or prompt.

### 6. rag_aya.py

The actual RAG pipeline. To use the pipeline, execute as below (the `--log` argument is optional).

```commandline
python3 rag_aya.py path_to_faiss_index --log
```

### 7. aya-23-8B-logs.jsonl

The log file where the user input, context, and the response is stored if you set the `--log` argument
when running `rag_aya.py`.

## Test scripts

These files don't affect the RAG pipeline, but are handy to test the components and the complete RAG pipeline.

- `test_vector_store.py`
