
# Corpus Key RAG Virtual Teaching Assistant

## Overview

The Corpus Key RAG Virtual Teaching Assistant is a modular system designed to enable semantic search over academic documents using transformer-based language models. This pipeline supports document preprocessing, embedding generation, and retrieval operations, and is particularly suited for educational environments that involve large collections of readings, lecture content, and instructional material.

This project was developed as part of a graduate-level consulting course at Purdue University to support a virtual assistant capable of answering student queries by referencing course materials.

## Features

* Extraction and preprocessing of content from PDF, DOCX, TXT, and LaTeX files
* Document chunking with overlapping windows to preserve context
* Embedding generation using BERT and Sentence-BERT (SBERT) models
* Semantic retrieval based on cosine similarity across embedding layers
* Bibliographic logging of source documents for transparency and evaluation

## Project Structure

The directory is organized as follows:

```
mokshda13-corpus_key_rag_virtual-ta/
├── ChopDocuments.py           # Preprocesses and chunks documents
├── CreateFinalData.py         # Combines chunks and embeddings into final dataset
├── EmbedDocuments.py          # Generates embeddings for text chunks
├── get_most_similar.py        # Retrieves top similar chunks for a given query
├── test.py                    # Embedding validation test
├── settings.txt               # Configuration file
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup Instructions

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/mokshda13-corpus_key_rag_virtual-ta.git
   cd mokshda13-corpus_key_rag_virtual-ta
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Configure `settings.txt` as needed for your local environment.

## Pipeline Execution

The system is designed to be executed in the following stages:

### Step 1: Preprocess and Chunk Documents

Run the following to tokenize, clean, and split documents into overlapping segments:

```
python ChopDocuments.py
```

### Step 2: Generate Embeddings

This step encodes each chunk using BERT and SBERT models:

```
python EmbedDocuments.py
```

### Step 3: Bundle Chunks and Embeddings

Merge chunk metadata with generated embeddings into unified output files:

```
python CreateFinalData.py
```

### Step 4: Retrieve Relevant Texts

Perform semantic search based on a query input to extract the most similar content chunks:

```
python get_most_similar.py
```

## Example Usage in Code

```python
from get_most_similar import Get_Most_Similar

retriever = Get_Most_Similar(
    openai_api_key_file="path/to/APIkey.txt",
    corpa_directory="path/to/project",
    dataname="chunks_ll",
    num_chunks=5
)

query = "What are the key principles of management consulting?"
results = retriever.get_most_similar(query, layer="last")
print(results)
```

## Output

* `chunks-originaltext.csv`: Cleaned and segmented text data
* `chunks_ll.npy` and `chunks_sll.npy`: Layer-based embeddings
* Bibliographic summaries for each query
* Retrieval statistics including average, minimum, maximum, and mode similarity scores

## Dependencies

Major libraries used in this project include:

* Transformers
* Torch
* Sentence-Transformers
* Pandas and NumPy
* NLTK
* PyPDF2 and python-docx
* OpenAI SDK

Refer to `requirements.txt` for the complete list.

## Acknowledgments

This project was developed under the supervision of Professor John F. Burr for the course "Fundamentals of Management Consulting" at Purdue University. It is intended for instructional and experimental use.
