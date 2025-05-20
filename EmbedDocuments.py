# EmbedDocuments.py
# This code embeds all the text chunks from ChopDocuments.py
import os
import time
import pandas as pd
import numpy as np
import openai
import torch

from transformers import BertModel, BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Read API key from file - change based on mapping
openai_api_key_file = r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing\APIkey.txt"
with open(openai_api_key_file, "r") as f:
    api_key = f.read().strip()
client = openai.OpenAI(api_key=api_key)

# embeddingmodel = "text-embedding-3-small"

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sbert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True,  output_attentions=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

# SFR_tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
# SFR_model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")
# SFR_target_layers = len(SFR_model.layers) - 1
# print(SFR_target_layers)

# token_use_dict = {}

# Now you can use the 'client' object to interact with OpenAI API
def make_embeddings_folder(corpa_directory):
    output_folders = [os.path.join(corpa_directory, folder) for folder in ["embeddings_ll", "embeddings_sll"]]

    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)

    return tuple(output_folders)

# load user settings and api key
def read_settings(file_name):
    settings = {}
    with open(file_name, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            settings[key] = value
    return settings

def get_bert_embeddings(embed_string):

    # Tokenize the input text string and input the tokenized text into BERT
    tokenized_text = bert_tokenizer(embed_string, padding=True, truncation=True, max_length=512, return_tensors='pt')

    with torch.no_grad():
        bert_output = bert_model(**tokenized_text)

    # Extract the last two hidden layers from BERT
    hidden_states = bert_output.hidden_states  

    last_layer = hidden_states[-1]  
    second_last_layer = hidden_states[-2]

    # Obtain the CLS embeddings for the last two layers - CLS embeddings are a representation of the whole input sequence
    # We do this because otherwise, the layers have a size of (1, x, 768) where x changes based on the number of tokens in
    # embed string. This would make it so we can't compute the cosine similarity of each layer with the prompt embedding.
    # By extracting the CLS embedding, we can standardize a layer size of (1, 768) which allows us to now compute layer-prompt
    # similarity scores

    last_layer_cls_embedding = last_layer[:, 0, :].numpy().T
    second_last_layer_cls_embedding = second_last_layer[:, 0, :].numpy().T

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # You can use other models too
    # embedding = model.encode([embed_string])

    # prompt = "list down the thirteen cardinal sins"
    # embed_1 = model.encode([embed_string])
    # embed_2 = model.encode([prompt])
    # tokenized_prompt = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # with torch.no_grad():
    #     outputs_p = sbert_model(**tokenized_prompt, output_hidden_states=True)

    # hidden_states_p = outputs_p.hidden_states  

    # last_layer_p = hidden_states_p[-1]  
    # last_layer_cls_embedding_p = last_layer_p[:, 0, :].numpy()
    # # print(last_layer_cls_embedding_p.shape)


    # similarity = cosine_similarity( embed_2, embed_2)[0][0]
    # print(similarity)
    # print("___________________________________________________")


    return (last_layer_cls_embedding, second_last_layer_cls_embedding)
    # return embedding


# Define the function to send a batch of input text to the OpenAI API and return the embeddings
def embed_input_text(embed_string):
    global token_use_dict
    global embeddingmodel
     
    response = client.embeddings.create(
        model=embeddingmodel,
        input=embed_string
    )

    print("working")
    embeddings = response.data
    embed_tokens = response.usage.total_tokens
    model = response.model

    for model in token_use_dict:
        if 'embed_tokens' not in token_use_dict[model]:
            token_use_dict[model]['embed_tokens'] = 0

        token_use_dict[model]['embed_tokens'] += embed_tokens
    else:
        if model not in token_use_dict:
                token_use_dict[model] ={}
        
        token_use_dict[model] = {'embed_tokens': embed_tokens}
    # Calculate total tokens

    return [response.embedding for response in embeddings]


def make_embeddings_array(textchunks_folder, embeddings_folder_ll, embeddings_folder_sll):
    for file in os.listdir(textchunks_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(textchunks_folder, file)
            df_chunks = pd.read_csv(file_path, encoding='utf-8', escapechar='\\')
            print(f"Loaded: {file_path}")
            
            # Embed the input text in batches of no more than MAX_TOKENS_PER_BATCH tokens each
            input_text_list = df_chunks.iloc[:, 1].tolist()

            embeddings_last_layer, embeddings_second_last_layer = [], []
            embedding = []

            for embed_string in input_text_list:
                layer_embeddings = get_bert_embeddings(embed_string)

                embeddings_last_layer.append(layer_embeddings[0])
                embeddings_second_last_layer.append(layer_embeddings[1])

            # Convert embeddings list to numpy array
            embeddings_last_layer_arr = np.array(embeddings_last_layer).reshape(len(embeddings_last_layer), -1)
            embeddings_second_last_layer_arr = np.array(embeddings_second_last_layer).reshape(len(embeddings_second_last_layer), -1)

            # Save the embeddings_array to the output_folder subfolder
            # Remove the file extension from the filename
            filename_without_extension = os.path.splitext(file)[0]

            npy_filename_ll = f"{filename_without_extension}_ll.npy"
            npy_filename_sll = f"{filename_without_extension}_sll.npy"

            output_path_ll = os.path.join(embeddings_folder_ll, npy_filename_ll)
            output_path_sll = os.path.join(embeddings_folder_sll, npy_filename_sll)

            np.save(output_path_ll, embeddings_last_layer_arr)
            np.save(output_path_sll, embeddings_second_last_layer_arr)

if __name__ == "__main__":
    print("This is module.py being run directly.")
    os.chdir(r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing")
    current_dir = os.getcwd()

    corpa_directory = os.path.join(current_dir)

    # Define the maximum number of tokens per batch to send to OpenAI for embedding per minute
    MAX_TOKENS_PER_BATCH = 250000
    settings_file_path = os.path.join(corpa_directory, "settings.txt")
    settings = read_settings(settings_file_path)
    
    # Load text data from Textchunks
    textchunks_folder = os.path.join(corpa_directory, "chunks")

    embedding_file_path = os.path.join(corpa_directory, "embeddings\\")

    embeddings_folders = make_embeddings_folder(corpa_directory)
    make_embeddings_array(textchunks_folder, embeddings_folders[0], embeddings_folders[1])
    

    
