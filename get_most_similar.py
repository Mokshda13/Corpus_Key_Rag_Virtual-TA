#get_most_similar.py
from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
import os
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# chunks -> answer

#openai_api_key_file = r"H:\My Drive\CorpusKey_Limited\keys\APIkey.txt"  #for testing in personal computer
#google_api_key_path = r"H:\My Drive\CorpusKey_Limited\keys\GoogleAPIkey.txt"
#client = OpenAI(openai_api_key_file)

df_chunks = None
embedding= None
embedding_ll = None
embedding_sll = None

class Get_Most_Similar:
    def __init__(self, openai_api_key_file, corpa_directory, dataname, num_chunks, embedding_model = "text-embedding-3-small"):
        # with open(openai_api_key_file, "r") as f:
        #     self.api_key = f.read().strip()
        # openai.api_key = self.api_key
        self.corpa_directory = corpa_directory
        self.dataname = dataname
        self.num_chunks = num_chunks
        # self.embedding_model = embedding_model
        self.bibliography = []
        self.embedding_token_usage = {}

    # def get_embeddings_from_openai(self, text_list: list) -> list: #this can be a list but I am using this as a string
    #     """

    #     Args:
    #         text_list (list): List of text blocks

    #     Response format:
    #     {
    #     "object": "list",
    #     "data": [
    #         {
    #         "object": "embedding",
    #         "index": 0,
    #         "embedding": [...]
    #         }
    #     ],
    #     "model": "text-embedding-ada-002-v2",
    #     "usage": {
    #         "prompt_tokens": 20,
    #         "total_tokens": 20
    #     }
    #     }

    #     Returns:
    #         list: Embeddings from openai
    #     """

    #     response = openai.embeddings.create(
    #         model=self.embedding_model,
    #         input=text_list
    #     )

    #     embedding = response.data[0].embedding
        
    #     # Extract token usage information
    #     model_name = response.model
        
    #     prompt_tokens = response.usage.prompt_tokens
        
    #     total_tokens = response.usage.total_tokens

    #     # Update the token usage dictionary
    #     if model_name not in self.embedding_token_usage:
    #         self.embedding_token_usage[model_name] = {'prompt_tokens': 0, 'total_tokens': 0}
        
    #     self.embedding_token_usage[model_name]['prompt_tokens'] += prompt_tokens
    #     self.embedding_token_usage[model_name]['total_tokens'] += total_tokens
    #     print(f"embeddings look like this: {embedding[:3]} but are longer")
    #     return embedding

    def embed_prompt_bert(self, prompt_string):
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

            config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True,  output_attentions=True)
            bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

            tokenized_prompt = bert_tokenizer(prompt_string, padding=True, truncation=True, max_length=512, return_tensors='pt')

            with torch.no_grad():
                bert_output = bert_model(**tokenized_prompt)
                        
            hidden_states = bert_output.hidden_states  
            prompt_last_layer = hidden_states[-1]  
            prompt_second_last_layer = hidden_states[-2]  

            prompt_ll_embedding = prompt_last_layer[:, 0, :].numpy().T
            prompt_sll_embedding = prompt_second_last_layer[:, 0, :].numpy().T

            prompt_embeddings = {'last_layer': prompt_ll_embedding, 'second_last_layer': prompt_sll_embedding}

            return prompt_embeddings

    def hierarchical_pooling(hidden_states, window_size=4):
        last_layer = hidden_states[-1]
        second_last = hidden_states[-2]
        
        def window_pool(layer):
            batch_size, seq_len, hidden_size = layer.shape
            # Pad sequence to be divisible by window_size
            pad_len = (window_size - seq_len % window_size) % window_size
            if pad_len > 0:
                padding = torch.zeros(batch_size, pad_len, hidden_size, device=layer.device)
                layer = torch.cat([layer, padding], dim=1)
            
            # Reshape to group tokens into windows
            windows = layer.view(batch_size, -1, window_size, hidden_size)
            # Pool within windows
            pooled = torch.max(windows, dim=2)[0]
            return pooled
        
        # Pool each layer separately
        pooled_last = window_pool(last_layer)
        pooled_second = window_pool(second_last)
        
        # Combine pooled representations
        return torch.cat([pooled_last, pooled_second], dim=-1)
    
    def get_token_usage(self):
        return self.embedding_token_usage
    
    def get_bibliography(self):
        return self.bibliography

    # Load the data        
    def load_df_chunks(self):
        """
        loads df in order - there's no index so the set is simply title, text, embedding
        """        
        global df_chunks, embedding
        embedding = None
        
        if embedding is None:
            df_chunk_file_path = os.path.join(self.corpa_directory, "bundle", "chunks-originaltext.csv")
            df_chunks = pd.read_csv(df_chunk_file_path)
            
            embedding_file_path = os.path.join(self.corpa_directory, "bundle", f"{self.dataname}.npy")
            embedding = np.load(embedding_file_path)
        return df_chunks

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def get_most_similar(self, embed_string, layer):
        """
        This is the same method of chunk retrieval for all situations with the exception that this retrieved titles as well
        even though this does not use titles for any purpose, this is an updated function

        method of similarity is dot product
        
        need to change to "text-embedding-3-small"

        args
            embed_string (str): queary string

        return
            most_similar (str): num_chunks number of text chunks from embedding

        """   
        # (1471,768) (768,1) 

        try:
            if layer not in {"last", "second_last"}:
                raise ValueError("Invalid input: layer must be either 'last' or 'second_last'")
            
            df_chunks = self.load_df_chunks()
            prompt_embeddings = self.embed_prompt_bert(embed_string)
            layer_embedding = prompt_embeddings[f'{layer}_layer']

            for idx, embed in enumerate(embedding):
                similarity = cosine_similarity(embed.reshape(-1, 1), layer_embedding)[0][0]
                df_chunks.at[idx, 'similarity'] = similarity

            df_chunks = df_chunks.sort_values(by='similarity', ascending=False)
            most_similar_df = df_chunks.head(self.num_chunks)            
            most_similar_df = most_similar_df.drop_duplicates(subset=['Title', 'Text']) 
            most_similar = '\n\n'.join(row[1] for row in most_similar_df.values) #takes second column from title, text and creates most_similar
            
            most_similar_titles = '\n\n'.join(row[0] for row in most_similar_df.values) #takes second column from title, text and creates most_similar
            print(most_similar_titles)

            # Calculating statistics
            similarities = df_chunks['similarity'].head(self.num_chunks)
            average_similarity = similarities.mean()
            min_similarity = similarities.min()
            max_similarity = similarities.max()
            mode_similarity = similarities.mode().iloc[0] if not similarities.mode().empty else "N/A"

            # Converting to string
            average_similarity_str = str(average_similarity)
            min_similarity_str = str(min_similarity)
            max_similarity_str = str(max_similarity)
            mode_similarity_str = str(mode_similarity)

            # Printing statistics
            print("Average Similarity:", average_similarity_str)
            print("Min Similarity:", min_similarity_str)
            print("Max Similarity:", max_similarity_str)
            print("Mode Similarity:", mode_similarity_str)
            
            most_similar_df = most_similar_df.drop_duplicates(subset=['Title', 'Text'])
            title_counts = most_similar_df['Title'].value_counts()
            title_df = pd.DataFrame({'Title': title_counts.index, 'Count': title_counts.values}).sort_values('Count', ascending=False)
            title_df_filtered = title_df[title_df['Count'] >= 3]
            titles = title_df_filtered['Title'].values.tolist()
            if len(titles) == 1:
                self.bibliography.append(str(titles[0]) + '\n' + "Used for: " + embed_string + ". Avg Similarity of: " + average_similarity_str + '\n')
            elif len(titles) == 0:
                self.bibliography.append(" No reference found for " + '\n' + embed_string + " with a similarity of " + average_similarity_str + '\n')
            else:
                self.bibliography.append(str(titles[0]) + '\n' + "Used for: "  + embed_string + ". Avg Similarity of: " + average_similarity_str + '\n' + str(titles[1]) + '\n' + "Used for: "  + embed_string + ". Avg Similarity of: " + average_similarity_str + '\n')
            return most_similar
            
        except Exception as e:
            # Handle other exceptions
            print(f"An error occurred: {str(e)}")
            return e
        
if __name__ == "__main__":
    
    #point this to your local storage
    os.chdir(r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing")
    current_dir = os.getcwd()
    corpa_directory = os.path.join(current_dir)


    num_chunks = 4
    dataname_ll = "chunks_ll"  #this version the dataname is hardcoded
    dataname_sll = "chunks_sll"  #this version the dataname is hardcoded

    embedding_model = "text-embedding-3-small"
    openai_api_key_file = r"C:\Users\Hadi\Desktop\Code\Projects\All Day-TA - Updated\embeddings_testing\APIkey.txt"

    processor_ll = Get_Most_Similar(openai_api_key_file, corpa_directory, dataname_ll, num_chunks, embedding_model)
    prompt = "list down the thirteen cardinal sins"
    results = processor_ll.get_most_similar(prompt, "last")
    print(results)

    print("_________________________________________________________________________________________________")

    processor_sll = Get_Most_Similar(openai_api_key_file, corpa_directory, dataname_sll, num_chunks, embedding_model)
    results = processor_sll.get_most_similar(prompt, "second_last")

    print(results)

    # data = np.load('bundle/chunks.npy')

    # dimensions = data.shape
    # print(dimensions)


    # def hierarchical_pooling(hidden_states, window_size=4):
    #     last_layer = hidden_states[-1]
    #     second_last = hidden_states[-2]
        
    #     def window_pool(layer):
    #         batch_size, seq_len, hidden_size = layer.shape
    #         # Pad sequence to be divisible by window_size
    #         pad_len = (window_size - seq_len % window_size) % window_size
    #         if pad_len > 0:
    #             padding = torch.zeros(batch_size, pad_len, hidden_size, device=layer.device)
    #             layer = torch.cat([layer, padding], dim=1)
            
    #         # Reshape to group tokens into windows
    #         windows = layer.view(batch_size, -1, window_size, hidden_size)
    #         # Pool within windows
    #         pooled = torch.max(windows, dim=2)[0]
    #         return pooled
        
    #     # Pool each layer separately
    #     pooled_last = window_pool(last_layer)
    #     pooled_second = window_pool(second_last)
        
    #     # Combine pooled representations
    #     return torch.cat([pooled_last, pooled_second], dim=-1)