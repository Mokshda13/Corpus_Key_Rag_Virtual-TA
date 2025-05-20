from tenacity import retry, wait_random_exponential, stop_after_attempt
import openai
import os
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.metrics.pairwise import cosine_similarity

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

prompt_string = "Hello, I am a graduate student at Purdue"

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True,  output_attentions=True)
bert_model = BertModel.from_pretrained('bert-base-uncased', config=config)

tokenized_text = bert_tokenizer(prompt_string, padding=True, truncation=True, max_length=512, return_tensors='pt')
