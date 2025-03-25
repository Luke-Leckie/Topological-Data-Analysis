#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pickle
import os
import sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import time
from tqdm import tqdm
import numpy as np

tqdm.pandas()
#sys.path.append('/home/ll16598/Documents/Altered_States_Reddit/model_pipeline/__pycache__')
#from quality import reconst_qual, topic_diversity, coherence_centroid, coherence_pairwise #written for this jupyter notebook


# In[2]:


# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You may choose a larger model if needed.
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def get_gpt2_logprob(context, target_word, model, tokenizer):
    """
    Compute the log probability of the target_word given the context using GPT-2.
    Note: If the target word tokenizes into multiple tokens, this function sums their log probabilities.
    """
    # Encode the context and target word separately.
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)
    
    # Concatenate context and target tokens.
    full_ids = context_ids + target_ids
    input_ids = torch.tensor([full_ids])
    
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits  # shape: [1, sequence_length, vocab_size]
    
    # For each token in the target, the prediction is made at the previous position.
    log_probs = []
    for i, target_id in enumerate(target_ids):
        pos = len(context_ids) + i  # Position in the sequence for the target token.
        token_logits = logits[0, pos - 1, :]
        token_log_probs = F.log_softmax(token_logits, dim=-1)
        log_prob = token_log_probs[target_id].item()
        log_probs.append(log_prob)
    return sum(log_probs)

def compute_mean_surprisal(text, max_window=5, window_step=10, model=model, tokenizer=tokenizer):
    """
    For a given text, this function computes the mean surprisal (i.e. negative log probability)
    for each context window size from 1 to max_window.
    
    Returns a dictionary where each key is 'mean_surprisal_window_X' (with X as the window size).
    """
    words = text.split()
    # Create a dictionary to collect surprisal values for each window size.
    window_surprisal = {w: [] for w in range(1, max_window+1)}
    
    # Start at index 1 since the first word has no preceding context.
    for i in range(1, len(words)):
        # For each possible window size (up to the number of words available)
        for w in range(1, min(max_window, i) + 1, window_step):
            context = " ".join(words[i - w:i])
            target_word = words[i]
            logprob = get_gpt2_logprob(context, target_word, model, tokenizer)
            if logprob is not None:
                # Surprisal is defined as the negative log probability.
                surprisal = -logprob
                window_surprisal[w].append(surprisal)
            # A short pause can help throttle computation for long texts.
            time.sleep(0.1)
    
    # Compute the mean surprisal for each window size.
    mean_surprisal = {}
    for w, surps in window_surprisal.items():
        if surps:
            mean_surprisal[f"mean_surprisal_window_{w}"] = np.mean(surps)
            mean_surprisal[f"std_surprisal_window_{w}"] = np.std(surps)
        else:
            mean_surprisal[f"mean_surprisal_window_{w}"] = None
            mean_surprisal[f"std_surprisal_window_{w}"] =None
    return mean_surprisal



# In[3]:


def apply_surprisal(df_test, max_win=50,win_step=10):
    df_surprisal = df_test['clean_text'].progress_apply(lambda x: compute_mean_surprisal(x, max_window=max_win, window_step=win_step))
    df_surprisal = pd.DataFrame(df_surprisal.tolist())
    df_test = pd.concat([df_test, df_surprisal], axis=1)
    return(df_test)

def join_text(my_list):
    text= " ".join(my_list)
    return text


# In[4]:


user='cluster'
if user=='luke':
    working_dir='/home/ll16598/Documents/POSTDOC/'
    save_df_dir='/home/ll16598/Documents/POSTDOC/TDA/TDA_cluster/atom_assigned_dfs'
    dir_array='/home/ll16598/Documents/POSTDOC/TDA/TDA_cluster/window_vectors'
    save_df_dir='/home/ll16598/Documents/POSTDOC/TDA/TDA_cluster/final_dfs'
    dir_array='/home/ll16598/Documents/POSTDOC/TDA/TDA_cluster/unprocessed_topic_model_span_vecs'

elif user=='cluster':
    working_dir='/N/u/lleckie/Quartz/work/TDA_cluster/'
    save_df_dir=working_dir+'final_dfs'
    dir_array=working_dir+'unprocessed_topic_model_span_vecs'
df_name = sys.argv[3]

# In[5]:


df_to_test_list=[]
data_name_list=[]
df_names=os.listdir(save_df_dir)
for d, df_name in enumerate(df_names):
    df=pd.read_csv(save_df_dir+'/'+df_name)
    
    data_name_list.append(df_name[:-4])
    with open(f'{dir_array}/utterance_{1}_{df_name[:-4]}_back_utterances_arrays.pkl', 'rb') as f:
        utterances=pickle.load(f)
    df['utterances']=utterances
    df['clean_text']=df['utterances'].apply(join_text)
    df_to_test_list.append(df)
    
    
df_index = df_names.index(df_name)
dat=df_to_test_list[df_index]

# In[6]:


save_dat_dir=working_dir+'surprisal_data/'
os.makedirs(save_dat_dir, exist_ok=True)

print('computing surprisal metrics for: ',df_name)
prob_dat=apply_surprisal(dat, max_win=50, win_step=10)
prob_dat.to_csv(save_dat_dir+f'{df_name}_prob.csv')
    


# In[ ]:




