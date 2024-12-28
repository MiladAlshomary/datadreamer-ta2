"""
    Checks To Do:
        1. Ensure that unique count of authors per document is 1
        2. Minimum number of documents per author must be 1

"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import os, sys
tqdm.pandas(desc="Processing Author Inclusion Flag [Based on Cosine Similarity]")  # Enable progress_apply for pandas

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ceiling_threshold = 0.2

ds_path = f"/data/araghavan/HIATUS/datadreamer-ta2/data/sadiri/{sys.argv[1]}"
split=f"{sys.argv[2]}"

author_clm = "authorIDs"

print(f"Processing in ds_path: {ds_path}\n\tFor Split: {split}\n\tWith Threshold: {ceiling_threshold}")
## Reads queries and candidates file for that particular split
df_paths = Path(ds_path).glob("{}*.jsonl".format(split))
df_combined = pd.concat([pd.read_json(path, lines=True) for path in df_paths if 'filtered' not in str(path)]).reset_index(drop=True)

# Ensure that unique count of authors per document is 1
assert df_combined[df_combined['authorIDs'].apply(lambda x: len(x))>1].shape[0]==0, "documents not containing 1 unique author counts, please rectify"

# Flatten authorIDs list to authorID str
df_combined['authorID'] = df_combined['authorIDs'].apply(lambda x: x[0])
print(f"Flattened authorIDs to authorID: {df_combined.shape}")

# Minimum Threshold of documents per author filtering
# Requirement for similarity calculation
df_combined = df_combined[df_combined.groupby('authorID')['documentID'].transform('count')>1]
print(f"Filtered Out Authors with 1 or Less Documents: {df_combined.shape}")

def chunk_generate_embedding():
    all_embeddings = []
    batch_size = 2048
    total_texts = df_combined.shape[0]
    # Encode texts within the chunk in batches
    for i in tqdm(range(0, total_texts, batch_size), desc=f"Encoding Texts: "):
        batch_texts = df_combined.loc[i:i + batch_size-1]['fullText'].tolist()
        # print(len(batch_texts))
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)

    print(f"Shape of all_embeddings: {np.array(all_embeddings).shape}")
    return [embedding.tolist() for embedding in all_embeddings]  # Convert each embedding to a list


df_combined['document_embeddings'] = chunk_generate_embedding()
print(df_combined.head())
# Aggregate documents and fullText to author level
author_level_docs_df = df_combined.groupby('authorID', as_index=False).agg({'documentID': list, 'fullText': list, "document_embeddings": list})
del df_combined
print(f"Aggregated at AuthorID level: {author_level_docs_df.shape}")
# import pdb; pdb.set_trace()

### Take 1
# ## Compute Embeddings First For All Text Documents Instead
# all_texts = author_level_docs_df['fullText'].sum()  # Flatten all documents across authors
# all_embeddings = model.encode(all_texts, batch_size=1024)  # Adjust batch size as needed
# embedding_offset = 0
# embeddings_list = []
# for texts in author_level_docs_df['fullText']:
#     num_texts = len(texts)
#     embeddings_list.append(all_embeddings[embedding_offset:embedding_offset + num_texts])
#     embedding_offset += num_texts
# author_level_docs_df['document_embeddings'] = embeddings_list
# ### Take 2
# all_texts = author_level_docs_df['fullText'].sum()  # Flatten all documents across authors
# batch_size = 2048  # Adjust as needed
# all_embeddings = []

# for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding Texts"):
#     batch_texts = all_texts[i:i + batch_size]
#     batch_embeddings = model.encode(batch_texts)
#     all_embeddings.append(batch_embeddings)

# print(len(all_embeddings), len(all_embeddings[0]))
# all_embeddings = np.vstack(all_embeddings)
# print(len(all_embeddings))
# embedding_offset = 0
# embeddings_list = []
# for texts in author_level_docs_df['fullText']:
#     num_texts = len(texts)
#     embeddings_list.append(all_embeddings[embedding_offset:embedding_offset + num_texts])
#     embedding_offset += num_texts

# # Add embeddings as a column in the DataFrame
# author_level_docs_df['document_embeddings'] = embeddings_list


## Check Author: Take 1
# def check_author_inclusion_flag(lmodel, row, ceiling_threshold=0.2):
#     ## Ensure documents and document_ids are lists
#     assert type(row['documentID']) == list, "Must pass document_ids as list"
#     assert type(row['fullText']) == list, "Must pass documents as list"
#     ## Get Embeddings from Model
#     document_embeddings = lmodel.encode(row['fullText'])
#     pairwise_sim = cosine_similarity(document_embeddings, dense_output=True)
#     # Get shape of document_ids for the matrix shape determination 
#     x = len(row['documentID'])
#     # Get the indices for the upper triangular part (excluding the diagonal)
#     upper_indices = np.triu_indices(x, k=1)
#     # For lower triangular, use: lower_indices = np.tril_indices(x, k=-1)
#     # Generate the pairs based on document indices
#     # documentpairs_cossim_dict = {(documents_ids[i], documents_ids[j]):pairwise_sim[i, j] for i, j in zip(*upper_indices) if }
#     for i, j in zip(*upper_indices):
#         if pairwise_sim[i, j] <= ceiling_threshold:
#             return True
#     return False

## Check Author: Take 2
def check_author_inclusion_flag_with_embeddings(row, ceiling_threshold=0.2):
    document_embeddings = np.vstack(row['document_embeddings'])
    pairwise_sim = cosine_similarity(document_embeddings)
    
    # Check only the upper triangular part (excluding the diagonal)
    x = len(row['document_embeddings'])
    upper_indices = np.triu_indices(x, k=1)
    for i, j in zip(*upper_indices):
        if pairwise_sim[i, j] <= ceiling_threshold:
            return True
    return False

# author_level_docs_df['include_author_flag'] = author_level_docs_df.apply(
#     lambda x: check_author_inclusion_flag(model, x, ceiling_threshold=ceiling_threshold), 
#     axis=1
# )
# author_level_docs_df['include_author_flag'] = author_level_docs_df.progress_apply(
#     lambda x: check_author_inclusion_flag(model, x, ceiling_threshold=ceiling_threshold),
#     axis=1
# )
author_level_docs_df['include_author_flag'] = author_level_docs_df.progress_apply(
    lambda x: check_author_inclusion_flag_with_embeddings(x, ceiling_threshold=ceiling_threshold),
    axis=1
)

print(f"Included Author Inclusion Flag: {author_level_docs_df.shape}")

authors_to_include = author_level_docs_df.loc[author_level_docs_df.include_author_flag == True]['authorID'].tolist()

author_level_docs_df.to_json(os.path.join(ds_path, split + '_info.json'))
print(f"Written to Disk Author Information for future reference: {os.path.join(ds_path, split + '_info.json')}")

del author_level_docs_df

df_paths = Path(ds_path).glob("{}*.jsonl".format(split))
for path in df_paths:
    ## Ignore previously created filtered files
    if 'filtered' in str(path):
        continue

    df = pd.read_json(path, lines=True)
    df['authorID'] = df['authorIDs'].apply(lambda x: x[0])
    print(str(path), '{} --> {}'.format(len(df), len(df[df.authorID.isin(authors_to_include)])))
    df = df[df.authorID.isin(authors_to_include)]
    with open(str(path).replace('.jsonl','_filtered.jsonl'), "w") as f:
        f.write(df.to_json(orient='records', lines=True))

## Done