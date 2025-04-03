import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import torch
from tqdm import tqdm

""""
Previously run this file: /data/araghavan/HIATUS/datadreamer-ta2/src/generate_luarmud_emb_create_chunk.py
See commands in that script to run it
Once done, run this to read chunks and perform hard positive filtering
"""
start_time = time.time()
chunks_path = "/data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/trainsadiri_luarmud_chunks/"
print(f"[{time.time() - start_time:.2f}s] Checking path: {chunks_path}")
ceiling_threshold = 0.4

df_arr = []
# Get sorted file list
file_list = sorted(os.listdir(chunks_path))

# Optional: Filter to only .jsonl files
file_list = [f for f in file_list if f.endswith(".jsonl")]

# Load in sorted order
for file in file_list:
    if not file.endswith(".jsonl"):
        continue  # optionally skip non-JSONL files
    print(f"\tProcessing file: {file}")
    df = pd.read_json(os.path.join(chunks_path, file), lines=True)
    df_arr.append(df)


print(f"[{time.time() - start_time:.2f}s] Finished reading chunks: {len(df_arr)}")
df = pd.concat(df_arr)

# -----------------------------
# Create Anchor-Positive Pairs (Within Author)
# -----------------------------
dfcm = df.merge(df, how="outer", on="authorID", suffixes=["_anchor", "_positive"])
print(f"[{time.time() - start_time:.2f}s] Merged Dataframe to generate author anchor-positive pairs")
print(f"[{time.time() - start_time:.2f}s] Now with shape: {dfcm.shape}")

# -----------------------------
# Filter out self-pairs and same-genre pairs
# -----------------------------
dfcm = dfcm.loc[
    (dfcm["documentID_anchor"] != dfcm["documentID_positive"]) &
    (dfcm["doc_xrbmtgc_genre_anchor"] != dfcm["doc_xrbmtgc_genre_positive"])
]
print(f"[{time.time() - start_time:.2f}s] Filtered same anchor-positive pair documents, same genre anchor-positive pair documents")
print(f"[{time.time() - start_time:.2f}s] Now with shape: {dfcm.shape}")

# -----------------------------
# Deduplicate Unordered Pairs per Author
# -----------------------------
doc_min = np.minimum(dfcm["documentID_anchor"], dfcm["documentID_positive"])
doc_max = np.maximum(dfcm["documentID_anchor"], dfcm["documentID_positive"])
dfcm.loc[:, "pair_key"] = (
    dfcm["authorID"].astype(str) + "__" +
    doc_min.astype(str) + "__" +
    doc_max.astype(str)
)
dfcm = dfcm.drop_duplicates(subset="pair_key").drop(columns="pair_key")
print(f"[{time.time() - start_time:.2f}s] Dedup anchor-positive pairs, now with shape: {dfcm.shape}")

# -----------------------------
# Compute Cosine Similarity (Vectorized)
# -----------------------------
def batched_cosine_similarity(a_embeddings, b_embeddings, batch_size=10000):
    scores = []
    for i in tqdm(range(0, len(a_embeddings), batch_size), desc="Cosine Similarity"):
        a_batch = a_embeddings[i:i+batch_size]
        b_batch = b_embeddings[i:i+batch_size]
        sim = np.sum(a_batch * b_batch, axis=1) / (
            np.linalg.norm(a_batch, axis=1) * np.linalg.norm(b_batch, axis=1)
        )
        scores.extend(sim)
    return np.array(scores)


anchor_embeddings = np.stack(dfcm["doc_luarmud_embedding_anchor"].values)
positive_embeddings = np.stack(dfcm["doc_luarmud_embedding_positive"].values)
print(f"[{time.time() - start_time:.2f}s] Generated pairwise similarity of anchor-positive pairs from their respective embeddings")

dfcm["similarity_score"] = batched_cosine_similarity(anchor_embeddings, positive_embeddings, batch_size=10000)
print(f"[{time.time() - start_time:.2f}s] Now with shape: {dfcm.shape}")

# -----------------------------
# Filter Low-Similarity Pairs
# -----------------------------
initial_count = dfcm.shape[0]
dfcm = dfcm.loc[dfcm["similarity_score"] < ceiling_threshold]
dfcm = dfcm.reset_index(drop=True)
print(f"[{time.time() - start_time:.2f}s] Hard positive filtering of considering anchor-positive pairs below threshold of: {ceiling_threshold}")
print(f"[{time.time() - start_time:.2f}s] Dropped {initial_count - dfcm.shape[0]} pairs above similarity threshold")
print(f"[{time.time() - start_time:.2f}s] Now with shape: {dfcm.shape}")

# -----------------------------
# Sample One Positive per Anchor per Author
# -----------------------------
fin_ans = dfcm.groupby(["authorID", "documentID_anchor"], group_keys=False).sample(n=1, random_state=42)
print(f"[{time.time() - start_time:.2f}s] Grouped author, anchors and sampled candidates to generate auth-anchor-positive pairs")

# -----------------------------
# Final Output
# -----------------------------
print(f"[{time.time() - start_time:.2f}s] Final sampled result shape: {fin_ans.shape}")
# Save or return as needed
fin_ans.to_json("final_pairs.jsonl", orient="records", lines=True, force_ascii=False)