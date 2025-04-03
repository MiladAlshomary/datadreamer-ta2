import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Configs
# -----------------------------
start_time = time.time()
embed_model_name = "gabrielloiseau/LUAR-MUD-sentence-transformers"
doc_genre_model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"
data_path = "/data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/train_sadiri_processed_with_embeddings_wo_ao3_filtered.jsonl"
batch_size = 1024
ceiling_threshold = 0.4

print(f"[{time.time() - start_time:.2f}s] Checking file path: {os.path.isfile(data_path)}")

# -----------------------------
# Load Models
# -----------------------------
embed_model = SentenceTransformer(embed_model_name, device="cuda")

# Load model/tokenizer manually
genre_tokenizer = AutoTokenizer.from_pretrained(doc_genre_model_name)
genre_model = AutoModelForSequenceClassification.from_pretrained(doc_genre_model_name).cuda()

# doc_genre_classifier = pipeline("text-classification", model=doc_genre_model_name, device="cuda")
print(f"[{time.time() - start_time:.2f}s] Loaded embedding and genre classification models")

# -----------------------------
# Batched Processing Function
# -----------------------------
# def batched_process(texts, embed_model, doc_genre_classifier, batch_size=64):
#     embeddings = []
#     genres = []

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i + batch_size]

#         # Embedding generation
#         batch_embeds = embed_model.encode(batch, convert_to_numpy=True, batch_size=batch_size, device="cuda")
#         batch_embeds = np.array(batch_embeds, copy=True)  # Ensures data is detached from GPU
#         embeddings.extend(batch_embeds)

#         # Genre classification
#         batch_genres = doc_genre_classifier(batch, truncation=True, batch_size=batch_size)
#         genres.extend([g["label"] for g in batch_genres])
#         torch.cuda.empty_cache()  # optional: clears memory right after each batch


#     return np.array(embeddings), genres

def classify_genre(batch):
    inputs = genre_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = genre_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    labels = [genre_model.config.id2label[p] for p in preds]
    return labels

# def batched_process(texts, embed_model, doc_genre_classifier, batch_size=64):
def batched_process(texts, embed_model, batch_size=64):
    embeddings = []
    genres = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Genre Batches and Embeddings", unit=" batch"):
    # for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Embedding generation (CPU move)
        batch_embeds = embed_model.encode(batch, convert_to_tensor=True, batch_size=batch_size)
        batch_embeds = batch_embeds.cpu().numpy()
        embeddings.extend(batch_embeds)

        # # Genre classification (stay in CPU memory)
        # batch_genres = doc_genre_classifier(batch, truncation=True, batch_size=batch_size)
        # genres.extend([g["label"] for g in batch_genres])
        batch_genres = classify_genre(batch)
        genres.extend(batch_genres)
        
        if (i // batch_size) % 10 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return np.array(embeddings), genres


# -----------------------------
# Load Data
# -----------------------------
df = pd.read_json(data_path, lines=True)#, nrows=100000)
print(f"[{time.time() - start_time:.2f}s] Read JSON: {data_path} with shape: {df.shape}")

# -----------------------------
# Apply Batched Embedding + Genre Inference
# -----------------------------
texts = df["fullText"].tolist()
# embeddings, genres = batched_process(texts, embed_model, doc_genre_classifier, batch_size=batch_size)
embeddings, genres = batched_process(texts, embed_model, batch_size=batch_size)
print(f"[{time.time() - start_time:.2f}s] Generated embeddings and genre for all docs with shape: {embeddings.shape}")

df["doc_embedding"] = list(embeddings)
df["doc_genre"] = genres

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
    (dfcm["doc_genre_anchor"] != dfcm["doc_genre_positive"])
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
anchor_embeddings = np.stack(dfcm["doc_embedding_anchor"].values)
positive_embeddings = np.stack(dfcm["doc_embedding_positive"].values)
print(f"[{time.time() - start_time:.2f}s] Generated pairwise similarity of anchor-positive pairs from their respective embeddings")

dfcm["similarity_score"] = np.diag(cosine_similarity(anchor_embeddings, positive_embeddings))
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
