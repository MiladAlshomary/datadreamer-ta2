from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

start_time = time.time()
sbert_model = "sentence-transformers/all-MiniLM-L6-v2"
luarmud_model = "gabrielloiseau/LUAR-MUD-sentence-transformers"
doc_genre_model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"
batch_size = 512

sbert_embed_model = SentenceTransformer(sbert_model, device="cuda")
luarmud_embed_model = SentenceTransformer(luarmud_model, device="cuda")

# Load model/tokenizer manually
genre_tokenizer = AutoTokenizer.from_pretrained(doc_genre_model_name)
genre_embed_model = AutoModelForSequenceClassification.from_pretrained(doc_genre_model_name).cuda()

def classify_genre(batch, genre_embed_model, genre_tokenizer):
    inputs = genre_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = genre_embed_model(**inputs)
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    labels = [genre_embed_model.config.id2label[p] for p in preds]
    return labels


# def batched_process(texts, embed_model, doc_genre_classifier, batch_size=64):
def batched_process(texts, luarmud_embed_model, sbert_embed_model, genre_embed_model, genre_tokenizer, batch_size=64):
    embeddings_luar = []
    embeddings_sbert = []
    genres = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Genre Batches and Embeddings", unit=" batch"):
    # for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # LUAR Embedding generation (CPU move)
        batch_luar_embeds = luarmud_embed_model.encode(batch, convert_to_tensor=True, batch_size=batch_size)
        batch_luar_embeds = batch_luar_embeds.cpu().numpy()
        embeddings_luar.extend(batch_luar_embeds)

        # SBERT Embedding generation (CPU move)
        batch_sbert_embeds = sbert_embed_model.encode(batch, convert_to_tensor=True, batch_size=batch_size)
        batch_sbert_embeds = batch_sbert_embeds.cpu().numpy()
        embeddings_sbert.extend(batch_sbert_embeds)

        # # Genre classification (stay in CPU memory)
        batch_genres = classify_genre(batch, genre_embed_model, genre_tokenizer)
        genres.extend(batch_genres)
        
        if (i // batch_size) % 10 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return np.array(embeddings_luar), np.array(embeddings_sbert), genres


# -----------------------------
# Load Data
# -----------------------------
biber_data_path = "../data/biber/genre_data/train_documents.jsonl"
biber_genre_path = "../data/biber/genre_data/final_train.csv"
biber_data_df = pd.read_json(biber_data_path, lines=True)
biber_genre_df = pd.read_csv(biber_genre_path)
biber_data_df.drop_duplicates(subset=['documentID'], inplace=True)
biber_genre_df.drop_duplicates(subset=['documentID'], inplace=True)
biber_data_genre_df = biber_data_df.merge(biber_genre_df[['documentID', 'predicted_genre']], how="inner", left_on="documentID", right_on="documentID")

df = biber_data_genre_df.copy()
print(f"[{time.time() - start_time:.2f}s] Merged DF's. now with shape: {df.shape}")

# -----------------------------
# Apply Batched Embedding + Genre Inference
# -----------------------------
texts = df["fullText"].tolist()
embeddings_luar, embeddings_sbert, genres = batched_process(
    texts, luarmud_embed_model, sbert_embed_model,
    genre_embed_model, genre_tokenizer, batch_size=batch_size
)
print(f"[{time.time() - start_time:.2f}s] Generated LUAR embeddings and genre for all docs with shape: {embeddings_luar.shape}")
print(f"[{time.time() - start_time:.2f}s] Generated SBERT embeddings and genre for all docs with shape: {embeddings_sbert.shape}")

df["doc_luarmud_embedding"] = list(embeddings_luar)
df["doc_sbertamllv2_embedding"] = list(embeddings_sbert)
df["doc_xrbmtgc_genre"] = genres

output_with_luar_sbert_genre_embeddings_path = "/data/araghavan/HIATUS/datadreamer-ta2/data/biber/genre_data/biber_train_luar_sbert_genre_embeddings_v001.jsonl"
df.to_json(output_with_luar_sbert_genre_embeddings_path, orient='records', lines=True)
print(f"Wrote dataframe: {df.shape} to path: {output_with_luar_sbert_genre_embeddings_path}")