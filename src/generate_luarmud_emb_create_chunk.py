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
"""
CUDA_VISIBLE_DEVICES=0 python generate_luarmud_emb_create_chunk.py --start_idx 0 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=0 python generate_luarmud_emb_create_chunk.py --start_idx 200000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=0 python generate_luarmud_emb_create_chunk.py --start_idx 400000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=0 python generate_luarmud_emb_create_chunk.py --start_idx 600000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=0 python generate_luarmud_emb_create_chunk.py --start_idx 800000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=0 python generate_luarmud_emb_create_chunk.py --start_idx 1000000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=1 python generate_luarmud_emb_create_chunk.py --start_idx 1200000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=1 python generate_luarmud_emb_create_chunk.py --start_idx 1400000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=1 python generate_luarmud_emb_create_chunk.py --start_idx 1600000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=1 python generate_luarmud_emb_create_chunk.py --start_idx 1800000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=1 python generate_luarmud_emb_create_chunk.py --start_idx 2000000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=1 python generate_luarmud_emb_create_chunk.py --start_idx 2200000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=2 python generate_luarmud_emb_create_chunk.py --start_idx 2400000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=2 python generate_luarmud_emb_create_chunk.py --start_idx 2600000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=2 python generate_luarmud_emb_create_chunk.py --start_idx 2800000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=2 python generate_luarmud_emb_create_chunk.py --start_idx 3000000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=2 python generate_luarmud_emb_create_chunk.py --start_idx 3200000 --process_numrows 200000;
CUDA_VISIBLE_DEVICES=2 python generate_luarmud_emb_create_chunk.py --start_idx 3400000 --process_numrows 200000;

Once done, run this file:
/data/araghavan/HIATUS/datadreamer-ta2/src/hard_batching_rankingloss_optimized_chunks.py
"""
from argparse import ArgumentParser
parser = ArgumentParser(description="generate and classify sadiri dataset")
parser.add_argument("--input_file", type=str, default="/data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/train_sadiri_processed_with_embeddings_wo_ao3_filtered.jsonl", help="Input JSONL file path")
parser.add_argument("--start_idx", type=int, required=True, help="Start Index to read")
parser.add_argument("--output_file", type=str, default="/data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/trainsadiri_luarmud_chunks/train_sadiri_processed_with_luarembeddings_wo_ao3_filtered.jsonl", help="Output JSONL file path")
parser.add_argument("--dataset_numrows", type=int, default=3522640, help="Number of Rows in the dataset")
parser.add_argument("--process_numrows", type=int, default=100000, help="Number of Rows in the dataset")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch Size to process for embedding generation")

args = parser.parse_args()
print(vars(args))
# -----------------------------
# Configs
# -----------------------------
start_time = time.time()
embed_model_name = "gabrielloiseau/LUAR-MUD-sentence-transformers"
doc_genre_model_name = "classla/xlm-roberta-base-multilingual-text-genre-classifier"
chunk_suffix = f"_{args.start_idx:0{len(str(args.dataset_numrows))}d}_{args.start_idx+args.process_numrows:0{len(str(args.dataset_numrows))}d}.jsonl"
intermediate_chunk_data_path = args.input_file.replace(".jsonl",chunk_suffix)
output_chunk_data_path = args.output_file.replace(".jsonl", chunk_suffix)
batch_size = args.batch_size

print(f"[{time.time() - start_time:.2f}s] Checking file path: {os.path.isfile(args.input_file)}")

# -----------------------------
# Load Models
# -----------------------------
embed_model = SentenceTransformer(embed_model_name, device="cuda")

# Load model/tokenizer manually
genre_tokenizer = AutoTokenizer.from_pretrained(doc_genre_model_name)
genre_model = AutoModelForSequenceClassification.from_pretrained(doc_genre_model_name).cuda()

# doc_genre_classifier = pipeline("text-classification", model=doc_genre_model_name, device="cuda")
print(f"[{time.time() - start_time:.2f}s] Loaded embedding and genre classification models")


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

import subprocess

# -----------------------------
# Create Range Chunk Data Using `sed`
# -----------------------------
start_line = args.start_idx + 1  # sed is 1-indexed
end_line = args.start_idx + args.process_numrows

sed_command = f"sed -n '{start_line},{end_line}p' {args.input_file} > {intermediate_chunk_data_path}"
print(f"[{time.time() - start_time:.2f}s] Running: {sed_command}")
subprocess.run(sed_command, shell=True, check=True)
print(f"Created Intermediate Chunk Data File: {intermediate_chunk_data_path}")
print(f"Exist Check: {os.path.exists(intermediate_chunk_data_path)}")
if not os.path.exists(intermediate_chunk_data_path):
    raise ValueError("Intermediate File Creation Failure")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_json(intermediate_chunk_data_path, lines=True)
print(f"[{time.time() - start_time:.2f}s] Read JSON: {intermediate_chunk_data_path} with shape: {df.shape}")

df.drop(columns=['embeddings'], inplace=True)
# -----------------------------
# Apply Batched Embedding + Genre Inference
# -----------------------------
texts = df["fullText"].tolist()
# embeddings, genres = batched_process(texts, embed_model, doc_genre_classifier, batch_size=batch_size)
embeddings, genres = batched_process(texts, embed_model, batch_size=batch_size)
print(f"[{time.time() - start_time:.2f}s] Generated embeddings and genre for all docs with shape: {embeddings.shape}")

df["doc_luarmud_embedding"] = list(embeddings)
df["doc_xrbmtgc_genre"] = genres

df.to_json(output_chunk_data_path, orient='records', lines=True)
print(f"Wrote dataframe: {df.shape} to path: {output_chunk_data_path}")

# -----------------------------
# Cleanup Intermediate Chunk File
# -----------------------------
if os.path.exists(intermediate_chunk_data_path):
    os.remove(intermediate_chunk_data_path)
    print(f"Deleted temporary chunk file: {intermediate_chunk_data_path}")
print(f"[{time.time() - start_time:.2f}s] Finished and cleaned up.")
