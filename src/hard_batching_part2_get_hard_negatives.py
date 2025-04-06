'''
BIG IDEA:
- choose a seed pair from one author, then for every other unique author,
  randomly sample one candidate pair, use FAISS semantic search to select the hardest negatives for the batch.
PSEUDOCODE:
1. Build mappings from document IDs to embeddings and full text.
2. For each author and group in positive pairs:
   - Add each pair to a candidate cache.
   - Record a reverse lookup that maps each pair (using doc1 and doc2) to its author and group.
   - Count the total number of pairs.
3. Initialize a progress bar with the total pair count.
4. While there are still positive pairs:
   a. Randomly select a seed author and choose one seed pair from that author’s cache.
   b. Retrieve the seed pair’s embedding.
   c. For all other authors, randomly pick candidate pairs from their caches.
   d. Use semantic_search_faiss (with corpus_precision set to 'binary') to find the top candidate pairs based on the seed embedding.
   e. Form a batch consisting of the seed pair and the selected candidate pairs.
   f. Remove the used pairs from your data structures using the reverse lookup.
   g. Update the progress bar and clean up any authors with no remaining pairs.
5. Close the progress bar and return the batches.
'''

import numpy as np
import pandas as pd
import random
import json
from tqdm import tqdm
import faiss
import sys
from sentence_transformers.quantization import semantic_search_faiss, quantize_embeddings


def hard_negative_batching(positive_nested, file_df, batch_size):
    """
    Optimized batching that precomputes quantized embeddings and builds a global FAISS index
    for candidate selection. Each candidate pair is assigned a unique candidate_id and added
    to the global index so that negatives can be retrieved without rebuilding the index.
    """
    # -----------------------------
    # 1. Build mappings from documentID to its style embedding and full text.
    doc_embedding_dict = {}
    doc_text_dict = {}
    doc_ids = []
    raw_embeddings = []
    for idx, row in file_df.iterrows():
        doc_id = row['documentID']
        emb = np.array(row['doc_luarmud_embedding'], dtype='float32')
        doc_embedding_dict[doc_id] = emb
        doc_text_dict[doc_id] = row['fullText']
        doc_ids.append(doc_id)
        raw_embeddings.append(emb)
    print("Completed: Mapped doc ID to style embedding / full text.")
    sys.stdout.flush()

    # -----------------------------
    # 2. Precompute quantized embeddings for each document once.
    # We stack all embeddings and quantize them in one go.
    raw_embeddings_array = np.stack(raw_embeddings, axis=0)
    # quantize_embeddings expects a batch and returns binary representations.
    quantized_embeddings_array = quantize_embeddings(raw_embeddings_array, precision='ubinary')
    # Create a lookup: doc_id -> precomputed quantized embedding.
    doc_quantized_dict = {doc_id: quantized_embeddings_array[i] for i, doc_id in enumerate(doc_ids)}
    print("Completed: Precomputed quantized embeddings for each document.")
    sys.stdout.flush()

    # -----------------------------
    # 3. Build candidate cache and global candidate lookup.
    # We'll assign a unique candidate_id to each candidate pair.
    candidate_cache = {}  # author -> list of (candidate_id, pair, group)
    global_candidate_lookup = {}  # candidate_id -> (author, pair, group)
    total_pairs = 0
    candidate_id_counter = 0

    for author, groups in positive_nested.items():
        candidate_cache[author] = []
        for group in ['group1', 'group2', 'group3', 'group4']:
            pairs = groups.get(group, [])
            total_pairs += len(pairs)
            for pair in pairs:
                candidate_cache[author].append((candidate_id_counter, pair, group))
                global_candidate_lookup[candidate_id_counter] = (author, pair, group)
                candidate_id_counter += 1
    print("Completed: Candidate cache and global candidate lookup built. Total pairs:", total_pairs)
    sys.stdout.flush()

    # -----------------------------
    # 4. Build a global FAISS binary index for all candidate embeddings.
    # We assume that the embedding for a candidate pair is the quantized embedding
    # of its 'doc1' field.
    candidate_ids = []
    candidate_embeddings_list = []
    for cid, (author, pair, group) in global_candidate_lookup.items():
        doc_id = pair['doc1']
        # It is assumed that every doc_id is in the doc_quantized_dict.
        candidate_emb = doc_quantized_dict.get(doc_id)
        if candidate_emb is not None:
            candidate_ids.append(cid)
            candidate_embeddings_list.append(candidate_emb)
    candidate_embeddings_array = np.stack(candidate_embeddings_list, axis=0)
    # For binary indices, FAISS expects d as the number of bits.
    # If each embedding is a uint8 vector of length L, then d = L * 8.
    L = candidate_embeddings_array.shape[1]
    d = L * 8

    # Create a binary index and wrap it with an IDMap to support removals.
    binary_index = faiss.IndexBinaryFlat(d)
    global_index = faiss.IndexBinaryIDMap(binary_index)
    candidate_ids_np = np.array(candidate_ids, dtype=np.int64)
    global_index.add_with_ids(candidate_embeddings_array, candidate_ids_np)
    print("Completed: Global FAISS index built with", global_index.ntotal, "candidates.")
    sys.stdout.flush()

    # -----------------------------
    # 5. Batching loop using the global index.
    batches = []
    with tqdm(total=total_pairs, desc="Processing Pairs") as pbar:
        while positive_nested:
            available_authors = list(positive_nested.keys())
            if not available_authors:
                break

            # Randomly select a seed author.
            seed_author = random.choice(available_authors)
            if not candidate_cache.get(seed_author):
                # Remove author if no candidates remain.
                positive_nested.pop(seed_author, None)
                candidate_cache.pop(seed_author, None)
                continue

            # Select a random seed candidate for the seed author.
            seed_candidate_entry = random.choice(candidate_cache[seed_author])
            seed_candidate_id, seed_pair, seed_group = seed_candidate_entry
            # Remove the seed candidate from candidate_cache and global index.
            candidate_cache[seed_author].remove(seed_candidate_entry)
            try:
                global_index.remove_ids(np.array([seed_candidate_id], dtype=np.int64))
            except Exception as e:
                # In case removal fails, log or pass.
                pass
            global_candidate_lookup.pop(seed_candidate_id, None)

            # Retrieve the seed embedding (use the original float embedding).
            seed_doc_id = seed_pair['doc1']
            seed_embedding = doc_embedding_dict.get(seed_doc_id)
            if seed_embedding is None:
                # If missing, skip this seed.
                continue

            # -----------------------------
            # Retrieve negative candidates from the global index.
            # We query for more than needed to allow for filtering.
            top_k_query = batch_size * 2
            results, search_time = semantic_search_faiss(
                query_embeddings=np.expand_dims(seed_embedding, axis=0),
                corpus_index=global_index,
                top_k=top_k_query,
                corpus_precision='ubinary',
                exact=False,
                rescore=False,
                output_index=False
            )

            selected_negatives = {}
            for res in results[0]:
                neg_cid = res["corpus_id"]
                # Check if candidate still exists.
                candidate_info = global_candidate_lookup.get(neg_cid)
                if candidate_info is None:
                    continue
                neg_author, neg_pair, neg_group = candidate_info
                # Exclude candidates from the seed author.
                if neg_author == seed_author:
                    continue
                # Ensure unique author per batch.
                if neg_author in selected_negatives:
                    continue
                # Accept this candidate.
                selected_negatives[neg_author] = (neg_cid, neg_pair, neg_group)
                # Remove the candidate from its author cache if present.
                if neg_author in candidate_cache:
                    candidate_cache[neg_author] = [
                        entry for entry in candidate_cache[neg_author] if entry[0] != neg_cid
                    ]
                try:
                    global_index.remove_ids(np.array([neg_cid], dtype=np.int64))
                except Exception as e:
                    pass
                global_candidate_lookup.pop(neg_cid, None)
                # Break if we have enough negatives.
                if len(selected_negatives) >= (batch_size - 1):
                    break

            # Form the batch: seed candidate plus selected negatives.
            batch = [seed_pair] + [info[1] for info in selected_negatives.values()]
            batches.append(batch)

            # -----------------------------
            # Remove used pairs from positive_nested.
            # For the seed candidate.
            seed_key = (seed_pair['doc1'], seed_pair['doc2'])
            if seed_author in positive_nested and seed_pair in positive_nested[seed_author].get(seed_group, []):
                positive_nested[seed_author][seed_group].remove(seed_pair)
            # For negatives.
            for neg in selected_negatives.values():
                neg_cid, neg_pair, neg_group = neg
                neg_author = global_candidate_lookup.get(neg_cid, (None,))[0]
                # Since we already removed from global_candidate_lookup,
                # use candidate_cache keys (which are authors).
                # We assume that neg_pair is in positive_nested.
                if neg_author and neg_author in positive_nested:
                    if neg_pair in positive_nested[neg_author].get(neg_group, []):
                        positive_nested[neg_author][neg_group].remove(neg_pair)

            pairs_removed = 1 + len(selected_negatives)
            pbar.update(pairs_removed)

            # Clean up authors with no remaining pairs.
            authors_to_remove = []
            for author in list(positive_nested.keys()):
                groups = positive_nested[author]
                if all(len(groups.get(g, [])) == 0 for g in ['group1', 'group2', 'group3', 'group4']):
                    authors_to_remove.append(author)
            for author in authors_to_remove:
                positive_nested.pop(author, None)
                candidate_cache.pop(author, None)

    print("Batching Completed.")
    sys.stdout.flush()
    return batches

# Load the original DataFrame from the JSON Lines file.
file_path = "/mnt/nlpgpu-io1/data/jiachzhu/projects/data/train_sadiri_processed_with_luarsbertembeddings_wo_ao3_filtered.jsonl"
print("Reading training dataset.")
sys.stdout.flush()
file_df = pd.read_json(file_path)
print("Training data loaded.")
sys.stdout.flush()

# Load the previously generated positives_nested JSON file.
print("Reading hard positive assignments.")
sys.stdout.flush()
positive_pairs_path = "../output/positives_nested.json"
with open(positive_pairs_path, "r") as f:
    positives_nested = json.load(f)
print("Hard positives loaded.")
sys.stdout.flush()

# Run the hard negative batching function.
batches = hard_negative_batching(positives_nested, file_df, batch_size=1024)

print("Dumping batches to storage.")
sys.stdout.flush()
# Save the resulting batches to a local JSON file.
with open("../output/batches.json", "w") as f:
    json.dump(batches, f, indent=2)
