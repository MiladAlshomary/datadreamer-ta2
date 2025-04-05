'''
PSEUDOCODE:

Function hard_negative_batching(positive_nested, file_df, batch_size):
    Build doc_embedding_dict mapping documentID to normalized embedding
    Build doc_text_dict mapping documentID to fullText  <-- NEW

    Initialize empty list batches

    WHILE positive_nested is not empty:
        available_authors = list of keys in positive_nested
        IF available_authors is empty, BREAK

        seed_author = randomly choose an author from available_authors
        seed_pair = randomly choose one positive pair from seed_author (from any non-empty group)
        IF no seed_pair found, remove seed_author and CONTINUE

        seed_doc_id = seed_pair['doc1']
        seed_embedding = doc_embedding_dict[seed_doc_id]
        IF seed_embedding not found, remove seed_pair and CONTINUE

        corpus_authors = available_authors excluding seed_author
        FOR each author in corpus_authors:
            candidate_pair = randomly choose one positive pair from that author (from any non-empty group)
            IF candidate_pair exists, add to candidate_pairs list

        IF candidate_pairs is empty:
            batch = [seed_pair]
        ELSE:
            Build candidate_embeddings from candidate_pairs using doc_embedding_dict
            Build FAISS index on candidate_embeddings
            Query FAISS with seed_embedding to get top (batch_size - 1) candidate indices
            selected_pairs = candidate_pairs corresponding to top indices
            batch = [seed_pair] + selected_pairs

        Append batch to batches

        Remove seed_pair and selected candidate_pairs from positive_nested
        Remove any authors with no remaining pairs from positive_nested

    RETURN batches
'''

import numpy as np
import pandas as pd
import random
import json
from tqdm import tqdm
from sentence_transformers.quantization import semantic_search_faiss


def hard_negative_batching(positive_nested, file_df, batch_size):
    """
    Create batches for hard negative ranking, while tracking progress of pair removal,
    and using sentence_transformers' semantic_search_faiss for candidate selection.

    This version combines the construction of candidate cache, reverse lookup, and total pair count.

    Parameters:
        positive_nested (dict): Nested dictionary mapping each author to a dictionary
                                of positive pair groups (group1, group2, group3, group4).
        file_df (pd.DataFrame): Original DataFrame containing at least:
                                'documentID', 'doc_luarmud_embedding', and 'fullText'.
        batch_size (int): Number of pairs to include in each batch. Each batch will have
                          one seed pair plus (batch_size - 1) negatives.

    Returns:
        List of batches. Each batch is a list of pair dictionaries.
    """
    # Build mappings from documentID to its style embedding and full text.
    doc_embedding_dict = {}
    doc_text_dict = {}
    for idx, row in file_df.iterrows():
        doc_id = row['documentID']
        emb = np.array(row['doc_luarmud_embedding'], dtype='float32')
        doc_embedding_dict[doc_id] = emb
        doc_text_dict[doc_id] = row['fullText']
    print("Completed: Mapped doc ID to style embedding / full text.")

    # Build candidate cache, reverse lookup, and count total pairs
    candidate_cache = {}
    reverse_lookup = {}
    total_pairs = 0
    for author, groups in positive_nested.items():
        candidate_cache[author] = []
        for group in ['group1', 'group2', 'group3', 'group4']:
            pairs = groups.get(group, [])
            total_pairs += len(pairs)
            for pair in pairs:
                candidate_cache[author].append((pair, group))
                key = (pair['doc1'], pair['doc2'])
                reverse_lookup[key] = (author, group)
    print("Completed: Candidate cache, reverse lookup, and total pair count built.")

    pbar = tqdm(total=total_pairs, desc="Processing Pairs")
    print("Completed: Progress bar initialized.")

    batches = []

    # Continue until positive_nested is empty.
    while positive_nested:
        available_authors = list(positive_nested.keys())
        if not available_authors:
            break

        # Randomly select a seed author.
        seed_author = random.choice(available_authors)
        seed_tuple = random.choice(candidate_cache[seed_author])
        candidate_cache[seed_author].remove(seed_tuple)
        seed_pair = seed_tuple[0]
        seed_doc_id = seed_pair['doc1']
        seed_embedding = doc_embedding_dict.get(seed_doc_id)

        corpus_authors = [a for a in available_authors if a != seed_author]
        candidate_pairs = []
        for author in corpus_authors:
            if candidate_cache.get(author) and len(candidate_cache[author]) > 0:
                candidate_tuple = random.choice(candidate_cache[author])
                candidate_pair, group = candidate_tuple
                candidate_cache[author].remove(candidate_tuple)
                candidate_pairs.append(candidate_pair)
        candidate_embeddings = [doc_embedding_dict.get(pair['doc1']) for pair in candidate_pairs]

        if len(candidate_embeddings) == 0:
            batch = [seed_pair]
        else:
            candidate_embeddings = np.stack(candidate_embeddings, axis=0)
            top_k = batch_size - 1
            results, search_time, _ = semantic_search_faiss(
                query_embeddings=np.expand_dims(seed_embedding, axis=0),
                corpus_embeddings=candidate_embeddings,
                top_k=top_k,
                corpus_precision='binary',
                exact=False,
                output_index=False
            )
            selected_pairs = []
            for res in results[0]:
                idx = res["corpus_id"]
                if idx < len(candidate_pairs):
                    selected_pairs.append(candidate_pairs[idx])
            batch = [seed_pair] + selected_pairs

        batches.append(batch)
        pairs_removed = len(batch)

        # Remove used pairs using reverse lookup.
        seed_key = (seed_pair['doc1'], seed_pair['doc2'])
        if seed_key in reverse_lookup:
            author, group = reverse_lookup[seed_key]
            positive_nested[author][group].remove(seed_pair)
            del reverse_lookup[seed_key]
        for candidate_pair in batch[1:]:
            candidate_key = (candidate_pair['doc1'], candidate_pair['doc2'])
            if candidate_key in reverse_lookup:
                author, group = reverse_lookup[candidate_key]
                positive_nested[author][group].remove(candidate_pair)
                del reverse_lookup[candidate_key]

        pbar.update(pairs_removed)
        authors_to_remove = []
        for author in list(positive_nested.keys()):
            if all(len(positive_nested[author].get(g, [])) == 0 for g in ['group1', 'group2', 'group3', 'group4']):
                authors_to_remove.append(author)
        for author in authors_to_remove:
            del positive_nested[author]
            candidate_cache.pop(author, None)

    pbar.close()
    return batches

# Load the original DataFrame from the JSON Lines file.
file_path = "/mnt/nlpgpu-io1/data/jiachzhu/projects/data/train_sadiri_processed_with_luarsbertembeddings_wo_ao3_filtered.jsonl"
file_df = pd.read_json(file_path)

# Load the previously generated positives_nested JSON file.
positive_pairs_path = "../output/positives_nested.json"
with open(positive_pairs_path, "r") as f:
    positives_nested = json.load(f)

# Run the hard negative batching function.
# (Assuming the function 'hard_negative_batching' is defined in your code)
batches = hard_negative_batching(positives_nested, file_df, batch_size=1024)

# Save the resulting batches to a local JSON file.
with open("../output/batches.json", "w") as f:
    json.dump(batches, f, indent=2)
