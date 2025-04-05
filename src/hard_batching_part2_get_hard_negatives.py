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
import faiss
import json
from tqdm import tqdm


def hard_negative_batching(positive_nested, file_df, batch_size):
    """
    Create batches for hard negative ranking, while tracking progress of pair removal.

    Parameters:
        positive_nested (dict): Nested dictionary mapping each author to a dictionary
                                of positive pair groups (group1, group2, group3, group4).
        file_df (pd.DataFrame): Original DataFrame containing at least:
                                'documentID', 'doc_sbertamllv2_embedding', and 'fullText'.
        batch_size (int): Number of pairs to include in each batch. Each batch will have
                          one seed pair plus (batch_size - 1) negatives.

    Returns:
        List of batches. Each batch is a list of pair dictionaries.
    """
    # Build a mapping from documentID to its normalized embedding
    doc_embedding_dict = {}
    for idx, row in file_df.iterrows():
        doc_id = row['documentID']
        emb = np.array(row['doc_luarmud_embedding'], dtype='float32')
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        doc_embedding_dict[doc_id] = emb

    # Build a mapping from documentID to full text
    doc_text_dict = {}
    for idx, row in file_df.iterrows():
        doc_id = row['documentID']
        doc_text_dict[doc_id] = row['fullText']

    # Compute total number of positive pairs initially for progress bar
    total_pairs = 0
    for author, groups in positive_nested.items():
        for group in ['group1', 'group2', 'group3', 'group4']:
            total_pairs += len(groups.get(group, []))

    pbar = tqdm(total=total_pairs, desc="Processing Pairs")

    batches = []

    # Continue until positive_nested is empty
    while positive_nested:
        # Get available authors (those that still have at least one pair)
        available_authors = list(positive_nested.keys())
        if not available_authors:
            break

        # Randomly select a seed author
        seed_author = random.choice(available_authors)
        seed_pair = None
        # Pick one positive pair from seed_author from any non-empty group
        for group in ['group1', 'group2', 'group3', 'group4']:
            if positive_nested[seed_author].get(group):
                seed_pair = random.choice(positive_nested[seed_author][group])
                break
        if seed_pair is None:
            del positive_nested[seed_author]
            continue

        # Get seed anchor's document ID and embedding (we use 'doc1' as anchor)
        seed_doc_id = seed_pair['doc1']
        seed_embedding = doc_embedding_dict.get(seed_doc_id)
        if seed_embedding is None:
            # Remove seed_pair if its embedding cannot be found and update progress bar
            for group in ['group1', 'group2', 'group3', 'group4']:
                if seed_pair in positive_nested[seed_author].get(group, []):
                    positive_nested[seed_author][group].remove(seed_pair)
                    pbar.update(1)
                    break
            if all(len(positive_nested[seed_author].get(g, [])) == 0 for g in ['group1', 'group2', 'group3', 'group4']):
                del positive_nested[seed_author]
            continue

        # Define corpus authors (all authors except the seed author)
        corpus_authors = [a for a in available_authors if a != seed_author]
        candidate_pairs = []
        candidate_authors = []
        # For each corpus author, pick one candidate pair (from any non-empty group)
        for author in corpus_authors:
            candidate_pair = None
            for group in ['group1', 'group2', 'group3', 'group4']:
                if positive_nested[author].get(group):
                    candidate_pair = random.choice(positive_nested[author][group])
                    break
            if candidate_pair:
                candidate_pairs.append(candidate_pair)
                candidate_authors.append(author)

        # Build candidate embeddings from the candidate pairs (using their doc1 anchors)
        candidate_embeddings = []
        for pair in candidate_pairs:
            doc_id = pair['doc1']
            emb = doc_embedding_dict.get(doc_id)
            if emb is not None:
                candidate_embeddings.append(emb)
            else:
                candidate_embeddings.append(np.zeros_like(seed_embedding))
        # If no candidates, form batch with only the seed pair.
        if len(candidate_embeddings) == 0:
            batch = [seed_pair]
        else:
            candidate_embeddings = np.stack(candidate_embeddings, axis=0)
            # Build FAISS index on candidate embeddings (using inner product as cosine similarity)
            d = candidate_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(candidate_embeddings)
            top_k = batch_size - 1  # we want batch_size pairs including the seed
            seed_embedding_np = np.expand_dims(seed_embedding, axis=0)
            D, I = index.search(seed_embedding_np, top_k)
            selected_pairs = []
            for idx in I[0]:
                if idx < len(candidate_pairs):
                    selected_pairs.append(candidate_pairs[idx])
            batch = [seed_pair] + selected_pairs

        batches.append(batch)

        # Count the number of pairs removed in this iteration (seed pair plus candidate pairs)
        pairs_removed = len(batch)

        # Remove used pairs from positive_nested:
        # Remove seed_pair from seed_author
        for group in ['group1', 'group2', 'group3', 'group4']:
            if seed_pair in positive_nested[seed_author].get(group, []):
                positive_nested[seed_author][group].remove(seed_pair)
                break
        # Remove each candidate pair used from its corresponding author
        for candidate_pair in batch[1:]:
            candidate_author = None
            for author in corpus_authors:
                for group in ['group1', 'group2', 'group3', 'group4']:
                    if candidate_pair in positive_nested.get(author, {}).get(group, []):
                        candidate_author = author
                        positive_nested[author][group].remove(candidate_pair)
                        break
                if candidate_author:
                    break

        # Update the progress bar by the number of pairs removed in this iteration
        pbar.update(pairs_removed)

        # Remove any authors that have no remaining pairs in any group
        authors_to_remove = []
        for author in list(positive_nested.keys()):
            if all(len(positive_nested[author].get(g, [])) == 0 for g in ['group1', 'group2', 'group3', 'group4']):
                authors_to_remove.append(author)
        for author in authors_to_remove:
            del positive_nested[author]

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
