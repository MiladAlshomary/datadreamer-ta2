'''
IDEA: break all possible positive pairs into 4 groups:
    - low style cos sim & different genre
    - low style cos sim only
    - different genre only
    - the rest

PSEUDOCODE:

function get_positives_nested(data_by_author, semantic_threshold):
    initialize empty dictionary "positives"
    for each author in data_by_author:
        if the number of documents < 2:
            continue to next author
        extract embeddings, genres, and document IDs from the DataFrame
        compute cosine similarity matrix for embeddings
        get non-redundant document pairs (upper triangle of matrix)

        initialize positives[author] as a dictionary with keys:
            "group1", "group2", "group3", "group4"

        for each pair (i, j):
            calculate cosine similarity 'sim' for the pair
            check if genres differ (diff_genre)
            create a pair_info dictionary with doc1, doc2, similarity, genre1, genre2

            if sim < semantic_threshold:
                if diff_genre is True:
                    add pair_info to positives[author]["group1"]
                else:
                    add pair_info to positives[author]["group2"]
            else:  # sim >= semantic_threshold
                if diff_genre is True:
                    add pair_info to positives[author]["group3"]
                else:
                    add pair_info to positives[author]["group4"]

    return positives

NOTE: Actual code compute groups for each author in parrallel
'''


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor
import json
from tqdm import tqdm


def process_author(args):
    """
    Process a single author: compute cosine similarities among all documents,
    then group positive pairs into four categories.

    Parameters:
        args (tuple): Contains (author, df, semantic_threshold)

    Returns:
        (author, groups): where groups is a dictionary with keys "group1", "group2", "group3", "group4".
    """
    author, df, semantic_threshold = args
    if len(df) < 2:
        return author, None

    # Convert embeddings to a 2D numpy array. Each embedding should be a list or array.
    embeddings = np.vstack(df['doc_sbertamllv2_embedding'].values)
    genres = df['doc_xrbmtgc_genre'].values
    doc_ids = df['documentID'].values
    n_docs = len(df)

    # Compute the cosine similarity matrix for all documents of this author.
    sim_matrix = cosine_similarity(embeddings)

    # Get the indices for the upper triangle (non-redundant pairs)
    i_idx, j_idx = np.triu_indices(n_docs, k=1)

    # Initialize groups for this author.
    groups = {
        "group1": [],  # low semantic (< threshold) and different genre
        "group2": [],  # low semantic (< threshold) and same genre
        "group3": [],  # high semantic (>= threshold) and different genre
        "group4": []  # high semantic (>= threshold) and same genre
    }

    # Iterate over each pair (i, j)
    for i, j in zip(i_idx, j_idx):
        sim = sim_matrix[i, j]
        diff_genre = (genres[i] != genres[j])
        pair_info = {
            "doc1": doc_ids[i],
            "doc2": doc_ids[j],
            "similarity": sim,
            "genre1": genres[i],
            "genre2": genres[j]
        }

        if sim < semantic_threshold:
            if diff_genre:
                groups["group1"].append(pair_info)
            else:
                groups["group2"].append(pair_info)
        else:  # sim >= semantic_threshold
            if diff_genre:
                groups["group3"].append(pair_info)
            else:
                groups["group4"].append(pair_info)

    return author, groups


def get_positives_nested_parallel(data_by_author, semantic_threshold=0.2, max_workers=None):
    """
    Process each author's data in parallel, returning a nested dictionary where each key is an author,
    and the value is a dictionary of four groups of positive pairs.

    Parameters:
        data_by_author (dict): Mapping from authorID to a DataFrame of documents.
        semantic_threshold (float): Cosine similarity threshold.
        max_workers (int, optional): Number of worker processes; if None, defaults to the number of cores.

    Returns:
        dict: Nested dictionary of the form:
              { author: {'group1': [...], 'group2': [...], 'group3': [...], 'group4': [...]}, ... }
    """
    positives = {}
    args_list = [(author, df, semantic_threshold) for author, df in data_by_author.items()]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Wrap the executor.map iterator with tqdm for a progress bar
        for author, groups in tqdm(executor.map(process_author, args_list), total=len(args_list),
                                   desc="Processing Authors"):
            if groups is not None:
                positives[author] = groups

    return positives


# group documents by authorID:
file_path = "/mnt/nlpgpu-io1/data/jiachzhu/projects/data/train_sadiri_processed_with_luarsbertembeddings_wo_ao3_filtered.jsonl"
file_df = pd.read_json(file_path)
data_by_author = {author: group for author, group in file_df.groupby('authorID')}

# Process in parallel using a semantic threshold (default is 0.2)
positives_nested = get_positives_nested_parallel(data_by_author, semantic_threshold=0.2, max_workers=10)

# save the resulting nested dictionary to a JSON file
with open("../output/positives_nested.json", "w") as f:
    json.dump(positives_nested, f, indent=2)
