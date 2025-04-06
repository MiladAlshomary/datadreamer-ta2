import os
import sys
import glob
import ast
import json
import jsonlines
import numpy as np
from transformers import TrainerCallback
from itertools import islice

# Add the parent directory of 'src' (which is the root directory) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_source')))
epoch_tracker = {}
epoch_tracker['epoch'] = 0
class EpochTrackerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_tracker['epoch'] = int(state.epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_tracker['epoch'] = int(state.epoch)


from collections import defaultdict
from datadreamer import DataDreamer
from datadreamer.steps import DataSource
from transformers import TrainerCallback
from sentence_transformers import losses
from peft import LoraConfig
from random import Random

from training_source.luar_utils import get_luar_trainer

def get_performs_data_generator(path, split):
    train_folder_path = path.replace('{split}', split)

    # Create variables for train dataset
    author_documents = defaultdict(list) # Maps from an author's true ID to a list of all their documents
    author_query_documents = defaultdict(list) # Maps from an author's true ID to a list of their query documents
    author_candidate_documents = defaultdict(list) # Maps from an author's true ID to a list of their candidate documents
    query_author_ids = set()
    candidate_author_ids = set()
    

    
    # Read query documents
    #'/burg/dsi/users/ma4608/hiatus_performers_data/sadiri/*/{split}
    for json_file in glob.glob(train_folder_path + '_queries_filtered.jsonl'):
        print(json_file)
        for line in jsonlines.open(json_file):
            author_id = line['authorIDs'][0]
            query_author_ids.add(author_id)
            author_documents[author_id].append(line['fullText'])
            author_query_documents[author_id].append(line['fullText'])

    print(f"Check Here: {train_folder_path}")
    

    # Read candidate documents
    for json_file in glob.glob(train_folder_path + '_candidates_filtered.jsonl'):
        print(json_file)
        for line in jsonlines.open(json_file):
            author_id = line['authorIDs'][0]
            candidate_author_ids.add(author_id)
            author_documents[author_id].append(line['fullText'])
            author_candidate_documents[author_id].append(line['fullText'])
    
    # Convert to lists
    query_author_ids = list(sorted(query_author_ids))
    candidate_author_ids = list(sorted(candidate_author_ids))

    author_documents = {x[0]: x[1] for x in author_documents.items() if len(x[1]) > 1}
    print("Dataset Statistics:", split, len(author_documents))

    
    # Get total number of rows per epoch
    # total_num_rows = len(author_documents)
    total_num_rows = min(len(author_documents), 1000)  # Cap at 1000 rows

    # Create data generator
    def data_generator():
        # If on the train split, do a new seed each epoch
        seed = epoch_tracker['epoch'] if split == 'train' else 0
        rand = Random(seed)
        
        # Create examples of anchors and positives from query-candidate pairs
        rows = []
        for author_id in author_documents:
            if len(author_documents[author_id]) >1:
                rows.append({
                    "anchors": author_documents[author_id][0],
                    "positives": author_documents[author_id][1]
                })
            # else:
            #     print(author_id, ' has ', len(author_documents[author_id]))

        rows = rows[:1000]  # Cap the rows at 1000
        # Assert we have the right number of rows
        assert len(rows) == total_num_rows

        # Shuffle and yield the rows
        rand.shuffle(rows)
        for row in rows:
            yield row

    return total_num_rows, data_generator


def hard_batch_creation_data_generator(path, split):
    train_folder_path = path.replace('{split}', split)

    # Create variables for train dataset
    author_documents = defaultdict(list) # Maps from an author's true ID to a list of all their documents
    author_query_documents = defaultdict(list) # Maps from an author's true ID to a list of their query documents
    author_candidate_documents = defaultdict(list) # Maps from an author's true ID to a list of their candidate documents
    query_author_ids = set()
    candidate_author_ids = set()
    

    
    # Read query documents
    #'/burg/dsi/users/ma4608/hiatus_performers_data/sadiri/*/{split}
    for json_file in glob.glob(train_folder_path + '_queries_filtered.jsonl'):
        print(json_file)
        for line in jsonlines.open(json_file):
            author_id = line['authorIDs'][0]
            query_author_ids.add(author_id)
            author_documents[author_id].append(line['fullText'])
            author_query_documents[author_id].append(line['fullText'])

    print(f"Check Here: {train_folder_path}")
    

    # Read candidate documents
    for json_file in glob.glob(train_folder_path + '_candidates_filtered.jsonl'):
        print(json_file)
        for line in jsonlines.open(json_file):
            author_id = line['authorIDs'][0]
            candidate_author_ids.add(author_id)
            author_documents[author_id].append(line['fullText'])
            author_candidate_documents[author_id].append(line['fullText'])
    
    # Convert to lists
    query_author_ids = list(sorted(query_author_ids))
    candidate_author_ids = list(sorted(candidate_author_ids))

    author_documents = {x[0]: x[1] for x in author_documents.items() if len(x[1]) > 1}
    print("Dataset Statistics:", split, len(author_documents))

    
    # Get total number of rows per epoch
    # total_num_rows = len(author_documents)
    total_num_rows = min(len(author_documents), 1000)  # Cap at 1000 rows

    # Create data generator
    def data_generator():
        # If on the train split, do a new seed each epoch
        seed = epoch_tracker['epoch'] if split == 'train' else 0
        rand = Random(seed)
        
        # Create examples of anchors and positives from query-candidate pairs
        rows = []
        for author_id in author_documents:
            if len(author_documents[author_id]) >1:
                rows.append({
                    "anchors": author_documents[author_id][0],
                    "positives": author_documents[author_id][1]
                })
            # else:
            #     print(author_id, ' has ', len(author_documents[author_id]))

        rows = rows[:1000]  # Cap the rows at 1000
        # Assert we have the right number of rows
        assert len(rows) == total_num_rows

        # Shuffle and yield the rows
        rand.shuffle(rows)
        for row in rows:
            yield row

    return total_num_rows, data_generator


def get_performs_data_generator_genre_batches(path, split, genres=None):
    """
    Data generator that alternates genres across batches while keeping a single genre per batch.
    :param path: Path to the dataset.
    :param split: Data split (e.g., "train" or "dev").
    :param genres: List of all possible genres (if not provided, infer from the data).
    """
    train_folder_path = path.replace("{split}", split)

    # Read documents by authors and genres
    author_documents_by_genre = defaultdict(lambda: defaultdict(list))  # {author_id: {genre: [documents]}}
    for json_file in glob.glob(train_folder_path + "_queries_filtered.jsonl"):
        for line in jsonlines.open(json_file):
            author_id = line["authorIDs"][0]
            genre = os.path.basename(os.path.dirname(json_file))
            author_documents_by_genre[author_id][genre].append(line["fullText"])

    # Filter authors with at least two documents across genres
    valid_authors = {
        author_id: genres
        for author_id, genres in author_documents_by_genre.items()
        if sum(len(docs) for docs in genres.values()) > 1
    }

    # Infer genres if not provided
    if genres is None:
        genres = set(
            genre for docs_by_genre in author_documents_by_genre.values() for genre in docs_by_genre.keys()
        )

    def data_generator():
        seed = epoch_tracker["epoch"] if split == "train" else 0
        rand = Random(seed)

        # Shuffle authors and genres for randomness
        author_ids = list(valid_authors.keys())
        rand.shuffle(author_ids)
        genres = list(genres)
        rand.shuffle(genres)

        # Alternate genres across batches
        for batch_genre in genres:
            for author_id in author_ids:
                # Skip authors without the current genre
                if batch_genre not in author_documents_by_genre[author_id]:
                    continue

                # Anchor from the batch genre
                anchor = rand.choice(author_documents_by_genre[author_id][batch_genre])

                # Positive from a different genre by the same author
                other_genres = [
                    genre for genre in author_documents_by_genre[author_id] if genre != batch_genre
                ]
                if not other_genres:
                    continue  # Skip if no other genres are available
                positive_genre = rand.choice(other_genres)
                positive = rand.choice(author_documents_by_genre[author_id][positive_genre])

                # Yield anchor-positive pair
                yield {
                    "anchors": anchor,
                    "positives": positive,
                }

    return len(valid_authors), data_generator

import pandas as pd
from random import Random

def get_performs_data_generator_random_batches(path, split, genres=None):
    """
    Data generator that alternates genres across batches while keeping a single genre per batch.
    :param path: Path to the dataset.
    :param split: Data split (e.g., "train" or "dev").
    :param genres: List of all possible genres (if not provided, infer from the data).
    """
    file_path = path.replace("{split}", split)
    print(f"Called to read file: {file_path}")
    
    # Load the data
    # with open(file_path, 'r') as f:
    #     first_1000_lines = list(islice(f, 10000))
    # # Parse the lines into a DataFrame
    # file_df = pd.DataFrame([json.loads(line) for line in first_1000_lines])
    file_df = pd.read_json(file_path, lines=True)
    print(f"Finished Loading file: {file_path}")
    
    file_df['authorID'] = file_df['authorIDs'].apply(lambda x: x[0])
    
    # Remove authors that don't meet minimum documents threshold count
    file_df = file_df[file_df.groupby('authorID')['documentID'].transform('count') > 1]
    
    # Aggregate documents at author level
    aggregated_file_df = file_df.groupby('authorID', as_index=False).agg({'documentID': list, 'fullText': list})
    # aggregated_file_df = aggregated_file_df.sample(n=200)
    print(f"Finished aggregating at author level: {aggregated_file_df} with shape: {aggregated_file_df.shape}")

    # Define the data generator
    def data_generator():
        seed = epoch_tracker["epoch"] if split == "train" else 0
        rand = Random(seed)

        # Shuffle the data for the current epoch
        shuffled_df = aggregated_file_df.sample(frac=1, random_state=seed)  # Use fixed seed for reproducibility
        for _, row in shuffled_df.iterrows():
            pair = rand.sample(row['fullText'], 2)
            yield {'anchors': pair[0], 'positives': pair[1]}

    return aggregated_file_df.shape[0], data_generator


def get_data_generator_for_combined_hrs(path, fold, split, split_percent=None):
    train_folder_path = path.replace('{fold}', fold).replace('{split}', split)
    
    # Create variables for train dataset
    author_global_ids = {} # Maps from query or candidate author ID to true author ID
    author_documents = defaultdict(list) # Maps from an author's true ID to a list of all their documents
    author_query_documents = defaultdict(list) # Maps from an author's true ID to a list of their query documents
    author_candidate_documents = defaultdict(list) # Maps from an author's true ID to a list of their candidate documents
    query_author_ids = set()
    candidate_author_ids = set()
    
    # Get global ID mapping for authors
    with open(os.path.join(train_folder_path, 'author-set-ids-map_P1_and_P2.json'), 'r') as fp:
        for k, v in json.load(fp).items():
            #print(k)
            source = ast.literal_eval(k)[1]
            if source.startswith('HRS1'):
                k = ast.literal_eval(ast.literal_eval(k)[0])[0]
            else:
                k = ast.literal_eval(k)[0]

            author_global_ids[k] = k
            author_global_ids[v] = k

    # Read query documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, '*_input_queries.jsonl'))[0]):
        #print(line.keys())
        author_id = author_global_ids[line['authorIDs'][0]]
        query_author_ids.add(author_id)
        author_documents[author_id].append(line['fullText'])
        author_query_documents[author_id].append(line['fullText'])
    

    # Read candidate documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, '*_input_candidates.jsonl'))[0]):
        author_id = author_global_ids[line['authorSetIDs'][0]]
        candidate_author_ids.add(author_id)
        author_documents[author_id].append(line['fullText'])
        author_candidate_documents[author_id].append(line['fullText'])
    
    # Convert to lists
    query_author_ids = list(sorted(query_author_ids))
    candidate_author_ids = list(sorted(candidate_author_ids))

    # Split train into internal train/val splits for hyperparameter tuning / early-stopping
    if split_percent is not None:
        bigger_split_percent = max(split_percent, 1-split_percent)
        index_of_split = 0 if bigger_split_percent == split_percent else 1
        query_author_ids = np.split(query_author_ids, [int(len(query_author_ids)*bigger_split_percent)])[index_of_split]
        candidate_author_ids = np.split(candidate_author_ids, [int(len(candidate_author_ids)*bigger_split_percent)])[index_of_split]
    print("Dataset Statistics:", fold, split, split_percent, len(query_author_ids), len(candidate_author_ids))

    # Get total number of rows per epoch
    total_num_rows = len(query_author_ids) * 2

    # Create data generator
    def data_generator():
        # If on the train split, do a new seed each epoch
        seed = epoch_tracker['epoch'] if split == 'train' else 0
        rand = Random(seed)
        
        # Create examples of anchors and positives from query-candidate pairs
        rows = []
        for query_author_id in query_author_ids:
            rows.append({
                "anchors": rand.choice(author_query_documents[query_author_id]),
                "positives": rand.choice(author_candidate_documents[query_author_id])
            })

        # Create augmented examples of anchors and positives from candidate-candidate pairs
        cc = 0
        while cc < len(query_author_ids):
            candidate_author_id = rand.choice(candidate_author_ids)
            if candidate_author_id in query_author_ids or len(author_candidate_documents[candidate_author_id]) < 2:
                continue
            docs = author_candidate_documents[candidate_author_id].copy()
            rand.shuffle(docs)
            rows.append({
                "anchors": docs[0],
                "positives": docs[1],
            })
            cc += 1
        
        # Assert we have the right number of rows
        assert len(rows) == total_num_rows

        # Shuffle and yield the rows
        rand.shuffle(rows)
        for row in rows:
            yield row

    return total_num_rows, data_generator


def get_path_until_stop(original_path, stop_dir):
    """
    Extracts the path up to and including the specified stop directory.

    :param original_path: The full original path as a string.
    :param stop_dir: The directory where the path extraction should stop.
    :return: The extracted path up to and including the stop directory.
    """
    # Split the path into parts
    path_parts = original_path.split(os.sep)
    
    # Find the index of the stop directory
    if stop_dir in path_parts:
        stop_index = path_parts.index(stop_dir) + 1
        return os.sep.join(path_parts[:stop_index])
    else:
        # If stop_dir not found, return the original path
        return original_path
    

def get_data_generator_for_k_fold_hrs(path, fold, split, split_percent=None):
    train_folder_path = path.replace('{fold}', fold).replace('{split}', split)
    print(f"Train Path Used: {train_folder_path}")
    
    # Create variables for train dataset
    author_global_ids = {} # Maps from query or candidate author ID to true author ID
    author_documents = defaultdict(list) # Maps from an author's true ID to a list of all their documents
    author_query_documents = defaultdict(list) # Maps from an author's true ID to a list of their query documents
    author_candidate_documents = defaultdict(list) # Maps from an author's true ID to a list of their candidate documents
    query_author_ids = set()
    candidate_author_ids = set()

    author_set_ids_file_dir = get_path_until_stop(train_folder_path, split)
    author_set_ids_file_path = os.path.join(author_set_ids_file_dir, 'author-set-ids-map.json')
    print(f"Author set Ids file used: {author_set_ids_file_path}")

    # Get global ID mapping for authors
    with open(author_set_ids_file_path, 'r') as fp:
        for k, v in json.load(fp).items():
            #print(k)
            source = ast.literal_eval(k)[1]
            if source.startswith('HRS1'):
                k = ast.literal_eval(ast.literal_eval(k)[0])[0]
            else:
                k = ast.literal_eval(k)[0]

            author_global_ids[k] = k
            author_global_ids[v] = k

    # Read query documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, '*_input_queries.jsonl'))[0]):
        #print(line.keys())
        author_id = author_global_ids[line['authorIDs'][0]]
        query_author_ids.add(author_id)
        author_documents[author_id].append(line['fullText'])
        author_query_documents[author_id].append(line['fullText'])
    

    # Read candidate documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, '*_input_candidates.jsonl'))[0]):
        author_id = author_global_ids[line['authorSetIDs'][0]]
        candidate_author_ids.add(author_id)
        author_documents[author_id].append(line['fullText'])
        author_candidate_documents[author_id].append(line['fullText'])
    
    # Convert to lists
    query_author_ids = list(sorted(query_author_ids))
    candidate_author_ids = list(sorted(candidate_author_ids))

    # Split train into internal train/val splits for hyperparameter tuning / early-stopping
    if split_percent is not None:
        bigger_split_percent = max(split_percent, 1-split_percent)
        index_of_split = 0 if bigger_split_percent == split_percent else 1
        query_author_ids = np.split(query_author_ids, [int(len(query_author_ids)*bigger_split_percent)])[index_of_split]
        candidate_author_ids = np.split(candidate_author_ids, [int(len(candidate_author_ids)*bigger_split_percent)])[index_of_split]
    print("Dataset Statistics:", fold, split, split_percent, len(query_author_ids), len(candidate_author_ids))

    # Get total number of rows per epoch
    total_num_rows = len(query_author_ids) * 2

    # Create data generator
    def data_generator():
        # If on the train split, do a new seed each epoch
        seed = epoch_tracker['epoch'] if split == 'train' else 0
        rand = Random(seed)
        
        # Create examples of anchors and positives from query-candidate pairs
        rows = []
        for query_author_id in query_author_ids:
            rows.append({
                "anchors": rand.choice(author_query_documents[query_author_id]),
                "positives": rand.choice(author_candidate_documents[query_author_id])
            })

        # Create augmented examples of anchors and positives from candidate-candidate pairs
        cc = 0
        while cc < len(query_author_ids):
            candidate_author_id = rand.choice(candidate_author_ids)
            if candidate_author_id in query_author_ids or len(author_candidate_documents[candidate_author_id]) < 2:
                continue
            docs = author_candidate_documents[candidate_author_id].copy()
            rand.shuffle(docs)
            rows.append({
                "anchors": docs[0],
                "positives": docs[1],
            })
            cc += 1
        
        # Assert we have the right number of rows
        assert len(rows) == total_num_rows

        # Shuffle and yield the rows
        rand.shuffle(rows)
        for row in rows:
            yield row

    return total_num_rows, data_generator


def get_data_generator_hard_batches(text_path, batches_path, split):
    """
    Data generator that yields hard negative batches as training examples.
    It loads batches from a JSON file and yields a dictionary with keys:
      'anchors': list of anchor texts,
      'positives': list of positive texts.

    The 'split' parameter is kept as a placeholder.

    Parameters:
        path (str): Path to the batches JSON file.
        split (str): Data split (e.g., "train" or "dev"); currently only 'train' is available.

    Returns:
        tuple: (num_batches, generator_function)
    """
    # Load the batches from the JSON file.
    batches_info_path = batches_path.replace("{split}", split)
    print(f"Reading batches info at: {batches_info_path}")
    sys.stdout.flush()
    with open(batches_info_path, "r") as f:
        batches = json.load(f)
    num_batches = len(batches)
    print("Completed: batches loaded.")
    sys.stdout.flush()

    print(f"Loading text info at: {text_path}")
    sys.stdout.flush()
    doc_text_dict = {}
    text_df = pd.read_json(text_path)
    for idx, row in text_df.iterrows():
        doc_id = row['documentID']
        doc_text_dict[doc_id] = row['fullText']
    print("Completed: text loaded")
    sys.stdout.flush()

    def data_generator():
        for batch in batches:
            anchors = []
            positives = []
            for pair in batch:
                anchor = doc_text_dict.get(pair.get("doc1"))
                positive = doc_text_dict.get(pair.get("doc2"))
                anchors.append(anchor)
                positives.append(positive)
            yield {"anchors": anchors, "positives": positives}

    return num_batches, data_generator
