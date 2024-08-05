# Train script
import os
import sys
import glob
import ast
import json
import jsonlines
import numpy as np
from collections import defaultdict
from datadreamer import DataDreamer
from datadreamer.steps import DataSource
from transformers import TrainerCallback
from sentence_transformers import losses
from SupContrastLoss import SupConLoss
from peft import LoraConfig
from random import Random

from luar_utils import get_luar_trainer
epoch_tracker = {}
epoch_tracker['epoch'] = 0
class EpochTrackerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_tracker['epoch'] = int(state.epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_tracker['epoch'] = int(state.epoch)

def get_data_generator(path, fold, split, split_percent=None):
    train_folder_path = path.replace('{fold}', fold).replace('{split}', split)
    
    # Create variables for train dataset
    author_global_ids = {} # Maps from query or candidate author ID to true author ID
    author_documents = defaultdict(list) # Maps from an author's true ID to a list of all their documents
    author_query_documents = defaultdict(list) # Maps from an author's true ID to a list of their query documents
    author_candidate_documents = defaultdict(list) # Maps from an author's true ID to a list of their candidate documents
    query_author_ids = set()
    candidate_author_ids = set()
    
    # Get global ID mapping for authors
    with open(os.path.join(train_folder_path, 'author-set-ids-map.json'), 'r') as fp:
        for k, v in json.load(fp).items():
            k = ast.literal_eval(ast.literal_eval(k)[0])[0]
            author_global_ids[k] = k
            author_global_ids[v] = k

    # Read query documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, 'TA2', 'hrs_*', 'data', '*_input_queries.jsonl'))[0]):    
        author_id = author_global_ids[line['authorIDs'][0]]
        query_author_ids.add(author_id)
        author_documents[author_id].append(line['fullText'])
        author_query_documents[author_id].append(line['fullText'])
    

    # Read candidate documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, 'TA2', 'hrs_*', 'data', '*_input_candidates.jsonl'))[0]):
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


def get_data_generator_for_supervised_contrastive_learning(path, fold, split, split_percent=None):
    train_folder_path = path.replace('{fold}', fold).replace('{split}', split)
    
    # Create variables for train dataset
    author_global_ids = {} # Maps from query or candidate author ID to true author ID
    author_documents = defaultdict(list) # Maps from an author's true ID to a list of all their documents
    author_query_documents = defaultdict(list) # Maps from an author's true ID to a list of their query documents
    author_candidate_documents = defaultdict(list) # Maps from an author's true ID to a list of their candidate documents
    query_author_ids = set()
    candidate_author_ids = set()

    # Get global ID mapping for authors
    with open(os.path.join(train_folder_path, 'author-set-ids-map.json'), 'r') as fp:
        for k, v in json.load(fp).items():
            k = ast.literal_eval(ast.literal_eval(k)[0])[0]
            author_global_ids[k] = k
            author_global_ids[v] = k

    all_author_ids = set(list(author_global_ids.keys()) + list(author_global_ids.values()))
    all_author_ids_to_labels = {id:i for i, id in enumerate(sorted(all_author_ids))}

    # Read query documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, 'TA2', 'hrs_*', 'data', '*_input_queries.jsonl'))[0]):    
        author_id = author_global_ids[line['authorIDs'][0]]
        query_author_ids.add(author_id)
        author_documents[author_id].append(line['fullText'])
        author_query_documents[author_id].append(line['fullText'])
    

    # Read candidate documents
    for line in jsonlines.open(glob.glob(os.path.join(train_folder_path, 'TA2', 'hrs_*', 'data', '*_input_candidates.jsonl'))[0]):
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
            for d in author_query_documents[query_author_id]:
                rows.append({
                    "anchors": d,
                    "others": "",
                    "labels": all_author_ids_to_labels[query_author_id] # anchors' labels
                })
                
            for d in author_candidate_documents[query_author_id]:
                rows.append({
                    "anchors": d,
                    "others": "",
                    "labels": all_author_ids_to_labels[query_author_id] # anchors' labels
                })

        # Shuffle and yield the rows
        rand.shuffle(rows)
        for row in rows:
            yield row

    return total_num_rows, data_generator