from datetime import datetime
from sadiri_training import get_performs_data_generator, EpochTrackerCallback, LoraConfig, get_luar_trainer, DataSource, DataDreamer, losses, get_performs_data_generator_random_batches
from SupContrastLoss import MultiPosConLoss
from RankingLoss import MultipleNegativesSymmetricRankingLoss
import os
from transformers import TrainerCallback
from random import Random
import pandas as pd
import argparse
import pandas as pd
import numpy as np
from random import Random
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from random import Random
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


epoch_tracker = {}
epoch_tracker['epoch'] = 0
class EpochTrackerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_tracker['epoch'] = int(state.epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_tracker['epoch'] = int(state.epoch)


def get_dev_random_batches(path):
    file_df = pd.read_json(path, lines=True)
    print(f"Finished Loading Dev file: {path}")
    
    file_df['authorID'] = file_df['authorIDs'].apply(lambda x: x[0])
    
    # Remove authors that don't meet minimum documents threshold count
    file_df = file_df[file_df.groupby('authorID')['documentID'].transform('count') > 1]
    
    # Aggregate documents at author level
    aggregated_file_df = file_df.groupby('authorID', as_index=False).agg({'documentID': list, 'fullText': list})
    
    print(f"Finished aggregating at author level: {aggregated_file_df} with shape: {aggregated_file_df.shape}")

    # Define the data generator
    def data_generator():
        seed = 0
        rand = Random(seed)

        # Shuffle the data for the current epoch
        shuffled_df = aggregated_file_df.sample(frac=1, random_state=seed)  # Use fixed seed for reproducibility
        for _, row in shuffled_df.iterrows():
            pair = rand.sample(row['fullText'], 2)
            yield {'anchors': pair[0], 'positives': pair[1]}

    return aggregated_file_df.shape[0], data_generator 


def get_anchorpos_random_batches(path, split):
    """
    Data generator that alternates genres across batches while keeping a single genre per batch.
    :param path: Path to the dataset.
    :param split: Data split (e.g., "train" or "dev").
    :param genres: List of all possible genres (if not provided, infer from the data).
    """
    print(f"Called get_anchorpos_random_batches with split: {split}")

    if split == "train":
        file_df = pd.read_json(path, lines=True)
        print(f"Finished Loading file: {path} with Shape: {file_df.shape}")
        
        ## Drop embedding columns to save memory
        file_df.drop(columns=[
            'doc_luarmud_embedding_anchor','doc_luarmud_embedding_positive',
            'doc_sbertamllv2_embedding_anchor','doc_sbertamllv2_embedding_positive',
        ], inplace=True)
        print(f"Dropped embedding columns to save memory, now with Shape: {file_df.shape}")

        ## Get AuthorID from authorIDs
        # file_df['authorID'] = file_df['authorIDs_anchor'].apply(lambda x: x[0])
        
        # # Remove authors that don't meet minimum documents threshold count
        # # Get group counts before filtering
        # author_group_sizes = file_df.groupby('authorID')['documentID_anchor'].count()
        # print(f"Total unique authors before filtering: {author_group_sizes.shape[0]}")

        # # Filter authors with more than 1 document
        # file_df = file_df.loc[file_df['authorID'].isin(author_group_sizes[author_group_sizes > 1].index)]
        # print(f"Filtered authors with >1 pairs: {file_df.shape} (Unique authors: {file_df['authorID'].nunique()})")

        # Group all anchor-positive pairs per author
        author_grouped = file_df.groupby('authorID').agg(list).reset_index()
        print(f"Grouped by authorID: {author_grouped.shape[0]} authors")
        print(f"Final DF, now shape: {author_grouped.shape}")

        # Define the data generator
        def data_generator():
            seed = epoch_tracker["epoch"] if split == "train" else 0
            rand = Random(seed)

            # Shuffle author rows each epoch
            shuffled_authors = author_grouped.sample(frac=1, random_state=seed)
            # import pdb; pdb.set_trace()
            for _, row in shuffled_authors.iterrows():
                # Randomly sample one index from the list of anchor-positive pairs for this author
                num_pairs = len(row['documentID_anchor'])
                idx = rand.randint(0, num_pairs - 1)

                yield {
                    'anchors': row['fullText_anchor'][idx],
                    'positives': row['fullText_positive'][idx],
                    'authorID': row['authorID']
                }

        return author_grouped.shape[0], data_generator
    
    else:
        return get_dev_random_batches(path)


def get_anchorpos_rankingloss_hardbatches(path, split):
    """
    Should be for: /data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/hard_batching_dataset_luaremb_wo_ao3_filtered/train_sadiri_hard_batches_v001.jsonl
    Data generator that alternates genres across batches while keeping a single genre per batch.
    :param path: Path to the dataset.
    :param split: Data split (e.g., "train" or "dev").
    :param genres: List of all possible genres (if not provided, infer from the data).
    """
    print(f"Called to read file: {path}")
    
    file_df = pd.read_json(path, lines=True)
    print(f"Finished Loading file: {path} with Shape: {file_df.shape}")

    if split == "train":
        ## Filter those batch_id's where their frequency is not 32
        batch_counts = file_df['batch_id'].value_counts()
        num_with_32 = (batch_counts == 32).sum()
        num_without_32 = (batch_counts != 32).sum()
        print(f"Number of Batches With 32: {num_with_32} | Without 32: {num_without_32}")

        bidcc = file_df.groupby('batch_id')['documentID_anchor'].count()
        good_batch_ids = bidcc[bidcc == 32].index
        file_df = file_df.loc[file_df['batch_id'].isin(good_batch_ids)] 
        print(f"Shape after dropping batch_ids not of 32 batch size: {file_df.shape}")

        batches = file_df['batch_id'].unique().tolist()
        # Define the data generator
        def data_generator():
            seed = epoch_tracker['epoch'] if split == 'train' else 0
            rand = Random(seed)
            rand.shuffle(batches)
            for batch in batches:
                batch_rows = file_df.loc[file_df['batch_id'] == batch]
                for _, anchpospair in batch_rows.iterrows():
                    yield {'anchors': anchpospair['fullText_anchor'], 'positives': anchpospair['fullText_positive']}

        return file_df.shape[0], data_generator
    else:
        return get_dev_random_batches(path) 
    

def get_anchorpos_hardmix_batches(
    path, 
    split, 
    sem_sim_threshold, 
    sty_sim_threshold, 
    batch_size=32
):
    """
    Construct each batch such that:
      1. ~30% of documents satisfy: semantic similarity < sem_sim_threshold
      2. ~30% of documents satisfy: style similarity > sty_sim_threshold, computed 
         by comparing the style embedding (doc_luarmud_embedding_anchor) between 
         documents from different authors.
      3. ~40% random selection from the remainder.
      (Ignoring genre similarity)

    Parameters:
    -----------
    path : str
        File path to read from.
    split : str
        'train' or 'dev' (or 'test'). If not 'train', use dev batch sampler.
    sem_sim_threshold : float
        Semantic similarity cutoff.
    sty_sim_threshold : float
        Style similarity cutoff.
    style_sim_df : pd.DataFrame
        (Not used here, provided for compatibility.)
    batch_size : int
        Size of each batch to sample.

    Returns:
    --------
    (int, generator):
        The total number of rows, and a generator that yields dictionary samples
        of the form:
        {
          'anchors': <text>,
          'positives': <text>,
          'authorID': <str or int>
        }
    """
    if split != "train":
        return get_dev_random_batches(path)

    # -------------------------
    # 1) LOAD THE DATA
    # -------------------------
    print(f"Thresholds are: {sem_sim_threshold}, {sty_sim_threshold}, batch size {batch_size}")
    file_df = pd.read_json(path, lines=True)
    print(f"Finished Loading file: {path} with shape: {file_df.shape}")
    file_df = file_df.copy()
    file_df["used"] = False

    # -------------------------
    # 2) DEFINE BATCH PROPORTIONS
    # -------------------------
    # For a batch_size=32:
    #   ~30% from semantic condition -> 9
    #   ~30% from style condition -> 9
    #   ~40% random -> 14
    num_sem = int(0.3 * batch_size)  # e.g. 9
    num_sty = int(0.3 * batch_size)  # e.g. 9

    def data_generator():
        # Set up a reproducible random seed if needed
        seed = epoch_tracker.get("epoch", 0) if split == "train" else 0
        rand = Random(seed)
        df_working = file_df.copy()
        
        # Create a progress bar using total number of rows in the DataFrame.
        total_rows = file_df.shape[0]
        pbar = tqdm(total=total_rows, desc="Documents Processed", leave=True)

        while True:
            df_remaining = df_working[~df_working["used"]]
            if len(df_remaining) < batch_size:
                break

            # --------------------------------------------------
            # A) SEMANTIC CONDITION:
            #    Select documents whose semantic similarity is below the threshold.
            # --------------------------------------------------
            sem_candidates = df_remaining[
                df_remaining["sbertamllv2_embedding_similarity_score"] < sem_sim_threshold
            ]
            sem_selected = sem_candidates.sample(
                n=min(num_sem, len(sem_candidates)),
                random_state=rand.randint(0, 999999)
            )

            # --------------------------------------------------
            # B) STYLE CONDITION:
            #    Select documents from authors that have high style similarity.
            #    We compute pairwise cosine similarity on the style embeddings 
            #    ("doc_luarmud_embedding_anchor") from a sample of candidates 
            #    (excluding those already selected semantically).
            # --------------------------------------------------
            candidate_pool = df_remaining[~df_remaining["documentID_anchor"].isin(sem_selected["documentID_anchor"])]
            sty_selected_doc_ids = set()
            max_attempts = 5  # limit the number of sampling attempts
            attempt = 0
            while len(sty_selected_doc_ids) < num_sty and attempt < max_attempts:
                sample_size = min(100, len(candidate_pool))
                if sample_size == 0:
                    break
                candidate_sample = candidate_pool.sample(n=sample_size, random_state=rand.randint(0, 999999))
                # Perform a cross join on the candidate sample with itself to generate pairs
                pairs = candidate_sample.merge(candidate_sample, how='cross', suffixes=("_p1", "_p2"))
                # Only consider pairs from different authors
                pairs = pairs[pairs["authorID_p1"] != pairs["authorID_p2"]]

                def compute_cosine(row):
                    vec1 = np.array(row["doc_luarmud_embedding_anchor_p1"]).reshape(1, -1)
                    vec2 = np.array(row["doc_luarmud_embedding_anchor_p2"]).reshape(1, -1)
                    return cosine_similarity(vec1, vec2)[0, 0]
                
                pairs["author_pair_doc_style_score"] = pairs.apply(compute_cosine, axis=1)
                # Filter to only those pairs with style similarity above the threshold
                high_sim_pairs = pairs[pairs["author_pair_doc_style_score"] > sty_sim_threshold]
                # Accumulate document IDs from both sides of the high-similarity pairs
                docs_from_pairs = set(high_sim_pairs["documentID_anchor_p1"].tolist() + 
                                      high_sim_pairs["documentID_anchor_p2"].tolist())
                sty_selected_doc_ids.update(docs_from_pairs)
                attempt += 1

            if len(sty_selected_doc_ids) > 0:
                sty_candidates = candidate_pool[candidate_pool["documentID_anchor"].isin(list(sty_selected_doc_ids))]
                sty_selected = sty_candidates.sample(
                    n=min(num_sty, len(sty_candidates)),
                    random_state=rand.randint(0, 999999)
                )
            else:
                # If no style candidates are found, use an empty DataFrame.
                sty_selected = candidate_pool.sample(n=0)

            # --------------------------------------------------
            # C) COMBINE & REMOVE DUPLICATES:
            # --------------------------------------------------
            condition_df = pd.concat([sem_selected, sty_selected], ignore_index=True)
            condition_df.drop_duplicates(subset="documentID_anchor", inplace=True)
            used_ids = condition_df["documentID_anchor"].unique().tolist()

            # --------------------------------------------------
            # D) RANDOM CONDITION:
            # --------------------------------------------------
            num_random = batch_size - len(used_ids)
            if num_random > 0:
                rand_candidates = df_remaining[
                    ~df_remaining["documentID_anchor"].isin(used_ids)
                ]
                rand_selected = rand_candidates.sample(
                    n=min(num_random, len(rand_candidates)),
                    random_state=rand.randint(0, 999999)
                )
                final_batch = pd.concat([condition_df, rand_selected], ignore_index=True)
            else:
                final_batch = condition_df.sample(n=batch_size, random_state=rand.randint(0, 999999))

            # Shuffle the final batch
            final_batch = final_batch.sample(frac=1, random_state=rand.randint(0, 999999))

            # Mark these as used so they are not selected again
            df_working.loc[
                df_working["documentID_anchor"].isin(final_batch["documentID_anchor"]),
                "used"
            ] = True

            # Update the progress bar
            pbar.update(len(final_batch))

            # Yield the final batch as a series of dictionary samples.
            for _, row in final_batch.iterrows():
                yield {
                    "anchors": row["fullText_anchor"],
                    "positives": row["fullText_positive"],
                    "authorID": row["authorID"]
                }
        pbar.close()

    return file_df.shape[0], data_generator


def train_datadreamer_ta2__on_performers_data(type_, train_path, dev_path, output_folder, used_loss, luar_model_path='./rrivera1849', batch_size=128, epochs=25, dict_thresholds={'sem_sim_threshold':0.1,'sty_sim_threshold':0.2}):
    if type_ == "anchorpos_rankingloss_hardbatches":
        train_num_rows, train_data_generator = get_anchorpos_rankingloss_hardbatches(train_path, "train")
        dev_num_rows, dev_data_generator = get_anchorpos_rankingloss_hardbatches(dev_path, "dev")
    elif type_ == "get_anchorpos_random_batches":
        train_num_rows, train_data_generator = get_anchorpos_random_batches(train_path, "train")
        dev_num_rows, dev_data_generator = get_anchorpos_random_batches(dev_path, "dev")

    elif type_ == "get_anchorpos_hardmix_batches":
        train_num_rows, train_data_generator = get_anchorpos_hardmix_batches(train_path, "train", dict_thresholds['sem_sim_threshold'], dict_thresholds['sty_sim_threshold'])
        dev_num_rows, dev_data_generator = get_anchorpos_hardmix_batches(dev_path, "dev", dict_thresholds['sem_sim_threshold'], dict_thresholds['sty_sim_threshold'])   

    with DataDreamer(output_folder):
        dataset = DataSource(
            "Train Data",
            data=train_data_generator,
            total_num_rows=train_num_rows,
        )
        dev_dataset = DataSource(
            "Dev Data",
            data=dev_data_generator,
            total_num_rows=dev_num_rows,
        )
    
        trainer = get_luar_trainer()(
            "LUAR Trainer",
            model_name=luar_model_path,
            peft_config=LoraConfig(),
            trust_remote_code=True,
            device='cuda',
            dtype="bfloat16",
            force=False, #so we can resume training if things shutsdown
        )
        # import pdb; pdb.set_trace() 
        trainer.train_with_positive_pairs(
            train_anchors=dataset.output["anchors"],
            train_positives=dataset.output["positives"],
            validation_anchors=dev_dataset.output["anchors"],
            validation_positives=dev_dataset.output["positives"],
            epochs=epochs,    
            batch_size=batch_size,
            loss=losses.MultipleNegativesSymmetricRankingLoss,
            learning_rate=0.0005,
            early_stopping_threshold=0.001,
            early_stopping_patience=5,
            eval_strategy='steps',
            logging_strategy='steps',
            save_strategy='steps',
            logging_steps=200,
            save_steps=600,
            eval_steps=600,
            resume_from_checkpoint=False,
            overwrite_output_dir=True,
            accelerator_config={
                "dispatch_batches": False,
            },
            callbacks=[EpochTrackerCallback()]
        )

        trainer.export_to_disk(output_folder + 'final_model', adapter_only=False)


output_path = '../output'
luar_model_path = '../training_source/rrivera1849/LUAR-MUD'

# dev_file_path = "../data/ta2_jan_2025_trian_data/dev_sadiri_processed_with_embeddings_wo_ao3_filtered.jsonl"
### ALWAYS CHECK BELOW 3 FOR EVERY RUN
## train_file_path = "../data/ta2_jan_2025_trian_data/hard_batching_dataset_luaremb_wo_ao3_filtered/train_sadiri_hard_batches_v001.jsonl"
# train_file_path = "../data/ta2_jan_2025_trian_data/anchor_pos_train_sadiri_luarmudsbertamllv2.jsonl"
# type_ = "get_anchorpos_random_batches"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LUAR with custom data generator")

    parser.add_argument('--train_file_path', type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument('--dev_file_path', type=str, required=True, help="Path to the dev JSONL file")
    parser.add_argument('--type_', type=str, choices=["get_anchorpos_random_batches", "anchorpos_rankingloss_hardbatches", "get_anchorpos_hardmix_batches"], required=True, help="Type of data generator to use")
    parser.add_argument('--model_op_dir_name', type=str, required=True, help="Name of the output model")

    parser.add_argument('--sem_sim_threshold', type=float, required=True, help="Semantic Similarity Threshold")
    parser.add_argument('--sty_sim_threshold', type=float, required=True, help="Stylistic Similarity Threshold")


    args = parser.parse_args()
    dict_thr = {'sem_sim_threshold':args.sem_sim_threshold, 'sty_sim_threshold':args.sty_sim_threshold}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_op_dir = os.path.join(output_path, args.model_op_dir_name, timestamp)

    train_datadreamer_ta2__on_performers_data(
        type_=args.type_,
        train_path=args.train_file_path,
        dev_path=args.dev_file_path,
        output_folder=model_op_dir, 
        used_loss='MultipleNegativesSymmetricRankingLoss',  # dummy
        luar_model_path=luar_model_path, 
        batch_size=32, 
        epochs=1,
        dict_thresholds=dict_thr,
    )
    print(f"Finished running for config: {vars(args)}")