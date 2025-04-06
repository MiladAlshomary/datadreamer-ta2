from datetime import datetime
from sadiri_training import get_performs_data_generator, EpochTrackerCallback, LoraConfig, get_luar_trainer, DataSource, DataDreamer, losses, get_performs_data_generator_random_batches
from SupContrastLoss import MultiPosConLoss
from RankingLoss import MultipleNegativesSymmetricRankingLoss
import os
from transformers import TrainerCallback
from random import Random
import pandas as pd
import argparse
import json
import numpy as np
from tqdm import tqdm
import faiss
import random

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


def get_samplinganchorpos_random_batches(path, split, args):
    """
    Data generator that alternates genres across batches while keeping a single genre per batch.
    :param path: Path to the dataset.
    :param split: Data split (e.g., "train" or "dev").
    :param genres: List of all possible genres (if not provided, infer from the data).
    """
    print(f"Called get_samplinganchorpos_random_batches with split: {split}")
    print(f"Called to read file: {path}")

    file_df = pd.read_json(path, lines=True)
    print(f"Finished Loading file: {path}")
    
    if 'authorID' not in file_df.columns:
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


def get_anchorpos_random_batches(path, split, args):
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


def get_anchorpos_cached_hardbatches(path, split, args):
    """
    Should be for: /data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/hard_batching_dataset_luaremb_wo_ao3_filtered/train_sadiri_hard_batches_v001.jsonl
    Data generator that alternates genres across batches while keeping a single genre per batch.
    :param path: Path to the dataset.
    :param split: Data split (e.g., "train" or "dev").
    :param genres: List of all possible genres (if not provided, infer from the data).
    """
    print(f"Called get_anchorpos_cached_hardbatches with split: {split}")
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


# def get_anchorpos_hardmix_batches(path, split, args):
# Refer to the function "get_anchorpos_hardmix_batches_optimized" in the code: "~/datadreamer-ta2/src/create_cached_hardbatches_pipeline.ipynb"


def train_datadreamer_ta2__on_performers_data(type_, train_path, dev_path, output_folder, args):
    if type_ not in GENERATOR_REGISTRY:
        raise ValueError(f"Invalid generator type: {type_}")
        
    generator_func = GENERATOR_REGISTRY[type_]
    train_num_rows, train_data_generator = generator_func(train_path, "train", args)
    dev_num_rows, dev_data_generator = generator_func(dev_path, "dev", args)
    print(f"Finished loading data generators: {train_num_rows} train rows, {dev_num_rows} dev rows")

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

        custom_lora_config = LoraConfig(
            r=args.lora_config_r,                          # Set new LoRA rank
            lora_dropout=args.lora_config_lora_dropout,              # Set dropout rate to 10%
            lora_alpha=args.lora_config_lora_alpha,                 # Adjust scaling factor (if needed)
            # target_modules=args.lora_config_target_modules.split(",")  # Specify which modules to modify
        )

        trainer = get_luar_trainer()(
            "LUAR Trainer",
            model_name=args.luar_model_path,
            peft_config=custom_lora_config,
            trust_remote_code=True,
            device='cuda',
            dtype="bfloat16",
            force=False, #so we can resume training if things shutsdown
        )

        if args.used_loss == "MultipleNegativesSymmetricRankingLoss":
            lossto_use = losses.MultipleNegativesSymmetricRankingLoss
        elif args.used_loss == "MultiPosConLoss":
            lossto_use = losses.MultiPosConLoss
        elif args.used_loss == "TripletLoss":
            lossto_use = losses.TripletLoss
        else:
            raise ValueError(f"Unknown loss type: {args.used_loss}")
        
        # import pdb; pdb.set_trace() 
        trainer.train_with_positive_pairs(
            train_anchors=dataset.output["anchors"],
            train_positives=dataset.output["positives"],
            validation_anchors=dev_dataset.output["anchors"],
            validation_positives=dev_dataset.output["positives"],
            epochs=args.epochs,    
            batch_size=args.batch_size,
            loss=lossto_use,
            learning_rate=args.learning_rate,
            early_stopping_threshold=args.early_stopping_threshold,
            early_stopping_patience=args.early_stopping_patience,
            eval_strategy='steps',
            logging_strategy='steps',
            save_strategy='steps',
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            resume_from_checkpoint=False,
            overwrite_output_dir=True,
            accelerator_config={
                "dispatch_batches": False,
            },
            callbacks=[EpochTrackerCallback()]
        )

        final_model_path = os.path.join(output_folder,'final_model')
        trainer.export_to_disk(final_model_path, adapter_only=False)
        print(f"Finished training and exporting model to {final_model_path}")


def none_or_int(value):
    if value.lower() == 'none':
        return None
    return int(value)


def write_config_to_file(config, file_path):
    """
    Write the configuration dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration written to {file_path}")
    

if __name__ == "__main__":
    GENERATOR_REGISTRY = {
        "get_samplinganchorpos_random_batches": get_samplinganchorpos_random_batches,
        "get_anchorpos_random_batches": get_anchorpos_random_batches,
        "get_anchorpos_cached_hardbatches": get_anchorpos_cached_hardbatches,
        ## implemented optimized version elsewhere, refer to function placeholder above - "get_anchorpos_hardmix_batches": get_anchorpos_hardmix_batches,
    }

    parser = argparse.ArgumentParser(description="Train LUAR with custom data generator")

    parser.add_argument('--train_file_path', type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument('--dev_file_path', type=str, required=True, help="Path to the dev JSONL file")
    parser.add_argument('--type_', type=str, choices=list(GENERATOR_REGISTRY.keys()), required=True, help="Type of data generator to use")
    parser.add_argument('--model_op_dir_name', type=str, required=True, help="Name of the output model")
    
    # parser.add_argument('--sem_sim_threshold', type=float, default=0.1, help="Semantic Similarity Threshold")
    # parser.add_argument('--sty_sim_threshold', type=float, default=0.3, help="Stylistic Similarity Threshold")
    parser.add_argument('--output_dir', type=str, default="../output", help="Output directory for the model")
    parser.add_argument('--luar_model_path', type=str, default='../training_source/rrivera1849/LUAR-MUD', help="Path to the LUAR model")

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs for training")
    parser.add_argument('--learning_rate', type=float, default=0.0005, help="Learning rate for training")
    parser.add_argument('--early_stopping_threshold', type=float, default=0.001, help="Early stopping threshold")
    parser.add_argument('--early_stopping_patience', type=none_or_int, default=None,help="Pass 'None/none' to disable early stopping. Early stopping patience.")
    parser.add_argument('--used_loss', type=str, default='MultipleNegativesSymmetricRankingLoss', help="Loss function to use")
    parser.add_argument('--logging_steps', type=int, default=500, help="Logging steps")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save steps")
    parser.add_argument('--eval_steps', type=int, default=1000, help="Evaluation steps")

    parser.add_argument('--lora_config_r', type=int, default=16, help="LoRA config r value")
    parser.add_argument('--lora_config_lora_alpha', type=int, default=16, help="LoRA config alpha value")
    parser.add_argument('--lora_config_lora_dropout', type=float, default=0.1, help="LoRA config dropout value")
    # parser.add_argument('--lora_config_target_modules', type=str, default='q_proj,v_proj', help="LoRA config target modules")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.model_op_dir = model_op_dir = os.path.join(args.output_dir, f"{args.model_op_dir_name}_{args.type_}_{timestamp}")

    # Save the args to a JSON file (for example, in the output directory)
    config_path = os.path.join(args.model_op_dir, "config.json")
    write_config_to_file(vars(args), config_path)    

    print(f"Started running for config: {vars(args)}")
    train_datadreamer_ta2__on_performers_data(
        type_=args.type_,
        train_path=args.train_file_path,
        dev_path=args.dev_file_path,
        output_folder=model_op_dir, 
        args=args
    )
    print(f"Finished running for config: {vars(args)}")