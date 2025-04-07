from src.sadiri_training import get_data_generator_hard_batches, DataSource, DataDreamer, get_luar_trainer, LoraConfig, EpochTrackerCallback, losses
import os
import sys
import pandas as pd


def train_datadreamer_ta2_on_hard_batched_data(text_path, output_folder, batches_path,
                                               luar_model_path, batch_size=256, epochs=25):
    """
    Train using hard-batched data loaded from batches.json.

    Parameters:
        fold (str): A placeholder pattern for the fold (not used in this generator).
        output_folder (str): Directory to store outputs.
        used_loss (str): Loss type (dummy placeholder here).
        batches_path (str): Path to the JSON file containing batches.
        luar_model_path (str): Path to the LUAR model.
        batch_size (int): Batch size used in training.
        epochs (int): Number of epochs.
    """
    # Get data generators for training and dev (using the same batches for now as placeholder)

    print(f"Loading text info at: {text_path}")
    sys.stdout.flush()
    doc_text_dict = {}
    text_df = pd.read_json(text_path)
    for idx, row in text_df.iterrows():
        doc_id = row['documentID']
        doc_text_dict[doc_id] = row['fullText']
    print("Completed: text loaded")
    sys.stdout.flush()

    train_num_examples, train_data_generator = get_data_generator_hard_batches(doc_text_dict, batches_path, "train")
    dev_num_examples, dev_data_generator = get_data_generator_hard_batches(doc_text_dict, batches_path, "dev")

    with DataDreamer(output_folder):
        dataset = DataSource(
            "Train Data",
            data=train_data_generator,
            total_num_rows=train_num_examples,
        )
        dev_dataset = DataSource(
            "Dev Data",
            data=dev_data_generator,
            total_num_rows=dev_num_examples,
        )

        trainer = get_luar_trainer()(
            "LUAR Trainer",
            model_name=luar_model_path,
            peft_config=LoraConfig(),
            trust_remote_code=True,
            device='cuda',
            dtype="bfloat16",
            force=False,
        )
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
            accelerator_config={"dispatch_batches": False},
            callbacks=[EpochTrackerCallback()]
        )

        trainer.export_to_disk(output_folder + 'final_model', adapter_only=False)


output_folder = '/mnt/nlpgpu-io1/data/jiachzhu/projects/datadreamer-ta2/output'
luar_model_path = '/mnt/nlpgpu-io1/data/jiachzhu/projects/model/LUAR-MUD'
model_op_dir = os.path.join(output_folder, 'sadiri_hard_batch_model_v1')
batches_path = "/mnt/nlpgpu-io1/data/jiachzhu/projects/datadreamer-ta2/output/{split}_batches.json" # this only store doc id like {'doc1': 'c93d1d1c-3357-2ba8-b604-7a76ffecd66b', 'doc2': '979f7248-6df3-8656-374c-6acfffb98866', 'similarity': 0.15085165411557855, 'genre1': 'Opinion/Argumentation', 'genre2': 'Forum'}
text_path = "/mnt/nlpgpu-io1/data/jiachzhu/projects/data/train_sadiri_processed_with_luarsbertembeddings_wo_ao3_filtered.jsonl" # this has text

train_datadreamer_ta2_on_hard_batched_data(
    text_path=text_path,
    output_folder=model_op_dir,
    batches_path=batches_path,
    luar_model_path=luar_model_path,
)

print("Training Completed.")
sys.stdout.flush()