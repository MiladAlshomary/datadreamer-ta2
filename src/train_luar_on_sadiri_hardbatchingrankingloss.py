from sadiri_training import get_performs_data_generator, EpochTrackerCallback, LoraConfig, get_luar_trainer, DataSource, DataDreamer, losses, get_performs_data_generator_random_batches, get_sadiri_batching_for_rankingloss
from SupContrastLoss import MultiPosConLoss
from RankingLoss import MultipleNegativesSymmetricRankingLoss
import os

def train_datadreamer_ta2__on_performers_data_with_hardbatch(fold, output_folder, used_loss, luar_model_path='./rrivera1849', batch_size=128, epochs=25):
    train_num_rows, train_data_generator = get_sadiri_batching_for_rankingloss(fold, "train")
    dev_num_rows, dev_data_generator = get_sadiri_batching_for_rankingloss(fold, "dev")
    # train_num_rows, train_data_generator = get_performs_data_generator(fold, "train")
    # dev_num_rows, dev_data_generator = get_performs_data_generator(fold, "dev")

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
            # train_negatives=dataset.output["negatives"],  # Explicit negatives
            # validation_negatives=dev_dataset.output["negatives"],  # Explicit validation negatives
            epochs=epochs,    
            batch_size=batch_size,
            loss=losses.MultipleNegativesSymmetricRankingLoss,
            # loss=losses.TripletLoss,
            learning_rate=0.0005,
            early_stopping_threshold=0.001,
            early_stopping_patience=5,
            eval_strategy='steps',
            logging_strategy='steps',
            save_strategy='steps',
            logging_steps=200,
            save_steps=600,
            eval_steps=600,
            # save_total_limit=3,
            resume_from_checkpoint=False,
            overwrite_output_dir=True,
            accelerator_config={
                "dispatch_batches": False,
            },
            callbacks=[EpochTrackerCallback()]
        )

        trainer.export_to_disk(output_folder + 'final_model', adapter_only=False)


# output_path = '/mnt/swordfish-pool2/milad/datadreamer-ta2/'
# luar_model_path = '/mnt/swordfish-pool2/milad/rrivera1849'
# fold = "/mnt/swordfish-pool2/milad/hiatus-data/performers-data/tmp-data/*/{split}"

output_path = '../output'
luar_model_path = '../training_source/rrivera1849/LUAR-MUD'
# luar_model_path = '../training_source/LUAR-MUD'
# luar_model_path = 'rrivera1849/LUAR-MUD'
# fold = "../data/sadiri/reddit/{split}"
fold = "/data/araghavan/HIATUS/datadreamer-ta2/data/ta2_jan_2025_trian_data/hard_batching_dataset_luaremb_wo_ao3_filtered/{split}_sadiri_hard_batches_v001.jsonl"
# fold = "../data/ta2_jan_2025_trian_data/{split}_sadiri_processed_with_embeddings_wo_ao3_filtered.jsonl"
model_op_dir = os.path.join(output_path, 'sadiri_hardbatching_rankingloss_wo_ao3_filtered_model_v001')
# model_op_dir = os.path.join(output_path, 'sadiri_random_batch_creation_model_v3_10epoch')


train_datadreamer_ta2__on_performers_data_with_hardbatch(
    fold=fold, 
    output_folder=model_op_dir, 
    used_loss='MultipleNegativesSymmetricRankingLoss', # dummy
    luar_model_path=luar_model_path, 
    batch_size=32, 
    epochs=3
)