from sadiri_training import get_performs_data_generator, EpochTrackerCallback, LoraConfig, get_luar_trainer, DataSource, DataDreamer, losses, get_data_generator_for_k_fold_hrs
from SupContrastLoss import MultiPosConLoss
from RankingLoss import MultipleNegativesSymmetricRankingLoss
import os

def train_datadreamer_ta2_k_fold_hrs(fold, output_folder, used_loss, luar_model_path='./rrivera1849', batch_size=128, epochs=25):
    print("Performing training on ta2 with hrs data")
    train_num_rows, train_data_generator = get_data_generator_for_k_fold_hrs(fold, "", "train")
    dev_num_rows, dev_data_generator = get_data_generator_for_k_fold_hrs(fold,  "", "dev")

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
            force=False, #so we can resume training
        )

        #loss = SupConLoss if used_loss=='SupConLoss' else losses.MultipleNegativesSymmetricRankingLoss
        loss = MultiPosConLoss if used_loss=='SupConLoss' else losses.MultipleNegativesSymmetricRankingLoss

        if used_loss == 'SupConLoss':
            trainer.train_with_labeled_pairs(
                train_anchors=dataset.output["anchors"],
                train_others= dataset.output["others"],
                train_labels=dataset.output["labels"],
                validation_anchors=dev_dataset.output["anchors"],
                validation_others=dev_dataset.output["others"],
                validation_labels=dev_dataset.output["labels"],
                epochs=epochs,    
                batch_size=batch_size,
                logging_steps=0.2,
                loss=loss,
                learning_rate=0.0005,
                early_stopping_threshold=0.001,
                early_stopping_patience=5,
                accelerator_config={
                    "dispatch_batches": False,
                },
                callbacks=[EpochTrackerCallback()]
            )
        else:
            trainer.train_with_positive_pairs(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                validation_anchors=dev_dataset.output["anchors"],
                validation_positives=dev_dataset.output["positives"],
                epochs=epochs,    
                batch_size=batch_size,
                loss=loss,
                learning_rate=0.0005,
                early_stopping_threshold=0.001,
                early_stopping_patience=5,
                overwrite_output_dir=True,
                accelerator_config={
                    "dispatch_batches": False,
                },
                callbacks=[EpochTrackerCallback()]
            )

        trainer.export_to_disk(output_folder + 'final_model', adapter_only=False)


# output_path = '/burg/old_dsi/users/ma4608/ajay-ta2-system/output'
# #luar_model_path = '/burg/old_dsi/users/ma4608/ajay-ta2-system/training_source/rrivera1849'
# luar_model_path= '/burg/old_dsi/users/ma4608/ajay-ta2-system/output/original_ta2_performers_data_model_50kfinal_model'
# fold = "/burg/old_dsi/users/ma4608/hiatus_data/hrs_data_combined/TA2/{split}"

output_path = '../output'
luar_model_path = '../output/sadiri_random_batch_creation_model_v4final_model'
# luar_model_path = "../training_source/rrivera1849/LUAR-MUD"
fold = "../data/hrs/english_TA2_p1_and_p2_{split}_20240207"
model_op_dir = os.path.join(output_path, 'combined_hrs_tuned_v5')

# train_datadreamer_ta2_k_fold_hrs(
#     fold=fold, 
#     output_folder=model_op_dir, 
#     used_loss='MultipleNegativesSymmetricRankingLoss',
#     luar_model_path=luar_model_path, 
#     batch_size=64, 
#     epochs=25
# )