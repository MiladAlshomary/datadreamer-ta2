import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from trainer import *
from SupContrastLoss import MultiPosConLoss
from RankingLoss import MultipleNegativesSymmetricRankingLoss


def train_datadreamer_ta2(fold, output_folder, used_loss, luar_model_path='./rrivera1849', batch_size=128, epochs=25):
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
            device='cuda:0',
            dtype="bfloat16",
            force=False,
        )

        #loss = SupConLoss if used_loss=='SupConLoss' else losses.MultipleNegativesSymmetricRankingLoss
        loss = MultiPosConLoss if used_loss=='SupConLoss' else losses.MultipleNegativesSymmetricRankingLoss

        if used_loss == 'SupConLoss':
            print('Using SupConLoss =======================')
            trainer.train_with_labeled_pairs(
                train_anchors=dataset.output["anchors"],
                train_others= dataset.output["others"],
                train_labels=dataset.output["labels"],
                validation_anchors=dev_dataset.output["anchors"],
                validation_others=dev_dataset.output["others"],
                validation_labels=dev_dataset.output["labels"],
                epochs=epochs,
                batch_size=batch_size,
                eval_strategy='epoch',
                logging_strategy='epoch',
                save_strategy='epoch',
                #save_steps=100,
                save_total_limit=3,
                #logging_steps=100,
                loss=loss,
                learning_rate=0.0001,
                early_stopping_threshold=0.001,
                early_stopping_patience=3,
                accelerator_config={
                    "dispatch_batches": False,
                },
                resume_from_checkpoint=True,
                overwrite_output_dir=False,
                callbacks=[EpochTrackerCallback()]
            )
        else:
            trainer.train_with_positive_pairs(
                train_anchors=dataset.output["anchors"],
                train_positives=dataset.output["positives"],
                validation_anchors=dev_dataset.output["anchors"],
                validation_positives=dev_dataset.output["positives"],
                gradient_checkpointing=True,
                epochs=epochs,    
                batch_size=batch_size,
                loss=loss,
                learning_rate=0.0001,
                early_stopping_threshold=0.001,
                early_stopping_patience=5,
                accelerator_config={
                    "dispatch_batches": False,
                },
                callbacks=[EpochTrackerCallback()]
            )


# output_path = '/mnt/swordfish-pool2/milad/datadreamer-ta2/'
# luar_model_path = '/mnt/swordfish-pool2/milad/rrivera1849'
output_path = '/burg/dsi/users/ma4608/ajay-ta2-system/output'
luar_model_path = '/burg/dsi/users/ma4608/ajay-ta2-system/training_source/rrivera1849'

fold = "/burg/dsi/users/ma4608/ajay-ta2-system/training_source/data/train-test-dev-split/{split}/official-query-candidate-format/"

train_num_rows, train_data_generator = get_data_generator_for_supervised_contrastive_learning(fold, "cross_genre_all", "train", split_percent=ast.literal_eval("None"))
dev_num_rows, dev_data_generator = get_data_generator_for_supervised_contrastive_learning(fold, "cross_genre_all", "dev", split_percent=ast.literal_eval("None"))

train_datadreamer_ta2(fold, output_path + '/supcon_ta2_model', 'SupConLoss', luar_model_path=luar_model_path, batch_size=128, epochs=25)
