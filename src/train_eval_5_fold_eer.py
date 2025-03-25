import sys
import json
import os
from datetime import datetime
sys.path.insert(0, '../../datadreamer-ta2/')
from author_attribution.primary_pausit_model import PAUSIT_MODEL
from author_attribution.sbert_luar import LuarSentenceTransformer

def get_current_timestamp():
    """Generate current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def log_metrics(output_folder, fold_num, eer, auc):
    """Logs fold metrics to a JSON file."""
    log_file = os.path.join(output_folder, "fold_metrics.json")

    # Create the dictionary entry for the current fold
    fold_metrics = {
        f"fold_{fold_num}": {
            "EER": round(eer, 3),
            "AUC": round(auc, 3)
        }
    }

    # Check if the log file exists
    if os.path.exists(log_file):
        # Load existing data
        with open(log_file, "r") as f:
            metrics_data = json.load(f)
    else:
        # Create a new dictionary if the file doesn't exist
        metrics_data = {}

    # Update metrics for the current fold
    metrics_data.update(fold_metrics)

    # Save the updated metrics back to the file
    with open(log_file, "w") as f:
        json.dump(metrics_data, f, indent=4)

timestamp_in_humread_format = get_current_timestamp()
# Paths for train, dev (validation), and test data
for fold in range(5):
    fold_num = str(fold + 1)
    print(f"In Fold: {fold_num}")

    train_data_path = f'../data/hrs/nikita_direct_splits_5_random_fold_splits_qc/crossGenre/fold_{fold_num}/train/TA2/hrs2_09-24-24_english_crossGenre-combined'
    dev_data_path = f'../data/hrs/nikita_direct_splits_5_random_fold_splits_qc/crossGenre/fold_{fold_num}/dev/TA2/hrs2_09-24-24_english_crossGenre-combined'
    test_data_path = f'../data/hrs/nikita_direct_splits_5_random_fold_splits_qc/crossGenre/fold_{fold_num}/test/TA2/hrs2_09-24-24_english_crossGenre-combined'
    output_path = f'/data/araghavan/HIATUS/datadreamer-ta2/output/hrs_trained_5_folds_random_eer_run_{timestamp_in_humread_format}/model_fold_{fold_num}'

    assert os.path.isdir(train_data_path) and os.path.isdir(dev_data_path) and os.path.isdir(test_data_path), "Nope"
    print(f"Train: {train_data_path}\nDev: {dev_data_path}\nTest: {test_data_path}")

    # Initialize the model with training and dev data
    app = PAUSIT_MODEL(
        train_data_path,
        dev_data_path,
        output_path,
        run_id=f'luar-train-dev-fold-{fold_num}',
        query_identifier='authorIDs',
        candidate_identifier='authorSetIDs',
        ratio=4
    )

    # Train the model with training and dev data
    app.train(training_epochs=40, luar_based=True, evaluation_steps=100, valid_from_training=False, batch_size=48, eer_prefix=f"fold_{fold_num}_{timestamp_in_humread_format}")

    # Evaluate the model on the test data
    eer_metrics = app.predict_and_evaluate(output_path, input_path=test_data_path, with_filtering=False, filter_prcnt=0.4, eer_prefix=f"fold_{fold_num}_{timestamp_in_humread_format}")

    print(f"All EER: {eer_metrics}")
    print(f"Test EER: {eer_metrics[0]:.3f}, Test AUC: {eer_metrics[1]:.3f}")

    # Log metrics to a file
    log_metrics(os.path.dirname(output_path), fold_num, eer_metrics[0], eer_metrics[1])
