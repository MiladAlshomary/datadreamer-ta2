import os
import shutil
import json
import argparse
from train_k_fold_hrs import *
import subprocess
from datetime import datetime

def parse_evaluation_metrics(stdout_log_path):
    """
    Parse evaluation metrics from the stdout log file.

    Args:
        stdout_log_path: Path to the stdout log file.

    Returns:
        A dictionary with parsed metrics.
    """
    metrics = {
        "Area Under ROC Curve": None,
        "partial Area Under ROC Curve": None,
        "Precision": None,
        "Recall": None,
        "F1": None,
        "Equal Error Rate": None,
        "Detection Cost Function": None,
        "TAR@0.05": None
    }

    with open(stdout_log_path, "r") as f:
        for line in f:
            for key in metrics.keys():
                if line.startswith(key):
                    metrics[key] = float(line.split()[-1])
    return metrics

def aggregate_metrics(all_metrics):
    """
    Aggregate metrics across all folds.

    Args:
        all_metrics: List of metrics dictionaries for each fold.

    Returns:
        A dictionary with the average metrics.
    """
    aggregated = {key: 0.0 for key in all_metrics[0].keys()}
    for metrics in all_metrics:
        for key, value in metrics.items():
            if value is not None:
                aggregated[key] += value

    num_folds = len(all_metrics)
    for key in aggregated.keys():
        aggregated[key] /= num_folds

    return aggregated

def get_current_timestamp():
    """Generate current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_detailed_directory_contents(path):
    """
    Get detailed directory contents including file sizes (like ls -l).

    Args:
        path: Directory path to list.

    Returns:
        A list of file names with their sizes.
    """
    contents = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            contents.append(f"{file_path} - {size} bytes")
    return contents

def run_kfold_evaluation(args):
    """
    Perform k-fold training and evaluation, storing results for reproducibility.

    Args:
        args: Parsed command-line arguments.
    """
    timestamp_in_yyyy_mm_dd_hh_mm_ss = get_current_timestamp()
    args.train_model_output_dir = f"{args.train_model_output_dir}_{timestamp_in_yyyy_mm_dd_hh_mm_ss}"
    args.eval_output_dir = os.path.join(args.eval_output_dir, f"run_{timestamp_in_yyyy_mm_dd_hh_mm_ss}")

    if os.path.exists(args.train_model_output_dir):
        shutil.rmtree(args.train_model_output_dir)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    os.makedirs(args.train_model_output_dir, exist_ok=True)

    fold_metrics = []

    # Save the current working directory
    original_cwd = os.getcwd()

    # Iterate through each fold directory
    for fold_name in sorted(os.listdir(args.train_fold_path)):
        fold_path = os.path.join(args.train_fold_path, fold_name)
        if not os.path.isdir(fold_path):
            print(f"Incorrect Fold Path: {fold_path}")
            continue

        print(f"Starting training for {fold_name}...")

        ## Fold Path: /data/araghavan/HIATUS/datadreamer-ta2/data/hrs/english_crossGenre_092424_kf_fold_splits_qc/crossGenre/fold_1
        ## Dev Path: /data/araghavan/HIATUS/datadreamer-ta2/data/hrs/english_crossGenre_092424_kf_fold_splits_qc/crossGenre/fold_1/dev/TA2/hrs2_09-24-24_english_crossGenre-combined/data
        train_dev_split_dir = os.path.join(fold_path, "{split}", "TA2", "hrs2_09-24-24_english_crossGenre-combined", "data")

        # Train the model for this fold using train and dev splits
        train_output_dir = os.path.join(args.train_model_output_dir, f"model_{fold_name}")
        train_datadreamer_ta2_k_fold_hrs(
            fold=train_dev_split_dir,
            output_folder=train_output_dir,
            used_loss='MultipleNegativesSymmetricRankingLoss',
            luar_model_path=args.train_model_path,
            batch_size=64,
            epochs=40
        )

        # Copy model content to the SIVs checkpoint directory
        os.makedirs(args.eval_trained_model_checkpoint_dir, exist_ok=True)
        model_dir = os.path.join(train_output_dir, "luar-trainer/_model/")
        if os.path.exists(model_dir):
            print("Previous Checkpoints Contents:")
            print("\n".join(get_detailed_directory_contents(args.eval_trained_model_checkpoint_dir)))

            print("Before Copy:")
            print("\n".join(get_detailed_directory_contents(model_dir)))

            shutil.copytree(model_dir, args.eval_trained_model_checkpoint_dir, dirs_exist_ok=True)

            print("After Copy:")
            print("\n".join(get_detailed_directory_contents(args.eval_trained_model_checkpoint_dir)))

        print(f"Training for {fold_name} completed. Starting evaluation...")

        # Change directory to where the main.py script is located
        os.chdir(os.path.dirname(args.eval_main_script_path))

        # Evaluation on the test split
        test_dir = os.path.join(fold_path, "test", "TA2", "hrs2_09-24-24_english_crossGenre-combined")
        test_data_dir = os.path.join(test_dir, "data")
        groundtruth_dir = os.path.join(test_dir, "groundtruth")

        output_fold_dir = os.path.join(args.eval_output_dir, f"output_{fold_name}")
        os.makedirs(output_fold_dir, exist_ok=True)

        command = [
            "python", "-u", args.eval_main_script_path,
            "--input-dir", test_data_dir,
            "--output-dir", output_fold_dir,
            "--ground-truth-dir", groundtruth_dir,
            "--run-id", fold_name,
            "--query-identifier", "authorIDs",
            "--candidate-identifier", "authorSetIDs", "-g",
            "-ta1", args.eval_ta1_approach,
            "-ta2", args.eval_ta2_approach,
            "--language", args.eval_language
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        # Change back to the original working directory
        os.chdir(original_cwd)

        # Store output and error logs for each fold
        stdout_log_path = os.path.join(output_fold_dir, "stdout.log")
        with open(stdout_log_path, "w") as stdout_file:
            stdout_file.write(result.stdout)

        with open(os.path.join(output_fold_dir, "stderr.log"), "w") as stderr_file:
            stderr_file.write(result.stderr)

        if result.returncode == 0:
            print(f"Evaluation for {fold_name} completed successfully.")
            metrics = parse_evaluation_metrics(stdout_log_path)
            fold_metrics.append(metrics)
        else:
            print(f"Evaluation for {fold_name} failed. Check logs in {output_fold_dir}.")

    # Aggregate metrics across all folds
    aggregated_metrics = aggregate_metrics(fold_metrics)

    # Save aggregated results
    aggregated_results_path = os.path.join(args.eval_output_dir, "aggregated_results.json")
    with open(aggregated_results_path, "w") as results_file:
        json.dump({"fold_metrics": fold_metrics, "aggregated_metrics": aggregated_metrics}, results_file, indent=4)

    print(f"K-fold training and evaluation completed. Aggregated results saved to {aggregated_results_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-fold training and evaluation")
    parser.add_argument("--train-model-path", required=True, help="Path to the base LUAR model for training")
    parser.add_argument("--train-fold-path", required=True, help="Path to the k-fold training data with {split} placeholder")
    parser.add_argument("--train-model-output-dir", required=True, help="Directory to store trained model outputs")
    parser.add_argument("--eval-trained-model-checkpoint-dir", required=True, help="Directory to copy trained model checkpoint for evaluation")
    parser.add_argument("--eval-main-script-path", required=True, help="Path to the main.py script for evaluation")
    parser.add_argument("--eval-output-dir", required=True, help="Directory to store evaluation results")
    parser.add_argument("--eval-ta1-approach", default="baseline_luar", help="TA1 approach to use for evaluation")
    parser.add_argument("--eval-ta2-approach", default="baseline", help="TA2 approach to use for evaluation")
    parser.add_argument("--eval-language", default="eng", help="Language to use for evaluation")

    args = parser.parse_args()
    run_kfold_evaluation(args)
