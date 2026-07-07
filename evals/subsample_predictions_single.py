import os
import argparse
import numpy as np
import pickle

from subsample_predictions import (
    extract_ground_truth,
    load_checkpoint_pred,
    do_subsamples_accuracy_task,
    do_subsamples_perplexity_task,
    do_subsamples_reading_eye_task,
    tasks
)

def main(args):
    print("Loading ground truth dict:")
    ground_truth_dict = extract_ground_truth(args.ground_truth_path)
    print("Done loading")

    checkpoint_path = args.model_path

    index_set_path = args.index_set_path
    # load the index sets for each task: 
    index_set_dict = {}
    for task in tasks:
        path_to_index_set = os.path.join(index_set_path, f"{task}_index_set.pkl")
        
        print("path_to_index_set: ", path_to_index_set)
        with open(path_to_index_set, "rb") as f:
            index_set_dict[task] = pickle.load(f)
    
    print("Done loading index sets")

    checkpoint_dir = os.path.basename(os.path.normpath(checkpoint_path))
    print("Now doing checkpoint:", checkpoint_path)

    run_all = len(args.tasks) == 0
    tasks_to_run = set(args.tasks)

    if run_all or "blimp" in tasks_to_run:
        #try:
        data = load_checkpoint_pred(checkpoint_path, "blimp")
        gt = ground_truth_dict["blimp"]
        save_path = os.path.join(checkpoint_path, tasks["blimp"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
        do_subsamples_accuracy_task(data, gt, save_path, index_set_dict["blimp"], "blimp")
        print("Done with blimp")
        #except Exception as e:
        #    print("Skipping blimp:", str(e))

    if run_all or "blimp_supplement" in tasks_to_run:
        try:
            data = load_checkpoint_pred(checkpoint_path, "blimp_supplement")
            gt = ground_truth_dict["blimp_supplement"]
            save_path = os.path.join(checkpoint_path, tasks["blimp_supplement"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(data, gt, save_path, index_set_dict["blimp_supplement"], "blimp_supplement")
            print("Done with blimp_supplement")
        except Exception as e:
            print("Skipping blimp_supplement:", str(e))

    if run_all or "entity_tracking" in tasks_to_run:
        try:
            data = load_checkpoint_pred(checkpoint_path, "entity_tracking")
            gt = ground_truth_dict["entity_tracking"]
            save_path = os.path.join(checkpoint_path, tasks["entity_tracking"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(data, gt, save_path, index_set_dict["entity_tracking"], "entity_tracking")
            print("Done with entity_tracking")
        except Exception as e:
            print("Skipping entity_tracking:", str(e))

    if run_all or "ewok" in tasks_to_run:
        try:
            data = load_checkpoint_pred(checkpoint_path, "ewok")
            gt = ground_truth_dict["ewok"]
            save_path = os.path.join(checkpoint_path, tasks["ewok"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(data, gt, save_path, index_set_dict["ewok"], "ewok")
            print("Done with ewok")
        except Exception as e:
            print("Skipping ewok:", str(e))

    if run_all or "wug" in tasks_to_run:
        try:
            data = load_checkpoint_pred(checkpoint_path, "wug")
            gt = ground_truth_dict["wug"]
            save_path = os.path.join(checkpoint_path, tasks["wug"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(data, gt, save_path, index_set_dict["wug"], "wug")
            print("Done with wug")
        except Exception as e:
            print("Skipping wug:", str(e))

    if run_all or "perplexity" in tasks_to_run:
        try:
            data = np.load(os.path.join(checkpoint_path, tasks["perplexity"]["eval_path"].lstrip("/"), "per_batch_losses.npy"))
            save_path = os.path.join(checkpoint_path, tasks["perplexity"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_perplexity_task(data, save_path, index_set_dict["perplexity"], "perplexity")
            print("Done with perplexity")
        except Exception as e:
            print("Skipping perplexity:", str(e))

    if run_all or "reading" in tasks_to_run or "eye_tracking" in tasks_to_run:
        try:
            gt = ground_truth_dict["reading_eye_tracking"]
            load_path = os.path.join(checkpoint_path, tasks["reading_eye_tracking"]["eval_path"].lstrip("/"), "prediction.jsonl")
            save_path = os.path.join(checkpoint_path, tasks["reading_eye_tracking"]["eval_path"].lstrip("/"), "subsampled_scores")
            do_subsamples_reading_eye_task(load_path, gt, save_path, index_set_dict["reading_eye_tracking"], "reading_eye_tracking")
            print("Done with reading_eye_tracking")
        except Exception as e:
            print("Skipping reading_eye_tracking:", str(e))

    print("✅ Done with:", checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample predictions for a single model checkpoint.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to single checkpoint directory.")
    parser.add_argument("--index_set_path", type=str, required=True, help="Path to index set.")
    parser.add_argument("--ground_truth_path", type=str, required=True, help="Path to ground truth eval data.")
    parser.add_argument("--tasks", nargs="*", default=[], help="Specific tasks to run. If none specified, runs all.")
    args = parser.parse_args()
    main(args)