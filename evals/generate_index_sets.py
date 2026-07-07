# this script should be run once before the for the bootstrap subsampling procedure is executed on all models, 
# it generates index sets for the subsampling procecure to make sure to have a paired bootstrap. 

import sys
import pickle
import argparse
import numpy as np
import os

sys.path.append("/cluster/home/janulm/thesis/ms-thesis/evals")

from subsample_predictions import (
    extract_ground_truth,
    tasks
)



def generate_index_sets_others(num_rows_eval_dataset, num_samples, seed=42):
    np.random.seed(seed)
    index_set = np.random.randint(0, num_rows_eval_dataset, size=(num_samples, num_rows_eval_dataset))
    return index_set


def generate_index_sets_acc(ground_truth, num_samples, seed=42):
    np.random.seed(seed)
    per_file_index_sets = []
    for file in ground_truth.keys():
        num_rows = len(ground_truth[file]["solutions"])
        index_set = np.random.randint(0, num_rows, size=(num_samples, num_rows))
        per_file_index_sets.append(index_set)
    return per_file_index_sets




def main(args): 

    print("Loading ground truth dict:")
    ground_truth_dict = extract_ground_truth(args.ground_truth_path)
    print("Done loading")

    print("Generating index sets: ")

    # check the output path: 
    if not os.path.exists(args.store_path):
        os.makedirs(args.store_path)

    
    for task_name in tasks:

        index_set_type = tasks[task_name]["index_type"]

        if index_set_type == "acc":
            index_set = generate_index_sets_acc(ground_truth_dict[task_name], args.subsample_size,42)
            # pickle this python object
            with open(os.path.join(args.store_path, f"{task_name}_index_set.pkl"), "wb") as f:
                pickle.dump(index_set, f)



        elif index_set_type == "reading_eye_tracking": 
            index_set = generate_index_sets_others(num_rows_eval_dataset=1726, num_samples=args.subsample_size, seed=42)
            with open(os.path.join(args.store_path, f"{task_name}_index_set.pkl"), "wb") as f:
                pickle.dump(index_set, f)

        elif index_set_type == "perplexity": 
            index_set = generate_index_sets_others(num_rows_eval_dataset=10518, num_samples=args.subsample_size, seed=42)
            with open(os.path.join(args.store_path, f"{task_name}_index_set.pkl"), "wb") as f:
                pickle.dump(index_set, f)
        else: 
            raise ValueError("Wrong index set type")

    print("Done generating index sets")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample predictions for a single model checkpoint.")

    parser.add_argument("--subsample_size", type=int, default=1000, help="Size of the subsample.")
    parser.add_argument("--store_path", type=str, default="/cluster/work/cotterell/janulm/thesis_data/index_sets/", help="Path to store the index sets.")

    parser.add_argument("--ground_truth_path", type=str, default="/cluster/home/janulm/thesis/evaluation-pipeline-2025/evaluation_data/full_eval/", help="Path to ground truth eval data.")
    
    args = parser.parse_args()
    main(args)