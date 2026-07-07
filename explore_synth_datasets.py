import argparse
from datasets import load_from_disk
import random


def main():
    parser = argparse.ArgumentParser(description="Explore a synthetic DatasetDict.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the combined synthetic dataset (output of combine_synthetic_datasets.py)")
    args = parser.parse_args()

    # Load the dataset dict
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # Try to find the train split (synthetic_train)
    if "synthetic_train" in dataset:
        train_ds = dataset["synthetic_train"]
    elif "train" in dataset:
        train_ds = dataset["train"]
    else:
        raise ValueError("No train split found in the dataset. Available splits: " + str(list(dataset.keys())))

    print("\nFirst 10 rows of the train split (text only):")
    for i in range(min(10, len(train_ds))):
        print(train_ds[i]["text"])
        print("\n\n###########\n\n")

    print("\n10 random rows from the train split (text only):")
    if len(train_ds) > 10:
        random_indices = random.sample(range(len(train_ds)), 10)
    else:
        random_indices = list(range(len(train_ds)))
    for idx in random_indices:
        print(train_ds[idx]["text"])
        print("\n\n###########\n\n")
    
    # print the total number of rows and other info about the dataset
    print(dataset)


if __name__ == "__main__":
    main()
