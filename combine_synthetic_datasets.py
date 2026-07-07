import os
import argparse
from datasets import Dataset, DatasetDict, concatenate_datasets
import shutil

def main(args):
    # Get all process directories
    process_dirs = [d for d in os.listdir(args.input_dir) if d.startswith("process_")]
    process_dirs.sort(key=lambda x: int(x.split("_")[1]))  # Sort by process number
    
    print(f"Found {len(process_dirs)} process directories")
    
    # Load all datasets
    datasets = []
    for process_dir in process_dirs:
        full_path = os.path.join(args.input_dir, process_dir)
        print(f"Loading dataset from {full_path}")
        dataset = Dataset.load_from_disk(full_path)
        datasets.append(dataset)
    
    # Concatenate all datasets
    print("Concatenating datasets...")
    combined_dataset = concatenate_datasets(datasets)
    print(f"Combined dataset size: {len(combined_dataset)}")
    
    # Split into train and test
    print("Splitting into train and test...")
    split_dict = combined_dataset.train_test_split(test_size=0.05)
    
    # Create final dataset dict
    final_dataset = DatasetDict({
        "synthetic_train": split_dict["train"],
        "synthetic_test": split_dict["test"]
    })
    
    # Save the final dataset
    print(f"Saving final dataset to {args.output_dir}")
    final_dataset.save_to_disk(args.output_dir, num_proc=6)
    
    # Clean up process directories
    if args.cleanup:
        print("Cleaning up process directories...")
        for process_dir in process_dirs:
            full_path = os.path.join(args.input_dir, process_dir)
            shutil.rmtree(full_path)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing process-specific datasets")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the combined dataset")
    parser.add_argument("--cleanup", action="store_true", help="Whether to remove process directories after combining")
    args = parser.parse_args()
    
    main(args) 