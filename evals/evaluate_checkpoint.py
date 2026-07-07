import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import json
import argparse
import logging
import torch
import math
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from accelerate import Accelerator
from data.data_processing import tokenize_dataset, group_token_dataset, prepare_dataloader
import tempfile
import shutil
from contextlib import contextmanager
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def temporary_cache_dir():
    """
    Context manager that creates a temporary directory and automatically cleans it up
    when the context is exited.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary cache directory: {temp_dir}")

def evaluate_perplexity(model_path, dataset_path, output_dir, args, save_predictions=False):
    """
    Evaluate a model's perplexity on a dataset
    
    Args:
        model_path (str): Path to the model directory
        dataset_path (str): Path to the dataset
        output_dir (str): Directory to save results
        args: Command line arguments
    """
    # check for the output dir and create it if it doesn't exist
    print(f"Initial output directory: {output_dir}")
    print(f"Model path: {model_path}")
    # Get the parent directory of model_path and then get its basename
    model_checkpoint_name = os.path.basename(os.path.dirname(model_path))
    print(f"Model checkpoint name: {model_checkpoint_name}")
    output_dir = os.path.join(output_dir, model_checkpoint_name)
    # now also add the directory of the directory of the dataset used
    dataset_split = os.path.basename(dataset_path)
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    dataset_description = dataset_name + "-" + dataset_split
    output_dir = os.path.join(output_dir, dataset_description)
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # create temp cache dir that automatically gets deleted after the evaluation is done
    with temporary_cache_dir() as short_term_cache_dir:
        try:
            # Create accelerator
            accelerator = Accelerator()
            
            # Load and prepare the dataset
            logger.info(f"Loading dataset from: {dataset_path}")
            dataset = Dataset.load_from_disk(dataset_path)
            
            # Load tokenizer from model path
            logger.info(f"Loading tokenizer from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set cache directory in args
            args.cache_dataset_dir = short_term_cache_dir
            
            # Process dataset
            logger.info("Tokenizing and preparing dataset...")
            tokenized_datasets = tokenize_dataset(tokenizer, dataset, accelerator, args)
            grouped_datasets = group_token_dataset(tokenizer, tokenized_datasets, accelerator, args)
            eval_dataloader = prepare_dataloader(grouped_datasets, args, accelerator, shuffle=False)
            
            # Load model
            logger.info(f"Loading model from: {model_path}")
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                config=None,
                torch_dtype="auto",
            )
            torch.cuda.empty_cache()
            
            # Set model to evaluation mode
            model.eval()
            
            # Prepare model and dataloader
            model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
            
            # Evaluate model
            logger.info("Computing perplexity...")
            losses = []
            with torch.no_grad():
                for step, batch in tqdm(enumerate(eval_dataloader), desc="Evaluating", total=len(eval_dataloader)):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # Gather losses from all processes if distributed
                    batch_size = batch["input_ids"].shape[0]
                    losses.append(accelerator.gather_for_metrics(loss.repeat(batch_size)))
            
            # Concatenate all losses
            losses = torch.cat(losses)
            
            if save_predictions:
                losses_np = losses.cpu().numpy()  # Convert to numpy array
                np.save(os.path.join(output_dir, "per_batch_losses.npy"), losses_np)
                logger.info(f"Saved per-batch losses to {os.path.join(output_dir, 'per_batch_losses.npy')}")
                print(f"Saved per-batch losses to {os.path.join(output_dir, 'per_batch_losses.npy')}")
            
            # Calculate mean loss and perplexity
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                logger.warning("Overflow when computing perplexity")
                perplexity = float("inf")
            
            logger.info(f"Evaluation completed. Loss: {eval_loss}, Perplexity: {perplexity}")
            
            # Save results
            output_file = os.path.join(output_dir, "best_temperature_report.txt")
            
            with open(output_file, 'w') as f:
                f.write(f"Model Path: {model_path}\n")
                f.write(f"Dataset Path: {dataset_path}\n")
                f.write(f"Loss: {eval_loss.item()}\n")
                f.write(f"Perplexity: {perplexity}\n")
            
            logger.info(f"Results saved to: {output_file}")
            
            # Clean up
            del model
            del eval_dataloader
            del tokenized_datasets
            del grouped_datasets
            del tokenizer
            del dataset
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint's perplexity on a dataset")
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--data_path", required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    
    # Add data processing arguments with defaults
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for tokenization")
    parser.add_argument("--preprocessing_num_workers", type=int, default=6, help="Number of workers for preprocessing")
    parser.add_argument("--group_texts_batch_size", type=int, default=1000, help="Batch size for grouping texts")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--dataloader_num_workers", type=int, default=6, help="Number of workers for dataloader")    
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions")

    args = parser.parse_args()

    if args.save_predictions:
        print("Will be saving predictions (flag set)")
    else:
        print("Will not be saving predictions (flag not set)")
    
    evaluate_perplexity(args.model_path, args.data_path, args.output_dir, args, args.save_predictions)

if __name__ == "__main__":
    main()
