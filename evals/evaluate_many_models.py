# this file will read in a path to a file that contains a json list of paths to models, and a path to the tokenizer config. It will then 
# evaluate the models on the tasks. [blimp, ewok] and also a custom dataset path for the synthetic data/real eval data... 
# it will automically load the models (for all epochs) in that model dir and evaluate them.... 

"""
Cursor Prompt was: 
Please help me get started with @evaluate_many_models.py , it should read in a path to file like this @experiment1_evals.json . In the end the goal is to go over all the models. the respective epoch subfolders and then runs the evals for the tasks for all models and the tasks for only for the the specific model on all models in (for each epoch). All the numbers/the scores should be saved to an output json file that is named same as the input file + an appended _result.json. Please get me started with reading in the json input file and the output file logic and looping over the different models and their epoch folders. (each epoch folder has the name epoch_i), Please add placeholder function calls for eval_ewok, eval_blimp and eval_dataset, that each take a path to a model dir (one with a epoch) already and then return a numbers or json object as a result. But please only make them placeholder functions for now.  
"""


import os
import json
import argparse
from pathlib import Path
import logging
import re
import subprocess
import tempfile
import sys
from lm_eval import simple_evaluate
from datasets import Dataset
from transformers import AutoTokenizer
from data.data_processing import tokenize_dataset, group_token_dataset, prepare_dataloader
from accelerate import Accelerator
import torch
import math
from tqdm import tqdm
import multiprocessing
import pickle
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def eval_blimp(model_path, tokenizer_path):
    """
    Function for evaluating a model on the blimp dataset
    
    Args:
        model_path (str): Path to the model directory for a specific epoch
        tokenizer_path (str): Path to the tokenizer
        
    Returns:
        dict: Dictionary containing evaluation results with accuracy and stderr
    """
    logger.info(f"Evaluating BLIMP for model at: {model_path}")
    
    # Run the evaluation
    logger.info("Running BLIMP evaluation...")
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},backend=causal,tokenizer={tokenizer_path}",
        tasks=["blimp_filtered", "blimp_supplement"],
        device="cuda:0",
        batch_size=128,
        log_samples=False,
    )
    
    # Extract the overall blimp scores
    blimp_results = {}
    
    if "groups" in results:
        for task in ["blimp_filtered", "blimp_supplement"]:
            if task in results["groups"]:
                acc = results["groups"][task]["acc,none"] if "acc,none" in results["groups"][task] else None
                acc_std = results["groups"][task]["acc_stderr,none"] if "acc_stderr,none" in results["groups"][task] else None
                logger.info(f"{task} evaluation completed. Accuracy: {acc}, Stderr: {acc_std}")
                blimp_results[task] = {"accuracy": acc, "stderr": acc_std, "all_results": results["results"]}
            else:
                logger.error(f"Could not find {task} results in the evaluation output")
                blimp_results[task] = {"accuracy": 0.0, "stderr": 0.0}
    else:
        logger.error("Could not find any BLIMP results in the evaluation output")
        blimp_results = {
            "blimp_filtered": {"accuracy": 0.0, "stderr": 0.0},
            "blimp_supplement": {"accuracy": 0.0, "stderr": 0.0}
        }
    del results
    torch.cuda.empty_cache()
    return blimp_results

def eval_ewok(model_path, tokenizer_path):
    """
    Function for evaluating a model on the ewok dataset
    
    Args:
        model_path (str): Path to the model directory for a specific epoch
        
    Returns:
        dict: Dictionary containing evaluation results with accuracy and stderr
    """
    logger.info(f"Evaluating EWOK for model at: {model_path}")
    

    # Run the evaluation
    logger.info("Running EWOK evaluation...")
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},backend=causal,tokenizer={tokenizer_path}",
        tasks=["ewok_filtered"],
        device="cuda:0",
        batch_size=512,
        log_samples=False,
        #output_path=output_path
    )
    
    # Extract the overall ewok_filtered score
    if "groups" in results and "ewok_filtered" in results["groups"]:
            acc = results["groups"]["ewok_filtered"]["acc,none"] if "acc,none" in results["groups"]["ewok_filtered"] else None
            acc_std = results["groups"]["ewok_filtered"]["acc_stderr,none"] if "acc_stderr,none" in results["groups"]["ewok_filtered"] else None
            torch.cuda.empty_cache()
            logger.info(f"EWOK evaluation completed. Accuracy: {acc}, Stderr: {acc_std}")
            res = {"accuracy": acc, "stderr": acc_std, "all_results": results["results"]}
            del results
            return res
    else:
        logger.error("Could not find ewok_filtered results in the evaluation output")
        del results
        torch.cuda.empty_cache()
        return {"accuracy": 0.0, "stderr": 0.0}


def eval_dataset_process(model_path, dataset_path, dataset_name, tokenizer_path, args_dict, return_dict):
    """
    Separate process function for evaluating a model on a custom dataset
    
    Args:
        model_path (str): Path to the model directory for a specific epoch
        dataset_path (str): Path to the dataset
        dataset_name (str): Name of the dataset
        tokenizer_path (str): Path to the tokenizer
        args_dict (dict): Dictionary containing arguments for dataset processing
        return_dict (dict): Dictionary to store the result
    """
    try:
        # Create a new accelerator in this process
        accelerator = Accelerator()
        
        # Convert args_dict back to an object with attributes
        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        args = Args(**args_dict)
        
        # Load and prepare the dataset
        logger.info(f"Loading and preparing dataset {dataset_name} for model at: {model_path}")
        dataset = Dataset.load_from_disk(dataset_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        tokenized_datasets = tokenize_dataset(tokenizer, dataset, accelerator, args)
        grouped_datasets = group_token_dataset(tokenizer, tokenized_datasets, accelerator, args)
        eval_dataloader = prepare_dataloader(grouped_datasets, args, accelerator, shuffle=False)
        
        # Load the model from path
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            config=None,
            torch_dtype="auto",
        )
        torch.cuda.empty_cache()
        # Set the model to evaluation mode
        model.eval()
        
        # Prepare the model
        model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
        
        # Evaluate the model
        losses = []
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_dataloader), desc=f"Evaluating on {dataset_name}", total=len(eval_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                # Gather losses from all processes if distributed
                batch_size = batch["input_ids"].shape[0]
                losses.append(accelerator.gather_for_metrics(loss.repeat(batch_size)))
        
        # Concatenate all losses
        losses = torch.cat(losses)
        
        # Calculate mean loss and perplexity
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            logger.warning(f"Overflow when computing perplexity for {dataset_name}")
            perplexity = float("inf")
        
        logger.info(f"{dataset_name} evaluation completed. Loss: {eval_loss}, Perplexity: {perplexity}")
        
        # Store results in the shared dictionary
        return_dict["loss"] = eval_loss.item()
        return_dict["perplexity"] = perplexity
        
        # Clean up to free memory
        del model
        del eval_dataloader
        del tokenized_datasets
        del grouped_datasets
        del tokenizer
        del dataset
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error in eval_dataset_process: {str(e)}")
        logger.error(traceback.format_exc())
        return_dict["error"] = str(e)


def eval_dataset(model_path, dataloader, dataset_name, accelerator):
    """
    Function for evaluating a model on a custom dataset - Kept for compatibility
    but now just returns an error message as this is replaced by the process version
    """
    logger.error(f"The direct eval_dataset function is deprecated. Use the process-based version instead.")
    return {"error": "Direct evaluation not supported. Use process-based evaluation."}


def get_tokenized_grouped_dataset(dataset_path, tokenizer_path, accelerator, args):
    """
    Get the tokenized grouped dataset
    """
    dataset = Dataset.load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokenized_datasets = tokenize_dataset(tokenizer, dataset, accelerator, args)
    grouped_datasets = group_token_dataset(tokenizer, tokenized_datasets, accelerator, args)
    
    eval_dataloader = prepare_dataloader(grouped_datasets, args, accelerator, shuffle=False)
    
    return eval_dataloader


def get_epoch_dirs(model_path, mode="epoch", min_step=0):
    """
    Get all epoch/step directories in the given model path
    
    Args:
        model_path (str): Path to the model directory
        mode (str): Either "epoch" or "step" to determine directory format
        min_step (int): Minimum step number to include (only applies in step mode)
        
    Returns:
        list: List of paths to epoch/step directories
    """
    # Check if the path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return []
    
    # Find all epoch/step directories
    dirs = []
    prefix = "epoch_" if mode == "epoch" else "step_"
    
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            try:
                num = int(os.path.basename(item).split("_")[1])
                if mode == "step" and num < min_step:
                    continue
                dirs.append(item_path)
            except (ValueError, IndexError):
                logger.warning(f"Skipping invalid directory format: {item}")
    
    # Sort by number
    dirs.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    logger.info(f"Found {len(dirs)} {mode} directories for model: {model_path}")
    return dirs

def evaluate_models(config_path, accelerator, args):
    """
    Evaluate all models specified in the config file
    
    Args:
        config_path (str): Path to the config file
        accelerator: The accelerator instance
        args: Command line arguments
    """
    # Load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract key information
    experiment_name = config.get("experiment_name", "unnamed_experiment")
    tokenizer_path = config.get("tokenizer_path")
    if not tokenizer_path:
        logger.warning("No tokenizer path specified in the config file. This is required for BLIMP and EWOK evaluations.")
    
    tasks_for_all_models = config.get("tasks_for_all_models", [])
    models = config.get("models", [])
    
    # Debug print of models to be evaluated
    logger.info("=" * 80)
    logger.info(f"Starting evaluation for experiment: {experiment_name}")
    logger.info(f"Mode: {args.mode}" + (f" (min_step: {args.min_step})" if args.mode == "step" else ""))
    logger.info(f"Common tasks for all models: {tasks_for_all_models}")
    logger.info("\nModels to be evaluated:")
    for i, model_config in enumerate(models, 1):
        model_name = model_config.get("model_name", "unnamed")
        model_path = model_config.get("model_path", "no_path")
        specific_datasets = model_config.get("model_specific_datasets", [])
        logger.info(f"\n{i}. Model: {model_name}")
        logger.info(f"   Path: {model_path}")
        if specific_datasets:
            logger.info("   Model-specific datasets:")
            for dataset in specific_datasets:
                logger.info(f"   - {dataset.get('dataset_name', 'unnamed')}: {dataset.get('dataset_path', 'no_path')}")
    logger.info("=" * 80 + "\n")
    
    # Prepare the results dictionary
    results = {
        "experiment_name": experiment_name,
        "tokenizer_path": tokenizer_path,
        "evaluated_models": []
    }
    
    # Convert args to dictionary for passing to subprocess
    args_dict = {key: getattr(args, key) for key in dir(args) if not key.startswith('_')}
    
    # Process each model
    for model_config in models:
        torch.cuda.empty_cache()
        model_name = model_config.get("model_name")
        model_path = model_config.get("model_path")
        model_specific_datasets = model_config.get("model_specific_datasets", [])
        
        logger.info(f"Processing model: {model_name} at {model_path}")
        
        # Get all epoch/step directories
        epoch_dirs = get_epoch_dirs(model_path, mode=args.mode, min_step=args.min_step)
        if not epoch_dirs:
            logger.warning(f"No {args.mode} directories found for model: {model_name}")
            continue
            
        # Print number of epochs/steps to be evaluated
        logger.info(f"Found {len(epoch_dirs)} {args.mode}s to evaluate for model {model_name}")
        if args.mode == "step":
            steps = [int(os.path.basename(d).split("_")[1]) for d in epoch_dirs]
            logger.info(f"Step range: {min(steps)} to {max(steps)}")
        
        # Evaluate each epoch/step
        model_results = {
            "model_name": model_name,
            "model_path": model_path,
            f"{args.mode}s": []
        }
        
        for dir_path in epoch_dirs:
            num = int(os.path.basename(dir_path).split("_")[1])
            logger.info(f"Evaluating {args.mode} {num} for model {model_name}")
            
            epoch_result = {
                args.mode: num,
                f"{args.mode}_path": dir_path,
                "task_results": {}
            }

            # Evaluate common tasks for all models
            for task in tasks_for_all_models:
                torch.cuda.empty_cache()
                if task.lower() == "ewok":
                    torch.cuda.empty_cache()
                    epoch_result["task_results"]["ewok"] = eval_ewok(dir_path, tokenizer_path)
                    torch.cuda.empty_cache()
                elif task.lower() == "blimp":
                    torch.cuda.empty_cache()
                    epoch_result["task_results"]["blimp"] = eval_blimp(dir_path, tokenizer_path)
                    torch.cuda.empty_cache()

            # Evaluate model-specific datasets in separate processes
            for dataset_config in model_specific_datasets:
                dataset_name = dataset_config.get("dataset_name")
                dataset_path = dataset_config.get("dataset_path")
                
                if dataset_name and dataset_path:
                    logger.info(f"Starting separate process for evaluating {dataset_name}")
                    
                    # Create a manager for sharing data between processes
                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()
                    
                    # Create and start the process
                    process = multiprocessing.Process(
                        target=eval_dataset_process,
                        args=(dir_path, dataset_path, dataset_name, tokenizer_path, args_dict, return_dict)
                    )
                    
                    process.start()
                    process.join()  # Wait for the process to complete
                    
                    # Check if there was an error
                    if "error" in return_dict:
                        logger.error(f"Error in subprocess for {dataset_name}: {return_dict['error']}")
                        epoch_result["task_results"][dataset_name] = {"error": return_dict["error"]}
                    else:
                        # Retrieve the results
                        epoch_result["task_results"][dataset_name] = {
                            "loss": return_dict.get("loss"),
                            "perplexity": return_dict.get("perplexity")
                        }
                        print(epoch_result["task_results"][dataset_name])
                        print(epoch_result["task_results"])
                    
                    # Force cleanup to ensure resources are released
                    manager.shutdown()
                    del return_dict
                    del manager
                    torch.cuda.empty_cache()
            
            
            
            model_results[f"{args.mode}s"].append(epoch_result)
            
            # Save intermediate results after each epoch to avoid losing progress on crashes
            intermediate_results = {
                "experiment_name": experiment_name,
                "tokenizer_path": tokenizer_path,
                "evaluated_models": results["evaluated_models"] + [model_results]
            }
            
            intermediate_output_path = config_path.replace(".json", f"_results_partial_{model_name}_epoch_{num}.json")
            with open(intermediate_output_path, 'w') as f:
                json.dump(intermediate_results, f, indent=4)
            
            logger.info(f"Saved intermediate results to: {intermediate_output_path}")
        
        results["evaluated_models"].append(model_results)
    
    # Construct output file path
    output_path = config_path.replace(".json", "_results.json")
    
    # Save results to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation complete. Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple models on various tasks")
    parser.add_argument("--config_path", help="Path to the config file containing model and task information", default="evals/eval_jobs_results/experiment1_evals.json")
    parser.add_argument("--cache_dataset_dir", help="Path to the directory to cache the tokenized grouped dataset", default=None)
    parser.add_argument("--seq_len", help="Sequence length", default=1024)
    parser.add_argument("--preprocessing_num_workers", help="Number of preprocessing workers", default=6)
    parser.add_argument("--group_texts_batch_size", help="Batch size for grouping texts", default=1000)
    parser.add_argument("--batch_size", help="Batch size", default=32)
    parser.add_argument("--dataloader_num_workers", help="Number of dataloader workers", default=6)
    parser.add_argument("--mode", help="Whether to use 'epoch' or 'step' based directories", choices=["epoch", "step"], default="epoch")
    parser.add_argument("--min_step", help="Minimum step number to include (only applies in step mode)", type=int, default=0)
    args = parser.parse_args()
    
    # Make sure we're using the spawn method for multiprocessing
    # This is important for proper CUDA memory management
    multiprocessing.set_start_method('spawn', force=True)
    
    accelerator = Accelerator()
    
    evaluate_models(args.config_path, accelerator, args)

if __name__ == "__main__":
    main() 