# this here seems to be a good start: 

# https://huggingface.co/blog/nroggendorff/train-with-llama-architecture 
# there they construct the llama model and train it on a custom dataset 

# the model needs the following additional tokens: (at least the ones they pass in the example)
# 1. pad (padding)
# 2. bos (begin of sequence)
# 3. eos (end of sequence)

# model architecture I want to use: https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaConfig
# # with the size parameters from GPT2-117M: https://huggingface.co/transformers/v2.2.0/pretrained_models.html

# As a starting point for this script, I use huggingfaces clm_no_trainer example: 
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

from model.load_model import load_babyllama_model
from tokenizer.train_tokenizer import load_llama_tokenizer
from datasets import DatasetDict, Dataset
import argparse

import time
import torch
import random
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, broadcast_object_list
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import datasets
import transformers
from transformers import LlamaForCausalLM, get_scheduler, SchedulerType, default_data_collator
from itertools import chain
import copy
import math
import os
import json
from data.data_processing import tokenize_dataset, group_token_dataset, prepare_dataloader
from datasets import concatenate_datasets
import re
import numpy as np
from torch.utils.data import Dataset
import threading
import sys

os.environ["WANDB_PROJECT"] = "train_llama"
os.environ["WANDB_ENTITY"] = "jannek-ulm"


def main(args):
    
    # Print GPU information
    print("\n===== GPU INFORMATION =====")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available, using CPU")
    print("===========================\n")
    
    # get the start time
    start_time = time.time()
    
    print("Start time:", start_time)
    # initialize the accelerator
    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "log_with": "wandb",
        "project_dir": "/home/janulm/Documents/ETH/SM12/Thesis/logs",
        "mixed_precision": args.mixed_precision,
        "step_scheduler_with_optimizer": False
    }

    # TODOOOO ADD THE NICE DATA AUGMENTATION THAT REGROUPS AND SHUFFLES THE DATA...
    # COMPARE AGAINST BASELINES (OTHER BABYLM Models, small GPT2), 
    # CHECK FOR THEIR perplexity
    # CHECK FOR OUR CD approach on some of our models. 
    # TRAIN A MODEL WITH COSINE WITH RESTARTS.

    accelerator = Accelerator(**accelerator_kwargs)
    
    # Print distributed training information
    if accelerator.is_main_process:
        print("\n===== DISTRIBUTED TRAINING SETUP =====")
        print(f"Accelerator state: {accelerator.state}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Current process index: {accelerator.process_index}")
        print(f"Is main process: {accelerator.is_main_process}")
        print(f"Distribution type: {accelerator.distributed_type}")
        print("======================================\n")
    
    # initialize logger
    accelerator.init_trackers(project_name="train_llama", config=args, init_kwargs={"wandb": {"entity": "janulm-organization", "name": args.model_name}})
    logger = get_logger(__name__)
    # logg the args: 
    logger.info("Args", args)
    print("Args", args)

    # make sure that the accelerator is on the correct device
    device = accelerator.device
    accelerator.log({"device": str(device)})
    accelerator.log({"args": vars(args)})
    accelerator.log({"mixed_precision": accelerator_kwargs.get("mixed_precision", "no")})

    print(f"Using device: {device}")
    print(f"Using mixed precision: {accelerator_kwargs.get('mixed_precision', 'no')}")

    # set the seed for reproducibility
    set_seed(args.seed)

    # set logging visibility? 
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    

    # now load the datasets
    if accelerator.is_main_process:
        raw_datasets = DatasetDict.load_from_disk(args.dataset_dir)
        print("Loaded datasets", raw_datasets)
        logger.info("Loaded datasets", raw_datasets)
    else:
        raw_datasets = None
    


    if accelerator.is_main_process: 
        if args.synthetic_dataset_dir:

            """
            synth_datasets = DatasetDict.load_from_disk(args.synthetic_dataset_dir)
            print("Loaded synthetic datasets", synth_datasets)
            logger.info("Loaded synthetic datasets", synth_datasets)
            # Explicitly use synthetic_train and ignore eval
            synth_train = synth_datasets["synthetic_train"]
            from data.data_processing import preprocess_synthetic_dataset
            synth_train = preprocess_synthetic_dataset(synth_train)
            """

            from data.data_processing import preprocess_synthetic_dataset
            synth_train_combined = []
            print("Loading synthetic datasets from", args.synthetic_dataset_dir, flush=True)
            dataset_dirs = args.synthetic_dataset_dir.split(" ")
            for dataset_dir in dataset_dirs:
                synth_datasets = DatasetDict.load_from_disk(dataset_dir)
                print("Loaded synthetic datasets", synth_datasets, "from", dataset_dir)
                logger.info("Loaded synthetic datasets", synth_datasets)
                # Explicitly use synthetic_train and ignore eval
                synth_train = synth_datasets["synthetic_train"]
                
                synth_train = preprocess_synthetic_dataset(synth_train)
                print("Preprocessed synthetic single train", synth_train)
                synth_train_combined.append(synth_train)
            synth_train_combined_dataset = concatenate_datasets(synth_train_combined)
            print("Preprocessed combined synthetic train", synth_train_combined_dataset)
            logger.info("Preprocessed synthetic train", synth_train_combined_dataset)
            
            synth_train = synth_train_combined_dataset
        else:
            synth_train = None
            print("No synthetic dataset provided, will not train on synthetic data")
        logger.info("No synthetic dataset provided, will not train on synthetic data")
    else:
        synth_train = None
    
    # now load the tokenizer
    tokenizer = load_llama_tokenizer(args.tokenizer_config_path)
    # activate that the tokenizer adds EOS tokens
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True

    """
    Process real and synthetic datasets separately and create data pools.
    """
    if accelerator.is_main_process:
        print("Processing real dataset...")
        # Process real dataset
        real_tokenized = tokenize_dataset(tokenizer, raw_datasets["train"], accelerator, args)
        real_pool = group_token_dataset(tokenizer, real_tokenized, accelerator, args, shuffle=True, shuffle_seed=args.seed)
    
        # do the same for the synthetic dataset    
        if synth_train is not None:
            print("Processing synthetic dataset...")
            # Process synthetic dataset
            synth_tokenized = tokenize_dataset(tokenizer, synth_train, accelerator, args)
            synthetic_pool = group_token_dataset(tokenizer, synth_tokenized, accelerator, args, shuffle=True, shuffle_seed=args.seed)
        else:
            synthetic_pool = None
        
    else:
        real_tokenized = None
        real_pool = None
        synth_tokenized = None
        synthetic_pool = None
    
    # Broadcast the pools to all processes
    if accelerator.num_processes > 1:
        object_list = [real_pool, synthetic_pool]
        broadcast_object_list(object_list, from_process=0)
        real_pool, synthetic_pool = object_list
        
    """
    Process eval dataset
    """
    if accelerator.is_main_process:
        print("Processing eval dataset...")
        eval_tokenized = tokenize_dataset(tokenizer, raw_datasets["test"], accelerator, args)
        eval_pool = group_token_dataset(tokenizer, eval_tokenized, accelerator, args, shuffle=False)
        print(f"Eval pool size: {len(eval_pool)}")
    else:
        eval_pool = None

    # Broadcast eval pool to all processes
    if accelerator.num_processes > 1:
        object_list = [eval_pool]
        broadcast_object_list(object_list, from_process=0)
        eval_pool = object_list[0]

    """
    Create the dataloaders 

    1. Real dataloader
    2. Synthetic dataloader
    3. Eval_real dataloader
    """      

    # check the batch sizes, and the number of workers 
    per_set_num_workers = int(args.dataloader_num_workers/2) if synthetic_pool is not None else int(args.dataloader_num_workers) # because we have two sets if using synthetic data
    assert per_set_num_workers >= 0, "Number of workers per set must be greater than 0"
    synth_batch_size = int(args.per_device_train_batch_size * args.synthetic_data_ratio) if synthetic_pool is not None else 0
    real_batch_size = int(args.per_device_train_batch_size - synth_batch_size)

    print("Synthetic batch size", synth_batch_size, flush=True)
    print("Real batch size", real_batch_size, flush=True)
    print("Total batch size", synth_batch_size + real_batch_size, "should be", args.per_device_train_batch_size, flush=True)
    assert synth_batch_size + real_batch_size == args.per_device_train_batch_size, "Total batch size should be the same as the per_device_train_batch_size"

    # Create real dataloader
    print(f"[rank {accelerator.process_index}] DEBUG: real_pool", real_pool, len(real_pool), flush=True)
    print(f"[rank {accelerator.process_index}] DEBUG: real batch size", real_batch_size, flush=True)
    print(f"[rank {accelerator.process_index}] DEBUG: per set num workers", per_set_num_workers, flush=True)


    """
    Now that we know the real and synth batch sizes, we can limit the real/synth pool to the max_safe_size
    and create the dataloaders. 
    """
            
            
    num_seq_per_global_step_real = accelerator.num_processes * real_batch_size * args.gradient_accumulation_steps
    num_safe_steps_real = len(real_pool) // num_seq_per_global_step_real
    real_pool = real_pool.select(range(num_safe_steps_real * num_seq_per_global_step_real))


    print(f"[rank {accelerator.process_index}] DEBUG: num_seq_per_global_step_real", num_seq_per_global_step_real, flush=True)
    print(f"[rank {accelerator.process_index}] DEBUG: limited the real pool to {num_safe_steps_real} global steps", flush=True)
    
    real_dataloader = prepare_dataloader(real_pool, args, accelerator, shuffle=True, batch_size=real_batch_size, num_workers=per_set_num_workers)


    if synthetic_pool is not None:
        num_seq_per_global_step_synth = accelerator.num_processes * synth_batch_size * args.gradient_accumulation_steps
        num_safe_steps_synth = len(synthetic_pool) // num_seq_per_global_step_synth
        synthetic_pool = synthetic_pool.select(range(num_safe_steps_synth * num_seq_per_global_step_synth))

        print(f"[rank {accelerator.process_index}] DEBUG: num_seq_per_global_step_synth", num_seq_per_global_step_synth, flush=True)
        print(f"[rank {accelerator.process_index}] DEBUG: limited the synthetic pool to {num_safe_steps_synth} global steps", flush=True)

        # Create synthetic dataloader
        synth_dataloader = prepare_dataloader(synthetic_pool, args, accelerator, shuffle=True, batch_size=synth_batch_size, num_workers=per_set_num_workers)
    else:
        synth_dataloader = None

    # Create eval dataloader
    eval_dataloader = prepare_dataloader(eval_pool, args, accelerator, shuffle=False)

    if accelerator.is_main_process:
        # Print tokenizer configuration for debugging
        print("\nTokenizer Configuration:")
        print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
        print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
        print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        print(f"add_bos_token: {tokenizer.add_bos_token}")
        print(f"add_eos_token: {tokenizer.add_eos_token}")

        print("Loaded tokenizer", tokenizer)
        logger.info("Tokenizer",tokenizer)
    # now load the model
    # either load an existing model or create a new one
    if args.checkpoint_path:
        model = LlamaForCausalLM.from_pretrained(
            args.checkpoint_path,
            config=None,
            torch_dtype="auto",
        )
        logger.info(f"Loaded model from {args.checkpoint_path}")
        print(f"Loaded model from {args.checkpoint_path}")
    elif args.init_model_config:
        model, config = load_babyllama_model(model_config_path=args.init_model_config)
        logger.info(f"Created new model from {args.init_model_config}")
        print(f"Created new model from {args.init_model_config}")
    else:
        raise ValueError("No model path or initial config provided")
    
    if accelerator.is_main_process:
        # go over the first 5 keys and print the shape of the parameter
        for key in list(model.state_dict().keys())[:5]:
            print(key, model.state_dict()[key].shape)
            # print the mean min max and std of the parameter
            print(model.state_dict()[key].mean(), model.state_dict()[key].min(), model.state_dict()[key].max(), model.state_dict()[key].std())


    # Try compiling the model before wrapping it with DDP
    if accelerator.is_main_process and args.compile_model:
        print("Compiling model forward function before DDP wrapping")
        logger.info("Compiling model forward function before DDP wrapping")

    torch.set_float32_matmul_precision('high')
    # this here gives quite a bit of a speedup, and also reduces memory usage by a lot
    if args.compile_model:
        # Compile with more conservative settings to avoid issues
        
        
        #model.forward = torch.compile(
        #    model.forward, 
        #    mode="max-autotune-no-cudagraphs",
        #    backend="inductor",
        #    fullgraph=False,
        #    dynamic=True
        #)
        

        model.forward = torch.compile(
            model.forward,
            mode="default",            # simpler and more stable
            backend="inductor",        # or experiment with "aot_eager"
            fullgraph=True,            # may help reduce memory by unifying graphs
            dynamic=False              # lock down sizes to simplify compilation
        )

        print(f"Process {accelerator.process_index}: Model compilation complete, proceeding to DDP wrapping")
        logger.info(f"Process {accelerator.process_index}: Model compilation complete, proceeding to DDP wrapping")

    # Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Calculate total training steps - now independent of GPU count
    if args.max_train_steps is None:
        raise ValueError("max_train_steps must be specified")
    
    # Get the lr scheduler - adjust warmup steps to be independent of GPU count
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps, # * accelerator.num_processes, # because each process takes a step, (its a shared counter object but still each process should take a step)
        num_training_steps=args.max_train_steps, # * accelerator.num_processes,
        scheduler_specific_kwargs={ "num_cycles": args.num_cycles } if args.lr_scheduler_type == "cosine_with_restarts" else {}
    )

    # Prepare everything with accelerator
    model, optimizer, lr_scheduler = accelerator.prepare(
        model, optimizer, lr_scheduler
    )

    # tracking: 
    experiment_config = vars(args)
    # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
    accelerator.init_trackers("train_llama", experiment_config)

    # compute the effictive total batch size
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Accelerator num processes = {accelerator.num_processes}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        print("***** Running training *****")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Accelerator num processes = {accelerator.num_processes}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")


    # Only show the progress bar once on each machine.
    # Initialize progress bar
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0

    # Synchronize all processes after preparation
    accelerator.wait_for_everyone()


    # Training loop
    model.train()
    total_loss = 0
    accumulation_loss = 0.0
    accumulation_count = 0

    # Initialize the iterators for the dataloaders
    real_iterator = iter(real_dataloader)
    synth_iterator = iter(synth_dataloader) if synth_dataloader is not None else None

    print(f"[rank {accelerator.process_index}] Initialized iterators for dataloaders",flush=True)
    print(f"[rank {accelerator.process_index}] Len dataloader: {len(real_dataloader)}",flush=True)

    # Function to regroup a dataset and create new dataloader
    def regroup_dataset(tokenized_data, batch_size, seed, dataset_name):
        # Regroup dataset with new seed
        if accelerator.is_main_process:
            print(f"Regrouping {dataset_name} dataset with new seed: {seed}",flush=True)
            pool = group_token_dataset(tokenizer, tokenized_data, accelerator, args, shuffle=True, shuffle_seed=seed)

            # limit the size of the pool to the max_safe_size
            samples_per_global_step = accelerator.num_processes * batch_size * args.gradient_accumulation_steps
        
            # Calculate maximum number of complete global steps we can do with current dataset
            original_size = len(pool)
            max_global_steps = original_size // samples_per_global_step
            max_safe_size = max_global_steps * samples_per_global_step
            pool = pool.select(range(max_safe_size))
            print(f"Original pool size: {original_size}",flush=True)
            print(f"Pool size after limiting: {len(pool)}",flush=True)

        else:
            print("Not on main process, so not regrouping",flush=True)
            pool = None
        accelerator.wait_for_everyone()
        # Broadcast the pool to all processes
        if accelerator.num_processes > 1:
            print("Broadcasting pool to all processes",flush=True)
            print(f"Pool: {pool}",flush=True)
            object_list = [pool]
            broadcast_object_list(object_list, from_process=0)
            pool = object_list[0]

        # Create new dataloader and iterator
        new_dataloader = prepare_dataloader(pool, args, accelerator, shuffle=True, batch_size=batch_size, num_workers=per_set_num_workers)
        new_iterator = iter(new_dataloader)
        print(f"Created new {dataset_name} dataloader and iterator",flush=True)
        return new_dataloader, new_iterator

    while completed_steps < args.max_train_steps:
        # Get a batch from the real dataloader
        try:
            #print(f"[rank {accelerator.process_index}] Getting batch from real dataloader, completed steps: {completed_steps}",flush=True)
            real_batch = next(real_iterator)
            #print(f"[rank {accelerator.process_index}] Got batch from real dataloader, completed steps: {completed_steps}",flush=True)
        except StopIteration:
            print(f"[rank {accelerator.process_index}] Stopping iteration for real dataloader, completed steps: {completed_steps}",flush=True)
            # Clean up old dataloader and iterator
            del real_dataloader, real_iterator
            # Regroup real dataset with new seed and create new dataloader
            new_seed = args.seed + completed_steps
            real_dataloader, real_iterator = regroup_dataset(real_tokenized, real_batch_size, new_seed, "real")
            real_batch = next(real_iterator)
        
        # Get a batch from the synthetic dataloader
        if synth_dataloader is not None:
            try:
                #print(f"[rank {accelerator.process_index}] Getting batch from synthetic dataloader, completed steps: {completed_steps}",flush=True)
                synth_batch = next(synth_iterator)
                #print(f"[rank {accelerator.process_index}] Got batch from synthetic dataloader, completed steps: {completed_steps}",flush=True)
            except StopIteration:
                print(f"[rank {accelerator.process_index}] Stopping iteration for synthetic dataloader, completed steps: {completed_steps}",flush=True)
                # Clean up old dataloader and iterator
                del synth_dataloader, synth_iterator
                # Regroup synthetic dataset with new seed and create new dataloader
                new_seed = args.seed + completed_steps
                synth_dataloader, synth_iterator = regroup_dataset(synth_tokenized, synth_batch_size, new_seed, "synthetic")
                synth_batch = next(synth_iterator)
        else:
            synth_batch = None
        
        # Combine batches
        if synth_batch is not None:
            combined_batch = {}
            for key in real_batch.keys():
                if key in synth_batch:
                    combined_batch[key] = torch.cat([real_batch[key], synth_batch[key]], dim=0)
                else:
                    raise ValueError(f"Key {key} not found in both real and synthetic batches")
        else:
            combined_batch = real_batch
        
        with accelerator.accumulate(model):
            outputs = model(**combined_batch)
            loss = outputs.loss
            loss_float = loss.detach().float()
            total_loss += loss_float
            
            # Track loss for current accumulation step
            accumulation_loss += loss_float
            accumulation_count += 1
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # Explicitly delete intermediate tensors to free memory
            del combined_batch, real_batch, synth_batch, outputs, loss

        if accelerator.sync_gradients:
            # Update scheduler and completed steps on all processes
            lr_scheduler.step()
            completed_steps += 1
            
            # Gather losses from all processes
            gathered_losses = accelerator.gather_for_metrics(accumulation_loss.clone().detach().to(accelerator.device))
            gathered_counts = accelerator.gather_for_metrics(torch.tensor(accumulation_count,device=accelerator.device))
            
            # Calculate global average loss
            global_avg_loss = gathered_losses.sum() / gathered_counts.sum()
            
            # Only update progress bar and log metrics on main process
            if accelerator.is_main_process:
                progress_bar.update(1)
                
                # Log metrics
                accelerator.log({
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/loss": global_avg_loss.item(),
                    "train/step": completed_steps
                },step=completed_steps)
                
                print(f"Training step {completed_steps} loss: {global_avg_loss.item()}")
                print(f"Training step {completed_steps} lr: {lr_scheduler.get_last_lr()[0]}")
            
            # Reset accumulation tracking (do this on all processes)
            accumulation_loss = 0.0
            accumulation_count = 0

            # Checkpointing (only on main process)
            if isinstance(args.checkpointing_steps, int) and accelerator.is_main_process:
                if completed_steps % args.checkpointing_steps == 0 and completed_steps != 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    
                    if args.save_only_model:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save
                        )
                    else:
                        accelerator.save_state(output_dir)

            # Evaluation (run on all GPUs, but log only on main process)
            if completed_steps % args.eval_steps == 0:
                # Wait for all processes to reach this point
                accelerator.wait_for_everyone()
                
                # Set model to eval mode on all processes
                model.eval()
                
                losses = []
                # Run evaluation on all processes
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    # Gather losses from all processes
                    losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

                # Concatenate all gathered losses
                losses = torch.cat(losses)
                
                # Calculate metrics
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                # Log results only on main process
                if accelerator.is_main_process:
                    logger.info(f"step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")
                    print(f"step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")
                    accelerator.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": perplexity,
                        "eval/step": completed_steps
                    },step=completed_steps)

                # Set model back to train mode on all processes
                model.train()
                
                # Wait for all processes to finish evaluation
                accelerator.wait_for_everyone()


    # Try to clean up, but don't fail if cleanup fails
    try:
        accelerator.end_training()
    except Exception as e:
        print(f"Warning: Cleanup failed with error: {e}")
        print("This is expected and can be safely ignored as training completed successfully.")
        # Force cleanup of CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Exit with success code since training completed
        sys.exit(0)

    end_time = time.time()
    if accelerator.is_main_process:
        print("End time:", end_time)
        print("Time taken:", end_time - start_time)
        logger.info(f"Time taken: {end_time - start_time}")
    exit()


if __name__ == "__main__":

    # argparser: 
    parser = argparse.ArgumentParser()
    # dataset path
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Path to the real dataset directory")
    parser.add_argument("--synthetic_dataset_dir", type=str, required=False, default=None,
                      help="Path to the synthetic dataset directory")
    parser.add_argument("--cache_dataset_dir", type=str, default=None)
    
    # Add new argument for synthetic data ratio
    parser.add_argument("--synthetic_data_ratio", type=float, default=None,
                      help="Ratio of synthetic data in each batch (0.0 to 1.0)")
    
    # Add eval steps argument
    parser.add_argument("--eval_steps", type=int, default=100,
                      help="Run evaluation every N steps")
    
    # Modify checkpointing steps help
    parser.add_argument("--checkpointing_steps", type=int, default=100,
                      help="Save a checkpoint every N steps. Set to 0 to disable checkpointing during training.")
    
    # option is either to load an existing model to continue training or to create a new model
    # resume training from a checkpoint
    # start from initial config
    parser.add_argument("--init_model_config",type=str,default="configs/babyllama_model.config",help="If the training should start from an model initial config file.")
    # make use of one argument for this (checkpoint_path)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint to load.")
    
    # tokenizer config path
    parser.add_argument("--tokenizer_config_path", type=str, default="configs/tokenizer_tinybabylm.config")
    
    parser.add_argument("--per_device_train_batch_size",type=int,default=16,help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size",type=int,default=16,help="Batch size (per device) for the evaluation dataloader.")

    parser.add_argument("--learning_rate",type=float,default=0.0005,help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay",type=float,default=0.1,help="Weight decay to use.")
    parser.add_argument("--max_train_steps",type=int,default=1000,help="Total number of training steps to perform.")

    parser.add_argument("--gradient_accumulation_steps",type=int,default=4,help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    # Add mixed precision argument
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10 and an Nvidia Ampere GPU."
    )

    # num warmup steps
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=150,
        help="Number of steps for the warmup in the lr scheduler."
    )
    # seed
    parser.add_argument("--seed",type=int,default=42,help="A seed for reproducible training.")

    # num workers
    parser.add_argument("--preprocessing_num_workers",type=int,default=16,help="The number of processes to use for the preprocessing.")
    # num workers for the dataloader
    parser.add_argument("--dataloader_num_workers",type=int,default=8,help="The number of processes to use for the dataloader.") 
    # block size
    parser.add_argument("--block_size",type=int,default=1024,help="The block size to use for the training. If none it will use min(tokenizer.model_max_length, model.config.max_position_embeddings)")
    
    # group texts batch size
    parser.add_argument("--group_texts_batch_size",type=int,default=1000,help="The batc h size to use for the group texts.")

    # overwrite cache
    parser.add_argument("--overwrite_cache",type=bool,default=False,help="If the cache should be overwritten.")

    # lr scheduler type
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    # num cycles
    parser.add_argument("--num_cycles",type=int,default=3,help="The number of cycles to use for the cosine with restarts scheduler.")

    # output directory
    parser.add_argument("--output_dir", type=str, default="../checkpoints/models/exp3/", help="The output directory where the model checkpoints will be written.")

    # compile model
    parser.add_argument("--compile_model",default=False,action="store_true",help="If the model should be compiled.")

    # save only model weights
    parser.add_argument("--save_only_model",default=True,action="store_true",help="If True, only saves model weights during checkpointing, not optimizer/scheduler state.")

    # Add model name argument for WandB
    parser.add_argument("--model_name", type=str, default="babyllama", help="Name of the model for WandB tracking")

    args = parser.parse_args()

    if args.synthetic_dataset_dir is None or args.synthetic_data_ratio == 0.0 or args.synthetic_data_ratio is None or args.synthetic_dataset_dir == "None" or args.synthetic_dataset_dir == "":
        args.synthetic_data_ratio = None
        args.synthetic_dataset_dir = None
    main(args)

