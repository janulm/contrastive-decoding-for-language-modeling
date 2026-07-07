import argparse
from transformers import LlamaConfig, LlamaForCausalLM
import torch
from accelerate import Accelerator
from tqdm import tqdm
import time
from logits_processors import ContrastiveLogitsProcessor, DefaultLogitsProcessor, NoContrastVHeadLogitsProcessor, ContrastiveTailLogitsProcessor, ContrastiveWithNoRepeatNGram
from tokenizer.train_tokenizer import load_llama_tokenizer
from datasets import Dataset, DatasetDict
from data.data_processing import prepare_dataloader
import os
from ensemble_generation import contrastive_generate
from accelerate.utils import broadcast_object_list, gather_object


def load_model_with_dropout(model_path, attention_dropout=None):
    """Load a model with optional attention dropout modification."""
    # Load the model configuration first
    config = LlamaConfig.from_pretrained(model_path)
    
    # If dropout is specified, modify the config
    if attention_dropout is not None:
        config.attention_dropout = attention_dropout
        print(f"Setting attention dropout to {attention_dropout} in config")
    
    # Load the model with the modified config
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype="auto"
    )
    
    return model


def tokenize_and_shorten_seed_dataset(seed_dataset, tokenizer, args):
    # tokenize each row, 
    # and truncate to initial_seed_length 
    # what should we do with the rows that are shorter than initial_seed_length? 
    
    # only the first numb_seeds should be used, drop the rest, (first shuffle)
    
    seed_dataset = seed_dataset.shuffle(seed=42)
    if args.numb_seeds is not None: # only use the first numb_seeds if not None
        print(f"Using only the first {args.numb_seeds} seeds")
        seed_dataset = seed_dataset.select(range(args.numb_seeds))
    print(f"Number of seeds after selection: {seed_dataset.num_rows}")
    args.numb_seeds = seed_dataset.num_rows

    def tokenize_function(examples):
        # First tokenize with padding disabled
        tokenized = tokenizer(examples["text"], 
            add_special_tokens=True,
            truncation=True,
            max_length=args.initial_seed_length,
            padding=False  # Don't pad yet
        )
        
        # Now pad sequences that are shorter than initial_seed_length
        for i in range(len(tokenized["input_ids"])):
            current_length = len(tokenized["input_ids"][i])
            if current_length < args.initial_seed_length:
                # Calculate how many padding tokens to add
                padding_length = args.initial_seed_length - current_length
                # Add padding tokens at the beginning
                tokenized["input_ids"][i] = [tokenizer.pad_token_id] * padding_length + tokenized["input_ids"][i]
                tokenized["attention_mask"][i] = [0] * padding_length + tokenized["attention_mask"][i]
        
        return tokenized
    
    tokenized_seed_dataset = seed_dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=args.preprocessing_num_workers,
        remove_columns=seed_dataset.column_names,
    )
    return tokenized_seed_dataset


def get_logits_processor(args):
    if args.cd_strategy == "no_contrast":
        logits_processor = None
    elif args.cd_strategy == "contrastive_dropout" or args.cd_strategy == "contrastive":
        logits_processor = ContrastiveLogitsProcessor(args.contrast_strength, args.alpha, args.temperature, args.penalty_value)
    elif args.cd_strategy == "no_contrast_v_head":
        logits_processor = NoContrastVHeadLogitsProcessor(args.temperature, args.alpha, args.penalty_value)
    elif args.cd_strategy == "contrastive_tail":
        logits_processor = ContrastiveTailLogitsProcessor(args.contrast_strength, args.alpha, args.temperature)
    elif args.cd_strategy == "contrastive_with_no_repeat_ngram":
        inner_cd_processor = ContrastiveLogitsProcessor(args.contrast_strength, args.alpha, args.temperature, args.penalty_value)
        logits_processor = ContrastiveWithNoRepeatNGram(inner_cd_processor, args.no_repeat_ngram_size) 
    else:
        raise ValueError(f"Invalid CD strategy: {args.cd_strategy}")
    return logits_processor



def decode_tokens(token_dataset, tokenizer, args):
    # decode the tokens to text
    def decode_function(examples):
        return {"text": tokenizer.decode(examples["input_ids"], skip_special_tokens=False)}
    
    text_dataset = token_dataset.map(decode_function, batched=False, num_proc=args.preprocessing_num_workers)
    return text_dataset


def main(args):
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # No gradient accumulation needed for generation
        mixed_precision="no",  # No mixed precision needed for generation
    )

    # Print distributed training information
    if accelerator.is_main_process:
        print("\n===== DISTRIBUTED GENERATION SETUP =====")
        print(f"Accelerator state: {accelerator.state}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Current process index: {accelerator.process_index}")
        print(f"Is main process: {accelerator.is_main_process}")
        print(f"Distribution type: {accelerator.distributed_type}")
        print("======================================\n")

    device = accelerator.device
    print(f"[rank {accelerator.process_index}] Using device: {device}")
    torch.set_float32_matmul_precision('high')

    # load the good,bad model and tokenizer
    good_model = load_model_with_dropout(args.good_model_path, None)
    good_model.eval()
    print(f"Good model loaded from {args.good_model_path}")
    
    if args.compile_model:
        good_model.forward = torch.compile(
            good_model.forward, 
            mode="max-autotune-no-cudagraphs",
            backend="inductor",
            fullgraph=False,
            dynamic=True
        )

    # do we even need a bad model? 
    if args.cd_strategy == "no_contrast" or args.cd_strategy == "no_contrast_v_head":
        bad_model = None
    else:
        bad_model_dropout = args.dropout_rate if args.cd_strategy == "contrastive_dropout" else None
        bad_model = load_model_with_dropout(args.bad_model_path, bad_model_dropout)
        print(f"Bad model loaded from {args.bad_model_path}")
        bad_model.train() # this is important for dropout to work

    # this here gives quite a bit of a speedup, and also reduces memory usage by a lot
    if bad_model is not None and args.compile_model:
        # Compile with more conservative settings to avoid issues
        bad_model.forward = torch.compile(
            bad_model.forward, 
            mode="max-autotune-no-cudagraphs",
            backend="inductor",
            fullgraph=False,
            dynamic=True
        )
    
    tokenizer = load_llama_tokenizer(args.tokenizer_config_path)
    print(f"[rank {accelerator.process_index}] Tokenizer loaded from {args.tokenizer_config_path}")
    print(f"[rank {accelerator.process_index}] Tokenizer: {tokenizer}")
    
    # load and process seed dataset on main process
    if accelerator.is_main_process:
        seed_dataset = Dataset.load_from_disk(args.seed_dataset_path)
        print(f"Seed dataset loaded from {args.seed_dataset_path}")
        print(seed_dataset)
    else:
        seed_dataset = None

    # print 10 first examples of the seed dataset
    if accelerator.is_main_process:
        print("Printing 10 first examples of the seed dataset:")
        for i in range(10):
            print(seed_dataset[i])

        # tokenize the seed dataset on main process
        tokenized_seed_dataset = tokenize_and_shorten_seed_dataset(seed_dataset, tokenizer, args)
        print(f"Tokenized seed dataset: {tokenized_seed_dataset}")
    else:
        tokenized_seed_dataset = None

    # print 10 first examples of the tokenized seed dataset
    if accelerator.is_main_process:
        print("Printing 10 first examples of the tokenized seed dataset:")
        for i in range(10):
            print(tokenized_seed_dataset[i])
    # Broadcast tokenized dataset to all processes
    if accelerator.num_processes > 1:
        object_list = [tokenized_seed_dataset]
        broadcast_object_list(object_list, from_process=0)
        tokenized_seed_dataset = object_list[0]

    # get a dataloader for the tokenized seed dataset
    dataloader = prepare_dataloader(tokenized_seed_dataset, args, accelerator, shuffle=False)
    print(f"[rank {accelerator.process_index}] Dataloader created with batch size {args.batch_size}")

    # how many batches does this dataloader have?
    print(f"[rank {accelerator.process_index}] Dataloader has {len(dataloader)} batches")
    # Prepare models and dataloader with accelerator
    good_model = accelerator.prepare(good_model)
    if bad_model is not None:
        bad_model = accelerator.prepare(bad_model)

    if accelerator.is_main_process:
        print("#########################")
        print("Computation of how many tokens will be generated")
        print("Numb of seeds: ", args.numb_seeds)
        print("Numb of generations per seed: ", args.numb_generations_per_seed)
        print("Numb of tokens per generation: ", args.numb_tokens_per_generation)
        print("Numb of tokens from initial seed: ", args.initial_seed_length)
    
        # compute the total number of tokens that will be generated
        total_tokens = args.numb_seeds * args.numb_generations_per_seed * args.numb_tokens_per_generation
        print(f"[rank {accelerator.process_index}] Total number of tokens that will be generated: {total_tokens}")
        tokens_this_process = len(dataloader) * args.numb_generations_per_seed * args.numb_tokens_per_generation * args.batch_size
        
        print(f"[rank {accelerator.process_index}] Total number of tokens that will be generated on this process: ~{tokens_this_process}")

    # construct the correct CD strategy that is used for generation
    logits_processor = get_logits_processor(args)
    print(f"Logits processor: {logits_processor}")
    
    # Only compile the logits processor if it's not None and safe to compile.
    # Do NOT compile wrappers that close over HF processors like NoRepeatNGram to avoid
    # graph-capture/shape dynamism issues at the end of generation.
    if logits_processor is not None:
        safe_to_compile = True
        name = type(logits_processor).__name__
        # Avoid compiling ContrastiveWithNoRepeatNGram wrapper
        if name == "ContrastiveWithNoRepeatNGram":
            safe_to_compile = False
            print("NOT COMPILING")
        if safe_to_compile:
            print("START COMPILING")
            logits_processor.__call__ = torch.compile(
                logits_processor.__call__, 
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=True,
                dynamic=True
            )
            print("END COMPILING")

    # Generate data in parallel
    all_generations_tokens = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating synthetic data", disable=not accelerator.is_local_main_process):
            input_ids = batch["input_ids"]
            # get the attention mask
            attention_mask = batch["attention_mask"]
            
            outputs = contrastive_generate(
                good_model=good_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                bad_model=bad_model,
                max_new_tokens=args.numb_tokens_per_generation,
                num_return_sequences=args.numb_generations_per_seed,
                logits_combiner=logits_processor,
                do_sample=True,
            )
            all_generations_tokens.extend(outputs)

    # Gather all generations to main process
    print(f"[rank {accelerator.process_index}] Gathering generations to main process")
    print(f"[rank {accelerator.process_index}] All generations tokens before gathering: {len(all_generations_tokens)}")
    print(f"[rank {accelerator.process_index}] Done with generation, waiting for others")
    accelerator.wait_for_everyone()
    print(f"[rank {accelerator.process_index}] Done with generation")

    # Each process writes its own part to disk
    process_output_dir = os.path.join(args.output_dataset_path, f"process_{accelerator.process_index}")
    os.makedirs(process_output_dir, exist_ok=True)

    # Create dataset from this process's generations
    token_dataset = Dataset.from_dict({"input_ids": all_generations_tokens})
    
    # Decode the tokens to text
    text_dataset = decode_tokens(token_dataset, tokenizer, args)
    
    # Save this process's dataset
    text_dataset.save_to_disk(process_output_dir)
    print(f"[rank {accelerator.process_index}] Saved dataset to {process_output_dir}")

    # Only the main process writes the description file
    if accelerator.is_main_process:
        description_file_path = os.path.join(args.output_dataset_path, "description.txt")
        os.makedirs(os.path.dirname(description_file_path), exist_ok=True)
        with open(description_file_path, "w") as f:
            f.write("#########################\n")
            f.write("Description of the generation\n")
            f.write(args.title + "\n")
            f.write("#########################\n")
            f.write(f"Parameters used for the generation: {args}\n")
            f.write("#########################\n")

        print(f"[rank {accelerator.process_index}]#########################")
        print(f"[rank {accelerator.process_index}] Synthetic data generated and saved to {args.output_dataset_path}")
        print(f"[rank {accelerator.process_index}]#########################")

    # Wait for all processes to finish
    #accelerator.wait_for_everyone()
    print(f"[rank {accelerator.process_index}] Done")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed_dataset_path", type=str, default="../data/tinybabylm_dataset/seed/")
    # synthetic dataset path
    parser.add_argument("--output_dataset_path", type=str, default="../data/tinybabylm_dataset_synthetic/")
    # arguments that manage the generation 
    # should take care of numb generations per seed, number of seeds, numb of tokens per generation, numb of tokens from initial seed, 
    parser.add_argument("--numb_generations_per_seed", type=int, default=8)
    # number of seeds, if None all seeds will be used
    parser.add_argument("--numb_seeds", type=int, default=None) 
    # number of tokens per generation   
    parser.add_argument("--numb_tokens_per_generation", type=int, default=400)
    # number of tokens from initial seed
    parser.add_argument("--initial_seed_length", type=int, default=20)
    # samplng strategy, (greedy, beam_search, sampling)
    


    
    # good model path
    parser.add_argument("--good_model_path", type=str, default=None)
    # bad model path 
    parser.add_argument("--bad_model_path", type=str, default=None)
    # tokenizer_config_path
    parser.add_argument("--tokenizer_config_path", type=str, default="configs/tokenizer_tinybabylm.config")
    
    # temperature
    parser.add_argument("--temperature", type=float, default=1.0)
    
    
    # use dropout on bad model? or no cd at all? 
    parser.add_argument("--cd_strategy", type=str, default=None, choices=["no_contrast", "contrastive_dropout", "contrastive", "no_contrast_v_head", "contrastive_tail","contrastive_with_no_repeat_ngram"])
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)

    # args.preprocessing_num_workers
    parser.add_argument("--preprocessing_num_workers", type=int, default=12)

    # title that summarizes the generation parameters
    parser.add_argument("--title", type=str, default="Classic CD Generation")
    
    # CD parameters, alpha, contrast strength, temperature, num_return_sequences
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--contrast_strength", type=float, default=0.5)
    parser.add_argument("--penalty_value", type=float, default=-20)
    # batch size
    parser.add_argument("--batch_size", type=int, default=32)
    # dataloader_num_workers
    parser.add_argument("--dataloader_num_workers", type=int, default=6)
    # compile the model
    parser.add_argument("--compile_model",default=False,action="store_true",help="If the model should be compiled.")
    args = parser.parse_args()
    

    # extend the output dataset path with the cd strategy
    args.output_dataset_path = args.output_dataset_path + "/"

    print("Starting generation with the following parameters:")
    print(args)
    main(args)