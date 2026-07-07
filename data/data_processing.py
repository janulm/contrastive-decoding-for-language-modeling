import copy
from itertools import chain
from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator


def tokenize_dataset(tokenizer, dataset, accelerator, args):
    """
    Tokenize a dataset for language modeling.
    
    Args:
        tokenizer: The tokenizer to use for tokenization
        dataset: The dataset to process
        accelerator: The accelerator for distributed training
        args: Arguments containing:
            - preprocessing_num_workers: Number of workers for preprocessing
        overwrite_cache: Whether to overwrite the cache
            
    Returns:
        A tokenized dataset
    """
    def tokenize_function(examples):
        return tokenizer(examples["text"], 
            add_special_tokens=True, # This should add BOS/EOS based on tokenizer settings, (pad tokens are not added)
            truncation=False,  # We'll handle truncation later after block_size is defined
        )

    # caching logic, if cache_dataset_dir is provided, save the grouped dataset to the cache, otherwise keep in memory
    if args.cache_dataset_dir is not None:
        cache_dir = args.cache_dataset_dir
        # generate uuid for cache file name
        import uuid
        import os
        cache_file_name = f"{uuid.uuid4()}.pt"
        cache_file_path = os.path.join(cache_dir, cache_file_name)
        keep_in_memory = False
        print(f"Caching grouped dataset to {cache_file_path}")
    else:
        keep_in_memory = True
        cache_dir = None
        cache_file_path = None
        print("Not caching tokenized dataset")
    
    if accelerator.is_main_process:
        tokenized_datasets = dataset.map(
            tokenize_function, 
            batched=True, 
            num_proc=args.preprocessing_num_workers,
            remove_columns=dataset.column_names, 
            #load_from_cache_file=not (overwrite_cache if overwrite_cache is not None else args.overwrite_cache),
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_path,
            desc="Tokenizing dataset"
        )
        return tokenized_datasets
    return None

def group_token_dataset(tokenizer, tokenized_dataset, accelerator, args, shuffle=False, shuffle_seed=42, overwrite_cache=False):
    """
    Group tokenized sequences into blocks of fixed size for training.
    
    Args:
        tokenizer: The tokenizer (needed for pad_token_id)
        tokenized_dataset: The tokenized dataset to process
        accelerator: The accelerator for distributed training
        args: Arguments containing:
            - seq_len: The sequence length for grouping
            - group_texts_batch_size: Batch size for grouping operation
        shuffle: Whether to shuffle the dataset before grouping
        shuffle_seed: Seed for shuffling
        overwrite_cache: Whether to overwrite the cache
            
    Returns:
        A grouped dataset with:
            - input_ids: Tokenized and grouped sequences
            - attention_mask: Attention masks for the sequences
            - labels: Labels for language modeling (same as input_ids with padding masked)
    """
    # check if seq_len or block_size is defined in args

    if "seq_len" in args:
        block_size = args.seq_len
    elif "block_size" in args:
        block_size = args.block_size
    else:
        raise ValueError("seq_len or block_size is not set")
    
    # Apply shuffling if requested
    if shuffle and shuffle_seed is not None:
        tokenized_dataset = tokenized_dataset.shuffle(seed=shuffle_seed)
    
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        num_complete_blocks = total_length // block_size
        
        result = {}
        for k, t in concatenated_examples.items():
            result[k] = [t[i:i+block_size] for i in range(0, num_complete_blocks*block_size, block_size)]
            
            # Handle remainder with padding
            remainder_length = total_length - (num_complete_blocks * block_size)
            if remainder_length > 0:
                remainder = t[num_complete_blocks * block_size:]
                padding = [tokenizer.pad_token_id if k == "input_ids" else 0] * (block_size - remainder_length)
                result[k].append(remainder + padding)
                
        result["labels"] = copy.deepcopy(result["input_ids"])
        
        # Set padding positions to -100 in labels
        if "attention_mask" in result:
            for i in range(len(result["labels"])):
                for j in range(len(result["labels"][i])):
                    if result["attention_mask"][i][j] == 0:
                        result["labels"][i][j] = -100
                        
        return result
    
    # caching logic, if cache_dataset_dir is provided, save the grouped dataset to the cache, otherwise keep in memory
    if args.cache_dataset_dir is not None:
        cache_dir = args.cache_dataset_dir
        # generate uuid for cache file name
        import uuid
        import os
        cache_file_name = f"{uuid.uuid4()}.pt"
        cache_file_path = os.path.join(cache_dir, cache_file_name)
        keep_in_memory = False
        print(f"Caching grouped dataset to {cache_file_path}")
    else:
        keep_in_memory = True
        cache_dir = None
        cache_file_path = None
        print("Not caching grouped dataset")

    if accelerator.is_main_process:
        grouped_dataset = tokenized_dataset.map(
            group_texts, 
            batched=True, 
            batch_size=args.group_texts_batch_size,
            num_proc=args.preprocessing_num_workers, 
            #load_from_cache_file=not (overwrite_cache if overwrite_cache is not None else args.overwrite_cache),
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_path,
            desc=f"Grouping texts in chunks of {block_size}"
        )
        return grouped_dataset
    return None

def tokenize_group_dataset(tokenizer, dataset, accelerator, args, shuffle=False, shuffle_seed=42, overwrite_cache=False):
    """
    Tokenize and group a dataset for language modeling (combined function for backward compatibility).
    
    This function performs two main operations:
    1. Tokenizes the text data using the provided tokenizer
    2. Groups the tokenized sequences into blocks of fixed size for training
    
    Args:
        tokenizer: The tokenizer to use for tokenization
        dataset: The dataset to process
        accelerator: The accelerator for distributed training
        args: Arguments containing:
            - preprocessing_num_workers: Number of workers for preprocessing
            - seq_len: The sequence length for grouping
            - group_texts_batch_size: Batch size for grouping operation
        shuffle: Whether to shuffle the dataset before grouping
        shuffle_seed: Seed for shuffling
        overwrite_cache: Whether to overwrite the cache
            
    Returns:
        A processed dataset with:
            - input_ids: Tokenized and grouped sequences
            - attention_mask: Attention masks for the sequences
            - labels: Labels for language modeling (same as input_ids with padding masked)
    """
    # Step 1: Tokenize the dataset
    tokenized_dataset = tokenize_dataset(tokenizer, dataset, accelerator, args, overwrite_cache)
    
    # Step 2: Group the tokenized dataset

    grouped_dataset = group_token_dataset(tokenizer, tokenized_dataset, accelerator, args, shuffle, shuffle_seed, overwrite_cache)
    return grouped_dataset

def prepare_dataloader(dataset, args, accelerator, shuffle=False, batch_size=None, num_workers=None):
    """
    Prepare a dataloader from a dataset with the given arguments.
    
    Args:
        dataset: The dataset to create a dataloader for
        args: Arguments containing:
            - batch_size: Batch size for the dataloader
            - dataloader_num_workers: Number of workers for the dataloader
        accelerator: The accelerator for distributed training
        shuffle: Whether to shuffle the data in the dataloader
            
    Returns:
        A prepared dataloader for the dataset
    """
    batch_size_args = args.batch_size if hasattr(args, 'batch_size') else args.per_device_train_batch_size
    if batch_size is None:  
        batch_size = batch_size_args

    if num_workers is None:
        num_workers = args.dataloader_num_workers

    print("Creating dataloader with batch size:", batch_size, "and num_workers:", num_workers)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=default_data_collator, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,  # Prefetch next batches
        persistent_workers=True if num_workers > 0 else False  # Keep workers alive between batches
    )
    print("DEBUG: dataloader", dataloader, "len", len(dataloader))
    # Prepare for distributed computing
    dataloader = accelerator.prepare(dataloader)
    print("DEBUG: dataloader after prepare", dataloader, "len", len(dataloader))
    return dataloader 


from datasets import Dataset, DatasetDict
import re
from tqdm import tqdm


def preprocess_synthetic_dataset(synth_train):
    """
    Preprocess the synthetic dataset to split at <s> and </s> tags, removing the input_ids column and just working on the text that contains the tags.
    
    Args:
        synth_train: The synthetic dataset to preprocess

    Returns:
        A processed dataset with:
            - text: The processed text, split at <s> and </s> tags
    """

    print("Synth train:", synth_train)
    # drop the column "input_ids"
    synth_train = synth_train.remove_columns("input_ids")
    print("Synth train after dropping input_ids:", synth_train)

    # Process the synthetic dataset: split at <s> and </s> tags
    def process_text_row(text):
        # Replace all variations of tag patterns
        # First normalize by ensuring consistent spacing around tags
        text = re.sub(r'<s>\s*', '<s>', text)
        text = re.sub(r'\s*</s>', '</s>', text)
        
        # Also handle the case where </s> is written as <\s>
        text = re.sub(r'<\\s>', '</s>', text)
        
        # Split text at <s> and </s> tags
        # Using regex pattern that matches both opening and closing tags
        segments = re.split(r'</?s>', text)
        
        # Filter out empty segments and strip whitespace
        valid_segments = []
        for segment in segments:
            segment = segment.strip()
            if segment and not segment.isspace():  # Check if segment is not empty or just whitespace
                valid_segments.append(segment)
        
        return valid_segments

    # Process all rows and create a new expanded dataset
    new_texts = []
    processed_count = 0
    skipped_count = 0
    segments_per_row = []
    
    for idx, row in tqdm(enumerate(synth_train), desc="Processing synthetic dataset", total=len(synth_train)):
        segments = process_text_row(row["text"])
        
        if segments:
            new_texts.extend(segments)
            segments_per_row.append(len(segments))
            processed_count += 1
        else:
            skipped_count += 1
                
    # Create a new dataset with the processed texts
    processed_synth_train = Dataset.from_dict({"text": new_texts})
    
    print(f"Original synthetic dataset: {len(synth_train)} rows")
    print(f"Processed synthetic dataset: {len(processed_synth_train)} rows")
    print(f"Rows with valid segments: {processed_count}")
    print(f"Rows with no valid segments: {skipped_count}")

    if segments_per_row:
        avg_segments = sum(segments_per_row) / len(segments_per_row)
        max_segments = max(segments_per_row)
        print(f"Average segments per row: {avg_segments:.2f}")
        print(f"Maximum segments per row: {max_segments}")
    
    # Print a few examples from the processed dataset
    for i in range(min(5, len(processed_synth_train))):
        print(f"Processed Text {i}:", processed_synth_train[i]["text"][:100] + "...")

    # Use the processed dataset for training
    return processed_synth_train