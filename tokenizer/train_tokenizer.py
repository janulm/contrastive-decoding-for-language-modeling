# 
#  taken from here: https://discuss.huggingface.co/t/how-to-train-a-llamatokenizer/64835/2 
#  as there seems to be a bug in the directly training a LlamaTokenizerFast

import json
import yaml
import argparse
from tqdm import tqdm
from datasets import DatasetDict
from tokenizers import SentencePieceBPETokenizer
from transformers import LlamaTokenizerFast, TrainingArguments, AutoTokenizer
import os 

seed = 42

def load_config(config_path):
    # read the config file as a json
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def train_llama_tokenizer(config_path):

    # load the config
    config = load_config(config_path)

    # this script is taken from taken from here: https://discuss.huggingface.co/t/how-to-train-a-llamatokenizer/64835/2 
    # and modified to work with the config file, it uses a workaround to train the tokenizer using the SentencePieceBPETokenizer, and then loads the tokenizer as a LlamaTokenizerFast
    # directly training a LlamaTokenizerFast seems to not be working
    # load the dataset
    dataset_dict = DatasetDict.load_from_disk(config["data_path"])
    print("Loaded dataset: ",dataset_dict)

    # Create a SentencePieceBPETokenizer
    tokenizer = SentencePieceBPETokenizer()

    # get the vocab size from the config
    vocab_size = int(config["vocab_size"])

    # Train the SentencePieceBPETokenizer on the dataset
    tokenizer.train_from_iterator(
        iterator=dataset_dict['train']['text'],
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>",  "<pad>"],
    )
    # the algorithm uses NFKC normalization. 

    # Save the tokenizer
    store_name = config["store_path"] + "-sentencepiece-tokenizer.json"
    # compute the directory name from the store_name
    store_dir = os.path.dirname(store_name)
    # create the directory if it doesn't exist
    os.makedirs(store_dir, exist_ok=True)
    tokenizer.save(store_name, pretty=True)
    print("Saved (SentencePieceBPETokenizer) tokenizer to: ", store_name)

    # cast the max_input_length to int
    max_input_length = int(config["max_input_length"])
    
    # Load the new tokenizer as a LlamaTokenizerFast
    new_llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file=config["store_path"] + "-sentencepiece-tokenizer.json",
        name_or_path=config["store_path"] + "-tokenizer",
        unk_token="<unk>",
        unk_token_id=0,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
        pad_token="<pad>",
        pad_token_id=3,
        padding_side="right",
        model_max_length=max_input_length,  # Set model_max_length directly instead of max_model_input_sizes
    )
    new_llama_tokenizer.update_post_processor()
    new_llama_tokenizer.save_pretrained(config["store_path"])
    print("Saved (LlamaTokenizerFast) tokenizer to: ", config["store_path"])

    return new_llama_tokenizer



def load_llama_tokenizer(config_path=None, tokenizer_dir=None):
    
    """
    Load a LlamaTokenizerFast from the path specified in the config file.
    
    Args:
        config_path (str): Path to the config file of the tokenizer to load
        
    Returns:
        LlamaTokenizerFast: The loaded tokenizer
    """
    # assert that either config_path or tokenizer_dir is provided, but not both
    assert config_path is not None or tokenizer_dir is not None, "Either config_path or tokenizer_dir must be provided"
    assert config_path is None or tokenizer_dir is None, "Only one of config_path or tokenizer_dir can be provided"

    if config_path is not None:
        # load the config
        config = load_config(config_path)
        tok_dir = config["store_path"]
        # Get max_input_length from config
        max_input_length = int(config["max_input_length"])
    elif tokenizer_dir is not None:
        tok_dir = tokenizer_dir
    else:
        raise ValueError("Either config_path or tokenizer_dir must be provided")

    # load the tokenizer from disk
    print("Loading LlamaTokenizerFast from: ", tok_dir)
    
    try:
        # Try to load with explicit LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained(
            tok_dir,
            padding_side="right",
            use_fast=True
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    except Exception as e:
        print(f"Error loading with LlamaTokenizerFast: {e}")
        print("Trying with AutoTokenizer as fallback...")
        
        # Fallback to AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tok_dir,
            padding_side="right",
            use_fast=True,
            model_max_length=max_input_length  # Set model_max_length explicitly
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer

def main(args):

    # load the config
    config = load_config(args.config)

    if args.train:
        # train the tokenizer
        print("Training tokenizer...")
        tokenizer = train_llama_tokenizer(args.config)
    
    # load the tokenizer
    print("Loading tokenizer...")
    tokenizer = load_llama_tokenizer(args.config)

    # Test the tokenizer
    test_text = "Hello, world! This is a test of the tokenizer. Hopefully this is unknown token <unk>. <统一码>"
    print("\nTest tokenization:")
    print(f"Original text: {test_text}")
    print(f"Tokenized: {tokenizer.tokenize(test_text)}")
    print(f"Token IDs: {tokenizer.encode(test_text)}")
    print(f"Decoded: {tokenizer.decode(tokenizer.encode(test_text))}")
    
    # Print tokenizer info
    print("\nTokenizer information:")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"Model max length: {tokenizer.model_max_length}")

if __name__ == "__main__":
    
    # load the config file that specifies how the tokenizer should be trained
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--train', action='store_true', help='Train a new tokenizer')
    args = parser.parse_args()

    main(args)
