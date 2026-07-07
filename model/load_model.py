# load a llama model from disk, and return it
# also need a method to initialize the model with the correct parameters. 

# how to configure the model? : taken from: https://huggingface.co/blog/nroggendorff/train-with-llama-architecture

import sys
import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM
import json 

# Add the parent directory to sys.path to allow importing from sibling packages
# This is needed when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import from the tokenizer package
from tokenizer.train_tokenizer import load_llama_tokenizer


def load_babyllama_model(model_config_path=None):
    # load the model config,
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    # load the tokenizer
    tokenizer = load_llama_tokenizer(config_path=model_config["tokenizer_config_path"])

    # load the model
    llama_model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_layers"],
        num_attention_heads=model_config["num_heads"],
        max_position_embeddings=model_config["max_position_embeddings"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        dtype=model_config["dtype"],
    )

    # CHECK FOR SEED...     
    model = LlamaForCausalLM(llama_model_config)


    # make sure the model is with correct dtype model_config["dtype"] is str(float16)
    if model_config["dtype"] == "float16":
        #print("Setting model to float16")
        model.to(torch.float16)
    elif model_config["dtype"] == "float32":
        #print("Setting model to float32")
        model.to(torch.float32)
    else:
        raise ValueError(f"Unsupported dtype: {model_config['dtype']}")

    # return the model and the config
    return model, llama_model_config




def main():
    

    model_config_path = "configs/babyllama_model.config"
    print("Creating randomly initialized model from initial config: ", model_config_path)
    model, config = load_babyllama_model(model_config_path=model_config_path)

    print("Model: ", model)
    # how many parameters does the model have?
    print("Number of parameters: ", model.num_parameters())
    # what about the datatype of this model? 
    print("Datatype of the model: ", model.dtype)

    # save the model to disk
    model.save_pretrained("../checkpoints/models/tiny-llama")
    
    print("Loading model from disk...")
    model = LlamaForCausalLM.from_pretrained("../checkpoints/models/tiny-llama",torch_dtype="auto")

    print("Model: ", model)
    # how many parameters does the model have?
    print("Number of parameters: ", model.num_parameters())
    # what about the datatype of this model? 
    print("Datatype of the model: ", model.dtype)


if __name__ == "__main__":
    main()