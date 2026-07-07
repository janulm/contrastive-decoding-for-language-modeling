

# script that loads in a dataset, tokenizer path and then tokenizes and computes the diversity score


import argparse
from datasets import load_from_disk
import json
import random
import nltk


def _calc_ngram_diversity_product(total_num_words,ds, n_values=(2, 3, 4)):
    """Compute product over n of (unique n-grams / total n-grams) using nltk.

    Args:
        texts (list[str]): Corpus documents.
        n_values (tuple[int, ...]): N-gram sizes to include in the product.

    Returns:
        tuple[dict[int, float], float]: Mapping n->ratio and the product value.
    """
    

    tokens = []
    text_idx = 0
    for i in range(len(ds)):
        #print("Tokens length initial: ", len(tokens),flush=True)
        #print("Text idx: ", text_idx)
        #print("Text: ", dataset["text"][text_idx])
        tokens.extend(ds[i]["text"].split())
        #print("Tokens length: ", len(tokens),flush=True)
        #print("Text idx: ", text_idx)
        text_idx += 1
        if len(tokens) >= total_num_words:
            break

    # remove the overhang words
    tokens = tokens[:total_num_words]
    print("Tokens length: ", len(tokens), "and rows used: ", text_idx)

    per_n_ratios = {}
    product = 1.0
    for n in n_values:
        ngrams = list(nltk.ngrams(tokens, n))
        total = len(ngrams)
        if total == 0:
            ratio = 1.0  # no n-grams of this size; neutral factor
        else:
            ratio = len(set(ngrams)) / total
        per_n_ratios[n] = ratio
        product *= ratio

    return per_n_ratios, product


def main(args):
    
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

    #print("\nFirst 10 rows of the train split (text only):")
    #for i in range(min(10, len(train_ds))):
        print(train_ds[i]["text"])
        print("\n\n###########\n\n")
    
    
    print("Loaded dataset: ", dataset)


    if args.shuffle_rows:
        print("Shuffling rows")
        # shuffle rows in dataset
        dataset = dataset.shuffle(42)
        print("Shuffled dataset: ", dataset)
    
    
    per_n, div_prod = _calc_ngram_diversity_product(args.total_num_words, train_ds, (2, 3, 4))
    for n in (2, 3, 4):
        print(f"text: ngram_diversity_ratio_n{n}: {per_n[n]}")
    print(f"text: ngram_diversity_product_2x3x4: {div_prod}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=False)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--shuffle_rows", type=bool, default=True)
    parser.add_argument("--total_num_words", type=int, default=100000)
    args = parser.parse_args()

    main(args)