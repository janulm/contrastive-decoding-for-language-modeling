import argparse
import math
import os
import time

from transformers import LlamaConfig, LlamaForCausalLM
import torch
from accelerate import Accelerator
from tqdm import tqdm
from datasets import Dataset

from logits_processors import (
    ContrastiveLogitsProcessor,
    DefaultLogitsProcessor,               
    NoContrastVHeadLogitsProcessor,
    ContrastiveTailLogitsProcessor,
    ContrastiveWithNoRepeatNGram,
    NoContrastWithNoRepeatNGram,
    TopPLogitsProcessor,
    TopKLogitsProcessor,
    ContrastiveWithTopP,
    ContrastiveWithTopK,
)
from tokenizer.train_tokenizer import load_llama_tokenizer
from data.data_processing import prepare_dataloader
from ensemble_generation import contrastive_generate


def load_model_with_dropout(model_path, attention_dropout=None):
    config = LlamaConfig.from_pretrained(model_path)
    if attention_dropout is not None:
        config.attention_dropout = attention_dropout
        print(f"Setting attention dropout to {attention_dropout} in config")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype="auto",
    )
    return model


def tokenize_and_shorten_seed_dataset(seed_dataset, tokenizer, args):
    """Tokenize seed chunks and left-pad to initial_seed_length; assumes dataset is already shuffled/selected/sharded."""
    print(f"Tokenizing {seed_dataset.num_rows} seeds for worker {args.worker_id}/{args.num_workers}")
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=args.initial_seed_length,
            padding=False,
        )
        # Left-pad to fixed length
        for i in range(len(tokenized["input_ids"])):
            cur_len = len(tokenized["input_ids"][i])
            if cur_len < args.initial_seed_length:
                pad = args.initial_seed_length - cur_len
                tokenized["input_ids"][i] = [tokenizer.pad_token_id] * pad + tokenized["input_ids"][i]
                tokenized["attention_mask"][i] = [0] * pad + tokenized["attention_mask"][i]
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
    elif args.cd_strategy == "no_contrast_top_p":
        logits_processor = TopPLogitsProcessor(args.top_p, args.temperature, args.penalty_value)
    elif args.cd_strategy == "no_contrast_top_k":
        logits_processor = TopKLogitsProcessor(args.top_k, args.temperature, args.penalty_value)
    elif args.cd_strategy == "no_contrast_with_no_repeat_ngram":
        logits_processor = NoContrastWithNoRepeatNGram(args.no_repeat_ngram_size)
    elif args.cd_strategy == "contrast_with_top_p":
        logits_processor = ContrastiveWithTopP(args.top_p, args.contrast_strength, args.alpha, args.temperature, args.penalty_value)
    elif args.cd_strategy == "contrast_with_top_k":
        logits_processor = ContrastiveWithTopK(args.top_k, args.contrast_strength, args.alpha, args.temperature, args.penalty_value)
    elif args.cd_strategy in ("contrastive_dropout", "contrastive"):
        logits_processor = ContrastiveLogitsProcessor(args.contrast_strength, args.alpha, args.temperature, args.penalty_value)
    elif args.cd_strategy == "no_contrast_v_head":
        logits_processor = NoContrastVHeadLogitsProcessor(args.temperature, args.alpha, args.penalty_value)
    elif args.cd_strategy == "contrastive_tail":
        logits_processor = ContrastiveTailLogitsProcessor(args.contrast_strength, args.alpha, args.temperature)
    elif args.cd_strategy == "contrastive_with_no_repeat_ngram":
        inner = ContrastiveLogitsProcessor(args.contrast_strength, args.alpha, args.temperature, args.penalty_value)
        logits_processor = ContrastiveWithNoRepeatNGram(inner, args.no_repeat_ngram_size)
    else:
        raise ValueError(f"Invalid CD strategy: {args.cd_strategy}")
    return logits_processor


def decode_tokens(token_dataset, tokenizer, args):
    def decode_function(examples):
        return {"text": tokenizer.decode(examples["input_ids"], skip_special_tokens=False)}
    text_dataset = token_dataset.map(decode_function, batched=False, num_proc=args.preprocessing_num_workers)
    return text_dataset


def main(args):
    # Single-process per Slurm task. Accelerator is fine (keeps your dataloader util intact).
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="no")
    device = accelerator.device
    print("\n===== GENERATION (sharded, single GPU) =====")
    print(f"Worker {args.worker_id} / {args.num_workers}")
    print(f"Device: {device}")
    print("===========================================\n")
    torch.set_float32_matmul_precision('high')

    # Load models/tokenizer
    good_model = load_model_with_dropout(args.good_model_path, None)
    good_model.eval()
    if args.compile_model:
        good_model.forward = torch.compile(
            good_model.forward,
            mode="max-autotune-no-cudagraphs",
            backend="inductor",
            fullgraph=False,
            dynamic=True,
        )

    if args.cd_strategy in ("no_contrast", "no_contrast_v_head", "no_contrast_with_no_repeat_ngram", "no_contrast_top_p", "no_contrast_top_k"):
        bad_model = None
    else:
        bad_model_dropout = args.dropout_rate if args.cd_strategy == "contrastive_dropout" else None
        bad_model = load_model_with_dropout(args.bad_model_path, bad_model_dropout)
        bad_model.train()

        if args.compile_model:
            bad_model.forward = torch.compile(
                bad_model.forward,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=False,
                dynamic=True,
            )

    tokenizer = load_llama_tokenizer(args.tokenizer_config_path)
    print(f"Tokenizer loaded from {args.tokenizer_config_path}")

    # ===== Load / shuffle / shard seeds deterministically =====
    from datasets import Dataset as HFDataset
    seed_dataset = HFDataset.load_from_disk(args.seed_dataset_path)
    total_rows = seed_dataset.num_rows
    #seed_dataset = seed_dataset.shuffle(seed=42)

    # Optional global cap on total seeds
    if args.numb_seeds is not None:
        cap = min(args.numb_seeds, total_rows)
        seed_dataset = seed_dataset.select(range(cap))
        total_rows = cap

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")
    if not (0 <= args.worker_id < args.num_workers):
        raise ValueError("--worker_id must be in [0, num_workers)")

    per_worker = math.ceil(total_rows / args.num_workers) if total_rows > 0 else 0
    start = args.worker_id * per_worker
    end = min(start + per_worker, total_rows)

    if start >= total_rows or start == end:
        # No work for this shard: still write an empty shard to keep combine script simple
        shard_dir = os.path.join(args.output_dataset_path, f"process_{args.worker_id}")
        os.makedirs(shard_dir, exist_ok=True)
        empty = Dataset.from_dict({"text": []})
        empty.save_to_disk(shard_dir)
        if args.worker_id == 0:
            # Only shard 0 writes / updates description
            description_file_path = os.path.join(args.output_dataset_path, "description.txt")
            os.makedirs(os.path.dirname(description_file_path), exist_ok=True)
            with open(description_file_path, "w") as f:
                f.write("#########################\n")
                f.write("Description of the generation\n")
                f.write(args.title + "\n")
                f.write("#########################\n")
                f.write(f"Parameters used for the generation: {args}\n")
                f.write("#########################\n")
        print(f"[worker {args.worker_id}] No assigned seeds. Wrote empty shard to {shard_dir}")
        return

    # Select this shard
    shard = seed_dataset.select(range(start, end))
    print(f"[worker {args.worker_id}] Using seeds [{start}:{end}) of {total_rows} (size={shard.num_rows})")
    args.numb_seeds = shard.num_rows  # for downstream logging

    # Tokenize and dataloader
    tokenized_seed_dataset = tokenize_and_shorten_seed_dataset(shard, tokenizer, args)
    dataloader = prepare_dataloader(tokenized_seed_dataset, args, accelerator, shuffle=False)
    print(f"[worker {args.worker_id}] Dataloader batches: {len(dataloader)} (batch_size={args.batch_size})")

    # Compile / prepare logits processor
    logits_processor = get_logits_processor(args)
    if logits_processor is not None:
        safe_to_compile = type(logits_processor).__name__ != "ContrastiveWithNoRepeatNGram" and type(logits_processor).__name__ != "NoContrastWithNoRepeatNGram"
        if safe_to_compile:
            logits_processor.__call__ = torch.compile(
                logits_processor.__call__,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                fullgraph=True,
                dynamic=True,
            )

    # Prepare models for device
    good_model = accelerator.prepare(good_model)
    if bad_model is not None:
        bad_model = accelerator.prepare(bad_model)

    # Generate
    all_generations_tokens = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"[worker {args.worker_id}] Generating synthetic data"):
            input_ids = batch["input_ids"]
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

    # Save shard
    shard_dir = os.path.join(args.output_dataset_path, f"process_{args.worker_id}")
    os.makedirs(shard_dir, exist_ok=True)

    token_dataset = Dataset.from_dict({"input_ids": all_generations_tokens})
    text_dataset = decode_tokens(token_dataset, tokenizer, args)
    text_dataset.save_to_disk(shard_dir)
    print(f"[worker {args.worker_id}] Saved shard to {shard_dir}")

    # Only worker 0 writes/updates the description file
    if args.worker_id == 0:
        description_file_path = os.path.join(args.output_dataset_path, "description.txt")
        os.makedirs(os.path.dirname(description_file_path), exist_ok=True)
        with open(description_file_path, "w") as f:
            f.write("#########################\n")
            f.write("Description of the generation\n")
            f.write(args.title + "\n")
            f.write("#########################\n")
            f.write(f"Parameters used for the generation: {args}\n")
            f.write("#########################\n")

        print("#########################")
        print("Synthetic data shards are being written by parallel workers.")
        print(f"Root output: {args.output_dataset_path}")
        print("#########################")

    print(f"[worker {args.worker_id}] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--seed_dataset_path", type=str, default="../data/tinybabylm_dataset/seed/")
    parser.add_argument("--output_dataset_path", type=str, default="../data/tinybabylm_dataset_synthetic/")

    # Generation controls
    parser.add_argument("--numb_generations_per_seed", type=int, default=8)
    parser.add_argument("--numb_seeds", type=int, default=None)  # optional global cap before sharding
    parser.add_argument("--numb_tokens_per_generation", type=int, default=400)
    parser.add_argument("--initial_seed_length", type=int, default=20)

    # Models / tokenizer
    parser.add_argument("--good_model_path", type=str, default=None)
    parser.add_argument("--bad_model_path", type=str, default=None)
    parser.add_argument("--tokenizer_config_path", type=str, default="configs/tokenizer_tinybabylm.config")

    # CD strategy + params
    parser.add_argument("--cd_strategy", type=str, default=None,
                        choices=["no_contrast", "contrastive_dropout", "contrastive", "no_contrast_v_head",
                                 "contrastive_tail", "contrastive_with_no_repeat_ngram", "no_contrast_with_no_repeat_ngram", "no_contrast_top_p", "no_contrast_top_k","contrast_with_top_p","contrast_with_top_k"])
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--contrast_strength", type=float, default=0.5)
    parser.add_argument("--penalty_value", type=float, default=-20)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=10)

    # Data loading
    parser.add_argument("--preprocessing_num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=6)

    # Compile
    parser.add_argument("--compile_model", default=False, action="store_true")

    # New: sharding across separate Slurm jobs
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of parallel jobs/shards.")
    parser.add_argument("--worker_id", type=int, default=0, help="This job's shard index in [0, num_workers).")

    # Meta
    parser.add_argument("--title", type=str, default="Classic CD Generation")

    args = parser.parse_args()

    # Ensure trailing slash like before
    args.output_dataset_path = args.output_dataset_path + "/"

    print("Starting generation with parameters:")
    print(args)
    main(args)