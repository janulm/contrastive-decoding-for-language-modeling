# combine_synthetic_datasets.py
import os, argparse, shutil, time, json, subprocess
from datasets import Dataset, DatasetDict, concatenate_datasets

def log(msg): print(msg, flush=True)

def rsync_dir(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["rsync","-a","--delete", f"{src}/", f"{dst}/"], check=True)

def main(args):
    # Discover shards
    process_dirs = [d for d in os.listdir(args.input_dir) if d.startswith("process_")]
    process_dirs.sort(key=lambda x: int(x.split("_")[1]))
    log(f"Found {len(process_dirs)} process directories")

    # Load shards (Arrow mmaps -> fast)
    datasets = []
    for d in process_dirs:
        p = os.path.join(args.input_dir, d)
        log(f"Loading dataset from {p}")
        datasets.append(Dataset.load_from_disk(p))

    # Concatenate and normalize indices
    log("Concatenating datasets...")
    t0 = time.time()
    combined = concatenate_datasets(datasets)
    log(f"Combined dataset size: {len(combined)}")
    log("Flattening indices for faster sequential write...")
    combined = combined.flatten_indices()
    log(f"Concatenate+flatten took {time.time()-t0:.1f}s")

    # Fast path: save one combined dataset + split indices (best for speed)
    if args.save_single_combined:
        os.makedirs(args.output_dir, exist_ok=True)
        tmp_out = os.path.join(args.tmp_dir, "combined_tmp") if args.tmp_dir else args.output_dir
        if os.path.exists(tmp_out): shutil.rmtree(tmp_out)

        log(f"Saving single combined dataset to {tmp_out} "
            f"(num_proc={args.save_num_proc}, max_shard_size={args.max_shard_size})")
        t = time.time()
        combined.save_to_disk(tmp_out, num_proc=args.save_num_proc, max_shard_size=args.max_shard_size)
        log(f"Save took {time.time()-t:.1f}s")

        # Persist split indices (deterministic)
        n = len(combined)
        test_n = max(1, int(n * args.test_size))
        rng = list(range(n))
        # simple deterministic split without extra shuffle pass
        test_idx = rng[:test_n]
        train_idx = rng[test_n:]
        with open(os.path.join(tmp_out, "split_indices.json"), "w") as f:
            json.dump({"train": train_idx, "test": test_idx}, f)

        if args.tmp_dir:
            final = args.output_dir
            if os.path.exists(final): shutil.rmtree(final)
            log(f"Rsyncing combined dataset from {tmp_out} -> {final}")
            rsync_dir(tmp_out, final)
            shutil.rmtree(tmp_out)

        # Optional cleanup of shard dirs
        if args.cleanup:
            log("Cleaning up process directories...")
            for d in process_dirs:
                shutil.rmtree(os.path.join(args.input_dir, d))
        log("Done!")
        return

    # Classic path: write DatasetDict (two full writes)
    log("Splitting into train and test...")
    # use a deterministic split without shuffling the whole table again
    n = len(combined)
    test_n = max(1, int(n * args.test_size))
    test = combined.select(range(test_n))
    train = combined.select(range(test_n, n))
    final = DatasetDict({"synthetic_train": train, "synthetic_test": test})

    # Choose target (tmp or final)
    save_root = os.path.join(args.tmp_dir, "combined_dsdd_tmp") if args.tmp_dir else args.output_dir
    if os.path.exists(save_root): shutil.rmtree(save_root)
    os.makedirs(save_root, exist_ok=True)

    log(f"Saving final dataset to {save_root} "
        f"(num_proc={args.save_num_proc}, max_shard_size={args.max_shard_size})")
    t = time.time()
    final.save_to_disk(save_root, num_proc=args.save_num_proc, max_shard_size=args.max_shard_size)
    log(f"Save took {time.time()-t:.1f}s")

    if args.tmp_dir:
        if os.path.exists(args.output_dir): shutil.rmtree(args.output_dir)
        log(f"Rsyncing dataset dict from {save_root} -> {args.output_dir}")
        rsync_dir(save_root, args.output_dir)
        shutil.rmtree(save_root)

    if args.cleanup:
        log("Cleaning up process directories...")
        for d in process_dirs:
            shutil.rmtree(os.path.join(args.input_dir, d))
    log("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cleanup", action="store_true")

    # Performance knobs
    p.add_argument("--max_shard_size", type=str, default="2GB",
                   help='Max Arrow shard size (e.g., "1GB", "500MB"). Bigger = fewer files = faster on network FS.')
    p.add_argument("--save_num_proc", type=int, default=1,
                   help="Writer processes. Use 1–2 on network FS; higher on local NVMe.")
    p.add_argument("--tmp_dir", type=str, default="",
                   help="If set (e.g., $SLURM_TMPDIR), write there first then rsync to output_dir.")
    p.add_argument("--save_single_combined", action="store_true",
                   help="Save one combined dataset plus split_indices.json (fastest).")
    p.add_argument("--test_size", type=float, default=0.05)
    args = p.parse_args()
    main(args)