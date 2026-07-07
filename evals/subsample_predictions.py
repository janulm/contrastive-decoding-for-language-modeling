import os
import argparse
import glob
import json
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib
import statsmodels.formula.api as smf
from functools import partial
import torch
from concurrent.futures import ProcessPoolExecutor

tasks = {
    "blimp": {"eval_path": "/main/zero_shot/causal/blimp/blimp_filtered/", "ground_truth_path": "blimp_filtered", "index_type": "acc"},
    "blimp_supplement": {"eval_path": "/main/zero_shot/causal/blimp/supplement_filtered/", "ground_truth_path": "supplement_filtered", "index_type": "acc"},
    "entity_tracking": {"eval_path": "/main/zero_shot/causal/entity_tracking/entity_tracking/", "ground_truth_path": "entity_tracking", "index_type": "acc"},
    "ewok": {"eval_path": "/main/zero_shot/causal/ewok/ewok_filtered/", "ground_truth_path": "ewok_filtered", "index_type": "acc"},
    "wug": {"eval_path": "/main/zero_shot/causal/wug/wug_adj_nominalization/", "ground_truth_path": "wug_adj_nominalization", "index_type": "acc"},
    "reading_eye_tracking": {"eval_path": "/main/zero_shot/causal/reading/", "ground_truth_path": "reading", "index_type": "reading_eye_tracking"},
    "perplexity": {"eval_path": "/tinybabylm_dataset-test/", "index_type": "perplexity"}
}

# READING WENT WRONG, REPEAT THE EVALUATION THERE....

# input is a path to a directory:, this directory contains directories that are "model_names" and each model_name directory contains more directories that contain the single checkpoints respectively, and their eval data. 


# the idea of this file is to subsample (subsampling with replacement of each single sample) the predictions of the different tasks (1000 times each), 
# hence we need to go and load the ground truth predictions of the tasks. 
# then we want to go over all checkpoints we are tasked to subsample
# for each checkpoint, go over each task.
#   for each task, load the checkpoints predictions, compare them with the ground truth predictions, and subsample the predictions with replacement.
#   save the subsampled scores to a new file.
#   # "subsampling_scores.npy" (it should be a list of scores, in most tasks its accuracy, for perplexity its perplexity) and for the reading and eye tracking its change in R^2? 


def extract_blimp_supplement_ground_truth(path):
    """
    Loads all .jsonl files in the given directory and returns a dictionary:
    {
        'filename': {
            'solutions': [
                {'id': 'filename_pairid', 'truth': sentence_good}, ...
            ]
        },
        ...
    }
    """
    blimp_supplement_gt = {}
    jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))
    for file_path in jsonl_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        solutions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                solutions.append({
                    'id': f"{file_name}_{idx}",
                    'truth': data['sentence_good']
                })
        blimp_supplement_gt[file_name] = {'solutions': solutions}
    return blimp_supplement_gt

def extract_blimp_ground_truth(path):
    """
    Loads all .jsonl files in the given directory and returns a dictionary:
    {
        'filename': {
            'solutions': [
                {'id': '(uid==filename)_i', 'truth': sentence_good}, ...
            ]
        },
        ...
    }
    """
    blimp_gt = {}
    # Find all .jsonl files in the directory
    jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))
    for file_path in jsonl_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        predictions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                uid = data.get('UID', file_name)
                pred_id = f"{uid}_{idx}"
                predictions.append({
                    'id': pred_id,
                    'truth': data['sentence_good']
                })
        blimp_gt[file_name] = {'solutions': predictions}
    return blimp_gt

def extract_entity_tracking_ground_truth(path):
    """
    Loads all .jsonl files in the given directory and returns a dictionary:
    For each file, for each unique numops (i), a new key is created as 'filename__i_ops',
    and the value is a dict with 'solutions' as a list of dicts with 'id' and 'truth'.
    The 'id' is 'filename__i_ops_j', where j is the counter for the row with the same numops.
    """
    entity_gt = {}
    jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))
    for file_path in jsonl_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        # For each numops, collect solutions in a separate list
        numops_solutions = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                i = data['numops']
                key = f"{file_name}_{i}_ops"
                if key not in numops_solutions:
                    numops_solutions[key] = []
                j = len(numops_solutions[key])
                solution_id = f"{file_name}_{i}_ops_{j}"
                numops_solutions[key].append({
                    'id': solution_id,
                    'truth': data['options'][0]
                })
        # Add each numops group as a separate file in the dict
        for key, solutions in numops_solutions.items():
            entity_gt[key] = {'solutions': solutions}
    return entity_gt

def extract_ewok_ground_truth(path):
    """
    Loads all .jsonl files in the given directory and returns a dictionary:
    {
        'filename': {
            'solutions': [
                {'id': 'filename_pairid', 'truth': sentence_good}, ...
            ]
        },
        ...
    }

    its a bit of a mess the EWok benchmark as it consits of Context 1, Context 2, Target 1 and Target 2, 
    and there are various variations possible on how to test the LLM, but what is tested is 
    Context 1 | Target 1 vs. Context 1 | Target 2 
    Hence Target 1 is always correct.!! 
    """
    ewok_gt = {}
    jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))
    for file_path in jsonl_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        solutions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                solution_id = f"{file_name}_{idx}"
                solutions.append({
                    'id': solution_id,
                    'truth': data['Target1']
                })
        ewok_gt[file_name] = {'solutions': solutions}
    return ewok_gt

def extract_wug_ground_truth(path): 
    """
    Loads all .jsonl files in the given directory and returns a dictionary:
    {
        'filename': {
            'solutions': [
                {'id': 'filename_pairid', 'truth': sentence_good}, ...
            ]
        },
        ...
    }
    """
    wug_gt = {}
    jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))
    for file_path in jsonl_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        solutions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                first_sentence = data['sentences'].split('\t')[0]
                solution_id = f"{file_name}_{idx}"
                solutions.append({
                    'id': solution_id,
                    'truth': first_sentence
                })
        wug_gt[file_name] = {'solutions': solutions}
    return wug_gt

def extract_reading_eye_tracking_ground_truth(path):
    # pred is -log(p) -> stored as Logrpob 
    # prev_pred is -log(p2) -> stored as Prev_Logprob
    # this is the only information needed from the models predictions and can be extracted. 
    # all other stuff is stored in the file: 
    path += "/reading_data.csv"
    df = pd.read_csv(path, dtype={'item': str})
    df["item"] = df["item"].fillna("None")
    #print(df.head())
    #print(df.columns)
    return df
    
def extract_ground_truth(ground_truth_path):
    # load the ground truth predictions from the ground truth path
    ground_truth_dict = {}
    
    ground_truth_dict["blimp"] = extract_blimp_ground_truth(ground_truth_path+tasks["blimp"]["ground_truth_path"])
    ground_truth_dict["blimp_supplement"] = extract_blimp_supplement_ground_truth(ground_truth_path+tasks["blimp_supplement"]["ground_truth_path"])
    ground_truth_dict["entity_tracking"] = extract_entity_tracking_ground_truth(ground_truth_path+tasks["entity_tracking"]["ground_truth_path"])
    ground_truth_dict["ewok"] = extract_ewok_ground_truth(ground_truth_path+tasks["ewok"]["ground_truth_path"])
    ground_truth_dict["wug"] = extract_wug_ground_truth(ground_truth_path+tasks["wug"]["ground_truth_path"])
    ground_truth_dict["reading_eye_tracking"] = extract_reading_eye_tracking_ground_truth(ground_truth_path+tasks["reading_eye_tracking"]["ground_truth_path"])
    # ppl does not need a ground truth and reading_eye_tracking contains the information for two tasks. 
    """
    # Print the first 3 items of the first 3 files for inspection
    blimp_gt = ground_truth_dict["wug"]
    blimp_keys = list(blimp_gt.keys())[:3]
    for key in blimp_keys:
        print(f"File: {key}")
        for item in blimp_gt[key]["solutions"][:3]:
            print(item)
        print()
    """
    #print(ground_truth_dict["reading_eye_tracking"].head())
    #print(ground_truth_dict["reading_eye_tracking"].columns)
    return ground_truth_dict

def load_checkpoint_pred(checkpoint_full_path, task_name): 
    # this works for blimp, blimp_supp, ewok, wug, entity_tracking
    # append the tasks["blimp"]["eval_path"] to the checkpath and load the file predictions.json
    # Its one large json object.
    blimp_pred_path = os.path.join(checkpoint_full_path, tasks[task_name]["eval_path"].lstrip("/"), "predictions.json")
    # load the file predictions.json (one large json object)
    with open(blimp_pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    return predictions


def do_subsamples_accuracy_task(prediction_data, ground_truth, store_path, index_set, task_name):
    # this works for blimp, blimp_supp, ewok, wug, entity_tracking
    # first checks that all the keys in the dicts are identical
    # then creates a n = (numb rows in the dataset) x 1 vector of 1 and 0's depending on the single row accuracy
    # then subsample from that vector with replacement: 
    # store a vector of size numb_samples+1 x 1 to the path, first entry is the mean without sampling.
    assert set(prediction_data.keys()) == set(ground_truth.keys()), "Mismatch between prediction and ground truth keys for task."

    per_file_bootstrap_means = []
    per_file_means = []

    #print("index_set,len",len(index_set))
    #print("prediction data len",len(prediction_data.keys()))

    for j, key in enumerate(prediction_data.keys()):
        pred_dict = {item['id']: item['pred'] for item in prediction_data[key]["predictions"]}
        gt_dict = {item['id']: item['truth'] for item in ground_truth[key]["solutions"]}
        assert len(pred_dict) == len(gt_dict), f"Mismatch between prediction and ground truth ids for task {key}."
        assert set(pred_dict.keys()) == set(gt_dict.keys()), f"Mismatch between prediction and ground truth ids for task {key}."

        # Compute accuracy vector for this file
        accuracy_vector = []
        for i in range(len(pred_dict)):
            key1 = list(pred_dict.keys())[i]
            key2 = list(gt_dict.keys())[i]
            assert key1 == key2, f"Mismatch between prediction and ground truth ids for task {key}."
            prediction = pred_dict[key1]
            truth = gt_dict[key2]
            accuracy_vector.append(1.0 if prediction == truth else 0.0)

        accuracy_vector = np.array(accuracy_vector)
        per_file_means.append(accuracy_vector.mean())

        # Bootstrapping for this file
        sampled_indices = index_set[j]

        #print("sampled_indices: ", sampled_indices.shape, "max: ", sampled_indices.max(), "min: ", sampled_indices.min())
        #print("accuracy_vector: ", accuracy_vector.shape, "max: ", accuracy_vector.max(), "min: ", accuracy_vector.min())
        sampled = accuracy_vector[sampled_indices]
        #print("sampled: ", sampled.shape, "max: ", sampled.max(), "min: ", sampled.min())
        bootstrap_means = sampled.mean(axis=1)  # shape: (num_samples,)
        per_file_bootstrap_means.append(bootstrap_means)

    # Now, per_file_bootstrap_means is a list of arrays, each of shape (num_samples,)
    # Stack to shape (num_files, num_samples)
    per_file_bootstrap_means = np.stack(per_file_bootstrap_means, axis=0)  # shape: (num_files, num_samples)

    # For each bootstrap sample, average across files
    final_bootstrap_means = per_file_bootstrap_means.mean(axis=0)  # shape: (num_samples,)

    # Prepend the "correct" mean (mean of per-file means)
    correct_mean = np.mean(per_file_means)
    final_bootstrap_means = np.concatenate([[correct_mean], final_bootstrap_means])

    np.save(store_path, final_bootstrap_means)
    #print("Task: ", task_name," mean (over categories mean) [[", correct_mean.round(4), "]]  min [[", final_bootstrap_means.min().round(4), "]] max [[", final_bootstrap_means.max().round(4), "]]")
    #print("shape of final_bootstrap_means: ", final_bootstrap_means.shape)


def do_subsamples_perplexity_task(per_batch_losses, store_path, index_set, task_name):
    # per_batch_losses: 1D numpy array of NLLs (negative log likelihoods)
    #print("prediction_data: ", per_batch_losses.shape)
    #print("mean of per_batch_losses: ", per_batch_losses.mean())
    #print("min of per_batch_losses: ", per_batch_losses.min())
    #print("max of per_batch_losses: ", per_batch_losses.max())

    # Compute the mean perplexity of the original data
    mean_perplexity = math.exp(per_batch_losses.mean())
    #print("mean perplexity: ", mean_perplexity)
    #print("per batch losses shape: ", per_batch_losses.shape, "max: ", per_batch_losses.max(), "min: ", per_batch_losses.min())
    # Sample indices: shape (num_samples, N)
    sampled_indices = index_set
    #print("sampled_indices shape: ", sampled_indices.shape, "max: ", sampled_indices.max(), "min: ", sampled_indices.min())
    # Gather samples and compute means: shape (num_samples,)
    sampled_means = per_batch_losses[sampled_indices].mean(axis=1)
    #print("sampled_means shape: ", sampled_means.shape, "max: ", sampled_means.max(), "min: ", sampled_means.min())
    # Convert means to perplexities
    sampled_ppls = np.exp(sampled_means)

    # Prepend the "true" mean perplexity
    all_ppls = np.concatenate([[mean_perplexity], sampled_ppls])

    # Save to file
    np.save(store_path, all_ppls)




############################

def do_reading_eye_task_single_sample(df):

    #
    # THIS PART OF THE CODE WAS TAKEN FROM THE BABYLM EVALUATION PIPELINE: 
    # https://github.com/babylm/evaluation-pipeline-2025
    # https://github.com/babylm/evaluation-pipeline-2025/blob/main/evaluation_pipeline/reading/run.py


    variables = ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound', 'self_paced_reading_time',  'ELAN', 'LAN', 'N400', 'P600', 'EPNP', 'PNP']

    #correlations = df[["pred"] + variables].corr()["pred"]
    #corr_file = args.output_dir / "correlations.txt"
    #with corr_file.open("w") as f:
    #    for index, values in correlations.items():
    #        if index != "pred":
    #            print(f"{index}\t{values:.4f}", file=f)

    #results = []
    report_values = []

    #for dv in variables:
    for dv in ['RTfirstfix', 'RTfirstpass', 'RTgopast', 'RTrightbound']:    
        # baseline model
        temp = df[[dv, "Subtlex_log10", "length", "context_length"]].dropna()
        # first fit baseline model without predictability
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        #aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length"]].dropna()
        # experimental model with iv
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + Subtlex_log10:length + Subtlex_log10:context_length + length:context_length + pred', data=temp).fit()
        #is_sig = float(OLS_model.tvalues["pred"])
        #the_p = float(OLS_model.pvalues["pred"])
        #the_B = float(OLS_model.params["pred"])
        R2_model = float(OLS_model.rsquared)
        #aic_model = float(OLS_model.aic)
        #results.append({
        #    "Predicted variable": dv,
        #    "Coefficient": the_B,
        #    "Number of standard deviations": is_sig,
        #    "P-value": the_p,
        #    "R2": R2_model,
        #    "Change in R2 from baseline": R2_model-R2_baseline,
        #    "AIC": aic_model,
        #    "Change in AIC from baseline": aic_model-aic_baseline,
        #})
        if "RT" in dv:
            report_values.append(((R2_model-R2_baseline)/(1-R2_baseline)) * 100)

    #predictability_file = args.output_dir / "predictive_power.jsonl"
    #with predictability_file.open("w") as fj:
    #    for res in results:
    #        print(json.dumps(res), file=fj)

    #predictability_file = args.output_dir / "report.txt"
    #with predictability_file.open("w") as fj:
    #    print(f"EYE TRACKING SCORE: {sum(report_values) / len(report_values):.2f}", file=fj)
    #print(f"EYE TRACKING SCORE: {sum(report_values) / len(report_values):.2f}")
    
    eye_tracking_score = sum(report_values) / len(report_values)

    #results = []

    #for dv in variables:
    for dv in ['self_paced_reading_time']: # THIS SHOULD MAKE IT FASTER AND KEEPS RESULTS IDENTICAL
        # baseline model
        temp = df[[dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        # first fit baseline model without predictability
        OLS_baseline = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred', data=temp).fit()
        R2_baseline = float(OLS_baseline.rsquared)
        #aic_baseline = float(OLS_baseline.aic)
        temp = df[["pred", dv, "Subtlex_log10", "length", "context_length", "prev_length", "prev_pred"]].dropna()
        # experimental model with iv
        OLS_model = smf.ols(formula=dv+' ~ Subtlex_log10 + length + context_length + prev_length + prev_pred + Subtlex_log10:length + Subtlex_log10:context_length + Subtlex_log10:prev_length + Subtlex_log10:prev_pred + length:context_length + length:prev_length + length:prev_pred + context_length:prev_length + context_length:prev_pred + prev_length:prev_pred + pred', data=temp).fit()
        #is_sig = float(OLS_model.tvalues["pred"])
        #the_p = float(OLS_model.pvalues["pred"])
        #the_B = float(OLS_model.params["pred"])
        R2_model = float(OLS_model.rsquared)
        #aic_model = float(OLS_model.aic)
        #results.append({
        #    "Predicted variable": dv,
        #    "Coefficient": the_B,
        #    "Number of standard deviations": is_sig,
        #    "P-value": the_p,
        #    "R2": R2_model,
        #    "Change in R2 from baseline": R2_model-R2_baseline,
        #    "AIC": aic_model,
        #    "Change in AIC from baseline": aic_model-aic_baseline,
        #})

        if "self" in dv:
            report_values = ((R2_model-R2_baseline)/(1-R2_baseline)) * 100

    #predictability_file = args.output_dir / "predictive_power_spillover.jsonl"
    #with predictability_file.open("w") as fj:
    #    for res in results:
    #        print(json.dumps(res), file=fj)

    #predictability_file = args.output_dir / "report.txt"
    #with predictability_file.open("a") as fj:
    #    print(f"SELF-PACED READING SCORE: {report_values:.2f}", file=fj)
    #print(f"SELF-PACED READING SCORE: {report_values:.2f}")

    reading_score = report_values 

    #print("got reading score", reading_score)
    #print("got eye tracking score:",eye_tracking_score)

    return reading_score, eye_tracking_score


############################

def _sample_and_score_reading_eye(args):
    reading_ground_truth, index_set = args
    sampled_indices = index_set
    #print("sampled_indices: ", sampled_indices.shape, "max: ", sampled_indices.max(), "min: ", sampled_indices.min())
    #print("reading_ground_truth: ", reading_ground_truth.shape)
    #print("reading_ground_truth: ", reading_ground_truth.head())
    sampled_df = reading_ground_truth.copy().iloc[sampled_indices].reset_index(drop=True)
    #print("sampled_df: ", sampled_df.shape)
    #print("sampled_df: ", sampled_df.head())
    return do_reading_eye_task_single_sample(sampled_df)


def do_subsamples_reading_eye_task(load_path, reading_ground_truth, save_path, index_set, task_name):
    # load the information from the load_path
    # do the sampling in for loop manner, 
    # compute score 
    # log scores.
    # store the np array.
   
    # print("load_path: ", load_path)
    # check if it has already been done: 
    
    predictions_df = pd.read_json(load_path, lines=True)

    # Rename the index column in both DataFrames
    ##########
    #print("gt: ",reading_ground_truth.head())
    #print("gt: ",reading_ground_truth.columns)
    #print("pred: ",predictions_df.head())
    #print("pred: ",predictions_df.columns)
    ###########
    
    reading_ground_truth = reading_ground_truth.rename(columns={'Unnamed: 0': 'Index'})
    predictions_df = predictions_df.rename(columns={'Sentence': 'item'})

    # sort to ensure alignment:
    reading_ground_truth = reading_ground_truth.sort_values('Index').reset_index(drop=True)
    predictions_df = predictions_df.sort_values('Index').reset_index(drop=True)

    # Copy over only the required columns
    reading_ground_truth['pred'] = predictions_df['Logprob']
    reading_ground_truth['prev_pred'] = predictions_df['Prev_Logprob']

    #print("Combined DataFrame head:")
    #print(reading_ground_truth.head())
    #print("Shape:", reading_ground_truth.shape)
    
    num_rows = reading_ground_truth.shape[0]

    # compute the scores for the original data
    reading_score, eye_tracking_score = do_reading_eye_task_single_sample(reading_ground_truth)
    reading_scores = [reading_score]
    eye_tracking_scores = [eye_tracking_score]

    # Prepare arguments for parallel execution
    #print("index_set: ", index_set.shape, "max: ", index_set.max(), "min: ", index_set.min())
    subsample_size = index_set.shape[0]
    args_list = [(reading_ground_truth.copy(), index_set[i]) for i in range(index_set.shape[0])]

    print(f"Starting ProcessPoolExecutor with {subsample_size} subsamples...")
    with ProcessPoolExecutor(2) as executor:
        results_iter = executor.map(_sample_and_score_reading_eye, args_list)
        for reading_score, eye_tracking_score in tqdm(results_iter, total=subsample_size, desc="Bootstrapping"):
            reading_scores.append(reading_score)
            eye_tracking_scores.append(eye_tracking_score)

    # Store the lists as a numpy array: shape (2, subsample_size+1)
    
    reading_scores = np.array(reading_scores)
    #print(reading_scores[0:4], "mean: ", reading_scores.mean(), "std: ", reading_scores.std(), "min: ", reading_scores.min(), "max: ", reading_scores.max())

    eye_tracking_scores = np.array(eye_tracking_scores)
    #print(eye_tracking_scores[0:4], "mean: ", eye_tracking_scores.mean(), "std: ", eye_tracking_scores.std(), "min: ", eye_tracking_scores.min(), "max: ", eye_tracking_scores.max())
    
    # save the results
    np.save(save_path+"_reading.npy", reading_scores)
    np.save(save_path+"_eye_tracking.npy", eye_tracking_scores)
    return
    

def main(args):


    print("Loading ground truth dict:")
    ground_truth_dict = extract_ground_truth(args.ground_truth_path)  
    print("Done loading")


    model_path = args.model_path
    subsample_size = args.subsample_size
    

    print("Now doing model: ",model_path)
    # List all directories in the model_name directory
    checkpoint_dirs = [name for name in os.listdir(model_path)
                        if os.path.isdir(os.path.join(model_path, name))]
    print("Found checkpoint directories:")
    for checkpoint_dir in checkpoint_dirs:
        
        checkpoint_full_path =  os.path.join(model_path, checkpoint_dir)
        print("Now doing checkpoint: ",checkpoint_dir, "on path: ", checkpoint_full_path)

        # load all the information for each task:
        try:
            blimp_data = load_checkpoint_pred(checkpoint_full_path,"blimp")
            blimp_gt = ground_truth_dict["blimp"]
            save_path = os.path.join(checkpoint_full_path, tasks["blimp"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(blimp_data, blimp_gt, save_path, subsample_size, "blimp")
        except:
            print("No blimp data found for checkpoint: ",checkpoint_dir)
            
        try:
            blimp_supp_data = load_checkpoint_pred(checkpoint_full_path,"blimp_supplement")
            blimp_supp_gt = ground_truth_dict["blimp_supplement"]
            save_path = os.path.join(checkpoint_full_path, tasks["blimp_supplement"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(blimp_supp_data, blimp_supp_gt, save_path, subsample_size, "blimp_supplement")
        except:
            print("No blimp_supplement data found for checkpoint: ",checkpoint_dir)
            
        try:    
            entity_tracking_data = load_checkpoint_pred(checkpoint_full_path,"entity_tracking")
            entity_tracking_gt = ground_truth_dict["entity_tracking"]
            save_path = os.path.join(checkpoint_full_path, tasks["entity_tracking"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(entity_tracking_data, entity_tracking_gt, save_path, subsample_size, "entity_tracking")
        except:
            print("No entity_tracking data found for checkpoint: ",checkpoint_dir)

        try:
            ewok_data = load_checkpoint_pred(checkpoint_full_path,"ewok")
            ewok_gt = ground_truth_dict["ewok"]
            save_path = os.path.join(checkpoint_full_path, tasks["ewok"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(ewok_data, ewok_gt, save_path, subsample_size, "ewok")
        except:
            print("No ewok data found for checkpoint: ",checkpoint_dir)

        try:
            wug_data = load_checkpoint_pred(checkpoint_full_path,"wug")
            wug_gt = ground_truth_dict["wug"]
            save_path = os.path.join(checkpoint_full_path, tasks["wug"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_accuracy_task(wug_data, wug_gt, save_path, subsample_size, "wug")
        except:
            print("No wug data found for checkpoint: ",checkpoint_dir)

        
        # perplexity
        try:
            perplexity_data = np.load(os.path.join(checkpoint_full_path, tasks["perplexity"]["eval_path"].lstrip("/"), "per_batch_losses.npy"))
            save_path = os.path.join(checkpoint_full_path, tasks["perplexity"]["eval_path"].lstrip("/"), "subsampled_scores.npy")
            do_subsamples_perplexity_task(perplexity_data, save_path, subsample_size, "perplexity")
        except:
            print("No perplexity data found for checkpoint: ",checkpoint_dir)
        
        # reading and eye_tracking
        try:
            reading_ground_truth = ground_truth_dict["reading_eye_tracking"]
            load_path = os.path.join(checkpoint_full_path, tasks["reading_eye_tracking"]["eval_path"].lstrip("/"), "prediction.jsonl")
            save_path = os.path.join(checkpoint_full_path, tasks["reading_eye_tracking"]["eval_path"].lstrip("/"), "subsampled_scores")
            do_subsamples_reading_eye_task(load_path, reading_ground_truth, save_path, subsample_size, "reading_eye_tracking")
        except:
            print("No reading_eye_tracking data found for checkpoint: ",checkpoint_dir)
            
        # read in the prediction.jsonl file. (extract the -log(p) and -log(p2) from the file)
                                
        # compute the n=subsample_size scores
        # store the np array.

        print("Done with checkpoint: ",checkpoint_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample predictions for all model checkpoints.")
    parser.add_argument("--model_path", type=str, help="Path to the directory containing model directories.")
    parser.add_argument("--subsample_size", type=int, default=1000, help="Size of the subsample.")
    parser.add_argument("--ground_truth_path", type=str, default="/cluster/home/janulm/thesis/evaluation-pipeline-2025/evaluation_data/full_eval/", help="Path to the directory containing the ground truth predictions.")
    args = parser.parse_args()
    print(args)
    
    main(args)

