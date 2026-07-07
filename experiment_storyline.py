import sys
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
import re
from scipy.stats import bootstrap
import scipy
import matplotlib.pyplot as plt
import os
import math as math
import concurrent.futures

# this lists all tasks that will be read in, # all bootstrap paths can be treated equivalently
# but the Reading, Eye and PPL values need to be read out seperately. from the AVG_ACC parts of the other tasks; 
# r


TASK_INFO = {
     "Perplexity": {
        "value_path": "/tinybabylm_dataset-test/best_temperature_report.txt", 
        "bootstrap_path": "/tinybabylm_dataset-test/subsampled_scores.npy",
        "readout_type": "perplexity",
        "best_type" : "min"
        },
    "BLiMP": {
        "value_path": "/main/zero_shot/causal/blimp/blimp_filtered/best_temperature_report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/blimp/blimp_filtered/subsampled_scores.npy",
        "readout_type": "avg_acc",
        "best_type" : "max"
        },
    "BLiMP Supp.": {
        "value_path": "/main/zero_shot/causal/blimp/supplement_filtered/best_temperature_report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/blimp/supplement_filtered/subsampled_scores.npy",
        "readout_type": "avg_acc",
        "best_type" : "max"
        },
    "Entity Tracking": {
        "value_path": "/main/zero_shot/causal/entity_tracking/entity_tracking/best_temperature_report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/entity_tracking/entity_tracking/subsampled_scores.npy",
        "readout_type": "avg_acc",
        "best_type" : "max"
        },
    "EWoK": {
        "value_path": "/main/zero_shot/causal/ewok/ewok_filtered/best_temperature_report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/ewok/ewok_filtered/subsampled_scores.npy",
        "readout_type": "avg_acc",
        "best_type" : "max"
        },
    "WUG": {
        "value_path": "/main/zero_shot/causal/wug/wug_adj_nominalization/best_temperature_report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/wug/wug_adj_nominalization/subsampled_scores.npy",
        "readout_type": "avg_acc",
        "best_type" : "max"
        },
    "Reading": {
        "value_path": "/main/zero_shot/causal/reading/report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/reading/subsampled_scores_reading.npy",
        "readout_type": "reading",
        "best_type" : "max"
        },
    "Eye Tracking": {
        "value_path": "/main/zero_shot/causal/reading/report.txt", 
        "bootstrap_path": "/main/zero_shot/causal/reading/subsampled_scores_eye_tracking.npy",
        "readout_type": "eye_tracking",
        "best_type" : "max"
        },
}

MEAN_REL_PERFORMANCE_KEY = "MRPvB" # = "Mean rel. performance vs Baseline on tasks except Perplexity."
MEAN_REL_PERFORMANCE_TASK_LIST = list(TASK_INFO.keys())
MEAN_REL_PERFORMANCE_TASK_LIST.remove("Perplexity")
SELECTION_REL_PERFORMANCE_KEY = "MRPvSel"  # Mean rel. performance vs selection means

def read_task_value(value_path: str, readout_type: str) -> float: 
    # load the contents from the file and then readout accordingly: 
    # types are: 
    # .
    try:
        file_contents = open(value_path, 'r').read()
    except Exception as e:
        print(f"Error reading file {value_path}: {e}")
        return float('nan')
    
    if readout_type == "perplexity":
        return float(file_contents.split("Perplexity: ")[1])
    elif readout_type == "reading":
        return float(file_contents.split("SELF-PACED READING SCORE: ")[1])
    elif readout_type == "eye_tracking":
        return float(file_contents.split("SELF-PACED READING SCORE: ")[0].split(":")[1])
    elif readout_type == "avg_acc":
        val = float(file_contents.split("### AVERAGE ACCURACY")[1])
        assert val < 100.0 and val > 0.0 and not np.isnan(val), f"Average accuracy is out of range: {val}"
        return val
    else:
        raise ValueError(f"Invalid readout type: {readout_type}")
    
 
def load_bootstrap_data(bootstrap_path: str, readout_type: str) -> np.ndarray:
    data = np.load(bootstrap_path)
    value = data[0]
    distr = data[1:]

    if readout_type == "avg_acc":
        value = value*100
        distr = distr*100

    return value, distr

def _process_model_dir(args):
    experiment_dir, model_dir = args
    import os  # Needed for subprocesses
    data = []
    full_model_dir = os.path.join(experiment_dir, model_dir)
    #print(f"Processing model: {model_dir}")
    seed = int(model_dir.split('_')[-1])
    model_name = "_".join(model_dir.split('_')[:-1])
    assert model_name+"_"+str(seed) == model_dir
    try:
        max_task_results = {"model_name": model_name, "seed": seed, "step": "max"}
        for step_dir in os.listdir(full_model_dir):
            step_num = int(step_dir.split('_')[1])
            checkpoint_data = {"model_name": model_name, "seed": seed, "step": step_num}
            for task_name, task_info in TASK_INFO.items():
                value_path = full_model_dir+"/"+step_dir+task_info['value_path']
                bootstrap_path = full_model_dir+"/"+step_dir+task_info['bootstrap_path']
                value = read_task_value(value_path, task_info['readout_type'])
                stats_mean, distr = load_bootstrap_data(bootstrap_path, task_info['readout_type'])
                assert abs(float(stats_mean.round(2)) - round(value,2)) <= 0.01+1e-7, f"Stats mean and value do not match for {task_name} at step {step_num}, value: {value}, stats_mean: {stats_mean}, stats_mean.round(2): {stats_mean.round(2)} round(value,2): {round(value,2)}"
                task_data = {
                    "value": value,
                    "stats_mean": stats_mean,
                    "distr": distr
                }
                checkpoint_data[task_name] = task_data
                if task_name in max_task_results:
                    curr_data_val = max_task_results[task_name]["value"]
                    if TASK_INFO[task_name]['best_type'] == "max":
                        if value > curr_data_val:
                            max_task_results[task_name] = task_data
                    elif TASK_INFO[task_name]['best_type'] == "min":
                        if value < curr_data_val:
                            max_task_results[task_name] = task_data
                    else:
                        raise ValueError(f"Invalid best_type: {TASK_INFO[task_name]['best_type']}")
                else:
                    max_task_results[task_name] = task_data
            
            # now that all tasks are processed, we can compute the mean task: 
            mean_task_value = np.mean([checkpoint_data[task_name]['value'] for task_name in TASK_INFO.keys() if TASK_INFO[task_name]['best_type'] == "max"])
            checkpoint_data["Mean Tasks"] = { "value": mean_task_value}
            checkpoint_data[MEAN_REL_PERFORMANCE_KEY] = { "value": np.nan}
            #print("added mean task: ", mean_task_value, "for model: ", model_name)
            
            data.append(checkpoint_data)
        data.append(max_task_results)
    except Exception as e:
        print(f"Error processing model {model_dir}: {e}")
    return data




def create_experiment_dataframe(experiment_dir: str) -> pd.DataFrame:
    """
    Create a DataFrame containing all experiment data.
    Args:
        experiment_dir: Path to the experiment directory containing model directories
    Returns:
        DataFrame with columns # model_name, seed , step, TASK_value, TASK_stats_mean, TASK_distr x {blimp, blimp supp, etc....}
    """



    data = []
    print("will process: ", experiment_dir)
    print("models: ", os.listdir(experiment_dir))
    model_dirs = os.listdir(experiment_dir)
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        results = list(executor.map(_process_model_dir, [(experiment_dir, model_dir) for model_dir in model_dirs]))
        print("got all model loading results")
    for model_data in results:
        data.extend(model_data)
    df = pd.DataFrame(data)
    return df


def max_seeds_to_mean_method(df):
    # for each unique model_name, find out how many seeds are there
    # then it computes the mean over the seeds and accumulates the distribution over the seeds for each task: 
    # then it returns a new dataframe with the mean and distribution for each task
    # first, we need to find the max seed for each model
    df = df[df['step'] == 'max']
    # first find out how many seeds are there for each model_name
    num_seeds = df.groupby('model_name')['seed'].nunique()
    print(num_seeds)
    # then it computes the mean over the seeds and accumulates the distribution over the seeds for each task: 
    # then it returns a new dataframe with the mean and distribution for each task

    unique_models = df['model_name'].unique()
    new_data = []
    for model_name in unique_models:
        model_data = df[df['model_name'] == model_name]
        # compute the mean over the seeds and accumulate the distribution over the seeds for each task: 
        # then it returns a new dataframe with the mean and distribution for each task
        
        model_mean_max = {"model_name": model_name, "step": "max", "seed": "mean", "num_seeds": len(model_data['seed'].unique())} 
        for task_name, task_info in TASK_INFO.items():
            task_data = {}
            mean_value = []
            stats_mean = []
            distr = []
            for seed in model_data['seed'].unique():
                seed_data_series = model_data[model_data['seed'] == seed]
                assert len(seed_data_series) == 1, f"Expected 1 row for seed {seed} but got {len(seed_data_series)}"
                seed_data = seed_data_series.iloc[0]
                #task_data[seed] = seed_data[task_name]
                #print("seed_data[task_name]: ", seed_data[task_name], "type: ", type(seed_data[task_name]))
                #print("seed_data[task_name]['value']: ", seed_data[task_name]['value'], "type: ", type(seed_data[task_name]['value']))
                mean_value.append(seed_data[task_name]['value'])
                distr.append(seed_data[task_name]['distr'])
                stats_mean.append(seed_data[task_name]['stats_mean'])
            # store the various values for the the aggregated ones and the seed means
            task_data["value"] = np.mean(mean_value)
            task_data["value_std"] = np.std(mean_value, ddof=1) / np.sqrt(len(mean_value))
            task_data["stats_mean"] = np.mean(stats_mean)
            #print("distr0shape", distr[0].shape)
            stacked_distr = np.concatenate(distr)
            array_distr = np.array(distr)
            #print("stracked distr. shape",stacked_distr.shape)
            task_data["distr"] = array_distr
            task_data["distr_mean"] = np.mean(array_distr, axis=0)
            task_data["distr_std"] = np.std(stacked_distr,ddof=1) / np.sqrt(len(distr))
            task_data["seed_means"] = mean_value
            model_mean_max[task_name] = task_data

        # now that all tasks are processed, we can compute the mean task: 
        mean_task_value = np.mean([model_mean_max[task_name]['value'] for task_name in TASK_INFO.keys() if TASK_INFO[task_name]['best_type'] == "max"])
        model_mean_max["Mean Tasks"] = { "value": mean_task_value} 
        model_mean_max[MEAN_REL_PERFORMANCE_KEY] = { "value": np.nan}
        #print("added mean task: ", mean_task_value, "for model: ", model_name)


        new_data.append(model_mean_max)
    df = pd.DataFrame(new_data)
    return df


import re

def escape_latex(text):
    """
    Escapes LaTeX special characters in a string.
    """
    if not isinstance(text, str):
        return text
    # Escape \ first to avoid double escaping
    text = text.replace('\\', r'\textbackslash{}')
    # Escape other special characters
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, escape in special_chars.items():
        text = text.replace(char, escape)
    return text

import copy

def mark_best_models(df_in):
    """
    Adds boolean columns to df indicating if the value is the best for each task.
    For 'perplexity', lower is better; for others, higher is better.
    """
    unique_tasks = list(TASK_INFO.keys())
    unique_tasks.append("Mean Tasks")
    df = df_in.copy()
    #df = df.applymap(copy.deepcopy) -> to silence deprecation warning
    df = df.map(copy.deepcopy)
    unique_tasks.insert(0, MEAN_REL_PERFORMANCE_KEY)
    for task in unique_tasks:
        if task not in df.columns:
            raise ValueError(f"Task {task} not found in dataframe")
        if task in TASK_INFO:
            task_best_type = TASK_INFO[task]["best_type"]
        else:
            task_best_type = "max" # this is the case for the mean task as well as MRPvB... as its not in the TASK_INFO...

        # go over all rows and check if the value is the best for that task, mark current best_index
        curr_best_index = None
        curr_best_value = None
        # THIS WEIRD initializer fixes the string comparison issue... 
        if task == MEAN_REL_PERFORMANCE_KEY:
            curr_best_value = np.nan
        

        for index, row in df.iterrows():
            if task_best_type == "min":
                if curr_best_value is None or row[task]["value"] < curr_best_value:
                    curr_best_index = index
                    curr_best_value = row[task]["value"]
            elif task_best_type == "max":
                if isinstance(row[task]["value"], str):
                    val = float(row[task]["value"].split("\\%")[0])
                    #print("extracted val: ", val, "for task: ", task, "at index: ", index, "curr_best_value: ", curr_best_value)
                    if not np.isnan(val):
                        #print("val is not nan")
                        if curr_best_index is None or val > curr_best_value:
                            curr_best_index = index
                            curr_best_value = val
                            #print(f"NEW curr_best_value: {curr_best_value} for task: {task} at index: {curr_best_index}")
                        else:
                            #print("val is not better than curr_best_value")
                            pass
                else:
                    if curr_best_value is None or row[task]["value"] > curr_best_value:
                        curr_best_index = index
                        curr_best_value = row[task]["value"]
            else:
                raise ValueError(f"Invalid best_type: {task_best_type}")
        
        #if task == MEAN_REL_PERFORMANCE_KEY:
        df.loc[curr_best_index, task]["is_best"] = True

    return df


def generate_latex_code(df, unique_tasks, table_caption="Results Table", table_label="results", include_num_seeds=True, include_mean_task=True,dual_column=True):
   
    table1 = generate_latex_code_table_single(df, unique_tasks, table_caption=table_caption, table_label=table_label, include_num_seeds=include_num_seeds, include_mean_task=include_mean_task,dual_column=True)
    table2 = generate_latex_code_table_single(df, [unique_tasks[0]], table_caption=table_caption, table_label=table_label+"Short", include_num_seeds=include_num_seeds, include_mean_task=include_mean_task,dual_column=False)
    table3 = generate_latex_code_table_single(df, unique_tasks, table_caption=table_caption, table_label=table_label+"NoRel", include_num_seeds=include_num_seeds, include_mean_task=include_mean_task,dual_column=True, include_relative_improvement=False)

    return table1 + "\n\n" + table2 + "\n\n" + table3

def generate_latex_code_table_single(df, unique_tasks_in, table_caption="Results Table", table_label="results", include_num_seeds=False, include_mean_task=True,dual_column=True, include_relative_improvement=True,tablePrefix="Iterative"):
    unique_tasks = copy.deepcopy(unique_tasks_in)
    
    if include_mean_task:
        # insert at index 0 (others should be pushed back in the list )
        unique_tasks.insert(0, MEAN_REL_PERFORMANCE_KEY)

    if not isinstance(unique_tasks, list):
        unique_tasks = list(unique_tasks)
    col_format = "l" + "|" 
    if include_mean_task:
        col_format += "l|" + "l" * (len(unique_tasks)-1)
    else:
        col_format += "l" * (len(unique_tasks))
    if include_num_seeds:
        col_format += "|l"
    lines = []
    lines.append("\\newcommand{\\table"+tablePrefix+table_label+"}{")
    lines.append("\\begin{table*}[!h]" if dual_column else "\\begin{table}[!h]") # FOR THESIS
    #lines.append("\\begin{table*}[!t]" if dual_column else "\\begin{table}[H]") FOR PAPER
    lines.append("\\centering")
    lines.append("\\begin{adjustbox}{max width=\\linewidth}")
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")
    # Escape header and add arrows
    header = [escape_latex("Name")]
    for task in unique_tasks:
        
        if task in TASK_INFO:
            task_best_type = TASK_INFO[task]["best_type"]
        else:
            task_best_type = "max" # this is the case for the mean task... as its not in the TASK_INFO...
        
        if task_best_type == "min":
            arrow = "$\\downarrow$"
        elif task_best_type == "max":
            arrow = "$\\uparrow$"
        else:
            raise ValueError(f"Invalid best_type: {task_best_type}")
        header.append(f"{escape_latex(task)}{arrow}")
    if include_num_seeds:
        header.append(escape_latex("n_seeds"))
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        row_items = [escape_latex(str(row['model_name']))]
        for task in unique_tasks:
            #print("will try to extract task: ", task, f"row: {row}")
            model_task_data = row.get(task, None)
            if not model_task_data:
                raise ValueError("Extraction issue for task")
            #print(model_task_data)
            # getting the numbers that describe the performance of that task/method
            mean = model_task_data.get("value",float('nan'))
            #
            value_std_err = model_task_data.get("value_std", float('nan'))
            
            distr_std_err = model_task_data.get("distr_std", float('nan'))
            #print("For task",task,"got distrstdd",distr_std_err, "value std_err", value_std_err)
            #
            is_best = model_task_data.get(f'is_best', False)
            ##
            significant_data = model_task_data.get(f"significant", False)
            if significant_data:
                is_significant = significant_data["result"] == "better" or significant_data["result"] == "worse"
                #print("FOUND SIGNIFICANT", significant_data["result"], significant_data["result"] == "better")
            else:
                is_significant = False
            relative_improvement = model_task_data.get("relative_improvement", float('nan'))

            if pd.notna(mean):
                if isinstance(mean, str):
                    cell = mean
                    if is_best:
                        cell = f"\\textbf{{{cell}}}"
                elif pd.notna(distr_std_err):
                    cell = f"{mean:.2f}$\\pm${distr_std_err:.2f}"
                else:
                    cell = f"{mean:.2f}"
                if is_best:
                    cell = f"\\textbf{{{cell}}}"
                # If significant, wrap in brackets (after bolding if present)
                if is_significant:
                    #print("FOUND SIGNIFICANT")
                    if cell.startswith("\\textbf{") and cell.endswith("}"):
                        cell = "\\textbf{"+cell[8:-1]+"}$^{*}$"
                    else:
                        cell = cell+"$^{*}$"
                if pd.notna(relative_improvement) and include_relative_improvement:
                    cell = f"{cell} ({(relative_improvement*100):.2f}\%)"
            else:
                cell = "-"
            row_items.append(str(cell))
        # num_seeds
        if include_num_seeds:
            num_seeds = row.get("num_seeds", float("nan"))
            if pd.notna(num_seeds):
                num_seeds = int(num_seeds)
            else:
                num_seeds = str(num_seeds)
            row_items.append(str(num_seeds))
        lines.append(" & ".join(row_items) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")
    lines.append("\\caption{"+escape_latex(table_caption) + ". $^{*}$ marks statistical significance; percentages indicate relative change vs Baseline." + "}")
    lines.append("\\label{"+escape_latex("tab:"+table_label)+"}")
    lines.append("\\end{table*}" if dual_column else "\\end{table}")
    lines.append("}")
    latex_code = "\n".join(lines)
    return latex_code

import os

def append_latex_to_file(latex_code, file_path):
    """
    Appends the given LaTeX code to the specified file.
    If the file does not exist, it will be created.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Append the LaTeX code to the file
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n% --- New Table ---\n")
        f.write(latex_code)
        f.write("\n\n")


def reset_latex_file(file_path):
    """
    Resets (clears) the specified LaTeX file. Use at the start of your script/notebook.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Clear the file and add a header comment
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("% All tables generated by script.\n\n")


from decimal import Decimal, ROUND_DOWN
def check_statistics_bootstrap(values_A,values_B, task, name_A,name_B, verbose,task_best_type):
    # the values are numpy arrays that are of of the shape (num_seeds,1000 = subsample_size) and are paired bootstrapped means subsampled from index sets on the tasks;
    # we now average out the noise across the seeds and compute the difference between the two arrays to get a sense of the statistical significance of the difference;
    
    #print("values_A: ", values_A.shape, "mean: ", values_A.mean(), "std: ", values_A.std())
    #print("values_B: ", values_B.shape, "mean: ", values_B.mean(), "std: ", values_B.std())
    values_A = values_A.mean(axis=0)
    values_B = values_B.mean(axis=0)
    
    # Assume a_samples and b_samples are your bootstrapped values
    diff = values_A - values_B
   
    lower = np.percentile(diff, 2.5, method="inverted_cdf")
    upper = np.percentile(diff, 97.5, method="inverted_cdf")

    #print("95% CI for (A - B):", (lower, upper))

    # compute the one sided p_value: 

    # count the indices in diff that are greater than 0: 
    
    # THERE IS A ROUNDING ISSUE WHEN PRINTING THE P_VALUE, it actually rounds up 
    

    def fmt_trunc(x, ndp=5):
        q = Decimal('1').scaleb(-ndp)         # 10**(-ndp)
        d = Decimal(str(x)).quantize(q, rounding=ROUND_DOWN)  # truncate toward zero
        return f"{d:.{ndp}f}"

    effect_size = diff.mean()
    
    # if the p_value is less than 0.05, then the difference is significant

    if task_best_type == "max":
        num_better = np.sum(diff > 0)
        num_worse = np.sum(diff < 0)
        # compute the p_value: 
        p_value_better = (num_better + 1) / (len(diff) + 1)
        p_value_worse = (num_worse + 1) / (len(diff) + 1)
        if lower > 0:
            if verbose:
                print(f"{name_B} is significantly worse (effect size {effect_size:.3f}) than {name_A} in {task} at 95% confidence with p_value {fmt_trunc(p_value_worse, 5)}, {num_worse} worse, {num_better} better, {len(diff)} total")
            return "worse", (lower, upper), diff, p_value_worse
        elif upper < 0:
            if verbose:
                print(f"{name_B} is significantly better (effect size {effect_size:.3f}) than {name_A} in {task} at 95% confidence with p_value {fmt_trunc(p_value_better, 5)}, {num_better} better, {num_worse} worse, {len(diff)} total")
            return "better", (lower, upper), diff, p_value_better
        else:
            return "no_difference", (lower, upper), diff, p_value_better
    elif task_best_type == "min":
        num_better = np.sum(diff < 0)
        num_worse = np.sum(diff > 0)
        # compute the p_value: 
        p_value_better = float(num_better + 1) / float(len(diff) + 1)
        p_value_worse = float(num_worse + 1) / float(len(diff) + 1)
        if lower > 0:
            if verbose:
                print(f"{name_B} is significantly better (effect size {effect_size:.3f}) than {name_A} in {task} at 95% confidence with p_value {fmt_trunc(p_value_better, 5)}, {num_better} better, {num_worse} worse, {len(diff)} total")
            return "better", (lower, upper), diff, p_value_better
        elif upper < 0:
            if verbose:
                print(f"{name_B} is significantly worse (effect size {effect_size:.3f}) than {name_A} in {task} at 95% confidence with p_value {fmt_trunc(p_value_worse, 5)}, {num_worse} worse, {num_better} better, {len(diff)} total")
            return "worse", (lower, upper), diff, p_value_worse
        else:
            return "no_difference", (lower, upper), diff, p_value_better
    else:
        raise ValueError(f"Invalid task_best_type: {task_best_type}")


def check_statistics(baseline_df, experiment_df, verbose=False):
    # this should go over all rows in the experiment_df and check if the value is significantly better than the baseline_df
    # for each task, 
    # we also want to compute the relative improvement and mark if the improvement is significant! 

    # check baseline only has one row:
    assert len(baseline_df) == 1, "Baseline df must have only one row"
    baseline_name = baseline_df.iloc[0]["model_name"]
    # go over all rows in the experiment_df and check if the value is significantly better than the baseline_df
    for index, row in experiment_df.iterrows():
        #print(row)
        current_model = row['model_name']
        
        mean_rel_performance_accumulator = []
        
        for task in TASK_INFO.keys():
            exp_task_data = row[task]
            baseline_task_data = baseline_df.iloc[0][task]
            # check if the value is significantly better than the baseline_df
            #print("baseline_task_data: ", baseline_task_data)
            #print("exp_task_data: ", exp_task_data)

            result, ci, diff, p_value = check_statistics_bootstrap(baseline_task_data["distr"],exp_task_data["distr"],task, baseline_name, current_model, verbose, TASK_INFO[task]["best_type"])
            experiment_df.loc[index, task]["significant"] = { "result": result, "ci": ci, "diff": diff, "p_value": p_value}
            
            if TASK_INFO[task]["best_type"] == "max":
                relative_improvement = (exp_task_data["value"] - baseline_task_data["value"]) / baseline_task_data["value"]
            elif TASK_INFO[task]["best_type"] == "min":
                relative_improvement = (baseline_task_data["value"] - exp_task_data["value"]) / baseline_task_data["value"]
            else:
                raise ValueError(f"Invalid best_type: {TASK_INFO[task]['best_type']}")
            experiment_df.loc[index, task]["relative_improvement"] = relative_improvement

            # accumulate the relative improvement for all tasks except Perplexity: 
            if task in MEAN_REL_PERFORMANCE_TASK_LIST:
                mean_rel_performance_accumulator.append(relative_improvement)

        # now compute the relative improvement of the mean task: 
        mean_task_data = row["Mean Tasks"]
        baseline_mean_task_data = baseline_df.iloc[0]["Mean Tasks"]
        relative_improvement = (mean_task_data["value"] - baseline_mean_task_data["value"]) / baseline_mean_task_data["value"]
        experiment_df.loc[index, "Mean Tasks"]["relative_improvement"] = relative_improvement
        experiment_df.loc[index, MEAN_REL_PERFORMANCE_KEY]["value"] = str(f"{float(np.mean(mean_rel_performance_accumulator) * 100.0):.2f}"+ "\%")

    return experiment_df


#########################################################################################
    # RARITY SCORE: Compute percentileofscore for each checkpoint (model_base, seed, step, task)
from scipy.stats import percentileofscore
"""
def rarity_score(df_split):
    # For each row in df_split, compute the percentile of its value among all values for the same task
    rarity_scores = []
    for idx, row in df_split.iterrows():
        task = row['task']
        value = row['value']
        # Get all values for this task
        task_values = df_split[df_split['task'] == task]['value']
        # For perplexity, lower is better, so use kind='rank' and invert percentile
        if task == 'Perplexity':
            percentile = 100 - percentileofscore(task_values, value, kind='rank')
        else:
            percentile = percentileofscore(task_values, value, kind='rank')
        rarity_scores.append(percentile)
    df_split['rarity_percentile'] = rarity_scores

    print("\nRarity (percentileofscore) for each checkpoint (model_base, seed, step, task):")
    print(df_split[['model_base', 'seed', 'step', 'task', 'value', 'rarity_percentile']].head(20))

    # Print perplexity checkpoints with lower perplexity values
    perplexity_df = df_split[df_split['task'] == 'Perplexity'].sort_values('value', ascending=True)
    print("\nCheckpoints with lowest perplexity:")
    print(perplexity_df[['model_base', 'seed', 'step', 'task', 'value', 'rarity_percentile']].head(20))

    # please combine the model_base, seed, step into checkpoint name and drop the other columns, 
    df_split['checkpoint'] = df_split.apply(lambda row: f"{row['model_base']}_seed{row['seed']}_step{row['step']}", axis=1)
    df_rarity = df_split[['checkpoint', 'task', 'value', 'rarity_percentile']].copy()
    print("\nRarity DataFrame with checkpoint names:")
    print(df_rarity.head(20))

    # this currently looks like this: Rarity DataFrame with checkpoint names:
    #                          checkpoint              task      value  rarity_percentile
    # 435   base_llama_8000_seed42_step500             blimp  64.560000             2.8125
    # 434   base_llama_8000_seed42_step500  blimp_supplement  57.990000             4.3750
    # 437   base_llama_8000_seed42_step500   entity_tracking  14.250000             6.2500
    # 436   base_llama_8000_seed42_step500              ewok  51.030000             7.5000
    # 432   base_llama_8000_seed42_step500        perplexity  46.961004             3.1250
    # 433   base_llama_8000_seed42_step500               wug  56.000000            38.4375
    # 387  base_llama_8000_seed42_step1000             blimp  67.930000            25.6250
    
    # now start with a new df (or initially a empty list of entries) and then go over all unique checkpoints, over each task, and fill the table with the columns {task}_score, {task}_rarity_percentile and then the mean_percentile and product_percentile for each checkpoint. 
    # one checkpoint should be one row in the df, then sort by mean_percentile and product_percentile and then print the top 10 checkpoints. 

    # Pivot the table to have one row per checkpoint, with columns for each task's score and rarity_percentile
    tasks = df_rarity['task'].unique()
    # Create a dict for each checkpoint
    checkpoint_rows = []
    for checkpoint in df_rarity['checkpoint'].unique():
        row = {'checkpoint': checkpoint}
        checkpoint_data = df_rarity[df_rarity['checkpoint'] == checkpoint]
        percentiles = []
        for task in tasks:
            task_row = checkpoint_data[checkpoint_data['task'] == task]
            if not task_row.empty:
                score = task_row['value'].values[0]
                rarity = task_row['rarity_percentile'].values[0]
                assert len(task_row) == 1, f"Expected 1 row for task {task} but got {len(task_row)}"
                row[f'{task}_score'] = score
                row[f'{task}_rarity_percentile'] = rarity
                percentiles.append(rarity)
            else:
                row[f'{task}_score'] = None
                row[f'{task}_rarity_percentile'] = None
        # Compute mean and product of percentiles (ignoring None)
        valid_percentiles = [p for p in percentiles if p is not None]
        if valid_percentiles:
            row['mean_percentile'] = np.mean(valid_percentiles)
        else:
            row['mean_percentile'] = None
        checkpoint_rows.append(row)
    df_checkpoints = pd.DataFrame(checkpoint_rows)

    # Sort by mean_percentile and product_percentile (descending)
    df_checkpoints_sorted_mean = df_checkpoints.sort_values('mean_percentile', ascending=False)

    print("\nTop 10 checkpoints by mean_percentile:")
    print(df_checkpoints_sorted_mean.head(10))
"""

import pandas as pd
from typing import Iterable, Tuple, Optional, List

def filter_and_rename(
    df: pd.DataFrame,
    pairs: Iterable[Tuple[str, str]],
    *,
    match_column: str = "model_name",
    rename_column: Optional[str] = None,
    regex: bool = False,
    case: bool = True,
    na: bool = False,
) -> pd.DataFrame:
    """
    Select rows whose `match_column` equals any value in `pairs`, rename them,
    and return a deep-copied subset.

    Args:
        df: Input DataFrame.
        pairs: Iterable of (value, new_name). Rows where `match_column` equals
               `value` are selected. The second becomes the value written to
               `rename_column` (or `match_column` if rename_column is None).
               Order matters: first match wins.
        match_column: Column to match against (default: 'model_name').
        rename_column: Column to write the new_name into. If None, uses match_column.
        regex: Ignored. Kept for backward-compatibility.
        case: Case sensitivity for equality (default: True).
        na: Ignored. NaNs never match in equality checks.

    Returns:
        A deep-copied DataFrame containing only the matched rows, with names
        rewritten according to `pairs`. Original `df` is not modified.
    """
    if match_column not in df.columns:
        raise ValueError(f"`{match_column}` not found in DataFrame columns.")
    target_col = rename_column or match_column

    # Track rows not yet taken so a row can't appear multiple times.
    remaining = pd.Series(True, index=df.index)
    parts: List[pd.DataFrame] = []

    # Ensure string dtype for matching column to avoid errors
    col = df[match_column].astype(str)

    for pattern, new_name in pairs:
        if case:
            match_series = col == str(pattern)
        else:
            match_series = col.str.lower() == str(pattern).lower()
        mask = remaining & match_series
        if not mask.any():
            continue
        chunk = df.loc[mask].copy(deep=True)      # deep copy of just the matched rows
        chunk[target_col] = new_name              # rename inside the copy
        parts.append(chunk)
        remaining &= ~mask                        # prevent duplicates: first match wins

    if not parts:
        # Return an empty deep-copied frame with same columns if nothing matched
        return df.iloc[0:0].copy(deep=True)

    out = pd.concat(parts, axis=0)
    return out.copy(deep=True)  # explicit deep copy of the result


def find_best_perplexity_for_each_seed(df):
    """
    For each unique (model_name, seed) combination, find the step with the lowest perplexity value.
    Returns a filtered dataframe containing only the rows with the best perplexity for each seed.
    
    Args:
        df: DataFrame with columns model_name, seed, step, and Perplexity (containing dict with 'value' key)
    
    Returns:
        DataFrame: Filtered dataframe with only the best perplexity checkpoints for each seed
    """
    best_checkpoints = []
    
    # Group by model_name and seed
    for (model_name, seed), group in df.groupby(['model_name', 'seed']):
        # Skip rows where step is 'max' as these are aggregated rows
        group = group[group['step'] != 'max']
        
        if group.empty:
            print("WARNING: No rows for model_name: ", model_name, "and seed: ", seed)
            
        # Find the row with the lowest perplexity value
        min_perplexity_idx = None
        min_perplexity_value = float('inf')
        
        for idx, row in group.iterrows():
            perplexity_value = row['Perplexity']['value']
            if perplexity_value < min_perplexity_value:
                min_perplexity_value = perplexity_value
                min_perplexity_idx = idx
        
        # Add the best checkpoint to our list
        if min_perplexity_idx is not None:
            best_checkpoints.append(df.loc[min_perplexity_idx].to_dict())
    
    # Create a new dataframe with only the best checkpoints
    if best_checkpoints:
        result_df = pd.DataFrame(best_checkpoints)
        return result_df
    else:
        return pd.DataFrame()


def compute_mean_rel_performance(df_in):
    """
    Compute per-task selection means over the provided selection (df_in), then for each row
    compute the relative performance vs those means for every task, and finally compute an
    aggregate mean relative performance across tasks excluding Perplexity.

    For tasks with best_type == "max": rel = (value - mean) / mean.
    For tasks with best_type == "min": rel = (mean - value) / mean.

    The function returns a deep-copied dataframe where:
      - Each task dict gains key 'relative_vs_selection_mean' (float)
      - Column MRPvB contains a dict with key 'value' holding the average relative improvement
        across tasks in MEAN_REL_PERFORMANCE_TASK_LIST (as a float fraction, not percent)
    """
    if df_in is None or len(df_in) == 0:
        return df_in

    # Compute selection means per task across the provided selection
    selection_task_means = {}
    for task_name in TASK_INFO.keys():
        vals = []
        for _, row in df_in.iterrows():
            task_cell = row.get(task_name)
            if isinstance(task_cell, dict):
                val = task_cell.get('value', np.nan)
                if pd.notna(val):
                    vals.append(float(val))
        selection_task_means[task_name] = float(np.mean(vals)) if len(vals) > 0 else float('nan')

    # Work on a deep copy at the row level and create shallow copies of task dicts
    df = df_in.copy(deep=True)
    # Ensure we don't mutate original task dicts
    for idx in df.index:
        for task_name in TASK_INFO.keys():
            cell = df.at[idx, task_name]
            if isinstance(cell, dict):
                df.at[idx, task_name] = dict(cell)

    # Pre-create the aggregate column to avoid alignment issues when assigning dicts
    df[SELECTION_REL_PERFORMANCE_KEY] = [None] * len(df)

    # Annotate rows with per-task relative performance vs selection mean
    for idx, row in df.iterrows():
        rel_accumulator = []
        for task_name, task_info in TASK_INFO.items():
            task_cell = row.get(task_name)
            if not isinstance(task_cell, dict):
                continue
            val = task_cell.get('value', np.nan)
            ref = selection_task_means.get(task_name, np.nan)
            if pd.isna(val) or pd.isna(ref) or ref == 0:
                rel = float('nan')
            else:
                if task_info['best_type'] == 'max':
                    rel = (float(val) - float(ref)) / float(ref)
                elif task_info['best_type'] == 'min':
                    rel = (float(ref) - float(val)) / float(ref)
                else:
                    raise ValueError(f"Invalid best_type: {task_info['best_type']}")
            # safely assign to a single cell with a dict present
            df.at[idx, task_name]['relative_vs_selection_mean'] = rel

            # accumulate only for non-perplexity tasks in the aggregate
            if task_name in MEAN_REL_PERFORMANCE_TASK_LIST and pd.notna(rel):
                rel_accumulator.append(rel)

        # Aggregate mean relative performance across non-Perplexity tasks
        mean_rel = float(np.mean(rel_accumulator)) if len(rel_accumulator) > 0 else float('nan')
        # Assign dict into single cell (use list to avoid Series alignment in edge cases)
        df.loc[[idx], SELECTION_REL_PERFORMANCE_KEY] = [{"value": mean_rel}]

    return df

def main():

    READ_FROM_PICKLE = True
    
    print(("\n########### SECTION: LOADING BASELINE #############\n"))

    
    if READ_FROM_PICKLE:
        baseline_data = pd.read_pickle("/cluster/home/janulm/thesis/baseline_data.pkl")
    else:
        baseline_data = create_experiment_dataframe("/cluster/work/cotterell/janulm/thesis_data/eval_results/baseline_training")
        baseline_data.to_pickle("/cluster/home/janulm/thesis/baseline_data.pkl")

    print(baseline_data.head(10), "in total: ", len(baseline_data))

    print(("\n########### SECTION: BASELINE- MEAN REL PERFORMANCE #############\n"))
    best_perplexity_checkpoints = find_best_perplexity_for_each_seed(baseline_data)
    print(best_perplexity_checkpoints.head(10), "in total: ", len(best_perplexity_checkpoints))

    best_perplexity_checkpoints_rel = compute_mean_rel_performance(best_perplexity_checkpoints)
    print(best_perplexity_checkpoints_rel.head(10), "in total: ", len(best_perplexity_checkpoints_rel))

    baseline_data_mean = max_seeds_to_mean_method(baseline_data) 

    # raise Exception("Stop here")
    print(("\n########### SECTION: BASELINE-RENAMED #############\n"))
    assert len(baseline_data_mean) == 1, "Baseline data mean should have only one row"
    # change the name to "Baseline"
    baseline_data_mean['model_name'] = 'Baseline'
    print(baseline_data_mean.head(10), "in total: ", len(baseline_data_mean))

    print(("\n########### SECTION: BASELINE-EXPERT #############\n"))
    
    expert_model_table = baseline_data[(baseline_data['model_name'] == 'base_llama_8000') & (baseline_data['step'] == 2500) & (baseline_data['seed'] == 43)]
    assert len(expert_model_table) == 1, "Expected 1 row for expert model table"
    # change the name to expert_model
    expert_model_table['model_name'] = 'Good'
    #print(expert_model_table)

    # combine the dataframe with the baseline_data_mean
    expert_model_table = pd.concat([expert_model_table, baseline_data_mean])
    print("combined expert_model_table and mean baseline: \n", expert_model_table)


    # defining the latex tables: 
    latex_file_path = "/cluster/home/janulm/thesis/ms-thesis/all_tables.tex"
    reset_latex_file(latex_file_path)

    unique_tasks = list(TASK_INFO.keys())

    # Usage in Jupyter:

    
    selection = filter_and_rename(expert_model_table, [("Good", "Good"), ("Baseline", "Baseline")])
    latex_code = generate_latex_code(selection, unique_tasks, table_caption="Baseline and Expert Model", table_label="BaselineExpert", include_mean_task=False)
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)

    
    print(("\n########### SECTION: LOADING EXPERIMENTS-RUN 1 #############\n"))

    if READ_FROM_PICKLE:
        experiment_data = pd.read_pickle("/cluster/home/janulm/thesis/experiment_data.pkl")
    else:
        experiment_data = create_experiment_dataframe("/cluster/work/cotterell/janulm/thesis_data/eval_results/8000_series_all")
        # backing up the experiment data
        experiment_data.to_pickle("/cluster/home/janulm/thesis/experiment_data.pkl")

    # sort the rows by model_name
    experiment_data = experiment_data.sort_values(by='model_name')

    print(experiment_data.head(10), "in total: ", len(experiment_data))


    print(("\n########### SECTION: EXPERIMENT- MEAN REL PERFORMANCE - GEN1 SYNTHETIC MODELS #############\n"))
    # filter for only rows where the model_name contains "_500_cs1_mr03" or "no_contrast"
    best_perplexity_checkpoints_rel_exp = find_best_perplexity_for_each_seed(experiment_data[experiment_data['model_name'].str.contains("_500_cs1_mr03") | experiment_data['model_name'].str.contains("no_contrast")])
    best_perplexity_checkpoints_rel_exp = compute_mean_rel_performance(best_perplexity_checkpoints_rel_exp)
    
    # sort the table by column SELECTION_REL_PERFORMANCE_KEY descending
    # Use key parameter to extract 'value' from dictionaries during sorting without modifying the DataFrame
    best_perplexity_checkpoints_rel_exp = best_perplexity_checkpoints_rel_exp.sort_values(
        by=SELECTION_REL_PERFORMANCE_KEY, 
        ascending=False,
        key=lambda x: x.apply(lambda val: val['value'] if isinstance(val, dict) and 'value' in val else val)
    )
    print(best_perplexity_checkpoints_rel_exp.head(20), "in total: ", len(best_perplexity_checkpoints_rel_exp))


    print(("\n########### SECTION: EXPERIMENT- MAX ACROSS STEPS #############\n"))

    # filter for only rows where the step="max"
    experiment_data = experiment_data[experiment_data['step'] == 'max']
    print(experiment_data.head(10), "in total: ", len(experiment_data))

    
    experiment_data_means = max_seeds_to_mean_method(experiment_data)
    
    
    # CHECK STATISTICS
    experiment_data_pre_stat = experiment_data_means.copy()
    experiment_data_means = check_statistics(baseline_data_mean, experiment_data_means)


    experiment_data_means = experiment_data_means.sort_values(by='model_name')
    
    experiment_data_means = pd.concat([baseline_data_mean, experiment_data_means])
    print(experiment_data_means.head(20), "in total: ", len(experiment_data_means))


    # generating latex table for all: 
    print(("\n############ SECTION: GENERATING LATEX TABLES ############\n"))


    latex_code = generate_latex_code(mark_best_models(experiment_data_means.copy()), unique_tasks, table_caption="All Models", table_label="AllModels")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)
    


    # MODEL SIZE EXPERIMENT
    print("BEFORE:",experiment_data_means["model_name"].unique())
    model_names = [("g2500_small_"+str(i)+"_cs1_mr03", "CD-Small-"+str(i)) for i in [10,20,50,100]]
    model_names.insert(0,("Baseline", "Baseline"))
    model_size_selection = filter_and_rename(experiment_data_means, model_names)
    print("AFTER:",model_size_selection["model_name"].unique())
    latex_code = generate_latex_code(mark_best_models(model_size_selection), unique_tasks, table_caption="Model Size Experiment", table_label="ModelSize")
    #latex_code = generate_latex_code(mark_best_models(experiment_data_means[experiment_data_means['model_name'].str.contains("small") | experiment_data_means['model_name'].str.contains("Baseline")].copy()), unique_tasks, table_caption="Model Size Experiment", table_label="ModelSize")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)
    


    # NO CONTRAST EXPERIMENT
    model_names = [("Baseline", "Baseline"), ("g2500_no_contrast_mr03","No-Contrast"), ("g2500_no_contrast_v_head_mr03","No-Contrast-V-Head")]
    model_names.extend([("g2500_no_contrast_topk_"+str(i)+"_mr03", "No-Contrast-TopK-"+str(i)) for i in [50,100,200]])
    model_names.extend([("g2500_no_contrast_topp_"+str(i)+"_mr03", "No-Contrast-TopP-"+str(i)) for i in [90,95,97]])
    model_no_contrast_selection = filter_and_rename(experiment_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(model_no_contrast_selection), unique_tasks, table_caption="No Contrast Experiment", table_label="NoContrast")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)
    

    # DROPOUT EXPERIMENT
    model_names = [("g2500_drop_0"+str(i)+"_cs1_mr03", "CD-Drop-0."+str(i)) for i in [1,3,5,7]]
    model_names.insert(0,("Baseline", "Baseline"))
    model_drop_selection = filter_and_rename(experiment_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(model_drop_selection), unique_tasks, table_caption="Dropout Experiment", table_label="Dropout")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)
    

    # EALIERT STEPS EXPERIMENT
    model_names = [("g2500_"+str(i)+"_cs1_mr03", "CD-Early-"+str(i)) for i in [500,1000,1500,2000]]
    model_names.insert(0,("Baseline", "Baseline"))
    model_early_selection = filter_and_rename(experiment_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(model_early_selection), unique_tasks, table_caption="Early Steps Experiment", table_label="EarlySteps")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)

    
    # MIXING RATIO EXPERIMENT
    
    model_names = [("g2500_500_cs1_mr0"+str(i), "CD-Synth-Ratio-0."+str(i)) for i in [1,2,3,4,5,6,7,8,9]]
    model_names.insert(0,("Baseline", "Baseline"))
    model_mixing_selection = filter_and_rename(experiment_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(model_mixing_selection), unique_tasks, table_caption="Mixing Ratio Experiment", table_label="MixingRatio")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)
     

    #selection = experiment_data_means[experiment_data_means['model_name'].str.contains("Baseline") |experiment_data_means['model_name'].str.contains("small_20") | experiment_data_means['model_name'].str.contains("g2500\_no\_contrast\_mr03") | experiment_data_means['model_name'].str.contains("drop_05_cs1_mr03") | experiment_data_means['model_name'].str.contains("g2500\_500\_cs1\_mr03")].copy()
    selection_names = [("Baseline", "Baseline"), ("g2500_small_20_cs1_mr03","CD-Small-20"), ("g2500_drop_07_cs1_mr03","CD-Drop-0.7"),("g2500_500_cs1_mr03","CD-Early-500")]
    selection = filter_and_rename(experiment_data_means, selection_names)
    latex_code = generate_latex_code(mark_best_models(selection), unique_tasks, table_caption="Overview of all Experiments", table_label="SummaryExperiments")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)

    


    #selection = experiment_data_means[experiment_data_means['model_name'].str.contains("Baseline") | experiment_data_means['model_name'].str.contains("g2500\_no\_contrast\_v\_head\_mr0")| experiment_data_means['model_name'].str.contains("g2500\_no\_contrast\_mr0") | experiment_data_means['model_name'].str.contains("g2500\_500\_cs1\_mr03")].copy()
    selection_names = [("Baseline", "Baseline"),("g2500_no_contrast_mr03","No-Contrast"),("g2500_no_contrast_v_head_mr03","No-Contrast-V-Head"),("g2500_500_cs1_mr03","CD-Early-500")]
    #,("g2500_no_contrast+contrast_mr03","No-Contrast+CD-Early-500") WE ALSO HAVE THIS, BUT SHOULDNT BE IN THE CURRENT VERSION OF THE PAPER
    selection = filter_and_rename(experiment_data_means, selection_names)
    latex_code = generate_latex_code(mark_best_models(selection.copy()), unique_tasks, table_caption="Overview of all Experiments", table_label="SummaryExperimentsSmall")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)


    # Top K and Top P Experiment: 
    print(("\n############ SECTION: TOP K AND TOP P EXPERIMENT ############\n"))

    model_names = [("Baseline", "Baseline"), ("g2500_no_contrast_mr03","No-Contrast"), ("g2500_no_contrast_v_head_mr03","No-Contrast-V-Head")]
    model_names.extend([("g2500_no_contrast_topk_"+str(i)+"_mr03", "No-Contrast-TopK-"+str(i)) for i in [50,100,200]])
    model_names.extend([("g2500_no_contrast_topp_"+str(i)+"_mr03", "No-Contrast-TopP-"+str(i)) for i in [90,95,97]])
    model_names.extend([("g2500_500_cs1_mr03","CD-Early-500")])
    model_names.extend([("g2500_contrast_early_500_topk_"+str(i)+"_mr03", "CD-Early-500-TopK-"+str(i)) for i in [50,100,200]])
    model_names.extend([("g2500_contrast_early_500_topp_"+str(i)+"_mr03", "CD-Early-500-TopP-"+str(i)) for i in [90,95,97]])
    model_top_k_p_selection = filter_and_rename(experiment_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(model_top_k_p_selection), unique_tasks, table_caption="Top K and Top P Experiment", table_label="TopKP")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)



    print(("\n############ SECTION: SECOND GEN MODELS ############\n"))

    if READ_FROM_PICKLE:
        second_gen_data = pd.read_pickle("/cluster/home/janulm/thesis/second_gen_training.pkl")
    else:
        second_gen_data = create_experiment_dataframe("/cluster/work/cotterell/janulm/thesis_data/eval_results/second_gen_training")
        second_gen_data.to_pickle("/cluster/home/janulm/thesis/second_gen_training.pkl")


    # sort the second_gen_data by model_name
    second_gen_data = second_gen_data.sort_values(by='model_name')
    print(second_gen_data.head(10), "in total: ", len(second_gen_data))


    print(("\n########### SECTION: EXPERIMENT- MEAN REL PERFORMANCE - NO ACCUMULATED - GEN2 SYNTHETIC MODELS #############\n"))

    best_perplexity_checkpoints_rel_exp = find_best_perplexity_for_each_seed(second_gen_data)
    best_perplexity_checkpoints_rel_exp = compute_mean_rel_performance(best_perplexity_checkpoints_rel_exp)
    
    # sort the table by column SELECTION_REL_PERFORMANCE_KEY descending
    # Use key parameter to extract 'value' from dictionaries during sorting without modifying the DataFrame
    best_perplexity_checkpoints_rel_exp = best_perplexity_checkpoints_rel_exp.sort_values(
        by=SELECTION_REL_PERFORMANCE_KEY, 
        ascending=False,
        key=lambda x: x.apply(lambda val: val['value'] if isinstance(val, dict) and 'value' in val else val)
    )
    print(best_perplexity_checkpoints_rel_exp.head(20), "in total: ", len(best_perplexity_checkpoints_rel_exp))

    print(("\n############ SECTION: SECOND GEN MODELS - NO ACCUMULATED - MAX ACROSS STEPS ############\n"))

    # filter for only rows where the step="max"
    second_gen_data = second_gen_data[second_gen_data['step'] == 'max']
    print(second_gen_data.head(10), "in total: ", len(second_gen_data))

    second_gen_data_means = max_seeds_to_mean_method(second_gen_data)

    # check statistics
    second_gen_data_pre_stat = second_gen_data_means.copy()
    second_gen_data_means = check_statistics(baseline_data_mean, second_gen_data_means)

    # sort the second_gen_data_means by model_name
    second_gen_data_means = second_gen_data_means.sort_values(by='model_name')
    second_gen_data_means = pd.concat([baseline_data_mean, second_gen_data_means])
    print(second_gen_data_means.head(len(second_gen_data_means)), "in total: ", len(second_gen_data_means))

    print(("\n############ SECTION: SECOND GEN MODELS - NO ACCUMULATED - LATEX TABLES ############\n"))

    print("BEFORE:",second_gen_data_means["model_name"].unique())



    latex_code = generate_latex_code(mark_best_models(second_gen_data_means.copy()), unique_tasks, table_caption="Second Gen Models", table_label="SecondGen")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)


    model_names = []
    model_names.append(("Baseline", "Baseline"))
    model_names.append(("double_first_gen_data_cs1_mr03","Double-First-Gen"))
    
    model_names.append(("sec_gen_no_contrast_cs1_mr03","2-GEN No-Contrast"))
    model_names.append(("sec_gen_acc_no_contrast_cs1_mr03","2-GEN Acc No-Contrast"))
    model_names.append(("sec_gen_no_contrast_cs1_n_gram_mr03","2-GEN No-Contrast-N-GRAM"))

    model_names.append(("sec_gen_acc_g5000_1500_cs1_mr03","2-GEN Acc E-1500"))
    model_names.append(("sec_gen_acc_g5000_500_ngram_cs1_mr03","2-GEN Acc E-500-NGRAM"))

    model_names.append(("sec_gen_g5000_1500_ngram_cs1_mr03","2-GEN G-1500-NGRAM"))
    model_names.append(("sec_gen_g5000_2000_cs1_mr03","2-GEN G-2000"))

    

    second_gen_selection = filter_and_rename(second_gen_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(second_gen_selection), unique_tasks, table_caption="Second Gen Models Summary", table_label="SecondGenSummary")
    append_latex_to_file(latex_code,latex_file_path)


    print(("\n############ SECTION: THIRD GEN MODELS ############\n"))

    if READ_FROM_PICKLE:
        third_gen_data = pd.read_pickle("/cluster/home/janulm/thesis/third_gen_training.pkl")
    else:
        third_gen_data = create_experiment_dataframe("/cluster/work/cotterell/janulm/thesis_data/eval_results/third_gen_training")
        third_gen_data.to_pickle("/cluster/home/janulm/thesis/third_gen_training.pkl")


    # sort the third_gen_data by model_name
    third_gen_data = third_gen_data.sort_values(by='model_name')
    print(third_gen_data.head(10), "in total: ", len(third_gen_data))


    print(("\n########### SECTION: EXPERIMENT- MEAN REL PERFORMANCE - THIRD GEN MODELS #############\n"))

    best_perplexity_checkpoints_rel_exp = find_best_perplexity_for_each_seed(third_gen_data)
    best_perplexity_checkpoints_rel_exp = compute_mean_rel_performance(best_perplexity_checkpoints_rel_exp)
    

    # sort the table by column SELECTION_REL_PERFORMANCE_KEY descending
    # Use key parameter to extract 'value' from dictionaries during sorting without modifying the DataFrame
    best_perplexity_checkpoints_rel_exp = best_perplexity_checkpoints_rel_exp.sort_values(
        by=SELECTION_REL_PERFORMANCE_KEY, 
        ascending=False,
        key=lambda x: x.apply(lambda val: val['value'] if isinstance(val, dict) and 'value' in val else val)
    )
    print(best_perplexity_checkpoints_rel_exp.head(20), "in total: ", len(best_perplexity_checkpoints_rel_exp))

    print(("\n############ SECTION: THIRD GEN MODELS - MAX ACROSS STEPS ############\n"))

    # filter for only rows where the step="max"
    third_gen_data = third_gen_data[third_gen_data['step'] == 'max']
    print(third_gen_data.head(10), "in total: ", len(third_gen_data))
    
    
    third_gen_data_means = max_seeds_to_mean_method(third_gen_data)

    # check statistics
    third_gen_data_pre_stat = third_gen_data_means.copy()
    third_gen_data_means = check_statistics(baseline_data_mean, third_gen_data_means)
    
    # sort the third_gen_data_means by model_name
    third_gen_data_means = third_gen_data_means.sort_values(by='model_name')
    third_gen_data_means = pd.concat([baseline_data_mean, third_gen_data_means])
    print(third_gen_data_means.head(20), "in total: ", len(third_gen_data_means))
    
    
    print(("\n############ SECTION: THIRD GEN MODELS - LATEX TABLES ############\n"))


    print("BEFORE:",third_gen_data_means["model_name"].unique())

    latex_code = generate_latex_code(mark_best_models(third_gen_data_means.copy()), unique_tasks, table_caption="Third Gen Models", table_label="ThirdGen")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)


    model_names = []
    model_names.append(("Baseline", "Baseline"))
    model_names.append(("third_gen_acc_no_contrast_cs1_mr03","3-GEN Acc No-Contrast"))
    model_names.append(("third_gen_g5000_1500_ngram_cs1_mr03","3-GEN G-1500-NGRAM"))
    model_names.append(("third_gen_g5000_1500_acc_cs1_mr03","3-GEN G-1500-ACC"))
    model_names.append(("third_gen_g5000_1500_acc_lasttwoonly_cs1_mr03","3-GEN G-1500-ACC Last Two Only"))
    model_names.append(("third_gen_g5000_500_acc_cs1_mr04","3-GEN G-500-ACC 0.4"))
    model_names.append(("third_gen_g5000_500_acc_cs1_mr05","3-GEN G-500-ACC 0.5"))
    third_gen_selection = filter_and_rename(third_gen_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(third_gen_selection), unique_tasks, table_caption="Third Gen Models Summary", table_label="ThirdGenSummary")
    append_latex_to_file(latex_code,latex_file_path)


    print(("\n############ SECTION: FOURTH GEN MODELS ############\n"))

    if READ_FROM_PICKLE:
        fourth_gen_data = pd.read_pickle("/cluster/home/janulm/thesis/fourth_gen_training.pkl")
    else:
        fourth_gen_data = create_experiment_dataframe("/cluster/work/cotterell/janulm/thesis_data/eval_results/fourth_gen_training")
        fourth_gen_data.to_pickle("/cluster/home/janulm/thesis/fourth_gen_training.pkl")


    # sort the third_gen_data by model_name
    fourth_gen_data = fourth_gen_data.sort_values(by='model_name')
    print(fourth_gen_data.head(10), "in total: ", len(fourth_gen_data))


    print(("\n########### SECTION: EXPERIMENT- MEAN REL PERFORMANCE - FOURTH GEN MODELS #############\n"))

    best_perplexity_checkpoints_rel_exp = find_best_perplexity_for_each_seed(fourth_gen_data)
    best_perplexity_checkpoints_rel_exp = compute_mean_rel_performance(best_perplexity_checkpoints_rel_exp)
    

    # sort the table by column SELECTION_REL_PERFORMANCE_KEY descending
    # Use key parameter to extract 'value' from dictionaries during sorting without modifying the DataFrame
    best_perplexity_checkpoints_rel_exp = best_perplexity_checkpoints_rel_exp.sort_values(
        by=SELECTION_REL_PERFORMANCE_KEY, 
        ascending=False,
        key=lambda x: x.apply(lambda val: val['value'] if isinstance(val, dict) and 'value' in val else val)
    )
    print(best_perplexity_checkpoints_rel_exp.head(20), "in total: ", len(best_perplexity_checkpoints_rel_exp))

    print(("\n############ SECTION: FOURTH GEN MODELS - MAX ACROSS STEPS ############\n"))

    # filter for only rows where the step="max"
    fourth_gen_data = fourth_gen_data[fourth_gen_data['step'] == 'max']
    print(fourth_gen_data.head(10), "in total: ", len(fourth_gen_data))
    
    
    fourth_gen_data_means = max_seeds_to_mean_method(fourth_gen_data)

    # check statistics
    fourth_gen_data_pre_stat = fourth_gen_data_means.copy()
    fourth_gen_data_means = check_statistics(baseline_data_mean, fourth_gen_data_means)
    
    # sort the third_gen_data_means by model_name
    fourth_gen_data_means = fourth_gen_data_means.sort_values(by='model_name')
    fourth_gen_data_means = pd.concat([baseline_data_mean, fourth_gen_data_means])
    print(fourth_gen_data_means.head(20), "in total: ", len(fourth_gen_data_means))
    
    
    print(("\n############ SECTION: FOURTH GEN MODELS - LATEX TABLES ############\n"))


    print("BEFORE:",fourth_gen_data_means["model_name"].unique())

    latex_code = generate_latex_code(mark_best_models(fourth_gen_data_means.copy()), unique_tasks, table_caption="Fourth Gen Models", table_label="FourthGen")
    #display(Markdown(f"```\n{latex_code}\n```"))
    append_latex_to_file(latex_code,latex_file_path)


    model_names = []
    model_names.append(("Baseline", "Baseline"))
    model_names.extend([("fourth_gen_g5000_"+str(i)+"_ngram_cs1_mr03","4-GEN G-"+str(i)+"-NGRAM") for i in [500, 1500, 2500]])
    model_names.extend([("fourth_gen_g5000_"+str(i)+"_acc_cs1_mr03","4-GEN G-"+str(i)+"-ACC") for i in [500, 1500, 2500]])
    fourth_gen_selection = filter_and_rename(fourth_gen_data_means, model_names)
    latex_code = generate_latex_code(mark_best_models(fourth_gen_selection), unique_tasks, table_caption="Fourth Gen Models Summary", table_label="FourthGenSummary")
    append_latex_to_file(latex_code,latex_file_path)


    return 
    

    print(("\n############ SECTION: STATISTICS PLOTS ############\n"))

    
    output_dir = "/cluster/home/janulm/thesis/ms-thesis/plots/statistics_new"   # Set your output directory here

    os.makedirs(output_dir, exist_ok=True)

    unique_models = experiment_data_means['model_name'].unique() # remove the base_llama_8000
    unique_models = [model for model in unique_models if model != 'Baseline']

    unique_tasks = TASK_INFO.keys()

    for model in unique_models:
        print("-" * 100)
        print(f"Model: {model}")

        available_tasks = unique_tasks
        n_tasks = len(available_tasks)
        if n_tasks == 0:
            continue

        fig, axes = plt.subplots(n_tasks, 2, figsize=(16, 3 * n_tasks), sharey=False)
        if n_tasks == 1:
            axes = [axes]

        for i, task in enumerate(available_tasks):
            

            baseline_task_data = baseline_data_mean.iloc[0][task]
            exp_task_data = experiment_data_means.loc[experiment_data_means['model_name'] == model].iloc[0][task]

            #print("baseline_task_data:################# ", baseline_task_data.keys())
            #print("exp_task_data:############## ", exp_task_data.keys())

            contrastive_values = np.array(exp_task_data["seed_means"])
            baseline_values = np.array(baseline_task_data["seed_means"])

            

            
            baseline_boot_means = baseline_task_data["distr_mean"]
            contrastive_boot_means = exp_task_data["distr_mean"]

            min_val = min(np.min(baseline_values), np.min(contrastive_values), np.min(baseline_boot_means), np.min(contrastive_boot_means))
            max_val = max(np.max(baseline_values), np.max(contrastive_values), np.max(baseline_boot_means), np.max(contrastive_boot_means))
            padding = 0.1 * (max_val - min_val) if max_val > min_val else 1.0

            ax = axes[i][0] if n_tasks > 1 else axes[0]

            # Plot histograms and get density (for line placement)
            bins = 60
            range_tuple = (min_val-padding, max_val+padding)
            hist_base, bin_edges = np.histogram(baseline_boot_means, bins=bins, range=range_tuple, density=True)
            hist_contr, _ = np.histogram(contrastive_boot_means, bins=bins, range=range_tuple, density=True)
            hist_max = max(np.max(hist_base), np.max(hist_contr))

            # Plot histograms
            ax.hist(baseline_boot_means, color='blue', bins=bins, range=range_tuple, density=True, alpha=0.3, label='Baseline boot. means')
            ax.hist(contrastive_boot_means, color='red', bins=bins, range=range_tuple, density=True, alpha=0.3, label='Contrastive boot. means')

            # Set a fixed y for number line and dots, above all histogram bars
            y_line = hist_max * 1.15
            dot_jitter = 0.04 * hist_max  # vertical separation for dots

            # Plot the horizontal number line
            ax.hlines(y=y_line, xmin=min_val-padding, xmax=max_val+padding, color='gray', linewidth=2, alpha=0.6)

            # Scatter dots for individual values, above/below the line
            ax.scatter(baseline_values, np.ones_like(baseline_values)* (y_line + dot_jitter), color='blue', label='Baseline', s=80, alpha=0.85)
            ax.scatter(contrastive_values, np.ones_like(contrastive_values)* (y_line - dot_jitter), color='red', label='Contrastive', s=80, alpha=0.85)

            # Plot means as vertical lines
            ax.axvline(baseline_values.mean(), color='blue', linestyle='--', label='Baseline Mean')
            ax.axvline(contrastive_values.mean(), color='red', linestyle='--', label='Contrastive Mean')

            # Formatting
            ax.set_ylabel('Density')
            ax.set_ylim(0, y_line + dot_jitter * 2)
            if i == n_tasks - 1:
                ax.set_xlabel('Value')

            significant_data = exp_task_data["significant"]

            if significant_data:
                if significant_data["result"] == "better":
                    significance_label = "Significantly Better"
                elif significant_data["result"] == "worse":
                    significance_label = "Significantly Worse"
                elif significant_data["result"] == "no_difference":
                    significance_label = "Not Significant"
                else:
                    assert False, "Invalid result"
            else:
                significance_label = "No data"
            
            ax.set_title(f"{task} - Value Distribution\n({significance_label})")
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

            # ---------- Statistic test histogram plot on the right ----------
            ax_stat = axes[i][1] if n_tasks > 1 else axes[1]
            stat_boot = np.array(significant_data["diff"])
            ci_low = significant_data["ci"][0]
            ci_high = significant_data["ci"][1]
            center = np.mean(stat_boot)
            min_stat, max_stat = np.min(stat_boot), np.max(stat_boot)
            stat_pad = 0.1 * (max_stat - min_stat) if max_stat > min_stat else 1.0

            ax_stat.hist(stat_boot, bins=60, range=(min_stat-stat_pad, max_stat+stat_pad), color='purple', alpha=0.5, density=True, label='Statistic boot. dist')
            ax_stat.axvline(0, color='black', linestyle=':', linewidth=2, label='Zero')
            ax_stat.axvline(ci_low, color='green', linestyle='--', linewidth=2, label='Conf. Interval Low')
            ax_stat.axvline(ci_high, color='green', linestyle='--', linewidth=2, label='Conf. Interval High')
            ax_stat.axvline(center, color='purple', linestyle='-', linewidth=2, label='Mean')
            ax_stat.set_yticks([])
            if i == n_tasks - 1:
                ax_stat.set_xlabel('Test Statistic Value')
            ax_stat.set_title(f"{task} - Bootstrap Statistic\n(CI: [{ci_low:.3g}, {ci_high:.3g}])")
            if i == 0:
                ax_stat.legend(loc='upper right', fontsize=8)

        plt.suptitle(f"{model} - All Tasks\n(Left: Distributions, Right: Statistic)", fontsize=18, y=1.04)
        plt.tight_layout()
        filename = f"{model}_statistics.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
        plt.show()



    print(("\n############ SECTION: STATISTICS: No Contrast vs Contrastive: ############\n"))

    # use the no contrastive model as the baseline and compare against the others: 

    
    no_contrast_df = experiment_data_pre_stat[experiment_data_pre_stat["model_name"].str.contains("no\_contrast\_mr03")].copy()
    contrast_df = experiment_data_pre_stat[experiment_data_pre_stat["model_name"].str.contains("g2500\_500\_cs1\_mr03")].copy()
    print(no_contrast_df)


    _ = check_statistics(no_contrast_df, contrast_df)



    return



    print(("\n############ SECTION: ? ############\n"))

    """
    ## Statistical Significance wrt to the baseline

    Lets now do a statistical test to see if the contrastive methods are better than the baseline. 

    We have 3 different synthetic data methods, and 1 baseline method. 
    For each task we have trained 10 models and now have the max value for each task, that each individual model achieved. 
    """


 
    """
    print(("\n############ SECTION: RUNNING STATISTICAL TESTS (OLD VERSION) ############\n"))    

    ### RUNNING STATISTICAL TESTS

    test_results = {}

    # add the MEAN for the tassk except ppl for the baseline model
    for model in unique_models_baseline:
        print("-"*100)
        print(f"Model: {model}")
        
        model_mean_task_accumulated = 0.0
        model_mean_task_count = 0

        # add all tasks except for perplexity to the model_mean_task_accumulated
        for task in unique_tasks:
            if task != 'Perplexity':
                model_mean_task_accumulated += np.array(baseline_list[(model, task)]).mean()
                model_mean_task_count += 1
            else: 
                print("Perplexity task found, skipping")

        model_mean_task_accumulated /= model_mean_task_count
        
        test_results[(model, 'mean_up')] = model_mean_task_accumulated
        print(f"Model {model} mean: {model_mean_task_accumulated}")

    # add the MEAN for the tassk except ppl for the contrastive models
    for model in unique_models:
        print("-"*100)
        print(f"Model: {model}")
        
        model_mean_task_accumulated = 0.0
        model_mean_task_count = 0

        # add all tasks except for perplexity to the model_mean_task_accumulated
        for task in unique_tasks:
            if task != 'Perplexity':
                model_mean_task_accumulated += np.array(contrastive_list[(model, task)]).mean()
                model_mean_task_count += 1
            else: 
                print("Perplexity task found, skipping")

        model_mean_task_accumulated /= model_mean_task_count
        
        test_results[(model, 'mean_up')] = model_mean_task_accumulated
        print(f"Model {model} mean: {model_mean_task_accumulated}")





    for model in unique_models:
        print("-"*100)
        print(f"Model: {model}")

        for task in unique_tasks:
        
            base_key = ('base_llama', task)
            contrastive_key = (model, task)

            if contrastive_key in contrastive_list and base_key in baseline_list:
                contrastive_values = contrastive_list[contrastive_key]
                baseline_values = baseline_list[base_key]
                
                verbose = False

                test1, res = check_statistics_1(contrastive_key, contrastive_values, base_key, baseline_values,n_resamples=10000, alpha=0.05, random_state=0, verbose=True)
                
                
                # now also compute the relative improvement of the method compared to the baseline
                relative_improvement = (np.array(contrastive_values).mean() - np.array(baseline_values).mean()) / np.array(baseline_values).mean()
                test_results[(model, task, 'relative_improvement')] = relative_improvement

                if test1:
                    print("Task: ", task, "Results:", test1)
                
                test_results[(model, task, 'significant')] = test1

    print(test_results)
    """

    # we now want to compute the flags "significant" and "relative_improvement" for each model_name and task
    # furthermore the other tables...


    print(("\n############ SECTION: ? ############\n"))

    # Now we want to create a combine table that has the num_seeds, different tasks, their performance, std_err and then if they are significant or not. 

    # go over the rows in combined_table and check if for each model_name and task if they are significant, add this information as a new column

    print(unique_tasks)
    for index, row in combined_table.iterrows():
        #print(row)

        current_model = row['model_name']
        if current_model in unique_models or current_model in unique_models_baseline:
            
            combined_table.at[index, "Mean Up"] = test_results[(current_model, 'mean_up')]

            for task in unique_tasks:
                if (current_model, task, 'significant') in test_results:
                    combined_table.at[index, task+'_significant'] = test_results[(current_model, task, 'significant')]
                    combined_table.at[index, task+'_relative_improvement'] = test_results[(current_model, task, 'relative_improvement')]
                    
                    #print(f"Model {current_model} for task {task} IS STORED IN TABLE: {test_results[(current_model, task)]}")
                else:
                    combined_table.at[index, task+'_significant'] = False
                    print(f"Model {current_model} for task {task} IS NOT STORED IN TABLE")

    print(combined_table)
    print(combined_table.columns)

    print(unique_tasks)
    unique_tasks.add("Mean Up")
    print(unique_tasks)

   


    print(("\n############ SECTION: ? ############\n"))

    ## Statistical Significance wrt to the synthetic data of the baseline model? 
    ### Is contrastive significantly better than plain sampling?


    for model in unique_models:
        print("-"*100)
        print(f"Model: {model}")
        for task in unique_tasks:
        
            base_key = ('g2500_no_contrast_mr03', task)
            contrastive_key = (model, task)

            if contrastive_key in contrastive_list and base_key in baseline_list:
                contrastive_values = contrastive_list[contrastive_key]
                baseline_values = baseline_list[base_key]
                
                verbose = False

                test1, res = check_statistics_1(contrastive_key, contrastive_values, base_key, baseline_values,n_resamples=10000, alpha=0.05, random_state=0, verbose=True)
                
                #test2 = bootstrap_diff_means(contrastive_key, contrastive_values, base_key, baseline_values, num_samples=10000, verbose=verbose)
                #test3 = paired_bootstrap_test(contrastive_key, contrastive_values, base_key, baseline_values, B=10000, alpha=0.05, verbose=verbose)
                if test1:
                    print("Task: ", task, "Results:", test1)

    print(("\n############ SECTION: ? ############\n"))

    # Best model of the syntheticly trained model

    #Use the same percentile of scores technique as for the baseline models. 


    print(rarity_score(contrastive_data))


    # potential model names: 
    # max mean percentile: g2500_no_contrast_mr03_seed42_step3500, g2500_500_cs1_mr03_seed44_step5000
    # min ppl: g2500_no_contrast_mr03_seed42_step5000, g2500_500_cs1_mr03_seed51_step5000


    # use the checkpoint column to filter for the correct ones.
    print(contrastive_data.head())
    print(contrastive_data.columns)

    #selection = contrastive_data[contrastive_data['checkpoint'].str.contains("g2500_no_contrast_mr03_seed42_step5000") | contrastive_data['checkpoint'].str.contains("g2500_no_contrast_mr03_seed42_step3500") | contrastive_data['checkpoint'].str.contains("g2500_500_cs1_mr03_seed44_step5000")].copy()

    potential_second_experts = contrastive_data[contrastive_data['checkpoint'].str.contains("g2500_no_contrast_mr03_seed42_step3500") | contrastive_data['checkpoint'].str.contains("g2500_500_cs1_mr03_seed44_step5000")].copy()
    print("Potential second experts:")
    print(potential_second_experts)


    # DO IT LIKE THIS::>>>>> 
    """
    # show table with model_name= base_llama_8000_43 and step = 2500

    expert_model_table = baseline_data[(baseline_data['model_name'] == 'base_llama_8000_43') & (baseline_data['step'] == 2500)].copy()
    # assert this table only has 1 row

    print(expert_model_table)

    num_tasks = len(expert_model_table['task'].unique())

    assert len(expert_model_table) == num_tasks, "Expert model table should only have num_tasks row"
    # rename all rows "model_name" to "expert_model"
    expert_model_table['model_name'] = 'expert_model'

    print(expert_model_table)


    expert_model_table = table_to_task_table_single_model(expert_model_table)
    print(expert_model_table)
    """

    print(("\n############ SECTION: ? ############\n"))
    """
    ## Small BabyLM models

    Now I wanted to train a few smaller LM's that as the classic CD paper also uses smaller models as the bad contrastive model. 
    I have trained 5 smaller models with model sizes 5,10,20,50,100 times smaller than the baseline model. 

    Lets see how the performed and to select the checkpoints for the contrastive part
    """
    """
    # THIS EVAL DOESNT EXIST YET (WASNT REQUIRED ANYMORE)
    path = "/cluster/work/cotterell/janulm/thesis_data/eval_results/baseline_training_small"

    small_baseline_data = create_experiment_dataframe(path)
    small_baseline_data = split_name_seed(small_baseline_data)
    small_baseline_list, small_baseline_table = get_best_values_list_and_table(small_baseline_data)
    print(small_baseline_list)
    print(small_baseline_table)

    """


    print(("\n############ SECTION: PLOTS ############\n"))

    # Plots for the tasks and the maxes and their distribution


    n_bootstrap = 10_000
    output_dir = "/Users/janulm/Documents/ETH/SM12/BabyLM Thesis/ms-thesis/plots/statistics"   # Set your output directory here

    os.makedirs(output_dir, exist_ok=True)

    for model in unique_models:
        print("-" * 100)
        print(f"Model: {model}")

        available_tasks = [
            task for task in unique_tasks
            if (model, task) in contrastive_list and ('base_llama', task) in baseline_list
        ]
        n_tasks = len(available_tasks)
        if n_tasks == 0:
            continue

        fig, axes = plt.subplots(n_tasks, 2, figsize=(16, 3 * n_tasks), sharey=False)
        if n_tasks == 1:
            axes = [axes]

        for i, task in enumerate(available_tasks):
            base_key = ('base_llama', task)
            contrastive_key = (model, task)

            contrastive_values = np.array(contrastive_list[contrastive_key])
            baseline_values = np.array(baseline_list[base_key])

            test1, res = check_statistics_1(
                contrastive_key, contrastive_values, base_key, baseline_values,
                n_resamples=10000, alpha=0.05, random_state=0, verbose=False
            )

            # Bootstrap sampling of the mean
            rng = np.random.default_rng(42)
            baseline_boot_means = rng.choice(baseline_values, (n_bootstrap, len(baseline_values)), replace=True).mean(axis=1)
            contrastive_boot_means = rng.choice(contrastive_values, (n_bootstrap, len(contrastive_values)), replace=True).mean(axis=1)

            min_val = min(np.min(baseline_values), np.min(contrastive_values), np.min(baseline_boot_means), np.min(contrastive_boot_means))
            max_val = max(np.max(baseline_values), np.max(contrastive_values), np.max(baseline_boot_means), np.max(contrastive_boot_means))
            padding = 0.1 * (max_val - min_val) if max_val > min_val else 1.0

            ax = axes[i][0] if n_tasks > 1 else axes[0]

            # Plot histograms and get density (for line placement)
            bins = 60
            range_tuple = (min_val-padding, max_val+padding)
            hist_base, bin_edges = np.histogram(baseline_boot_means, bins=bins, range=range_tuple, density=True)
            hist_contr, _ = np.histogram(contrastive_boot_means, bins=bins, range=range_tuple, density=True)
            hist_max = max(np.max(hist_base), np.max(hist_contr))

            # Plot histograms
            ax.hist(baseline_boot_means, color='blue', bins=bins, range=range_tuple, density=True, alpha=0.3, label='Baseline boot. means')
            ax.hist(contrastive_boot_means, color='red', bins=bins, range=range_tuple, density=True, alpha=0.3, label='Contrastive boot. means')

            # Set a fixed y for number line and dots, above all histogram bars
            y_line = hist_max * 1.15
            dot_jitter = 0.04 * hist_max  # vertical separation for dots

            # Plot the horizontal number line
            ax.hlines(y=y_line, xmin=min_val-padding, xmax=max_val+padding, color='gray', linewidth=2, alpha=0.6)

            # Scatter dots for individual values, above/below the line
            ax.scatter(baseline_values, np.ones_like(baseline_values)* (y_line + dot_jitter), color='blue', label='Baseline', s=80, alpha=0.85)
            ax.scatter(contrastive_values, np.ones_like(contrastive_values)* (y_line - dot_jitter), color='red', label='Contrastive', s=80, alpha=0.85)

            # Plot means as vertical lines
            ax.axvline(baseline_values.mean(), color='blue', linestyle='--', label='Baseline Mean')
            ax.axvline(contrastive_values.mean(), color='red', linestyle='--', label='Contrastive Mean')

            # Formatting
            ax.set_ylabel('Density')
            ax.set_ylim(0, y_line + dot_jitter * 2)
            if i == n_tasks - 1:
                ax.set_xlabel('Value')
            significance_label = "Significant" if test1 else "Not Significant"
            ax.set_title(f"{task} - Value Distribution\n({significance_label})")
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

            # ---------- Statistic test histogram plot on the right ----------
            ax_stat = axes[i][1] if n_tasks > 1 else axes[1]
            stat_boot = res.bootstrap_distribution.flatten()
            ci_low = res.confidence_interval.low
            ci_high = res.confidence_interval.high
            center = np.mean(stat_boot)
            min_stat, max_stat = np.min(stat_boot), np.max(stat_boot)
            stat_pad = 0.1 * (max_stat - min_stat) if max_stat > min_stat else 1.0

            ax_stat.hist(stat_boot, bins=60, range=(min_stat-stat_pad, max_stat+stat_pad), color='purple', alpha=0.5, density=True, label='Statistic boot. dist')
            ax_stat.axvline(0, color='black', linestyle=':', linewidth=2, label='Zero')
            ax_stat.axvline(ci_low, color='green', linestyle='--', linewidth=2, label='Conf. Interval Low')
            ax_stat.axvline(ci_high, color='green', linestyle='--', linewidth=2, label='Conf. Interval High')
            ax_stat.axvline(center, color='purple', linestyle='-', linewidth=2, label='Mean')
            ax_stat.set_yticks([])
            if i == n_tasks - 1:
                ax_stat.set_xlabel('Test Statistic Value')
            ax_stat.set_title(f"{task} - Bootstrap Statistic\n(CI: [{ci_low:.3g}, {ci_high:.3g}])")
            if i == 0:
                ax_stat.legend(loc='upper right', fontsize=8)

        plt.suptitle(f"{model} - All Tasks\n(Left: Distributions, Right: Statistic)", fontsize=18, y=1.04)
        plt.tight_layout()
        filename = f"{model}_statistics.png".replace("/", "_")
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
        plt.show()

    print(("\n############ SECTION: ? ############\n"))
    print(("\n############ SECTION: ? ############\n"))


    print(("\n############ SECTION: ? ############\n"))
    print(("\n############ SECTION: ? ############\n"))

    print(contrastive_data)

if __name__ == "__main__":
    main()