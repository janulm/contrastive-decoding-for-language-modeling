"""
This module extracts experiment data from model directories and creates a pandas DataFrame
with the structure [model_name, step, task] = value.
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore

# Task name mapping: maps long task names to short names
TASK_NAME_MAPPING = {
    'blimp/blimp_filtered': 'BLiMP',
    'blimp/supplement_filtered': 'BLiMP Supp.',
    'entity_tracking/entity_tracking': 'Entity Tracking',
    'ewok/ewok_filtered': 'EWoK',
    'perplexity': 'Perplexity',
    'wug/wug_adj_nominalization': 'WUG',
}


def get_step_directories(model_dir: str) -> List[str]:
    """Get all subdirectories that match the pattern 'step_x' in the given model directory."""
    step_dirs = []
    for item in os.listdir(model_dir):
        full_path = os.path.join(model_dir, item)
        if os.path.isdir(full_path) and re.match(r'step_\d+', item):
            step_dirs.append(full_path)
    return sorted(step_dirs)


def extract_perplexity(step_dir: str) -> Optional[float]:
    """Extract perplexity value from tinybabylm_dataset-test directory."""
    tinybabylm_dir = os.path.join(step_dir, "tinybabylm_dataset-test")
    if os.path.exists(tinybabylm_dir):
        report_path = os.path.join(tinybabylm_dir, "best_temperature_report.txt")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Perplexity:"):
                            try:
                                return float(line.split(":")[1].strip())
                            except ValueError:
                                print(f"Warning: Could not parse perplexity value in {report_path}")
            except Exception as e:
                print(f"Error reading {report_path}: {e}")
    return None


def extract_reading_scores(step_dir: str) -> Optional[Dict[str, float]]:
    """Extract eye tracking and self-paced reading scores from the reading/report.txt file."""
    report_path = os.path.join(
        step_dir,
        "main",
        "zero_shot",
        "causal",
        "reading",
        "report.txt"
    )
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                lines = f.readlines()
                scores = {}
                for line in lines:
                    if line.startswith("EYE TRACKING SCORE:"):
                        try:
                            scores["Eye Tracking"] = float(line.split(":")[1].strip())
                        except ValueError:
                            print(f"Warning: Could not parse EYE TRACKING SCORE in {report_path}")
                    elif line.startswith("SELF-PACED READING SCORE:"):
                        try:
                            scores["Reading"] = float(line.split(":")[1].strip())
                        except ValueError:
                            print(f"Warning: Could not parse SELF-PACED READING SCORE in {report_path}")
                if scores:
                    return scores
        except Exception as e:
            print(f"Error reading {report_path}: {e}")
    return None


def extract_task_results(step_dir: str) -> Dict[str, Dict[str, float]]:
    """Extract task results from main/zero_shot/causal directory."""
    results = {}
    
    def process_directory(current_dir: str) -> None:
        """Recursively process directories to find best_temperature_report.txt files."""
        for item in os.listdir(current_dir):
            full_path = os.path.join(current_dir, item)
            
            if os.path.isdir(full_path):
                report_path = os.path.join(full_path, "best_temperature_report.txt")
                if os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                last_lines = lines[-3:]
                                if "### AVERAGE ACCURACY" in last_lines[0]:
                                    try:
                                        accuracy = float(last_lines[1].strip())
                                        task_name = os.path.basename(os.path.dirname(full_path))
                                        if task_name not in results:
                                            results[task_name] = {}
                                        results[task_name][item] = accuracy
                                    except ValueError:
                                        print(f"Warning: Could not parse accuracy value in {report_path}")
                    except Exception as e:
                        print(f"Error reading {report_path}: {e}")
                else:
                    process_directory(full_path)
    
    main_dir = os.path.join(step_dir, "main", "zero_shot", "causal")
    if os.path.exists(main_dir):
        process_directory(main_dir)
    else:
        print(f"Warning: Could not find {main_dir}")
    
    return results


def create_experiment_dataframe(experiment_dir: str) -> pd.DataFrame:
    """
    Create a DataFrame containing all experiment data.
    
    Args:
        experiment_dir: Path to the experiment directory containing model directories
        
    Returns:
        DataFrame with columns [model_name, step, task] and values for accuracy/perplexity
    """
    data = []
    
    # Find all model directories
    for model_dir in os.listdir(experiment_dir):
        full_model_dir = os.path.join(experiment_dir, model_dir)
        if not os.path.isdir(full_model_dir) or model_dir == "plots":
            continue
            
        print(f"Processing model: {model_dir}")
        
        # Process each step directory
        for step_dir in get_step_directories(full_model_dir):
            step_name = os.path.basename(step_dir)
            step_num = int(step_name.split('_')[1])
            
            # Extract perplexity
            perplexity = extract_perplexity(step_dir)
            if perplexity is not None:
                task_short = TASK_NAME_MAPPING.get('perplexity')
                if task_short is None:
                    raise ValueError(f"Task name 'perplexity' not found in TASK_NAME_MAPPING")
                data.append({
                    'model_name': model_dir,
                    'step': step_num,
                    'task': task_short,
                    'value': perplexity
                })
            
            # Extract reading scores
            reading_scores = extract_reading_scores(step_dir)
            if reading_scores is not None:
                for score_name, score_value in reading_scores.items():
                    data.append({
                        'model_name': model_dir,
                        'step': step_num,
                        'task': score_name,
                        'value': score_value
                    })
            
            # Extract task results
            task_results = extract_task_results(step_dir)
            for task_name, subtasks in task_results.items():
                for subtask_name, accuracy in subtasks.items():
                    full_task_name = f"{task_name}/{subtask_name}"
                    task_short = TASK_NAME_MAPPING.get(full_task_name)
                    if task_short is None:
                        raise ValueError(f"Task name '{full_task_name}' not found in TASK_NAME_MAPPING")
                    data.append({
                        'model_name': model_dir,
                        'step': step_num,
                        'task': task_short,
                        'value': accuracy
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    if not df.empty:
        # Sort by model_name, step, and task
        df = df.sort_values(['model_name', 'step', 'task'])
    
    return df




def main():
    """Example usage of the module."""
    import argparse
    parser = argparse.ArgumentParser(description='Extract experiment data into a DataFrame')
    parser.add_argument('experiment_dir', help='Path to the experiment directory')
    parser.add_argument('--output', help='Path to save the DataFrame (CSV format)')
    parser.add_argument('--output-seed-analysis', help='Path to save the seed analysis DataFrame (CSV format)')
    args = parser.parse_args()
    
    all_data_df = create_experiment_dataframe(args.experiment_dir)
    
    # this dataframe is sorted contains the information: 
    # model_name, step, task, value e.g 

    #   model_name  step              task      value
    # 43  base_llama_42   500             blimp  63.030000
    # 44  base_llama_42   500  blimp_supplement  56.350000
    # 45  base_llama_42   500   entity_tracking  14.350000
    # 46  base_llama_42   500              ewok  50.580000
    # 42  base_llama_42   500        perplexity  47.014217
    # 

    if args.output:
        
        print(f"DataFrame saved to {args.output}")
        all_data_df.to_csv(args.output, index=False)
    

    print(all_data_df.head(10))
    print(all_data_df.info())


    # split the column model_name into model_name and seed in a new table.. 
    # currently it is: 
    #      model_name  step              task      value
    # 717  base_llama_42   500             blimp  63.030000
    # 716  base_llama_42   500  blimp_supplement  56.350000
    # 719  base_llama_42   500   entity_tracking  14.350000
    # 718  base_llama_42   500              ewok  50.580000

    # Split model_name into model_name and seed (assuming format: name_seed)
    df_split = all_data_df.copy()
    df_split[['model_base', 'seed']] = df_split['model_name'].str.rsplit('_', n=1, expand=True)
    # Optionally, convert seed to int if appropriate
    try:
        df_split['seed'] = df_split['seed'].astype(int)
    except Exception:
        raise TypeError("Seed is not always an int")  # If seed is not always an int, keep as string
    # drop column model_name
    df_split = df_split.drop(columns=['model_name'])
    print(df_split.head(20))



    # we now want to create a series of dataframes to construct the following tables/plots: 

    # 1. a number of plots (for each task) of all models for a given step + the mean and std/sqrt(n) of the mean across the seeds aggregated over each step.
    # 2. for (each task/model name/seed) tuple, get the best value across all steps (max, except for perplexity)
    #    now take these "best values" (over the different seeds) and create a new dataframe with {model_name, task, mean, std_err (std/sqrt(n)), n_seeds}

    # 3. given that one now has a list of bests of each (task/model name), we use boosting to get confidence intervals for the mean of the best values ??? WHAT DOES THAT MEAN? 
    # 4. Furthermore we can now go and compute the "rarity" of each model_name/seed/step/task checkpoint and then also compute the "overall rarity" of each model_name/seed/step
    #    this should give a good indication of the quality of the (should have goodness relative to all scores for that task and also mean and product overall rarity of all tasks)

    #########################################################################################
    # PLOT 1: 
    # get the n different tasks and then plot n plots on top of each other, each plot is a model and the y-axis is the accuracy/perplexity for all models on that task. 
    # + the mean and std/sqrt(n) of the mean across the seeds aggregated over each step.

    # now we compute the table that contains the means across the time steps each model_base,task,step (over the seeds)
    # Group by model_base, task, step and aggregate over seeds
    grouped = df_split.groupby(['model_base', 'task', 'step'])['value']
    summary_mean_step_based_df = grouped.agg(
        mean_value='mean',
        std_value='std',
        n_seeds='count'
    ).reset_index()
    summary_mean_step_based_df['stderr'] = summary_mean_step_based_df['std_value'] / np.sqrt(summary_mean_step_based_df['n_seeds'])

    print("Aggregated table (mean/std/stderr/n_seeds) for each (model_base, task, step):")
    print(summary_mean_step_based_df.head(20))

    # plot both the df_split and the summary_mean_step_based_df in one image (n plots stacked horizontally). the std-err should be shown as a shaded area around the mean.
    # and the mean line should be thicker that the single seed lines.

    unique_tasks = df_split['task'].unique()
    unique_models = df_split['model_base'].unique()
    n_tasks = len(unique_tasks)
    fig_height = max(5 * n_tasks, 8)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(16, fig_height))
    if n_tasks == 1:
        axes = [axes]

    plt.style.use('default')
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })

    fig.suptitle('All models and seeds per task, with mean ± stderr', fontsize=16, y=0.98)

    for ax, task in zip(axes, unique_tasks):
        for model in unique_models:
            # Plot individual seed runs (thin lines)
            model_seed_data = df_split[(df_split['model_base'] == model) & (df_split['task'] == task)]
            for seed in model_seed_data['seed'].unique():
                seed_data = model_seed_data[model_seed_data['seed'] == seed]
                if not seed_data.empty:
                    ax.plot(
                        seed_data['step'],
                        seed_data['value'],
                        label=f'{model}_seed{seed}',
                        alpha=0.4,
                        linewidth=1
                    )
            # Plot mean ± stderr (thick line with shaded area)
            mean_data = summary_mean_step_based_df[(summary_mean_step_based_df['model_base'] == model) & (summary_mean_step_based_df['task'] == task)]
            if not mean_data.empty:
                mean_data = mean_data.sort_values('step')
                x = mean_data['step']
                y = mean_data['mean_value']
                yerr = mean_data['stderr']
                line, = ax.plot(x, y, label=f'{model} (mean)', linewidth=3)
                color = line.get_color()
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        ax.set_title(f'{task} across Training Steps')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the figure in the 'plots' subdirectory of experiment_dir
    plots_dir = os.path.join(args.experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = 'all_models_seeds_per_task.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")


    #########################################################################################
    # TABLE 2: 
    # 2. for (each task/model name/seed) tuple, get the best value across all steps (max, except for perplexity)
    #    now take these "best values" (over the different seeds) and create a new dataframe with {model_name, task, mean, std_err (std/sqrt(n)), n_seeds}

    # start with the df_split, which looks like this: 
    #     step              task      value  model_base  seed
    # 717   500             blimp  63.030000  base_llama    42
    # 716   500  blimp_supplement  56.350000  base_llama    42
    # 719   500   entity_tracking  14.350000  base_llama    42
    # 718   500              ewok  50.580000  base_llama    42
    # 714   500        perplexity  47.014217  base_llama    42
    # 715   500               wug  65.000000  base_llama    42

    # we want to get the best value for each (model_base, task, seed) tuple across all steps (max, except for perplexity)

    # Get the best value for each (model_base, task, seed) tuple across all steps (max, except for perplexity)
    best_values = []
    for (model_base, task, seed), group in df_split.groupby(['model_base', 'task', 'seed']):
        if task == TASK_NAME_MAPPING.get('perplexity'):
            best_value = group['value'].min()
        else:
            best_value = group['value'].max()
        best_values.append({
            'model_base': model_base,
            'task': task,
            'seed': seed,
            'best_value': best_value
        })
    best_values_df = pd.DataFrame(best_values)
    print("\n\n\nBest value for each (model_base, task, seed):")
    print(best_values_df.head(20))

    # Now aggregate these best values over seeds for each (model_base, task)
    summary_best_df = best_values_df.groupby(['model_base', 'task'])['best_value'].agg(
        mean='mean',
        std='std',
        n_seeds='count'
    ).reset_index()
    summary_best_df['std_err'] = summary_best_df['std'] / np.sqrt(summary_best_df['n_seeds'])
    print("Summary of best values (mean/std/std_err/n_seeds) for each (model_base, task):")
    print(summary_best_df.head(20))

    #########################################################################################

    # we now want to figure out the "best" model checkpoint to generate synthetic data...
    # to asses this we want to gather all values recorded across the whole table for each experiment (independend of model_base, seed, step etc)
    # these values should be used to get some kind of distribution over the values of the different tasks. 
    # HOW SHOULD I Compute a rarity for e checkpoint? 

    #########################################################################################
    # RARITY SCORE: Compute percentileofscore for each checkpoint (model_base, seed, step, task)

    # For each row in df_split, compute the percentile of its value among all values for the same task
    rarity_scores = []
    for idx, row in df_split.iterrows():
        task = row['task']
        value = row['value']
        # Get all values for this task
        task_values = df_split[df_split['task'] == task]['value']
        # For perplexity, lower is better, so use kind='rank' and invert percentile
        if task == TASK_NAME_MAPPING.get('perplexity'):
            percentile = 100 - percentileofscore(task_values, value, kind='rank')
        else:
            percentile = percentileofscore(task_values, value, kind='rank')
        rarity_scores.append(percentile)
    df_split['rarity_percentile'] = rarity_scores

    print("\nRarity (percentileofscore) for each checkpoint (model_base, seed, step, task):")
    print(df_split[['model_base', 'seed', 'step', 'task', 'value', 'rarity_percentile']].head(20))

    # Print perplexity checkpoints with lower perplexity values
    perplexity_df = df_split[df_split['task'] == TASK_NAME_MAPPING.get('perplexity')].sort_values('value', ascending=True)
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
    
    


if __name__ == "__main__":
    main() 