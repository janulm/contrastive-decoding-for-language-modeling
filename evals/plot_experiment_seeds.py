"""
Plot experiment results with seed analysis, showing mean values and standard errors.
Also plots individual runs for each model.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from extract_experiment_dataframe import create_experiment_dataframe, create_seed_analysis_dataframe
import numpy as np


def get_experiment_name(experiment_dir: str) -> str:
    """
    Extract experiment name from the experiment directory path.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        str: Experiment name
    """
    # Get the last directory name from the path
    experiment_name = os.path.basename(os.path.normpath(experiment_dir))
    return experiment_name


def plot_individual_runs(
    df: pd.DataFrame,
    output_dir: str,
    experiment_name: str,
    tasks: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    steps: Optional[List[int]] = None,
    figsize: tuple = (12, 24),
    dpi: int = 300
) -> None:
    """
    Plot individual runs for each model without seed aggregation.
    
    Args:
        df: DataFrame with experiment data
        output_dir: Directory to save the plots
        experiment_name: Name of the experiment
        tasks: List of tasks to plot (if None, plot all tasks)
        models: List of model base names to plot (if None, plot all models)
        steps: List of steps to plot (if None, plot all steps)
        figsize: Figure size (width, height)
        dpi: DPI for the output figures
    """
    # Filter data if specified
    if tasks:
        df = df[df['task'].isin(tasks)]
    if models:
        df = df[df['model_name'].isin(models)]
    if steps:
        df = df[df['step'].isin(steps)]
    
    # Get unique tasks and models
    unique_tasks = df['task'].unique()
    unique_models = df['model_name'].unique()
    
    # Dynamically set figure height
    n_tasks = len(unique_tasks)
    fig_height = max(5 * n_tasks, 8)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(figsize[0], fig_height))
    if n_tasks == 1:
        axes = [axes]  # Make axes iterable for single plot case
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': figsize,
        'figure.dpi': dpi,
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
    
    # Add main title
    fig.suptitle(f'Experiment: {experiment_name}\nIndividual Runs', fontsize=16, y=0.98)
    
    # Plot each task in its subplot
    for ax, task in zip(axes, unique_tasks):
        # Plot each model
        for model in unique_models:
            model_data = df[(df['model_name'] == model) & (df['task'] == task)]
            
            if not model_data.empty:
                ax.plot(
                    model_data['step'],
                    model_data['value'],
                    label=model,
                    marker='o'
                )
        
        # Customize the subplot
        ax.set_title(f'{task} across Training Steps')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_individual_runs.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved individual runs plot to {output_path}")


def plot_model_task_table(
    seed_df: pd.DataFrame,
    output_dir: str,
    experiment_name: str,
    figsize: tuple = (12, 8),
    dpi: int = 1200,
    export_latex: bool = False
) -> None:
    """
    Create two table plots:
    1. First table: For each model/task, first average across seeds at each step, then find max/min of these means across steps
    2. Second table: For each model/task, first find max/min across steps for each seed, then average these maxes/mins across seeds
    
    For perplexity, shows minimum value, for all other tasks shows maximum value.
    Includes standard error with ± symbol. Best values are shown in bold.
    
    Args:
        seed_df: DataFrame with seed analysis data
        output_dir: Directory to save the plot
        experiment_name: Name of the experiment
        figsize: Figure size (width, height)
        dpi: DPI for the output figures
        export_latex: Whether to export LaTeX tables
    """
    # Get the original DataFrame from the seed analysis DataFrame
    df = seed_df.reset_index()
    
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Get unique models and tasks
    unique_models = sorted(df['model_base'].unique())
    unique_tasks = sorted(df['task'].unique())
    
    # Adjust figure size based on number of tasks
    width = max(20, len(unique_tasks) * 3)  # At least 3 units per task
    height = max(16, len(unique_models) * 1.6)  # Double height for two tables
    plt.figure(figsize=(width, height), dpi=dpi)
    
    # Add main title
    plt.suptitle(f'Experiment: {experiment_name}\nModel-Task Performance Tables', fontsize=16, y=0.98)
    
    # Create two subplots
    gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # First table: max/min of means across steps
    # For each model/task/step, we already have the mean and std_err from seed_df
    # We just need to find the max/min across steps
    first_table_data = []
    for model in unique_models:
        for task in unique_tasks:
            # Get all steps for this model/task
            model_task_data = df[(df['model_base'] == model) & (df['task'] == task)]
            if not model_task_data.empty:
                # Find the optimal step (max for non-perplexity, min for perplexity)
                if task == 'perplexity':
                    optimal_idx = model_task_data['mean'].idxmin()
                else:
                    optimal_idx = model_task_data['mean'].idxmax()
                optimal_data = model_task_data.loc[optimal_idx]
                first_table_data.append({
                    'model_base': model,
                    'task': task,
                    'mean': optimal_data['mean'],
                    'std_err': optimal_data['std_err']
                })
    
    # Convert to DataFrame for easier manipulation
    first_table_df = pd.DataFrame(first_table_data)
    
    # Create first table
    cell_text1 = []
    for model in unique_models:
        row = []
        for task in unique_tasks:
            mask = (first_table_df['model_base'] == model) & (first_table_df['task'] == task)
            if mask.any():
                row_data = first_table_df[mask].iloc[0]
                row.append(f"{row_data['mean']:.2f} ± {row_data['std_err']:.2f}")
            else:
                row.append("N/A")
        cell_text1.append(row)
    
    # Find best values for first table
    best_values1 = {}
    for task in unique_tasks:
        task_data = first_table_df[first_table_df['task'] == task]
        if task == 'perplexity':
            best_values1[task] = task_data.loc[task_data['mean'].idxmin()]
        else:
            best_values1[task] = task_data.loc[task_data['mean'].idxmax()]
    
    # Create first table
    table1 = ax1.table(
        cellText=cell_text1,
        rowLabels=unique_models,
        colLabels=unique_tasks,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    
    # Make best values bold in first table
    for i, model in enumerate(unique_models):
        for j, task in enumerate(unique_tasks):
            mask = (first_table_df['model_base'] == model) & (first_table_df['task'] == task)
            if mask.any():
                row_data = first_table_df[mask].iloc[0]
                if row_data.equals(best_values1[task]):
                    table1[(i+1, j)].set_text_props(weight='bold')
    
    # Second table: mean of maxes/mins across seeds
    # We need to get the raw data to work with individual seeds
    # Let's read the original data again
    original_df = create_experiment_dataframe(os.path.dirname(output_dir))
    
    # Extract model base names and seeds
    original_df['model_base'] = original_df['model_name'].str.replace(r'_\d+$', '', regex=True)
    original_df['seed'] = original_df['model_name'].str.extract(r'_(\d+)$').astype(int)
    
    # For each model/task/seed, find the optimal value across steps
    second_table_data = []
    for model in unique_models:
        for task in unique_tasks:
            # Get all data for this model/task
            model_task_data = original_df[(original_df['model_base'] == model) & (original_df['task'] == task)]
            if not model_task_data.empty:
                # For each seed, find the optimal value
                seed_optimals = []
                for seed in model_task_data['seed'].unique():
                    seed_data = model_task_data[model_task_data['seed'] == seed]
                    if task == 'perplexity':
                        optimal_value = seed_data['value'].min()
                    else:
                        optimal_value = seed_data['value'].max()
                    seed_optimals.append(optimal_value)
                
                # Calculate mean and standard error across seeds
                mean_value = np.mean(seed_optimals)
                std_err = np.std(seed_optimals) / np.sqrt(len(seed_optimals))
                
                second_table_data.append({
                    'model_base': model,
                    'task': task,
                    'mean': mean_value,
                    'std_err': std_err
                })
    
    # Convert to DataFrame for easier manipulation
    second_table_df = pd.DataFrame(second_table_data)
    
    # Create second table
    cell_text2 = []
    for model in unique_models:
        row = []
        for task in unique_tasks:
            mask = (second_table_df['model_base'] == model) & (second_table_df['task'] == task)
            if mask.any():
                row_data = second_table_df[mask].iloc[0]
                row.append(f"{row_data['mean']:.2f} ± {row_data['std_err']:.2f}")
            else:
                row.append("N/A")
        cell_text2.append(row)
    
    # Find best values for second table
    best_values2 = {}
    for task in unique_tasks:
        task_data = second_table_df[second_table_df['task'] == task]
        if task == 'perplexity':
            best_values2[task] = task_data.loc[task_data['mean'].idxmin()]
        else:
            best_values2[task] = task_data.loc[task_data['mean'].idxmax()]
    
    # Create second table
    table2 = ax2.table(
        cellText=cell_text2,
        rowLabels=unique_models,
        colLabels=unique_tasks,
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    
    # Make best values bold in second table
    for i, model in enumerate(unique_models):
        for j, task in enumerate(unique_tasks):
            mask = (second_table_df['model_base'] == model) & (second_table_df['task'] == task)
            if mask.any():
                row_data = second_table_df[mask].iloc[0]
                if row_data.equals(best_values2[task]):
                    table2[(i+1, j)].set_text_props(weight='bold')
    
    # Adjust table styles
    for table in [table1, table2]:
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.4, 1.8)
    
    # Add titles
    ax1.set_title('Max/Min of Means Across Steps\n(First average across seeds at each step, then find max/min)\nFormat: mean ± std_err\nBest values in bold', pad=20)
    ax2.set_title('Mean of Maxes/Mins Across Seeds\n(First find max/min for each seed using raw values, then average across seeds)\nFormat: mean ± std_err\nBest values in bold', pad=20)
    
    # Hide axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{experiment_name}_model_task_table.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model-task table plot to {output_path}")
    
    if export_latex:
        def latex_escape(s):
            return str(s).replace('_', '\\_').replace('%', '\\%')
        def latex_table(table_df, best_values, caption, label):
            # Mapping from long task names to short names
            task_short_names = {
                'blimp/blimp_filtered': 'blimp',
                'blimp/supplement_filtered': 'b_supp',
                'entity_tracking/entity_tracking': 'entity',
                'ewok/ewok_filtered': 'ewok',
                'perplexity': 'ppl',
                'wug/wug_adj_nominalization': 'wug',
            }
            def short_name(t):
                return latex_escape(task_short_names.get(t, t))
            header = ' & '.join(["Model"] + [short_name(t) for t in unique_tasks]) + r" \\"
            lines = [r"\begin{table}[H]", r"\centering", r"\begin{tabular}{l" + "c"*len(unique_tasks) + "}", r"\toprule", header, r"\midrule"]
            for i, model in enumerate(unique_models):
                row = [latex_escape(model)]
                for j, task in enumerate(unique_tasks):
                    mask = (table_df['model_base'] == model) & (table_df['task'] == task)
                    if mask.any():
                        row_data = table_df[mask].iloc[0]
                        value = f"{row_data['mean']:.2f} $\\pm$ {row_data['std_err']:.2f}"
                        if row_data.equals(best_values[task]):
                            value = r"\textbf{" + value + "}"
                        row.append(value)
                    else:
                        row.append("N/A")
                lines.append(' & '.join(row) + r" \\" )
                lines.append("")
            lines += [r"\bottomrule", r"\end{tabular}", f"\caption{{{caption}}}", f"\label{{{label}}}", r"\end{table}"]
            return '\n'.join(lines)
        latex1 = latex_table(first_table_df, best_values1, "Max/Min of Means Across Steps (mean $\\pm$ std err)", f"tab:{experiment_name}_means")
        latex2 = latex_table(second_table_df, best_values2, "Mean of Maxes/Mins Across Seeds (mean $\\pm$ std err)", f"tab:{experiment_name}_maxmins")
        latex_path = os.path.join(output_dir, f"{experiment_name}_model_task_tables.tex")
        with open(latex_path, 'w') as f:
            f.write("% Table 1: Max/Min of Means Across Steps\n")
            f.write(latex1)
            f.write("\n\n% Table 2: Mean of Maxes/Mins Across Seeds\n")
            f.write(latex2)
        print(f"Saved LaTeX tables to {latex_path}")


def plot_experiment_seeds(
    experiment_dir: str,
    output_dir: str,
    tasks: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    steps: Optional[List[int]] = None,
    figsize: tuple = (12, 24),
    dpi: int = 1200,
    export_latex: bool = False
) -> None:
    """
    Plot experiment results with seed analysis.
    
    Args:
        experiment_dir: Path to the experiment directory
        output_dir: Directory to save the plots
        tasks: List of tasks to plot (if None, plot all tasks)
        models: List of model base names to plot (if None, plot all models)
        steps: List of steps to plot (if None, plot all steps)
        figsize: Figure size (width, height)
        dpi: DPI for the output figures
        export_latex: Whether to export LaTeX tables
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get experiment name
    experiment_name = get_experiment_name(experiment_dir)
    
    # Get experiment data
    df = create_experiment_dataframe(experiment_dir)
    seed_df = create_seed_analysis_dataframe(df)
    
    # First, plot individual runs
    plot_individual_runs(df, output_dir, experiment_name, tasks, models, steps, figsize, dpi)
    
    # Filter data if specified
    if tasks:
        seed_df = seed_df[seed_df['task'].isin(tasks)]
    if models:
        seed_df = seed_df[seed_df['model_base'].isin(models)]
    if steps:
        seed_df = seed_df[seed_df['step'].isin(steps)]
    
    # Create the model-task table plot
    plot_model_task_table(seed_df, output_dir, experiment_name, figsize=(max(12, len(seed_df['task'].unique()) * 2), max(8, len(seed_df['model_base'].unique()) * 0.5)), dpi=dpi, export_latex=export_latex)
    
    # Get unique tasks and models
    unique_tasks = seed_df['task'].unique()
    unique_models = seed_df['model_base'].unique()
    
    # Dynamically set figure height
    n_tasks = len(unique_tasks)
    fig_height = max(5 * n_tasks, 8)
    fig, axes = plt.subplots(n_tasks, 1, figsize=(figsize[0], fig_height))
    if n_tasks == 1:
        axes = [axes]  # Make axes iterable for single plot case
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': figsize,
        'figure.dpi': dpi,
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
    
    # Add main title
    fig.suptitle(f'Experiment: {experiment_name}\nSeed Analysis', fontsize=16, y=0.98)
    
    # Plot each task in its subplot
    for ax, task in zip(axes, unique_tasks):
        # Plot each model
        for model in unique_models:
            model_data = seed_df[(seed_df['model_base'] == model) & (seed_df['task'] == task)]
            
            if not model_data.empty:
                # Sort by step to ensure correct plotting
                model_data = model_data.sort_values('step')
                x = model_data['step']
                y = model_data['mean']
                yerr = model_data['std_err']
                # Plot mean as a line
                line, = ax.plot(x, y, label=model, marker='o')
                # Add error bars
                ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=line.get_color(), capsize=5, elinewidth=1, alpha=0.8)
                # Add fill between mean ± std_err (trapezoidal background)
                color = line.get_color()
                ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.2)
        
        # Customize the subplot
        ax.set_title(f'{task} across Training Steps (with Seed Analysis)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{experiment_name}_seed_analysis.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved seed analysis plot to {output_path}")


def main():
    """Example usage of the script."""
    import argparse
    parser = argparse.ArgumentParser(description='Plot experiment results with seed analysis')
    parser.add_argument('experiment_dir', help='Path to the experiment directory')
    parser.add_argument('output_dir', help='Directory to save the plots')
    parser.add_argument('--tasks', nargs='+', help='List of tasks to plot')
    parser.add_argument('--models', nargs='+', help='List of model base names to plot')
    parser.add_argument('--steps', nargs='+', type=int, help='List of steps to plot')
    parser.add_argument('--figsize', nargs=2, type=int, default=[12, 24], help='Figure size (width height)')
    parser.add_argument('--dpi', type=int, default=1200, help='DPI for the output figures')
    parser.add_argument('--export-latex-table', action='store_true', help='Export LaTeX tables for model-task results')
    args = parser.parse_args()
    
    plot_experiment_seeds(
        args.experiment_dir,
        args.output_dir,
        tasks=args.tasks,
        models=args.models,
        steps=args.steps,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        export_latex=args.export_latex_table
    )


if __name__ == "__main__":
    main() 