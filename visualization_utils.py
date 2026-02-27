# UIT-ROUND v1.3.14
"""
Seaborn visualization utilities for ROUND benchmark suite.
Provides consistent styling and data transformation functions.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from typing import Dict, List, Optional, Tuple, Union

# ... (rest of imports)

# Color Schemes
CLASSIC_PALETTE = {
    'ROUND': '#FF4B4B',
    'GRU': '#4B4BFF'
}

ENHANCED_PALETTE = {
    'ROUND': sns.color_palette("rocket")[3],
    'GRU': sns.color_palette("mako")[3]
}

def setup_seaborn_theme(style='darkgrid', palette='classic'):
    """
    Configure Seaborn theme for ROUND benchmarks.

    Args:
        style: 'darkgrid' (default), 'dark', 'whitegrid', 'white'
        palette: 'classic' (original colors) or 'enhanced' (perceptually uniform)

    Returns:
        dict: Color palette to use for plotting
    """
    color_palette = CLASSIC_PALETTE if palette == 'classic' else ENHANCED_PALETTE

    sns.set_theme(
        style=style,
        context='paper',
        rc={
            'figure.facecolor': '#0e1117',
            'axes.facecolor': '#0e1117',
            'grid.alpha': 0.1,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'legend.fontsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white'
        }
    )
    return color_palette

def prepare_comparison_data(
    round_stats: Union[List[List[float]], np.ndarray],
    gru_stats: Union[List[List[float]], np.ndarray],
    epochs: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Convert ROUND vs GRU statistics to tidy DataFrame.

    Args:
        round_stats: List of runs, each run is list of accuracies
        gru_stats: List of runs, each run is list of accuracies
        epochs: Optional epoch array, defaults to 0..len(run)-1

    Returns:
        DataFrame with columns: ['Epoch', 'Accuracy', 'Model', 'Run']
    """
    # Convert to list if numpy array
    if isinstance(round_stats, np.ndarray):
        round_stats = round_stats.tolist()
    if isinstance(gru_stats, np.ndarray):
        gru_stats = gru_stats.tolist()

    if epochs is None:
        epochs = np.arange(len(round_stats[0]))

    records = []
    for model_name, stats in [('ROUND', round_stats), ('GRU', gru_stats)]:
        for run_idx, run_data in enumerate(stats):
            for epoch_idx, acc in enumerate(run_data):
                records.append({
                    'Epoch': epochs[epoch_idx] if isinstance(epochs, np.ndarray) else epoch_idx,
                    'Accuracy': acc,
                    'Model': model_name,
                    'Run': run_idx
                })
    return pd.DataFrame(records)

def plot_benchmark_comparison(
    df: pd.DataFrame,
    title: str,
    palette: Dict[str, str],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6),
    errorbar: str = 'sd',
    ylabel: str = 'Accuracy',
    xlabel: str = 'Epochs'
) -> None:
    """
    Create standard ROUND vs GRU comparison plot.

    Args:
        df: DataFrame from prepare_comparison_data()
        title: Plot title
        palette: Color dictionary {'ROUND': color, 'GRU': color}
        output_path: Full path for saving figure
        figsize: Figure dimensions
        errorbar: 'sd' for std dev, ('ci', 95) for 95% CI
        ylabel: Y-axis label
        xlabel: X-axis label
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=df,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        palette=palette,
        linewidth=2.5,
        errorbar=errorbar,
        err_style='band',
        err_kws={'alpha': 0.15},
        ax=ax
    )

    ax.set_title(title, fontsize=18, color='white', weight='bold')
    ax.set_xlabel(xlabel, fontsize=16, color='white')
    ax.set_ylabel(ylabel, fontsize=16, color='white')
    ax.legend(loc='best', frameon=True, fancybox=True, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmark_with_runs(
    df: pd.DataFrame,
    title: str,
    palette: Dict[str, str],
    output_path: str,
    figsize: Tuple[int, int] = (12, 6),
    errorbar: str = 'sd',
    ylabel: str = 'Accuracy',
    xlabel: str = 'Epochs',
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Create ROUND vs GRU comparison plot showing individual runs as faint lines.
    Used by benchmark_topology.py.

    Args:
        df: DataFrame from prepare_comparison_data()
        title: Plot title
        palette: Color dictionary {'ROUND': color, 'GRU': color}
        output_path: Full path for saving figure
        figsize: Figure dimensions
        errorbar: 'sd' for std dev, ('ci', 95) for 95% CI
        ylabel: Y-axis label
        xlabel: X-axis label
        ylim: Optional y-axis limits as (ymin, ymax)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot individual runs (faint)
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        units='Run',
        estimator=None,
        alpha=0.15,
        linewidth=1,
        palette=palette,
        legend=False,
        ax=ax
    )

    # Overlay mean with error bands
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Accuracy',
        hue='Model',
        linewidth=2.5,
        errorbar=errorbar,
        err_kws={'alpha': 0.15},
        palette=palette,
        ax=ax
    )

    ax.set_title(title, fontsize=18, color='white', weight='bold')
    ax.set_xlabel(xlabel, fontsize=16, color='white')
    ax.set_ylabel(ylabel, fontsize=16, color='white')

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def prepare_multi_word_data(
    hist_dict: Dict[str, List[float]],
    words: List[str],
    ep_axis: np.ndarray,
    model_name: str
) -> pd.DataFrame:
    """
    Convert word-by-word history to tidy DataFrame.
    Used by benchmark_phase_lock.py.

    Args:
        hist_dict: Dictionary mapping word -> list of accuracies
        words: List of words
        ep_axis: Epoch axis array
        model_name: 'ROUND' or 'GRU'

    Returns:
        DataFrame with columns: ['Epoch', 'Accuracy', 'Word', 'Model']
    """
    records = []
    for word in words:
        for epoch_idx, acc in enumerate(hist_dict[word]):
            records.append({
                'Epoch': ep_axis[epoch_idx],
                'Accuracy': acc,
                'Word': word,
                'Model': model_name
            })
    return pd.DataFrame(records)

def plot_multi_word_comparison(
    hist_r: Dict[str, List[float]],
    hist_g: Dict[str, List[float]],
    words: List[str],
    ep_axis: np.ndarray,
    hidden_size_r: int,
    hidden_size_g: int,
    output_path: str,
    word_colors: List[str]
) -> None:
    """
    Create 2-panel plot for word-by-word learning curves.
    Used by benchmark_phase_lock.py.

    Args:
        hist_r: ROUND history dict (word -> accuracies)
        hist_g: GRU history dict (word -> accuracies)
        words: List of words
        ep_axis: Epoch axis array
        hidden_size_r: ROUND hidden size
        hidden_size_g: GRU hidden size
        output_path: Full path for saving figure
        word_colors: List of colors for each word
    """
    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(14, 12))

    # Prepare DataFrames
    df_round = prepare_multi_word_data(hist_r, words, ep_axis, 'ROUND')
    df_gru = prepare_multi_word_data(hist_g, words, ep_axis, 'GRU')

    # Create word palette
    word_palette = {word: word_colors[i % len(word_colors)] for i, word in enumerate(words)}

    # Plot ROUND panel
    sns.lineplot(
        data=df_round,
        x='Epoch',
        y='Accuracy',
        hue='Word',
        palette=word_palette,
        linewidth=2,
        ax=ax_r
    )
    ax_r.set_title(f"ROUND - Phase Angle Lock ({hidden_size_r} Neurons)",
                   color='#FF5555', fontsize=20, weight='bold')
    ax_r.legend(loc='lower left', fontsize=12, ncol=3)
    ax_r.set_xlabel('Epochs', fontsize=16, color='white')
    ax_r.set_ylabel('Accuracy', fontsize=16, color='white')

    # Plot GRU panel
    sns.lineplot(
        data=df_gru,
        x='Epoch',
        y='Accuracy',
        hue='Word',
        palette=word_palette,
        linewidth=2,
        linestyle='--',
        ax=ax_g
    )
    ax_g.set_title(f"GRU - Standard Gating ({hidden_size_g} Neurons)",
                   color='#5555FF', fontsize=20, weight='bold')
    ax_g.legend(loc='lower left', fontsize=12, ncol=3)
    ax_g.set_xlabel('Epochs', fontsize=16, color='white')
    ax_g.set_ylabel('Accuracy', fontsize=16, color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def update_readme_metadata(readme_path: str, results_dir: str):
    """
    Finds the latest result UID and updates the Batch UID and image paths in README.md.
    """
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Find latest UID (assuming format YYYY-MM-DD_HHMM_...)
    subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not subdirs:
        print("No results found in directory.")
        return

    latest_uid_folder = sorted(subdirs)[-1]
    # Extract UID (everything before the first underscore or the whole thing)
    batch_uid = latest_uid_folder 
    
    print(f"Updating README with latest Batch UID: {batch_uid}")

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Batch UID
    content = re.sub(r"\*\*Batch UID:\*\* `[^`]+`", f"**Batch UID:** `{batch_uid}`", content)

    # 2. Update Image Paths
    # We look for paths like data/UIT_<OLD_UID>/plots/<NAME>_<OLD_UID>.png
    # or results/<OLD_UID>/plots/<NAME>_<OLD_UID>.png
    
    def replacer(match):
        prefix = match.group(1) # ![Caption](
        folder_base = match.group(2) # e.g. results/ or data/UIT_
        old_uid_1 = match.group(3)
        full_plot_name = match.group(4)
        old_uid_2 = match.group(5)
        ext = match.group(6)
        
        # Mapping README keys to Actual Script Output prefixes
        # (README prefix -> Actual Filename prefix)
        mappings = {
            "crystalline_loop": "verification_report",
            "sandwich_duel": "sandwich_duel_story",
            "prism_stack": "prism_stack_duel",
            "color_algebra": "color_algebra_duel",
            "sine_waves": "riemannian_recovery"
        }
        
        plot_name = full_plot_name
        # If the image in README is ![...](.../crystalline_loop_...), 
        # we know it needs to link to verification_report_...
        for key, actual_prefix in mappings.items():
            if full_plot_name.startswith(key):
                plot_name = actual_prefix
                break
        
        # New standard is 'results/<UID>/plots/<NAME>_<UID>.<EXT>'
        return f"{prefix}results/{batch_uid}/plots/{plot_name}_{batch_uid}{ext}"

    # Regex to catch image patterns
    # Group 1: ![...] (
    # Group 2: results/ or data/UIT_
    # Group 3: The OLD UID folder
    # Group 4: The plot name (everything before the final underscore)
    # Group 5: The OLD UID in filename
    # Group 6: Extension
    pattern = r"(\!\[[^\]]*\]\()((?:results/|data/)(?:UIT_)?)(\w+[\w\-]*)/plots/(\w+[\w\-]*)_(\w+[\w\-]*)(\.\w+)"
    
    new_content = re.sub(pattern, replacer, content)

    if new_content == content:
        print("Warning: No image paths were updated. Check regex patterns.")

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    print("README update complete.")
