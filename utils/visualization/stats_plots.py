import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


def make_box_plots(
        data: List[List[float]],
        labels: List[str],
        output_save_path: str = None,
        multiple_subplots: bool = False,
):
    """
    Generate box plots from multiple datasets.

    Parameters
    -----
    data : List[List[float]]
        A list of lists. Each sublist contains numerical values.
    labels : List[str]
        A list of labels corresponding to each dataset (sublist).
    output_save_path : str, optional
        Save path for generated plot. Plot displayed if None.
    multiple_subplots : bool, optional
        Multiple subplots used for each box-plot if True, else uses the same subplot for all box-plots.
        False will correspond to situations where datasets are of made of values of the same metric for instance
        (e.g. test MSEs for different models). True could correspond to situation where each dataset
        contains values of a different metric (e.g. test MSE, DTW and PCK for one model).
        Stacked side-by-side.
    """
    n = len(data)
    if multiple_subplots:
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 6), constrained_layout=True)
        if n == 1:
            axes = [axes]  # Ensure iterable for a single subplot
        for ax, values, label in zip(axes, data, labels or [None] * n):
            ax.boxplot(values, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'),
                       medianprops=dict(color='red', linewidth=2), whiskerprops=dict(color='blue', linestyle='--'),
                       capprops=dict(color='blue'), flierprops=dict(marker='o', color='black', alpha=0.5))
            if label:
                ax.set_title(label)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xticks([])
    else:
        plt.figure(figsize=(n * 1.5, 6))
        plt.boxplot(data, tick_labels=labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='blue', linestyle='--'),
                    capprops=dict(color='blue'),
                    flierprops=dict(marker='o', color='black', alpha=0.5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    if output_save_path:
        plt.savefig(output_save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_histograms(
        data: List[List[float]],
        labels: List[str],
        bins: int = 20,
        output_save_path: str = None,
):
    """
    Generate histograms for multiple datasets. Stacked side-by-side.

    Parameters
    -----
    Parameters
    -----
    data : List[List[float]]
        A list of lists. Each sublist contains numerical values.
    labels : List[str]
        A list of labels corresponding to each dataset (sublist).
    bins : int, optional
        Number of bins for the histograms (default is 20).
    output_save_path : str, optional
        Save path for generated plot. Plot displayed if None.
    """
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]  # Ensure iterable for single subplot
    for ax, values, label in zip(axes, data, labels):
        ax.hist(values, bins=bins, color='skyblue', edgecolor='black', alpha=0.75)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        if label:
            ax.set_xlabel(label)
    axes[0].set_ylabel("samples")

    if output_save_path:
        plt.savefig(output_save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_box_plot_histo_comparison(
        data: Dict[str, Dict[str, List[float]]],
        bins: int = 20,
        output_save_path: str = None,
        show_outliers: bool = False,
        show_mean: bool = True,
):
    """
    `data` is of the form:
            {
                NAME1: {
                        METRIC1: [val1, val2, ..., valN],
                        METRIC2: [val1', val2', ..., valN'],
                        ...
                      },
                ...
            }
    The saved plot at `output_save_path` will be a .png image with:
        - for each metric a subplot with the box-plots for NAME1, NAME2, ...
        - below for each metric a subplot with the histograms for NAME1, NAME2, ...
    """
    # --- fancy colors
    palette = sns.color_palette("Set2", n_colors=len(data))  # colormaps names: "Set2", "colorblind", "husl", "cubehelix"

    # --- plots
    names = list(data.keys())
    metrics = list(next(iter(data.values())).keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(nrows=2, ncols=n_metrics, figsize=(n_metrics * 3, 6))

    for i, metric in enumerate(metrics):

        # --- box-plots (one axis per metric | multiple box-plots in one axis)
        outliers_kwargs = {"showfliers": False}
        if show_outliers:
            outliers_kwargs = {"showfliers": show_outliers, "flierprops": dict(marker='o', color='black', alpha=0.5)}
        mean_kwargs = {}
        if show_mean:
            mean_kwargs = {"showmeans": show_mean, "meanprops": dict(marker="v", markerfacecolor='black', markeredgecolor="black")}
        boxplot = axes[0, i].boxplot(
            [data[name][metric] for name in names], patch_artist=True,
            boxprops=dict(alpha=0.5),
            medianprops=dict(color='red', linewidth=2), whiskerprops=dict(color='black', linestyle='--'),
            capprops=dict(color='grey'),
            **outliers_kwargs,
            **mean_kwargs
        )

        for patch, color in zip(boxplot['boxes'], palette):
            patch.set_facecolor(color)

        axes[0, i].yaxis.grid(True, linestyle='--', alpha=0.7)
        axes[0, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[0, i].set_title(metric.upper().replace("_", " "))

        # --- histograms
        for name, color in zip(names, palette):
            axes[1, i].hist(data[name][metric], bins=bins, color=color, edgecolor='black', alpha=0.5)
            axes[1, i].xaxis.grid(True, linestyle='--', alpha=0.7)
            axes[1, i].yaxis.grid(True, linestyle='--', alpha=0.7)
            axes[1, i].set_xlabel(metric.replace("_", " "))

    # --- adding shared labels
    axes[0, 0].set_ylabel("Metric Value")
    axes[1, 0].set_ylabel("N Samples")

    # --- adding a legend (with the names of each dict of data - typically models/cases' names)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.5) for color in palette]
    fig.legend(handles, names, loc='upper right', bbox_to_anchor=(1.2, 0.9))

    plt.tight_layout()

    # --- saving if an output path was provided
    if output_save_path:
        plt.savefig(output_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    else:
        plt.show()


if __name__ == "__main__":
    # ========= Tests ==========
    import numpy as np

    # DATA
    my_data = [
        [1, 2, 1, 3, 4, 2, 1, 2, 9, 9, 2, 3, 4, 5, 5, 1, 1, 1],
        [1, 11, 1, 34, 4, 34, 1, 2, 34, 56, 56, 56, 56, 5, 5, 1, 1, 1],
        [.91, 1.91, 1.9, 3.94, 4, 3.94, 1, 2.9, 34, 5.96, 56, .956, .956, 5, 5.9, 1.9, 1, 1],
    ]
    my_labels = [
        "metric1",
        "metric2",
        "metric3",
    ]

    my_data_dict = {
        "model1": {f"metric{k}": vals for k, vals in enumerate(my_data)},
    }
    for m in [2, 3, 4]:
        my_data_dict[f"model{m}"] = {
            n: (
                    np.array(v) + np.random.random(len(v)) * (max(v) - min(v))
            ).tolist() for n, v in my_data_dict["model1"].items()
        }

    # BOX-PLOTS
    make_box_plots(my_data, my_labels, "./tests_outputs/test_box_plots.png")
    make_box_plots(my_data, my_labels, "./tests_outputs/test_box_plots2.png", True)

    # HISTOGRAMS
    make_histograms(my_data, my_labels, 10, "./tests_outputs/test_histograms.png")

    # MODEL RESULTS COMPARISON
    make_box_plot_histo_comparison(my_data_dict, 5,"./tests_outputs/test_compare_results.png")