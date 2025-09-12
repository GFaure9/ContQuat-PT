import os
from typing import Dict


def make_markdown_results_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_folder: str,
    name: str = "results_table",
    precision: int = 2,
):
    """
    Generate and save a Markdown (.md) table summarizing evaluation results.

    Each row corresponds to a case/model, and each column to a metric.
    Metric values are shown in LaTeX math format as '$mean^{\\pm std}$' if both are available,
    otherwise as '$mean$'. Missing values are represented as '$ - $'.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict[str, float]]]
        Dictionary mapping case names to metric results. Each metric should map to
        a dictionary with keys 'mean' and optionally 'std'.

        Example:
        {
            "model_A": {
                "dtw": {"mean": 0.93, "std": 0.01},
                "pck": {"mean": 0.87}
            },
            "model_B": {
                "dtw_mje": {"mean": 0.76, "std": 0.03}
            }
        }

    output_folder : str
        Path to the directory where the Markdown file will be saved.

    name : str, optional
        Name of the output Markdown file (without extension). Default is "results_table".

    precision : int, optional
        Number of decimal digits to display for mean and std. Default is 2.
    """
    if not results:
        raise ValueError("Results dictionary is empty.")

    os.makedirs(output_folder, exist_ok=True)

    formatted_results = {
        case: {metric.upper().replace("_", "-"): val for metric, val in metrics.items()}
        for case, metrics in results.items()
    }

    all_metrics = sorted({metric for case in formatted_results.values() for metric in case})
    header = ["Case"] + all_metrics
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]

    fmt = f"{{:.{precision}f}}"

    for case_name, metrics in sorted(formatted_results.items()):
        row = [case_name]
        for metric in all_metrics:
            if metric in metrics:
                mean = metrics[metric].get("mean")
                std = metrics[metric].get("std")
                if std is not None:
                    row.append(f"${fmt.format(mean)}^{{\\pm {fmt.format(std)}}}$")
                else:
                    row.append(f"${fmt.format(mean)}$")
            else:
                row.append("$ - $")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    file_path = os.path.join(output_folder, f"{name}.md")

    with open(file_path, "w") as f:
        f.write(table)


if __name__ == "__main__":
    # ----- TEST
    my_results = {
        "model_A": {
            "dtw": {"mean": 0.9351, "std": 0.0123},
            "PCK": {"mean": 0.8745}
        },
        "model_B": {
            "DTW": {"mean": 0.9123, "std": 0.0141},
            "dtw_mje": {"mean": 0.7612, "std": 0.0305}
        },
        "model_C": {
            "pck": {"mean": 0.9012, "std": 0.0087}
        }
    }

    make_markdown_results_table(my_results, "./tests_outputs", name="test_table")
