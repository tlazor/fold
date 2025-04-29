from datetime import datetime
import io
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

signal = "embedding"
# Read the input data from a file (replace 'your_file.txt' with the actual file path)
file_path = f"{signal}_output.txt"


def parse_multiple_runs(file_path):
    # Initialize a dictionary to hold data for each run
    runs_data = {}

    # Read the file
    with open(file_path, "r") as f:
        content = f.read()

    # Hard-code the sections for the three runs
    run_names = ["compute_overlaps", "kl_divergence_matrix", "mae_matrix"]
    next_run_names = run_names[1:] + [None]

    for run_name, next_run_name in zip(run_names, next_run_names):
        # Extract the data for the current run
        run_data_pattern = rf"{run_name}\s*(.*?)\s*(?={next_run_name}|$)"
        run_data_match = re.search(run_data_pattern, content, re.DOTALL)

        if run_data_match:
            run_data = run_data_match.group(1)
        else:
            print(f"Run {run_name} not found in the file.")
            continue

        # Split the data by 'layer' keyword for the current run
        layers_data = []
        layers = re.split(r"layer=(\d+)", run_data)

        for j in range(
            1, len(layers), 2
        ):  # Skip the first element, which is before the first 'layer'
            layer_number = int(layers[j])
            layer_table = layers[j + 1]

            # Read the table using pandas
            table_str = layer_table.strip()
            data = pd.read_csv(
                io.StringIO(table_str), sep="|", engine="python", skipinitialspace=True
            )
            # Remove dashes
            data = data.drop(0).reset_index(drop=True)

            # Clean column names
            data.columns = [col.strip() for col in data.columns]

            data["p_pval"] = pd.to_numeric(data["p_pval"], errors="coerce")
            data["s_pval"] = pd.to_numeric(data["s_pval"], errors="coerce")

            # Add layer number to each entry
            data["layer"] = layer_number

            # Append this layer's data to the layers_data list
            layers_data.append(data)

        # Concatenate all the layers' data for this run
        run_dataframe = pd.concat(layers_data, ignore_index=True)
        runs_data[run_name] = run_dataframe

    return runs_data


# Parse the data from the file
runs_data = parse_multiple_runs(file_path)

# Plotting for each run
for run_name, data in runs_data.items():
    print(f"Plotting for run: {run_name}")

    metrics = data["metric"].unique()

    print(f"{metrics=}")

    # Create a plot for each metric
    for metric in metrics:
        print(f"Plotting {metric=}")
        # Filter the data for the current metric
        metric_data = data[data["metric"] == metric]

        # Create a new figure for each metric
        plt.figure(figsize=(10, 6))

        # Plot p_pval vs layer
        plt.plot(
            metric_data["layer"].tolist(),
            metric_data["p_pval"].tolist(),
            label="p_pval",
            marker="o",
        )

        print(metric_data["layer"].tolist(), metric_data["p_pval"].tolist())

        # Plot s_pval vs layer
        plt.plot(
            metric_data["layer"], metric_data["s_pval"], label="s_pval", marker="s"
        )

        # Add labels and title
        plt.xlabel("Layer")
        plt.ylabel("P-Value")
        plt.title(f"{run_name} - {metric.strip()} ({signal})")
        plt.legend()

        plot_dir = Path(".") / "plots"

        if not plot_dir.exists():
            plot_dir.mkdir(parents=True)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]

        # Create a filename with the current date and time
        filename = plot_dir / f"plot_{current_time}.png"

        plt.savefig(filename)
        plt.show()
        plt.clf()
