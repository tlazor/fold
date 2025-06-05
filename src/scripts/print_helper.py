from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


def save_fig(filename):
    plot_dir = Path(".") / "plots"

    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-2]

    # Create a filename with the current date and time
    filename = plot_dir / f"{filename}_{current_time}.png"
    plt.savefig(filename)
