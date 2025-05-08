import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from pathlib import Path




def palette_gen(n_colors):
    palette = plt.get_cmap("tab10")
    curr_idx = 0
    while curr_idx < 10:
        yield palette.colors[curr_idx]
        curr_idx += 1


def plot_losses(
    train_losses: List[np.ndarray],
    input_names: List[Path],
    plot_path: Path):
    """Plots training loss curves."""

    train_losses = np.array(train_losses).T
    n_epochs = len(train_losses[0])
    x_epochs = np.arange(n_epochs)

    total_recon_loss = train_losses[: len(input_names)].sum(axis=0)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    n_lines = len(train_losses) + 1

    gen = palette_gen(n_lines)

    if n_lines > 10:
        ax1.plot(x_epochs, total_recon_loss, lw=1.5, c=next(gen))
        plt.title("Total Reconstruction Loss")
    else:
        for loss, name in zip(train_losses[: len(input_names)], input_names):
            ax1.plot(x_epochs, loss, label=name.name, lw=1.5, c=next(gen))
        ax1.plot(x_epochs, total_recon_loss, label="Reconstruction Total", lw=1.5, c=next(gen))
        plt.title("Reconstruction Losses")
    fig.legend()

    plt.xlabel("Epochs")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.set_yscale("log")
    plt.grid(which="minor", axis="y")
    plt.tight_layout()

    plt.savefig(plot_path)


def save_losses(
    train_losses: List[np.ndarray],
    input_names: List[Path],
    save_path: Path):
    """Saves training loss data in a .tsv file."""

    train_losses = np.array(train_losses).T
    n_epochs = len(train_losses[0])
    x_epochs = np.arange(n_epochs)

    index = input_names
    data = pd.DataFrame(train_losses, index=input_names, columns=x_epochs).T
    data.to_csv(save_path, sep="\t")