import json
import time
import os
import math
import warnings
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any

import typer
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.multiprocessing

from alphaoracle.utils.embeddings_args import EmbeddingsArgsParser
from alphaoracle.utils.embeddings_plots import plot_losses, save_losses
from alphaoracle.utils.embeddings_data_preprocess import DataPreprocess
from alphaoracle.utils.embeddings_sampler import SampleStates, NeighborSamplerWithWeights
from alphaoracle.utils.multiGAT import MultiViewGAT
from alphaoracle.utils.weightedGAT import masked_scaled_mse


class EmbeddingsTrainer:
    def __init__(self, config: Union[Path, dict]):
        """Defines the relevant training and forward pass logic for extracting embeddings.

        A model is trained by calling `train()` and the resulting gene/protein embeddings are
        obtained by calling `forward()`.

        Args:
            config (Union[Path, dict]): Path to config file or dictionary containing config
                parameters.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        typer.secho("Using CPU") if self.device.type != "cuda" else typer.secho("Using CUDA")

        self.params = self.parse_config(config)  # parse configuration and load into `params` namespace
        (
            self.index,
            self.masks,
            self.weights,
            self.adj
        ) = self.preprocess_inputs()
        self.train_loaders = self.make_train_loaders()
        self.inference_loaders = self.make_inference_loaders()
        self.model, self.optimizer = self.init_model()

    def parse_config(self, config: Union[Path, dict]) -> Any:
        cp = EmbeddingsArgsParser(config)
        return cp.parse()

    def preprocess_inputs(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, List]:
        preprocessor = DataPreprocess(
            self.params.input_names,
            delimiter=self.params.delimiter,
        )
        return preprocessor.process()

    def make_train_loaders(self) -> List[NeighborSamplerWithWeights]:
        """Create data loaders for training."""
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[self.params.neighbor_sample_size] * self.params.gat_shapes["n_layers"],
                batch_size=self.params.batch_size,
                sampler=SampleStates(torch.arange(len(self.index))),
                shuffle=False,
            )
            for ad in self.adj
        ]

    def make_inference_loaders(self) -> List[NeighborSamplerWithWeights]:
        """Create data loaders for inference."""
        return [
            NeighborSamplerWithWeights(
                ad,
                sizes=[-1] * self.params.gat_shapes["n_layers"],  # all neighbors
                batch_size=1,
                sampler=SampleStates(torch.arange(len(self.index))),
                shuffle=False,
            )
            for ad in self.adj
        ]

    def init_model(self) -> Tuple[MultiViewGAT, torch.optim.Optimizer]:
        """Initialize the model and optimizer."""
        model = MultiViewGAT(
            len(self.index),
            self.params.gat_shapes,
            self.params.embedding_size,
            len(self.adj),
        )
        model.apply(self._init_model_weights)

        # Load pretrained model
        if self.params.pretrained_model_path:
            typer.echo("Loading pretrained model...")
            missing_keys, unexpected_keys = model.load_state_dict(
                torch.load(self.params.pretrained_model_path, map_location=self.device),
                strict=False
            )
            if missing_keys:
                warnings.warn(
                    "The following parameters were missing from the provided pretrained model:"
                    f"{missing_keys}"
                )
            if unexpected_keys:
                warnings.warn(
                    "The following unexpected parameters were provided in the pretrained model:"
                    f"{unexpected_keys}"
                )

        # Push model to device
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.params.learning_rate, weight_decay=0.0)

        return model, optimizer

    def _init_model_weights(self, model: torch.nn.Module) -> None:
        """Initialize model weights using Kaiming uniform."""
        if hasattr(model, "weight") and isinstance(model.weight, torch.Tensor):
            torch.nn.init.kaiming_uniform_(model.weight, a=0.1)

    def train(self, verbosity: Optional[int] = 1) -> None:
        """Trains model to obtain gene embeddings.

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # Track losses per epoch.
        train_loss = []

        best_loss = None
        best_state = None

        # Train model.
        for epoch in range(self.params.epochs):

            time_start = time.time()

            # Track average loss across batches
            epoch_losses = np.zeros(len(self.adj))

            _, losses = self.train_step()

            epoch_losses = [
                ep_loss + b_loss.item() / (len(self.index) / self.params.batch_size)
                for ep_loss, b_loss in zip(epoch_losses, losses)
            ]

            if verbosity:
                progress_string = self.create_progress_string(epoch, epoch_losses, time_start)
                typer.echo(progress_string)

            train_loss.append(epoch_losses)

            # Store best parameter set
            if not best_loss or sum(epoch_losses) < best_loss:
                best_loss = sum(epoch_losses)
                state = {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "best_loss": best_loss,
                }
                best_state = state

        self.train_loss, self.best_state = train_loss, best_state

    def train_step(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Defines training behaviour."""

        # Get random integers for batch.
        rand_int = SampleStates.step(len(self.index))
        int_splits = torch.split(rand_int, self.params.batch_size)

        # Initialize loaders to current batch.
        batch_loaders = self.train_loaders
        mask_splits = torch.split(self.masks[rand_int], self.params.batch_size)

        # List of losses.
        losses = [torch.tensor(0.0, device=self.device) for _ in range(len(batch_loaders))]

        # Get the data flow for each input, stored in a tuple.
        for batch_idx, (batch_masks, node_ids) in enumerate(zip(mask_splits, int_splits)):
            # Move mask tensors to the device
            batch_masks = batch_masks.to(self.device)

            # Sample from each loader for the current batch
            data_flows = [loader.sample(node_ids) for loader in batch_loaders]

            self.optimizer.zero_grad()

            output, _, _ = self.model(data_flows, batch_masks, device=self.device)
            recon_losses = [
                masked_scaled_mse(
                    output,
                    self.adj[i],
                    self.weights[i],
                    node_ids,
                    batch_masks[:, i],
                    device=self.device
                )
                for i in range(len(self.adj))
            ]

            losses = [loss + curr_loss for loss, curr_loss in zip(losses, recon_losses)]
            loss_sum = sum(recon_losses)

            loss_sum.backward()
            self.optimizer.step()

        return output, losses

    def create_progress_string(
            self, epoch: int, epoch_losses: List[float], time_start: float) -> str:
        """Creates a training progress string to display."""
        sep = "|"

        progress_string = (
            f"{'Epoch'}: {epoch + 1} {sep} "
            f"{'Loss Total'}: {sum(epoch_losses):.6f} {sep} "
        )
        if len(self.adj) <= 10:
            for i, loss in enumerate(epoch_losses):
                progress_string += f"{f'Loss {i + 1}'}: {loss:.6f} {sep} "
        progress_string += f"{'Time (s)'}: {time.time() - time_start:.4f}"
        return progress_string

    def forward(self, verbosity: int = 1) -> None:
        """Runs the forward pass on the trained model to extract embeddings.

        Args:
            verbosity (int): 0 to supress printing (except for progress bar), 1 for regular printing.
        """

        # Begin inference
        self.model.load_state_dict(
            self.best_state["state_dict"]
        )  # Recover model with lowest reconstruction loss
        if verbosity:
            typer.echo(
                (
                    f"""Loaded best model from epoch {f"{self.best_state['epoch']}"} """
                    f"""with loss {f"{self.best_state['best_loss']:.6f}"}"""
                )
            )

        self.model.eval()
        SampleStates.step(len(self.index), random=False)
        emb_list = []

        # Create a list to collect all nodes
        nodes_to_process = list(zip(self.masks, range(len(self.index))))

        # Use typer's progressbar
        with typer.progressbar(
                nodes_to_process,
                label=f"{'Forward Pass'}:",
                length=len(self.index),
        ) as progress:
            for mask, node_idx in progress:
                # Sample individually for each node
                mask = mask.reshape((1, -1)).to(self.device)
                node_tensor = torch.tensor([node_idx], dtype=torch.long)

                # Sample from each loader for the current node
                data_flows = [loader.sample(node_tensor) for loader in self.inference_loaders]

                # Forward pass
                with torch.no_grad():
                    _, emb, learned_scales = self.model(data_flows, mask, evaluate=True, device=self.device)
                    emb_list.append(emb.detach().cpu().numpy())

        # Concatenate embeddings
        emb = np.concatenate(emb_list)
        emb_df = pd.DataFrame(emb, index=self.index)

        # Create output directory if it doesn't exist
        out_path = self.params.output_name
        out_path.mkdir(exist_ok=True)

        # Save embeddings
        emb_df.index.name = "node"
        emb_df.to_csv(out_path / "Embeddings.csv", sep=",")

        # Free memory (necessary for sequential runs)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Output loss plot
        if self.params.plot_loss:
            if verbosity:
                typer.echo("Plotting loss...")
            plot_losses(
                self.train_loss,
                self.params.input_names,
                out_path / "loss.png"
            )

        # Save losses per epoch
        if self.params.save_loss_data:
            if verbosity:
                typer.echo("Saving loss data...")
            save_losses(
                self.train_loss,
                self.params.input_names,
                out_path / "loss.tsv"
            )

        # Save model
        if self.params.save_model:
            if verbosity:
                typer.echo("Saving model...")
            torch.save(self.model.state_dict(), out_path / "model.pt")

        # Save internal learned network scales
        if verbosity:
            typer.echo("Saving network scales...")
        learned_scales = pd.DataFrame(
            learned_scales.detach().cpu().numpy(), columns=self.params.input_names
        ).T
        learned_scales.to_csv(
            out_path / "network_scales.tsv", header=False, sep="\t"
        )

        typer.echo("Complete!")
