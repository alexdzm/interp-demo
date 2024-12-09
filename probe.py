import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


class LinearProbe(nn.Module):
    """Simple linear probe for classifying activations"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Binary classification

    def forward(self, x):
        return self.linear(x)


def train_probe(
    dataloader: DataLoader,
    input_dim: int,
    device: torch.device,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
) -> Tuple[LinearProbe, List[float]]:
    """Train a linear probe to detect sycophancy"""

    # Initialize probe
    probe = LinearProbe(input_dim).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            activations = batch["activation"].to(device)
            labels = batch["label"].float().unsqueeze(1).to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = probe(activations)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Record mean loss for epoch
        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        print(f"Epoch {epoch+1} Loss: {mean_loss:.4f}")

    return probe, losses


def evaluate_probe(
    probe: LinearProbe, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Evaluate probe performance"""
    probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            activations = batch["activation"].to(device)
            labels = batch["label"].to(device)

            # Make predictions
            outputs = probe(activations)
            predictions = outputs.squeeze() > 0

            # Calculate accuracy
            correct += (predictions == labels).sum().item()
            total += len(labels)

    accuracy = correct / total
    return {"accuracy": accuracy, "total_samples": total}
