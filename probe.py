import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import json
import os


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


def save_probe_and_results(
    save_dir: str,
    model_name: str,
    dataset_name: str,
    hook: str,
    probe: torch.nn.Module,
    results: dict,
    losses: Optional[list] = None
) -> None:
    """Save probe model and evaluation results to disk."""
    # Create directory structure similar to activations
    probe_dir = os.path.join(save_dir, model_name, "probes")
    results_dir = os.path.join(save_dir, model_name, "results")
    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Extract layer number from hook name
    layer_num = int(hook.split('.')[1])
    
    # Save probe model
    probe_path = os.path.join(
        probe_dir,
        f"{dataset_name}_layer{layer_num}_probe.pt"
    )
    torch.save(probe.state_dict(), probe_path)
    
    # Prepare results for JSON serialization
    # Convert any tensors or numpy arrays to Python types
    json_results = {
        'hook': hook,
        'results': {
            dataset: {
                metric: float(value) if hasattr(value, 'item') else value
                for metric, value in dataset_results.items()
            }
            for dataset, dataset_results in results.items()
        },
        'losses': [float(loss) if hasattr(loss, 'item') else loss for loss in losses] if losses else None
    }
    
    # Save results as JSON
    results_path = os.path.join(
        results_dir,
        f"{dataset_name}_layer{layer_num}_results.json"
    )
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)


def load_probe_results(
    save_dir: str,
    model_name: str,
    dataset_name: Optional[str] = None,
    layer: Optional[int] = None
) -> dict:
    """Load and parse probe results, optionally filtering by dataset and layer."""
    results_dir = os.path.join(save_dir, model_name, "results")
    all_results = {}
    
    # List all result files
    result_files = os.listdir(results_dir)
    
    for file in result_files:
        if not file.endswith('_results.json'):
            continue
            
        # Parse filename
        file_dataset = file.split('_layer')[0]
        file_layer = int(file.split('layer')[-1].split('_')[0])
        
        # Skip if doesn't match filters
        if dataset_name and dataset_name != file_dataset:
            continue
        if layer is not None and layer != file_layer:
            continue
            
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
            
        # Organize results by dataset and layer
        if file_dataset not in all_results:
            all_results[file_dataset] = {}
        all_results[file_dataset][file_layer] = data
    
    return all_results
