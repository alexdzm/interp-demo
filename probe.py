import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import json
import os
import matplotlib.pyplot as plt


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
    probe: LinearProbe, dataloader: DataLoader, device: torch.device, num_repeats: int = 5
) -> Dict[str, float]:
    """Evaluate probe performance with multiple repeats and bootstrap confidence intervals"""
    probe.eval()
    all_predictions = []
    all_labels = []
    
    # Collect all predictions and labels
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            activations = batch["activation"].to(device)
            labels = batch["label"].to(device)
            
            # Make predictions
            outputs = probe(activations)
            predictions = outputs.squeeze() > 0
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    total = len(all_labels)
    
    # Calculate overall accuracy
    accuracy = (all_predictions == all_labels).mean()
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_accs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(total, size=total, replace=True)
        bootstrap_preds = all_predictions[indices]
        bootstrap_labels = all_labels[indices]
        bootstrap_acc = (bootstrap_preds == bootstrap_labels).mean()
        bootstrap_accs.append(bootstrap_acc)
    
    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    
    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "total_samples": total
    }


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


def plot_accuracies_by_layer(save_dir: str, model_name: str, dataset_name: str = 'mixed'):
    """Plot test accuracies across layers with confidence intervals"""
    # Load results
    results = load_probe_results(save_dir, model_name, dataset_name)
    
    # Extract layers and accuracies with confidence intervals
    layers = sorted(results[dataset_name].keys())
    mixed_accs = [results[dataset_name][layer]['results']['mixed']['accuracy'] for layer in layers]
    mixed_ci_lower = [results[dataset_name][layer]['results']['mixed']['ci_lower'] for layer in layers]
    mixed_ci_upper = [results[dataset_name][layer]['results']['mixed']['ci_upper'] for layer in layers]
    
    pol_accs = [results[dataset_name][layer]['results']['politics']['accuracy'] for layer in layers]
    pol_ci_lower = [results[dataset_name][layer]['results']['politics']['ci_lower'] for layer in layers]
    pol_ci_upper = [results[dataset_name][layer]['results']['politics']['ci_upper'] for layer in layers]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot accuracies with confidence intervals
    plt.plot(layers, mixed_accs, 'b-o', label='Mixed Dataset')
    plt.fill_between(layers, mixed_ci_lower, mixed_ci_upper, color='b', alpha=0.2)
    
    plt.plot(layers, pol_accs, 'r-o', label='Politics Dataset')
    plt.fill_between(layers, pol_ci_lower, pol_ci_upper, color='r', alpha=0.2)
    
    # Add labels and title
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Probe Accuracy by Layer ({model_name})')
    plt.legend()
    plt.grid(True)
    
    # Add horizontal line at 0.5 for random chance
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Set y-axis limits with some padding
    plt.ylim(0.3, 1.1)
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mixed Dataset - Mean Accuracy: {np.mean(mixed_accs):.3f} (95% CI: {np.mean(mixed_ci_lower):.3f}-{np.mean(mixed_ci_upper):.3f})")
    print(f"Politics Dataset - Mean Accuracy: {np.mean(pol_accs):.3f} (95% CI: {np.mean(pol_ci_lower):.3f}-{np.mean(pol_ci_upper):.3f})")
    
    # Find best layers
    best_mixed_idx = np.argmax(mixed_accs)
    best_pol_idx = np.argmax(pol_accs)
    
    print(f"\nBest Layer for Mixed: {layers[best_mixed_idx]} (Acc: {max(mixed_accs):.3f}, CI: {mixed_ci_lower[best_mixed_idx]:.3f}-{mixed_ci_upper[best_mixed_idx]:.3f})")
    print(f"Best Layer for Politics: {layers[best_pol_idx]} (Acc: {max(pol_accs):.3f}, CI: {pol_ci_lower[best_pol_idx]:.3f}-{pol_ci_upper[best_pol_idx]:.3f})")
