from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import numpy as np
from dataset import create_dataloaders
from activations import collect_and_save_activations, load_activation_dataloaders
from sae_lens import SAE
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset

import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def analyze_sae_features(
    sae: SAE, dataloader: DataLoader, top_k: int = 10
) -> List[Tuple[int, float]]:
    """Analyze which SAE features correlate most with sycophancy"""
    sae.eval()
    all_encoded = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing data"):
            activations = batch["activation"].to(sae.device)
            labels = batch["label"].to(sae.device)
            encoded = sae.encode(activations)
            all_encoded.append(encoded)
            all_labels.append(labels)

    encoded = torch.cat(all_encoded, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Calculate correlation between each feature and sycophancy
    correlations = []
    for i in range(encoded.shape[1]):  # iterate over features
        feature_acts = encoded[:, i]
        correlation = torch.corrcoef(torch.stack([feature_acts, labels.float()]))[0, 1]
        if not torch.isnan(correlation):
            correlations.append((i, correlation.item()))

    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations[:top_k]


def evaluate_sae_features(
    sae: SAE,
    dataloader: DataLoader,
    top_features: List[int],
) -> Dict[str, float]:
    """Evaluate how well top SAE features predict sycophancy with bootstrap confidence intervals"""
    sae.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            activations = batch["activation"].to(sae.device)
            labels = batch["label"].to(sae.device)
            encoded = sae.encode(activations)

            # Use mean activation of top features as prediction
            feature_acts = encoded[:, top_features]
            predictions = (feature_acts.mean(dim=1) > 0).float()

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
        indices = np.random.choice(total, size=total, replace=True)
        bootstrap_preds = all_predictions[indices]
        bootstrap_labels = all_labels[indices]
        bootstrap_acc = (bootstrap_preds == bootstrap_labels).mean()
        bootstrap_accs.append(bootstrap_acc)

    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)

    return {
        "accuracy": float(accuracy),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "total_samples": total,
    }


def save_sae_results(
    save_dir: str,
    model_name: str,
    dataset_name: str,
    hook: str,
    results: dict,
    top_features: List[Tuple[int, float]],
) -> None:
    """Save SAE evaluation results to disk."""
    results_dir = os.path.join(save_dir, model_name, "sae_results")
    os.makedirs(results_dir, exist_ok=True)

    # Extract layer number from hook name
    layer_num = int(hook.split(".")[1])

    # Prepare results for JSON serialization
    json_results = {
        "hook": hook,
        "results": {dataset_name: results},
        "top_features": [
            {"feature_idx": int(idx), "correlation": float(corr)}
            for idx, corr in top_features
        ],
    }

    # Save results as JSON
    results_path = os.path.join(
        results_dir, f"{dataset_name}_layer{layer_num}_results.json"
    )
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)


def main():
    # Setup - force CUDA if available
    save_dir = ".data/sae"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available, using CPU")

    # Load model and SAE
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)

    # Create dataloaders using your existing pipeline
    dataloaders = create_dataloaders(
        dataset_names=["mixed"],
        batch_size=128,
        train_split=0.9,
        max_samples=1000,  # Adjust as needed
    )

    # These are chosen from middling l0 values and from later layers
    target_layers = [25, 24, 23]
    sae_ids = [
        "layer_25/width_16k/average_l0_116",
        "layer_24/width_16k/average_l0_73",
        "layer_23/width_16k/average_l0_75",
    ]
    hooks = [f"blocks.{layer}.hook_resid_post" for layer in target_layers]

    # Collect and save activations if not already done
    collect_and_save_activations(
        model=model,
        train_dataloader=dataloaders["mixed"]["train"],
        test_dataloader=dataloaders["mixed"]["test"],
        hooks=hooks,
        save_dir=save_dir,
        dataset_name="mixed",
    )

    # Analyze SAEs from different layers
    results = []

    for idx, layer in enumerate(target_layers):
        print(f"\nAnalyzing layer {layer}")

        # Load SAE using correct method from notebook
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res", sae_id=sae_ids[idx], device=device
        )

        # Load activation dataloaders
        activation_loaders = load_activation_dataloaders(
            save_dir=save_dir,
            model_name=model.cfg.model_name,
            dataset_name="mixed",
            hook=f"blocks.{layer}.hook_resid_post",
            batch_size=32,
        )

        # Find most correlated features
        top_features = analyze_sae_features(sae, activation_loaders["train"])
        print("\nTop correlated features:")
        for feature_idx, correlation in top_features:
            print(f"Feature {feature_idx}: correlation = {correlation}")

        top_features_indices = [
            f[0] for f in top_features[:1]
        ]  # only keep the top fixture
        # Evaluate on test set with bootstrapping
        test_results = evaluate_sae_features(
            sae, activation_loaders["test"], top_features_indices
        )

        # Save results
        save_sae_results(
            save_dir=save_dir,
            model_name=model.cfg.model_name,
            dataset_name="mixed",
            hook=f"blocks.{layer}.hook_resid_post",
            results=test_results,
            top_features=top_features[:1],
        )

        results.append(
            {
                "layer": layer,
                "top_features": top_features[:1],
                "test_results": test_results,
            }
        )

        print(f"Test accuracy: {test_results['accuracy']:.2%}")
        print(
            f"95% CI: [{test_results['ci_lower']:.2%}, {test_results['ci_upper']:.2%}]"
        )

        # Calculate basic L0 stats using your activation data
        with torch.no_grad():
            batch = next(iter(activation_loaders["train"]))
            activations = batch["activation"].to(sae.device)
            feature_acts = sae.encode(activations)
            l0 = (feature_acts > 0).float().sum(-1).detach()
            print(f"Average L0: {l0.mean().item()}")

    # Print summary
    print("\nSummary of results:")
    for result in results:
        print(f"\nLayer {result['layer']}:")
        print(f"Test accuracy: {result['test_results']['accuracy']:.2%}")
        print(
            f"95% CI: [{result['test_results']['ci_lower']:.2%}, {result['test_results']['ci_upper']:.2%}]"
        )
        print("Top features and correlations:")
        for feature_idx, correlation in result["top_features"]:
            print(f"Feature {feature_idx}: {correlation:.3f}")


if __name__ == "__main__":
    main()
