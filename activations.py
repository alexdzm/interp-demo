from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformer_lens import HookedTransformer, ActivationCache
from tqdm.auto import tqdm
import os
import pickle

class ActivationDataset(Dataset):
    """Dataset for storing and loading model activations"""
    
    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            activations: Tensor of shape [n_samples, d_model]
            labels: Binary tensor of shape [n_samples] (1 for sycophantic, 0 for not)
        """
        self.activations = activations
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.labels)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "activation": self.activations[idx],
            "label": self.labels[idx]
        }

def collect_and_save_activations(
    model: HookedTransformer,
    dataloader: DataLoader,
    hooks: List[str],
    save_dir: str,
    dataset_name: str,
    print_outputs: bool = False
) -> None:
    """Collect activations from model and save to disk
    
    Args:
        model: HookedTransformer model
        dataloader: DataLoader containing samples
        hooks: List of hook points to collect activations from (e.g. ['blocks.0.hook_resid_post'])
        save_dir: Directory to save activations
        dataset_name: Name of dataset (for saving)
        print_outputs: Whether to print the questions and answers being processed
    """
    device = next(model.parameters()).device
    all_activations = {hook: [] for hook in hooks}
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting activations for {dataset_name}"):
            questions = batch['question']
            syc_answers = batch['sycophantic_answer']
            non_syc_answers = batch['non_sycophantic_answer']
            
            # Get activations for both types of answers
            for answers, label in [(syc_answers, 1), (non_syc_answers, 0)]:
                batch_activations = {hook: [] for hook in hooks}
                for q, a in zip(questions, answers):
                    prompt = f"Question: {q}\n{a}"
                    
                    if print_outputs:
                        print(f"\nProcessing {'sycophantic' if label == 1 else 'non-sycophantic'} example:")
                        print(prompt)
                        print("-" * 80)
                    
                    # Get activations
                    tokens = model.to_tokens(prompt)
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=hooks
                    )
                    
                    # Get activation for the last token only
                    for hook in hooks:
                        activation = cache[hook][:, -1]
                        batch_activations[hook].append(activation)
                
                # Stack batch activations for each hook
                for hook in hooks:
                    hook_activations = torch.cat(batch_activations[hook], dim=0)
                    all_activations[hook].append(hook_activations.cpu())
                all_labels.extend([label] * len(questions))
    
    # Convert to tensors
    for hook in hooks:
        all_activations[hook] = torch.cat(all_activations[hook], dim=0)
    all_labels = torch.tensor(all_labels)
    
    # Save to disk
    os.makedirs(f"{save_dir}/{model.cfg.model_name}", exist_ok=True)
    save_path = os.path.join(save_dir, f"{model.cfg.model_name}/{dataset_name}_hooks.pkl")  # Changed filename
    with open(save_path, 'wb') as f:
        pickle.dump({
            'activations': all_activations,  # Now a dict of hook -> activations
            'labels': all_labels,
            'hooks': hooks  # Save the hook names for reference
        }, f)
    
    print(f"Saved {len(all_labels)} samples to {save_path}")

def load_activation_dataloaders(
    save_dir: str,
    model_name: str,
    dataset_name: str,
    hook: str,  # Changed from layer to hook
    batch_size: int = 32,
    train_split: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """Load saved activations and create train/test dataloaders
    
    Args:
        save_dir: Directory containing saved activations
        model_name: Name of the model used to generate activations
        dataset_name: Name of dataset to load
        hook: Which hook's activations to load (e.g. 'blocks.0.hook_resid_post')
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        shuffle: Whether to shuffle training data
        num_workers: Number of workers for dataloaders
        seed: Random seed
        
    Returns:
        Dict containing train and test dataloaders
    """
    # Load saved activations
    save_path = os.path.join(save_dir, f"{model_name}/{dataset_name}_hooks.pkl")
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    
    if hook not in data['hooks']:
        raise ValueError(f"Hook {hook} not found in saved data. Available hooks: {data['hooks']}")
    
    # Create dataset using the specified hook's activations
    dataset = ActivationDataset(data['activations'][hook], data['labels'])
    
    # Split into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    return {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    } 