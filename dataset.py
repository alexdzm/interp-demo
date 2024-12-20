from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
import requests
import random


class SycophancyDataset(Dataset):
    """PyTorch Dataset for Sycophancy data"""

    def __init__(self, samples: List[Dict], max_samples: Optional[int] = None):
        """Initialize dataset with list of samples

        Args:
            samples: List of dictionaries containing questions and answers
            max_samples: Maximum number of samples to use (for quick testing)
        """
        # Flatten samples to handle multiple non-sycophantic answers
        self.samples = []
        for sample in samples:
            non_sycophantic_answers = sample["answer_not_matching_behavior"]
            # Convert to list if it's a single string
            if isinstance(non_sycophantic_answers, str):
                non_sycophantic_answers = [non_sycophantic_answers]

            # Create a new sample for each non-sycophantic answer
            for answer in non_sycophantic_answers:
                self.samples.append(
                    {
                        "question": sample["question"],
                        "sycophantic_answer": sample["answer_matching_behavior"],
                        "non_sycophantic_answer": answer,
                    }
                )

        # Optionally limit number of samples
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        sample = self.samples[idx]
        return {
            "question": sample["question"],
            "sycophantic_answer": sample["sycophantic_answer"],
            "non_sycophantic_answer": sample["non_sycophantic_answer"],
        }


def create_dataloaders(
    dataset_names: Optional[List[str]] = None,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """Create PyTorch DataLoaders for specified datasets with train/test splits

    Args:
        dataset_names: List of dataset names to load ('nlp', 'philosophy', 'politics')
        batch_size: Batch size for DataLoaders
        train_split: Fraction of data to use for training (0-1)
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for DataLoader
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to use per dataset (for quick testing)

    Returns:
        Dictionary mapping dataset names to dict containing train and test DataLoaders
    """
    # URLs for the datasets
    URLS = {
        "nlp": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_nlp_survey.jsonl",
        "philosophy": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_philpapers2020.jsonl",
        "politics": "https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        "mixed": ["nlp", "philosophy"],
    }

    if dataset_names is None:
        dataset_names = list(URLS.keys())

    torch.manual_seed(seed)
    dataloaders = {}

    for name in dataset_names:
        if name not in URLS:
            raise ValueError(
                f"Unknown dataset name: {name}. Valid options are: {list(URLS.keys())}"
            )

        try:
            if name == "mixed":
                # Handle mixed dataset specially
                all_samples = []
                for sub_dataset in URLS[name]:
                    response = requests.get(URLS[sub_dataset])
                    samples = [
                        json.loads(line) for line in response.text.strip().split("\n")
                    ]
                    all_samples.extend(samples)

                # Shuffle the combined samples
                random.seed(seed)
                random.shuffle(all_samples)

                # Create PyTorch Dataset with optional sample limit
                pytorch_dataset = SycophancyDataset(
                    all_samples, max_samples=max_samples
                )
            else:
                # Original code for single datasets
                response = requests.get(URLS[name])
                samples = [
                    json.loads(line) for line in response.text.strip().split("\n")
                ]
                pytorch_dataset = SycophancyDataset(samples, max_samples=max_samples)

            # Calculate split sizes
            total_size = len(pytorch_dataset)
            train_size = int(train_split * total_size)
            test_size = total_size - train_size

            # Split into train and test
            train_dataset, test_dataset = random_split(
                pytorch_dataset,
                [train_size, test_size],
                generator=torch.Generator().manual_seed(seed),
            )

            dataloaders[name] = {
                "train": DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                ),
                "test": DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                ),
            }

        except Exception as e:
            print(f"Error loading {name} dataset: {str(e)}")

    return dataloaders
