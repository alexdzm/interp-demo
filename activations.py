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
        return {"activation": self.activations[idx], "label": self.labels[idx]}


def collect_and_save_activations(
    model: HookedTransformer,
    train_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader],
    hooks: List[str],
    save_dir: str,
    dataset_name: str,
    print_outputs: bool = False,
) -> None:
    """Collect activations from model and save to disk separately for train and test datasets."""

    def collect_activations(dataloader, split_name):
        all_activations = {hook: [] for hook in hooks}
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc=f"Collecting activations for {dataset_name} ({split_name})",
            ):
                questions = batch["question"]
                syc_answers = batch["sycophantic_answer"]
                non_syc_answers = batch["non_sycophantic_answer"]

                for answers, label in [(syc_answers, 1), (non_syc_answers, 0)]:
                    batch_activations = {hook: [] for hook in hooks}
                    for q, a in zip(questions, answers):
                        prompt = f"Question: {q}\n{a}"

                        if print_outputs:
                            print(
                                f"\nProcessing {'sycophantic' if label == 1 else 'non-sycophantic'} example:"
                            )
                            print(prompt)
                            print("-" * 80)

                        tokens = model.to_tokens(prompt)
                        _, cache = model.run_with_cache(tokens, names_filter=hooks)

                        for hook in hooks:
                            activation = cache[hook][:, -1]
                            batch_activations[hook].append(activation)

                    for hook in hooks:
                        hook_activations = torch.cat(batch_activations[hook], dim=0)
                        all_activations[hook].append(hook_activations.cpu())
                    all_labels.extend([label] * len(questions))

        for hook in hooks:
            all_activations[hook] = torch.cat(all_activations[hook], dim=0)
        all_labels = torch.tensor(all_labels)

        os.makedirs(f"{save_dir}/{model.cfg.model_name}", exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"{model.cfg.model_name}/{dataset_name}_{split_name}_activations.pkl",
        )
        with open(save_path, "wb") as f:
            pickle.dump(
                {"activations": all_activations, "labels": all_labels, "hooks": hooks},
                f,
            )

        print(f"Saved {len(all_labels)} {split_name} samples to {save_path}")

    # Collect activations for train
    collect_activations(train_dataloader, "train")

    # Collect activations for test if test_dataloader is not None
    if test_dataloader is not None:
        collect_activations(test_dataloader, "test")


def load_activation_dataloaders(
    save_dir: str,
    model_name: str,
    dataset_name: str,
    hook: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Load saved activations and create train/test dataloaders separately."""

    def load_split(split_name):
        save_path = os.path.join(
            save_dir, f"{model_name}/{dataset_name}_{split_name}_activations.pkl"
        )
        with open(save_path, "rb") as f:
            data = pickle.load(f)

        if hook not in data["hooks"]:
            raise ValueError(
                f"Hook {hook} not found in saved data. Available hooks: {data['hooks']}"
            )

        dataset = ActivationDataset(data["activations"][hook], data["labels"])

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(shuffle if split_name == "train" else False),
            num_workers=num_workers,
        )

    return {"train": load_split("train"), "test": load_split("test")}
