from dataset import create_dataloaders

# Create dataloaders for all datasets
dataloaders = create_dataloaders(
    batch_size=32, train_split=0.9, shuffle=True, num_workers=4, seed=42
)

# Print some statistics
for dataset_name, loaders in dataloaders.items():
    # Calculate total samples for train and test
    train_samples = len(loaders["train"].dataset)
    test_samples = len(loaders["test"].dataset)
    total_samples = train_samples + test_samples

    print(f"\nDataset: {dataset_name}")
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_samples}")
    print(f"Test samples: {test_samples}")

    # Print example sample
    batch = next(iter(loaders["train"]))
    print("\nExample sample:")
    print(f"Question: {batch['question'][0]}")
    print(f"Sycophantic answer: {batch['sycophantic_answer'][0]}")
    print(f"Non-sycophantic answer: {batch['non_sycophantic_answer'][0]}")
