from datasets import load_dataset
import os

def download_and_prepare(save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    print("Downloading Dataset...")
    dataset = load_dataset("ParsBench/parsinlu-machine-translation-fa-en-alpaca-style")

    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)   # 90% train, 10% test
    train_valid = dataset["train"].train_test_split(test_size=0.1, seed=42)  # From train, make validation

    dataset = {
        "train": train_valid["train"],
        "validation": train_valid["test"],
        "test": dataset["test"],
    }

    # Save dataset to disk
    from datasets import DatasetDict
    dataset = DatasetDict(dataset)
    dataset.save_to_disk(save_dir)

    print("Dataset downloaded successfully!")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Dataset saved to {save_dir}")


if __name__ == "__main__":
    download_and_prepare()