import datasets
import argparse
import json
import os

def save_dataset_to_json(dataset, save_path):
    dataset = dataset.to_list()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=4)

_CATEGORIES = [
    "Art_and_Entertainment",
    "Lifestyle_and_Personal_Development",
    "Society_and_Culture"
]

_SPLITS = [
    "train",
    "validation",
    "test"
]

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_save_directory", type=str, required=True)
parser.add_argument("--cache_dir", type=str, default="./cache")

if __name__ == "__main__":
    args = parser.parse_args()
    for category in _CATEGORIES:
        for split in _SPLITS:
            dataset = datasets.load_dataset(
                "alireza7/LaMP-QA",
                category,
                split=split,
                cache_dir=args.cache_dir
            )
            save_path = f"{args.dataset_save_directory}/{category}_{split}.json"
            save_dataset_to_json(dataset, save_path)
        