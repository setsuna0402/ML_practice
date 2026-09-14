"""Combine precomputed PINN datasets stored as compressed NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from visualise_validation import parse_args


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the X/Y arrays from one precomputed dataset."""
    try:
        with np.load(path) as data:
            if "X" not in data or "Y" not in data:
                raise ValueError("file must contain both 'X' and 'Y' arrays")
            X = np.asarray(data["X"], dtype=np.float64)
            Y = np.asarray(data["Y"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load {path}: {exc}") from exc

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"{path}: X and Y must both be two-dimensional")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"{path}: X and Y must have the same number of rows")
    return X, Y


def combine_datasets(input_paths: list[Path], output_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate compatible datasets along their sample axis and save them."""
    if len(input_paths) < 2:
        raise ValueError("provide at least two input dataset files")

    output_path = output_path.resolve()
    resolved_inputs = [path.resolve() for path in input_paths]
    if output_path in resolved_inputs:
        raise ValueError("the output file must be different from every input file")

    datasets = [load_dataset(path) for path in input_paths]
    first_X, first_Y = datasets[0]
    for path, (X, Y) in zip(input_paths[1:], datasets[1:]):
        if X.shape[1] != first_X.shape[1]:
            raise ValueError(
                f"{path}: X has {X.shape[1]} columns; expected {first_X.shape[1]}"
            )
        if Y.shape[1] != first_Y.shape[1]:
            raise ValueError(
                f"{path}: Y has {Y.shape[1]} columns; expected {first_Y.shape[1]}"
            )

    combined_X = np.concatenate([X for X, _ in datasets], axis=0)
    combined_Y = np.concatenate([Y for _, Y in datasets], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=combined_X,
        Y=combined_Y,
        n_trajectories=10000,
        source_files=np.asarray([str(path) for path in input_paths]),
    )
    return combined_X, combined_Y


def main() -> None:
    file_name_a = Path("./grackle_relaxed_training_data_5000_a.npz")
    file_name_b = Path("./grackle_relaxed_training_data_5000_b.npz")
    input_files = [file_name_a, file_name_b]
    output_file = Path("./grackle_relaxed_training_data_10000.npz")
    if file_name_a.exists() and file_name_b.exists():
        print(f"Combining {file_name_a} and {file_name_b} into {output_file}")
    else:
        raise FileNotFoundError(
            f"One or both input files do not exist: {file_name_a}, {file_name_b}"
        )
    X, Y = combine_datasets(input_files, output_file)
    print(f"Saved {output_file} with X={X.shape}, Y={Y.shape}")


if __name__ == "__main__":
    main()
