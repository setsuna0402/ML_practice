import h5py
import numpy as np


# ---------------------------------------------------------------------
# Files to merge
# ---------------------------------------------------------------------
FILE_A = "grackle_relaxed_training_data_5000_a.h5"
FILE_B = "grackle_relaxed_training_data_5000_b.h5"
OUTPUT_FILE = "grackle_relaxed_training_data_merged_10000.h5"


GROUPS_TO_MERGE = ("initial_condition", "value", "pde")


def _same_attribute(a, b):
    """Return True if two HDF5 attribute values are identical."""
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


def merge_precomputed_datasets(file_a, file_b, output_file):
    with h5py.File(file_a, "r") as fa, \
         h5py.File(file_b, "r") as fb, \
         h5py.File(output_file, "w") as fout:

        # -------------------------------------------------------------
        # Merge data groups along the trajectory axis.
        # -------------------------------------------------------------
        for group_name in GROUPS_TO_MERGE:
            if group_name not in fa or group_name not in fb:
                raise ValueError(f"Both files must contain /{group_name}.")

            ga = fa[group_name]
            gb = fb[group_name]

            names_a = set(ga.keys())
            names_b = set(gb.keys())
            if names_a != names_b:
                raise ValueError(
                    f"Dataset names differ in /{group_name}:\n"
                    f"  only in A: {sorted(names_a - names_b)}\n"
                    f"  only in B: {sorted(names_b - names_a)}"
                )

            gout = fout.create_group(group_name)

            for name in sorted(names_a):
                data_a = np.asarray(ga[name])
                data_b = np.asarray(gb[name])

                if data_a.ndim != data_b.ndim:
                    raise ValueError(
                        f"Dimension mismatch for /{group_name}/{name}: "
                        f"{data_a.shape} vs {data_b.shape}"
                    )

                if data_a.shape[1:] != data_b.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for /{group_name}/{name}: "
                        f"{data_a.shape} vs {data_b.shape}"
                    )

                merged = np.concatenate((data_a, data_b), axis=0)

                gout.create_dataset(
                    name,
                    data=merged,
                    compression="gzip",
                    shuffle=True,
                )

        # -------------------------------------------------------------
        # Copy/check metadata.
        # -------------------------------------------------------------
        if "meta" not in fa or "meta" not in fb:
            raise ValueError("Both files must contain /meta.")

        meta_a = fa["meta"].attrs
        meta_b = fb["meta"].attrs
        meta_out = fout.create_group("meta")

        keys_a = set(meta_a.keys())
        keys_b = set(meta_b.keys())
        print("Meta in FILE one")
        print(keys_a)
        if keys_a != keys_b:
            raise ValueError(
                "Metadata attribute names differ between the two files."
            )

        n_a = int(meta_a["n_trajectories"])
        n_b = int(meta_b["n_trajectories"])
        seed_a = int(meta_a["seed"])
        seed_b = int(meta_b["seed"])

        # seed and n_trajectories are expected to differ.
        ignored_keys = {"seed", "n_trajectories"}

        for key in sorted(keys_a - ignored_keys):
            if not _same_attribute(meta_a[key], meta_b[key]):
                raise ValueError(
                    f"Metadata mismatch for attribute {key!r}:\n"
                    f"  A: {meta_a[key]}\n"
                    f"  B: {meta_b[key]}"
                )

            meta_out.attrs[key] = meta_a[key]

        meta_out.attrs["n_trajectories"] = n_a + n_b
        meta_out.attrs["source_seeds"] = np.asarray(
            [seed_a, seed_b], dtype=np.int64
        )

    print("Merged:")
    print(f"  {file_a}: {n_a} trajectories, seed={seed_a}")
    print(f"  {file_b}: {n_b} trajectories, seed={seed_b}")
    print("Saved:")
    print(f"  {output_file}: {n_a + n_b} trajectories")


if __name__ == "__main__":
    merge_precomputed_datasets(FILE_A, FILE_B, OUTPUT_FILE)
