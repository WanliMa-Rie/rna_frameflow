import torch
import numpy as np
import pathlib
from rna_backbone_design.data.rna_cluster_dataset import RNAClusterDataset


def test_dataset_loading():
    # Setup dummy data structure
    data_dir = "data/test_cluster_root"
    cluster_dir = pathlib.Path(data_dir) / "cluster_1"
    struct_dir = cluster_dir / "structure"
    embed_dir = cluster_dir / "embedding"
    struct_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy PDB (very minimal)
    pdb_content = """ATOM      1  O4'   A A   1      10.000  10.000  10.000  1.00  0.00           O
ATOM      2  C4'   A A   1      11.000  11.000  11.000  1.00  0.00           C
ATOM      3  C3'   A A   1      12.000  12.000  12.000  1.00  0.00           C
TER
"""
    with open(struct_dir / "test.pdb", "w") as f:
        f.write(pdb_content)

    # Create dummy embeddings
    np.save(embed_dir / "test_single.npy", np.zeros((1, 128)))
    np.save(embed_dir / "test_pair.npy", np.zeros((1, 1, 64)))

    dataset = RNAClusterDataset(data_dir=data_dir)
    print(f"Dataset size: {len(dataset)}")

    try:
        item = dataset[0]
        print("Success! Keys in item:")
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error during __getitem__: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataset_loading()
