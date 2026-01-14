import os
import sys
import torch
import numpy as np
import pathlib
import shutil

# Add repo root to path
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from rna_backbone_design.data.rna_cluster_dataset import RNAClusterDataset
from rna_backbone_design.data import utils as du


def create_dummy_embeddings(data_dir):
    """
    The dataset loader requires 'embedding' folder with *_single.npy and *_pair.npy.
    We create these dummies to verify the dataloader logic.
    """
    cluster_dir = pathlib.Path(data_dir) / "cluster_0"
    embedding_dir = cluster_dir / "embedding"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    # Check if there is a feature file to gauge length
    feature_dir = cluster_dir / "features"
    pkl_files = list(feature_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {feature_dir}. Cannot determine length.")
        # Create a dummy length
        length = 50
    else:
        # Load one to get length
        data = du.read_pkl(str(pkl_files[0]))
        length = len(data["aatype"])
        print(f"Found PKL file with length {length}")

    # Create dummy embeddings
    # sizes from typical models: single [L, 1280] or similar, pair [L, L, 64]
    single_emb = np.random.rand(length, 640).astype(np.float32)
    pair_emb = np.random.rand(length, length, 128).astype(np.float32)

    np.save(embedding_dir / "dummy_single.npy", single_emb)
    np.save(embedding_dir / "dummy_pair.npy", pair_emb)
    print(f"Created dummy embeddings in {embedding_dir}")
    return embedding_dir


def main():
    data_dir = "data/rna_ensemble_data"

    # 1. Setup Data
    # Ensure we have what the dataset expects
    if not os.path.exists(data_dir):
        print(
            f"Data directory {data_dir} does not exist. Please run preprocessing first."
        )
        return

    # Create dummy embeddings since our preprocessor doesn't make them yet
    try:
        created_embed_dir = create_dummy_embeddings(data_dir)
    except Exception as e:
        print(f"Failed to create dummy embeddings: {e}")
        return

    # 2. Initialize Dataset
    print("\nInitializing RNAClusterDataset...")
    try:
        dataset = RNAClusterDataset(data_dir=data_dir, split="train")
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    print(f"Dataset length: {len(dataset)}")

    # 3. Test __getitem__
    print("\nTesting __getitem__(0)...")
    try:
        item = dataset[0]
    except Exception as e:
        print(f"__getitem__ failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Inspect Output
    print("\nKeys in returned item:")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} (dtype={v.dtype})")
        else:
            print(f"  {k}: {type(v)}")

    # 5. Sanity Checks
    print("\nRunning Sanity Checks...")

    # Check Torsions shape (should be [L, 8, 2])
    torsions = item["torsion_angles_sin_cos"]
    if torsions.shape[1:] == (8, 2):
        print("✅ Torsion angles shape is correct: [L, 8, 2]")
    else:
        print(
            f"❌ Torsion angles shape mismatch: expected [L, 8, 2], got {torsions.shape}"
        )

    # Check Frames
    trans = item["trans_1"]
    rots = item["rotmats_1"]
    if trans.shape[1] == 3 and rots.shape[1:] == (3, 3):
        print("✅ Frame shapes are correct")
    else:
        print(f"❌ Frame shape mismatch: trans {trans.shape}, rots {rots.shape}")

    # Check Embeddings match length
    seq_len = item["aatype"].shape[0]
    if item["single_embeds"].shape[0] == seq_len:
        print(f"✅ Embedding length matches sequence length ({seq_len})")
    else:
        print(
            f"❌ Length mismatch: seq {seq_len} vs embed {item['single_embeds'].shape[0]}"
        )

    print("\nVerification Complete.")


if __name__ == "__main__":
    main()
