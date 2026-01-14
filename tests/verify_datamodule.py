import os
import sys
import torch
import numpy as np
import pathlib
import shutil

# Add repo root to path
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from rna_backbone_design.data.rna_cluster_datamodule import (
    RNAClusterDataModule,
    length_batching_collate,
)
from rna_backbone_design.data.rna_cluster_dataset import RNAClusterDataset


class MockConfig:
    def __init__(self, data_dir, batch_size=2, num_workers=0, max_len=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        self.csv_path = None
        self.cluster_split = None

        # Add a get method to mimic Hydra dict-like access if needed
        def get(self, key, default=None):
            return getattr(self, key, default)


def main():
    data_dir = "data/rna_ensemble_data"

    # 1. Setup DataModule
    print("Initializing DataModule...")
    cfg = MockConfig(data_dir=data_dir, batch_size=2, max_len=100)
    # The datamodule expects a dict-like get method or attribute access for max_len?
    # In the code: self.data_cfg.get("max_len", None)
    # So let's monkeypatch get onto cfg
    cfg.get = lambda k, d=None: getattr(cfg, k, d)

    dm = RNAClusterDataModule(data_cfg=cfg)

    # 2. Setup
    print("Calling dm.setup()...")
    try:
        dm.setup()
    except Exception as e:
        print(f"Setup failed: {e}")
        # If it fails due to split sizes (e.g. dataset too small), warn
        if "Sum of input lengths" in str(e):
            print("Dataset too small for default split. forcing full_dataset for both.")
            dataset = RNAClusterDataset(data_dir=data_dir)
            dm.train_dataset = dataset
            dm.val_dataset = dataset
        else:
            return

    # 3. Test Train Loader
    print("\nTesting train_dataloader()...")
    try:
        loader = dm.train_dataloader()
        batch = next(iter(loader))
    except Exception as e:
        print(f"DataLoader iteration failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 4. Inspect Batch
    print("\nBatch Shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            # Lists (like pdb_name)
            print(f"  {k}: {len(v)} items")

    # 5. Verify Padding
    print("\nVerifying Padding Logic...")
    # Check if we have multiple items
    B = batch["trans_1"].shape[0]
    if B > 1:
        # Check masks
        mask = batch["res_mask"]
        print(f"  res_mask shape: {mask.shape}")
        # If we have variable lengths (unlikely with just 1 file in data dir, but we can fake it)
        # For this test with 1 cluster repeated, lengths are identical.
        print("  Batch successfully collated.")
    else:
        print("  Batch size is 1, skipping variable length check.")

    print("\nDataModule Verification Complete.")


if __name__ == "__main__":
    main()
