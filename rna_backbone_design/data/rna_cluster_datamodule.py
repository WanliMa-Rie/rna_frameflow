import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from rna_backbone_design.data.rna_cluster_dataset import RNAClusterDataset


def length_batching_collate(batch):
    """
    Simple collate function that stacks features.
    Assumes batch items are already cropped/padded to same length
    OR handles variable length by padding (if implemented).

    Since RNAClusterDataset currently returns variable length sequences
    (unless max_length is forced), we usually need to pad here.
    """
    # Just use simple stacking if lengths are equal,
    # otherwise we need a custom padder.
    # For now, let's assume the user configures max_length or
    # we implement basic padding logic here.

    # Check if lengths vary
    lengths = [b["trans_1"].shape[0] for b in batch]
    max_len = max(lengths)

    # Keys to pad
    keys_1d = ["res_mask", "sequence"]  # [L]
    keys_2d = ["trans_1", "single_embeds", "torsion_angles_mask"]  # [L, D]
    keys_3d = [
        "rotmats_1",
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
    ]  # [L, 3, 3], [L, 7, 2]
    keys_pair = ["pair_embeds"]  # [L, L, D]

    padded_batch = {}

    # Initialize basic lists for keys not in padding logic (like names)
    padded_batch["pdb_name"] = [b["pdb_name"] for b in batch]
    padded_batch["is_na_residue_mask"] = torch.zeros(
        len(batch), max_len
    )  # Required by model logic

    # Pre-allocate tensors
    # To handle heterogeneous batches properly we need masking

    # Let's inspect one item to get dimensions
    item0 = batch[0]
    B = len(batch)

    # 1. 2D features [B, L, D]
    for k in keys_2d:
        if k not in item0:
            continue
        dim = item0[k].shape[-1]
        out = torch.zeros(B, max_len, dim, dtype=item0[k].dtype)
        for i, b in enumerate(batch):
            l = lengths[i]
            out[i, :l] = b[k]
        padded_batch[k] = out

    # 2. 1D features [B, L]
    for k in keys_1d:
        if k not in item0:
            continue
        out = torch.zeros(B, max_len, dtype=item0[k].dtype)
        for i, b in enumerate(batch):
            l = lengths[i]
            out[i, :l] = b[k]
        padded_batch[k] = out

    # 3. 3D features [B, L, N, M]
    for k in keys_3d:
        if k not in item0:
            continue
        dims = item0[k].shape[1:]  # e.g. (3,3) or (10,2)
        out = torch.zeros(B, max_len, *dims, dtype=item0[k].dtype)
        for i, b in enumerate(batch):
            l = lengths[i]
            out[i, :l] = b[k]
        padded_batch[k] = out

    # 4. Pair features [B, L, L, D]
    for k in keys_pair:
        if k not in item0:
            continue
        dim = item0[k].shape[-1]
        out = torch.zeros(B, max_len, max_len, dim, dtype=item0[k].dtype)
        for i, b in enumerate(batch):
            l = lengths[i]
            out[i, :l, :l] = b[k]
        padded_batch[k] = out

    # Fill is_na_residue_mask (assuming all are NA for this dataset)
    # The dataset returns `res_mask` which is 1s for present residues.
    # We can just copy that.
    if "res_mask" in padded_batch:
        padded_batch["is_na_residue_mask"] = padded_batch["res_mask"].clone()

    return padded_batch


class RNAClusterDataModule(LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_cfg = data_cfg

        # Parse data directory from config or default
        self.data_dir = getattr(data_cfg, "data_dir", "data_ensemble/rna_ensemble_data")
        self.batch_size = data_cfg.batch_size
        self.num_workers = data_cfg.num_workers

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Instantiate the full dataset
        full_dataset = RNAClusterDataset(
            data_dir=self.data_dir,
            split="train",  # We'll split manually
            max_length=self.data_cfg.get("max_len", None),  # Optional filtering
        )

        # Simple random split for now (e.g., 90/10)
        # In production, use strict validation sets from metadata
        total_size = len(full_dataset)
        val_size = int(total_size * 0.1)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=length_batching_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,  # Validation can be same batch size
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=length_batching_collate,
            pin_memory=True,
        )
