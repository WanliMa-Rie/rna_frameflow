import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
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
    keys_1d = ["res_mask", "aatype"]  # [L]
    keys_2d = ["trans_1", "single_embedding", "torsion_angles_mask"]  # [L, D]
    keys_3d = [
        "rotmats_1",
        "torsion_angles_sin_cos",
        "alt_torsion_angles_sin_cos",
    ]  # [L, 3, 3], [L, 7, 2]
    keys_pair = ["pair_embedding"]  # [L, L, D]

    padded_batch = {}

    # Initialize basic lists for keys not in padding logic (like names)
    padded_batch["pdb_name"] = [b["pdb_name"] for b in batch]
    if "cluster_id" in batch[0]:
        padded_batch["cluster_id"] = [b["cluster_id"] for b in batch]
    if "gt_c4_ensemble" in batch[0]:
        padded_batch["gt_c4_ensemble"] = [b.get("gt_c4_ensemble", None) for b in batch]
        padded_batch["gt_ensemble_size"] = [b.get("gt_ensemble_size", None) for b in batch]
    padded_batch["is_na_residue_mask"] = torch.ones(len(batch), max_len, dtype=torch.bool)

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
        self.test_dataset = None

    def setup(self, stage=None):
        return_ensemble = bool(self.data_cfg.get("return_ensemble", False))
        max_ensemble_conformers = self.data_cfg.get("max_ensemble_conformers", None)

        if stage in (None, "fit", "validate"):
            self.train_dataset = RNAClusterDataset(
                data_dir=self.data_dir,
                split="train",
                max_length=self.data_cfg.get("max_len", None),
                return_ensemble=False,
                max_ensemble_conformers=max_ensemble_conformers,
            )
            self.val_dataset = RNAClusterDataset(
                data_dir=self.data_dir,
                split="val",
                max_length=self.data_cfg.get("max_len", None),
                return_ensemble=return_ensemble,
                max_ensemble_conformers=max_ensemble_conformers,
            )
        if stage == "test" or stage is None:
            self.test_dataset = RNAClusterDataset(
                data_dir=self.data_dir,
                split="test",
                max_length=self.data_cfg.get("max_len", None),
                return_ensemble=return_ensemble,
                max_ensemble_conformers=max_ensemble_conformers,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=length_batching_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=length_batching_collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=length_batching_collate,
            pin_memory=True,
        )
