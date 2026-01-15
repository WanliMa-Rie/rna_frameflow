import hashlib
from typing import Any, Dict, List, Optional
import pathlib
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from rna_backbone_design.data import parsing
from rna_backbone_design.data import data_transforms
from rna_backbone_design.data import utils as du
from rna_backbone_design.data import nucleotide_constants as nc
from rna_backbone_design.data.rigid_utils import Rigid


class RNAClusterDataset(Dataset):
    """
    Dataset that iterates over clusters of RNA structures.
    For each cluster, it randomly samples one PDB structure (conformer)
    and pairs it with the cluster's shared embeddings.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",  # "train", "val", "test"
        max_length: Optional[int] = None,
        overfit: bool = False,
    ):
        """
        Args:
            data_dir: Path to the root directory containing cluster folders.
                      (e.g., 'data_ensemble/rna_ensemble_data')
            split: Data split to use (train, val, or test).
            max_length: Optional maximum sequence length to filter or crop.
            overfit: If True, restricts the dataset to a small subset for debugging.
        """
        self.data_dir = pathlib.Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.overfit = overfit

        # Gather all cluster directories
        all_clusters = sorted(
            [
                d
                for d in self.data_dir.iterdir()
                if d.is_dir() and d.name.startswith("cluster_")
            ]
        )

        # Stable hash-based split
        self.clusters = []
        for cluster_dir in all_clusters:
            # Use MD5 hash of the cluster name for stable splitting
            cluster_name = cluster_dir.name
            hash_val = int(hashlib.md5(cluster_name.encode()).hexdigest(), 16)
            percent = hash_val % 100

            if split == "train":
                if percent < 80:
                    self.clusters.append(cluster_dir)
            elif split == "val":
                if 80 <= percent < 90:
                    self.clusters.append(cluster_dir)
            elif split == "test":
                if percent >= 90:
                    self.clusters.append(cluster_dir)
            else:
                raise ValueError(f"Invalid split: {split}")

        if self.overfit:
            self.clusters = self.clusters[:2]

        print(
            f"RNAClusterDataset ({self.split}): Found {len(self.clusters)} clusters (out of {len(all_clusters)}) in {self.data_dir}"
        )

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_idx = idx % len(self.clusters)
        cluster_dir = None
        pkl_path = None
        raw_feats = None

        for offset in range(len(self.clusters)):
            candidate_cluster_dir = self.clusters[(start_idx + offset) % len(self.clusters)]
            feature_dir = candidate_cluster_dir / "features"
            pkl_files = list(feature_dir.glob("*.pkl"))
            if not pkl_files:
                continue

            candidate_pkl_path = random.choice(pkl_files)
            candidate_raw_feats = du.read_pkl(str(candidate_pkl_path))

            embedding_dir = candidate_cluster_dir / "embedding"
            single_files = sorted(list(embedding_dir.glob("*_single.npy")))
            pair_files = sorted(list(embedding_dir.glob("*_pair.npy")))
            if not single_files or not pair_files:
                continue

            cluster_dir = candidate_cluster_dir
            pkl_path = candidate_pkl_path
            raw_feats = candidate_raw_feats
            break

        if raw_feats is None or pkl_path is None or cluster_dir is None:
            raise RuntimeError(
                f"Unable to find valid (features + embedding) sample for split={self.split} under {self.data_dir}"
            )

        # Re-compute Geometry (Frames & Torsions) on-the-fly to match training pipeline
        # -----------------------------------------------------------------------------
        # 1. Convert to tensor
        aatype_global = torch.from_numpy(raw_feats["aatype"]).long()
        atom_positions = torch.from_numpy(raw_feats["atom_positions"]).double()
        atom_mask = torch.from_numpy(raw_feats["atom_mask"]).double()
        atom_deoxy = torch.from_numpy(raw_feats["atom_deoxy"]).bool()

        # 2. Slice to NA atoms (23)
        # We assume the dataset contains RNA residues.
        NUM_NA_RESIDUE_ATOMS = 23
        tensor_feats = {
            "aatype": aatype_global,
            "all_atom_positions": atom_positions[:, :NUM_NA_RESIDUE_ATOMS],
            "all_atom_mask": atom_mask[:, :NUM_NA_RESIDUE_ATOMS],
            "atom_deoxy": atom_deoxy,
        }

        # Cache atom23 positions
        tensor_feats["atom23_gt_positions"] = tensor_feats["all_atom_positions"]

        # 3. Apply data transforms
        tensor_feats = data_transforms.make_atom23_masks(tensor_feats)
        data_transforms.atom23_list_to_atom27_list(
            tensor_feats, ["all_atom_positions", "all_atom_mask"], inplace=True
        )
        tensor_feats = data_transforms.atom27_to_frames(tensor_feats)
        tensor_feats = data_transforms.atom27_to_torsion_angles()(tensor_feats)

        # Extract calculated features
        # rigidgroups_gt_frames: [L, 11, 4, 4]
        gt_frames_tensor = tensor_feats["rigidgroups_gt_frames"]
        # torsion_angles: [L, 10, 2]
        gt_torsions_sin_cos = tensor_feats["torsion_angles_sin_cos"]
        gt_torsions_mask = tensor_feats["torsion_angles_mask"]

        # -----------------------------------------------------------------------------

        atom23_mask = tensor_feats["all_atom_mask"]
        res_mask_base = (atom23_mask.sum(dim=-1) > 0).to(torch.float32)

        # Extract basic info for length alignment
        seq_len = aatype_global.shape[0]

        embedding_dir = cluster_dir / "embedding"
        single_files = sorted(list(embedding_dir.glob("*_single.npy")))
        pair_files = sorted(list(embedding_dir.glob("*_pair.npy")))
        single_embedding = torch.from_numpy(np.load(single_files[0])).float()
        pair_embedding = torch.from_numpy(np.load(pair_files[0])).float()

        # 4. Align Lengths (Features vs Embeddings)
        emb_len = single_embedding.shape[0]
        min_len = min(seq_len, emb_len)

        # Helper to slice all relevant keys
        def slice_tensor(t, length):
            return t[:length]

        # Load and Slice Features
        # We assume these exist because we put them there in preprocessing
        aatype = slice_tensor(aatype_global, min_len)
        res_mask = slice_tensor(res_mask_base, min_len)

        # rigidgroups_gt_frames: [L, 11, 4, 4] -> we slice to 1 for backbone frame if needed,
        # but original code used 'frames_all = Rigid.from_tensor_4x4(rigidgroups_gt_frames)'
        # so let's keep it as is.
        rigidgroups_gt_frames = slice_tensor(gt_frames_tensor, min_len)

        # torsion_angles: [L, 10, 2] -> we will slice to 8 later
        torsion_angles_sin_cos = slice_tensor(gt_torsions_sin_cos, min_len)
        torsion_angles_mask = slice_tensor(gt_torsions_mask, min_len)

        single_embedding = single_embedding[:min_len]
        pair_embedding = pair_embedding[:min_len, :min_len]

        # 5. Cropping
        if self.max_length is not None and min_len > self.max_length:
            if self.split == "train":
                start_idx = random.randint(0, min_len - self.max_length)
            else:
                start_idx = 0
            end_idx = start_idx + self.max_length

            aatype = aatype[start_idx:end_idx]
            res_mask = res_mask[start_idx:end_idx]
            rigidgroups_gt_frames = rigidgroups_gt_frames[start_idx:end_idx]
            torsion_angles_sin_cos = torsion_angles_sin_cos[start_idx:end_idx]
            torsion_angles_mask = torsion_angles_mask[start_idx:end_idx]
            single_embedding = single_embedding[start_idx:end_idx]
            pair_embedding = pair_embedding[start_idx:end_idx, start_idx:end_idx]

            min_len = self.max_length

        # 6. Extract SE(3) Targets from Precomputed Frames
        # rigidgroups_gt_frames is [L, 1, 4, 4] (Flat Rigids) or just [L, 1, 4, 4] for just backbone?
        # In preprocessing we used data_transforms.atom27_to_frames which produces [L, 1, 4, 4] for 'rigidgroups_gt_frames'
        # corresponding to the backbone frame.
        frames_all = Rigid.from_tensor_4x4(rigidgroups_gt_frames)
        frame_0 = frames_all[:, 0]  # [L]
        trans_1 = frame_0.get_trans()
        rotmats_1 = frame_0.get_rots().get_rot_mats()

        is_na_residue_mask = res_mask > 0.5

        return {
            "aatype": aatype,  # [L]
            "trans_1": trans_1.float(),  # [L, 3]
            "rotmats_1": rotmats_1.float(),  # [L, 3, 3]
            "torsion_angles_sin_cos": torsion_angles_sin_cos[
                :, :8
            ].float(),  # [L, 8, 2]
            "torsion_angles_mask": torsion_angles_mask[:, :8].float(),  # [L, 8]
            "res_mask": res_mask,
            "is_na_residue_mask": is_na_residue_mask,
            "single_embedding": single_embedding,
            "pair_embedding": pair_embedding,
            "pdb_name": pkl_path.stem,
        }
