import torch
import numpy as np
import os
import pathlib
import argparse
from rna_backbone_design.data import parsing
from rna_backbone_design.data import utils as du

def calculate_rmsd(pos1, pos2):
    """Calculate RMSD between two sets of positions [N, 3]."""
    return np.sqrt(np.mean(np.sum((pos1 - pos2)**2, axis=-1)))

def run_eval(generated_dir, data_dir):
    """
    generated_dir: directory where inference_se3_flows.py saved samples.
                   Expects files like 'sample_{pdb_name}.pdb'
    data_dir: root directory containing clusters.
    """
    generated_dir = pathlib.Path(generated_dir)
    data_dir = pathlib.Path(data_dir)
    
    # 1. Gather all generated PDBs
    gen_files = list(generated_dir.glob("*.pdb"))
    if not gen_files:
        # Search in subdirs (some scripts use length_L subdirs)
        gen_files = list(generated_dir.rglob("*.pdb"))
    
    results = []
    
    for gen_path in gen_files:
        # Expected name: sample_cluster_XXXX_0.pdb or similar
        # Extract pdb_name
        name = gen_path.stem
        if name.startswith("sample_"):
            pdb_name = name.replace("sample_", "")
        else:
            pdb_name = name
            
        # 2. Locate ground truth
        # pdb_name is like 'cluster_123_0'
        # Cluster dir is 'cluster_123'
        cluster_id = "_".join(pdb_name.split("_")[:2])
        gt_path = data_dir / cluster_id / "structure" / f"{pdb_name}.pdb"
        
        if not gt_path.exists():
            print(f"Ground truth not found for {pdb_name}: {gt_path}")
            continue
            
        # 3. Parse PDBs
        try:
            gen_parsed = parsing.parse_pdb(str(gen_path))
            gt_parsed = parsing.parse_pdb(str(gt_path))
            
            # Use C4' atoms for RMSD (index 3 in compact format)
            # Or use specific atom names if needed
            gen_pos = gen_parsed["atom_positions"]
            gt_pos = gt_parsed["atom_positions"]
            
            # Align lengths if needed (though they should match)
            min_len = min(gen_pos.shape[0], gt_pos.shape[0])
            gen_c4 = gen_pos[:min_len, 3] # C4' is at index 3
            gt_c4 = gt_pos[:min_len, 3]
            
            rmsd = calculate_rmsd(gen_c4, gt_c4)
            results.append({"name": pdb_name, "rmsd": rmsd})
            print(f"{pdb_name}: RMSD = {rmsd:.3f} A")
            
        except Exception as e:
            print(f"Error evaluating {pdb_name}: {e}")

    if results:
        rmsds = [r["rmsd"] for r in results]
        print("\n" + "="*30)
        print(f"Evaluation Summary ({len(results)} samples)")
        print(f"Mean RMSD: {np.mean(rmsds):.3f} A")
        print(f"Median RMSD: {np.median(rmsds):.3f} A")
        print(f"Min RMSD: {np.min(rmsds):.3f} A")
        print(f"Max RMSD: {np.max(rmsds):.3f} A")
        print("="*30)
    else:
        print("No samples successfully evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated PDBs")
    parser.add_argument("--data_dir", type=str, default="data_ensemble/rna_ensemble_data", help="Data root")
    args = parser.parse_args()
    
    run_eval(args.gen_dir, args.data_dir)
