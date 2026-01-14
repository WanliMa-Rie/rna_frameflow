现在的代码是一个 de novo RNA backbone generation 的问题. 这其实和我们的有些区别. 我想要做的是一个 RNA ensemble generation 问题. 想要通过学习 ensemble dataset, 来学习 ensemble 信息. 因此数据输入和验证的逻辑可能要进行修改.
首先，我们先来关注模型。我先陈述一些事实：
1. 现在的模型是一个基于 SE(3) flow matching 的 generative model。
2. 模型输入是加噪后的结构数据（在 `interpolant.py` 中可以看到，通过 `corrupt_batch` 来对输入的 ground truth batch $x_1$ 进行线性插值加噪，得到 noisy batch $x_t$）。
3. 这个代码是一个 do novo RNA backbone generation. 因此在推理时，其输入只需要时长度信息以及采样的噪声即可。模型在任何时候都不需要 sequence 的信息。并且生成的只是骨架，而不包含任何的核苷酸信息。
基于此，我们要对代码进行如下更改：

### 输入数据
1. 输入数据改为 ensemble dataset. 具体来说，数据位于 `data_ensemble` 文件夹中，格式如下：
```
Dataset utilities for RNA cluster directory format.

This module provides a dataset class for loading RNA structure data organized
in the cluster directory format:

    cluster_XXXX_id/
    ├── input_data.json         # Index of structures: {cluster_id: [[structure_id, score1, score2], ...]}
    ├── msa/
    │   └── representative.a3m  # Representative MSA
    ├── structure/
    │   ├── structure1.pdb      # PDB structure files
    │   └── ...
    └── embedding/
        ├── representative_single.npy   # Single representation (L, 384)
        └── representative_pair.npy     # Pair representation (L, L, 128)
```
2. 之前 ensemble 的数据处理代码为 `data_ensemble/rna_cluster_dataset.py` 和 `data_ensemble/rna_cluster_utils.py`. 现在我想将其放在当前的 context 下。具体的逻辑是，为每一个 RNA 结构，保存如下信息：

```
*   structure_id: 结构的 ID。
*   cluster_id: 结构所属的 cluster ID。
*   cluster_dir: 结构所属的 cluster 目录。
*   structure_dir: 结构所属的 structure 目录。
*   representative_id: 代表性结构的 ID。
*   aatype: 序列信息的整数编码。
*   atom_positions: N_res, N_atoms, 3 所有原子坐标。
*   bb_positions: N_res, 3 骨架中心坐标。
*   res_mask / bb_mask: N_res 掩码，指示哪些残基是有效的。
*   chain_index: 链的索引。
*   atom37_positions: 符合 AlphaFold 标准的原子坐标格式。
*   single_embeds: (L, 384) 的单体表示。
*   pair_embeds: (L, L, 128) 的配对表示。
... (如果有遗漏的你来补充)
```

在一个大的整体的 pkl 中。

3. 在训练中，类似 RNAFrameFlow 通过 data_transforms 来对数据进行转换。但是 feats 多了几个元素。总共包括：

```
trans_1
rotmats_1
torsion_angles_sin_cos
res_mask
is_na_residue_mask
single_embeds
pair_embeds
... (如果有遗漏的你来补充)
```

4. 接入现在的模型 pipeline.


### 模型改进



