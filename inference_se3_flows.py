"""
Inference script for Conditional RNA-FrameFlow.
Iterates through the test split of the RNAClusterDataset and generates structures.
"""

import os
import time
import numpy as np
import hydra
import torch
import GPUtil
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf

import rna_backbone_design.utils as eu
from rna_backbone_design.models.flow_module import FlowModule
from rna_backbone_design.data.rna_cluster_datamodule import RNAClusterDataModule

torch.set_float32_matmul_precision("high")
log = eu.get_pylogger(__name__)


class Sampler:
    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        if os.path.isdir(ckpt_path):
            ckpts = [
                os.path.join(ckpt_path, f)
                for f in os.listdir(ckpt_path)
                if f.endswith(".ckpt")
            ]
            if len(ckpts) != 1:
                raise ValueError(
                    f"inference.ckpt_path is a directory but contains {len(ckpts)} .ckpt files: {ckpt_path}"
                )
            ckpt_path = ckpts[0]
        ckpt_dir = os.path.dirname(ckpt_path)
        # Attempt to load config from checkpoint dir
        config_path = os.path.join(ckpt_dir, "config.yaml")
        if not os.path.exists(config_path):
            # Fallback to flashipa named one if exists
            config_path = os.path.join(ckpt_dir, "config_flashipa.yaml")
        
        if os.path.exists(config_path):
            ckpt_cfg = OmegaConf.load(config_path)
            # Set-up config.
            OmegaConf.set_struct(cfg, False)
            OmegaConf.set_struct(ckpt_cfg, False)
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
        
        cfg.experiment.checkpointer.dirpath = "./"

        self._cfg = cfg
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up directories to write results to
        self._output_dir = os.path.join(
            self._infer_cfg.output_dir,
            self._infer_cfg.name,
        )
        os.makedirs(self._output_dir, exist_ok=True)
        log.info(f"Saving results to {self._output_dir}")
        
        # Save merged config for reproducibility
        with open(os.path.join(self._output_dir, "inference_config.yaml"), "w") as f:
            OmegaConf.save(config=self._cfg, f=f)

        # Read checkpoint and initialize module.
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path, cfg=cfg
        )

        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg
        self._flow_module._output_dir = self._output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(order="memory", limit=8)[
            : self._infer_cfg.num_gpus
        ]
        if not devices:
            devices = [0]
        log.info(f"Using devices: {devices}")

        # Use DataModule to get test dataloader
        datamodule = RNAClusterDataModule(self._cfg.data_cfg)
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()

        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            strategy="ddp" if len(devices) > 1 else "auto",
            devices=devices if torch.cuda.is_available() else "auto",
        )

        start_time = time.time()
        trainer.predict(self._flow_module, dataloaders=dataloader)
        elapsed_time = time.time() - start_time
        log.info(f"Finished in {elapsed_time:.2f}s")
        log.info(
            f"Generated samples are stored here: {self._output_dir}"
        )


@hydra.main(
    version_base=None, config_path="configs", config_name="inference"
)
def run(cfg: DictConfig) -> None:
    if cfg.inference.run_inference:
        log.info("Starting inference")
        sampler = Sampler(cfg)
        sampler.run_sampling()


if __name__ == "__main__":
    run()
