import argparse
import logging
import typing as tp
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf

from separator import Separator
from data import EvalSourceSeparationDataset
from utils.utils_inference import load_pl_state_dict
from utils.utils_test import compute_SDRs, bleed_full
from torch.cuda.amp import autocast


class EvaluateProgram:
    CFG_PATH = "{}/tb_logs/hparams.yaml"
    CKPT_DIR = "{}/weights"

    def __init__(
        self,
        run_dir: str,
        data_path,
        ckpt_path,
        duration = None,
        device: str = "cuda",
    ):
        
        # paths
        self.cfg_path = Path(self.CFG_PATH.format(run_dir))
        self.ckpt_dir = Path(self.CKPT_DIR.format(run_dir))
        self.ckpt_path = ckpt_path

        # config params
        self.cfg = OmegaConf.load(self.cfg_path)
        logger.info(f"Used model: {self.cfg_path}")

        if data_path:
            self.cfg.test_dataset.in_fp = data_path
        
        if duration:
            self.cfg.test_dataset.win_size = duration
            self.cfg.test_dataset.hop_size = duration

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        self.target = self.cfg.train_dataset.target


        logger.info("Initializing the dataset...")
        self.dataset = EvalSourceSeparationDataset(mode="test", instruments=self.cfg.train_dataset.instruments, **self.cfg.test_dataset)
        logger.info("Initializing the separator...")
        self.cfg["audio_params"] = self.cfg.test_dataset
        self.sep = Separator(self.cfg, None)
        _ = self.sep.eval()
        _ = self.sep.to(self.device)

    def run_one_ckpt(self) -> tp.Dict[str, np.ndarray]:
        metrics = {instr: defaultdict(list) for instr in self.target}

        for y, y_tgt in self.dataset:
            # send to device
            y = y.to(self.device)

            # run inference on mixture
            with autocast():
                y_hat = self.sep(y).cpu()

            for i, instr in enumerate(self.target):
                cSDR, uSDR = compute_SDRs(y_hat[i], y_tgt[i])
                bleedless, fullness = bleed_full(y_hat[i], y_tgt[i], n_fft=self.cfg.featurizer.direct_transform.n_fft, hop_length=self.cfg.featurizer.direct_transform.hop_length)

                metrics[instr]["cSDR"].append(cSDR)
                metrics[instr]["uSDR"].append(uSDR)
                metrics[instr]["bleedless"].append(bleedless)
                metrics[instr]["fullness"].append(fullness)
            
        # compute and save metrics
        for i, instr in enumerate(self.target):
            metrics[instr]["cSDR"] = np.array(metrics[instr]["cSDR"])
            metrics[instr]["uSDR"] = np.array(metrics[instr]["uSDR"])
            metrics[instr]["bleedless"] = np.array(metrics[instr]["bleedless"])
            metrics[instr]["fullness"] = np.array(metrics[instr]["fullness"])

        return metrics

    def run(self) -> None:
        # iterate over checkpoints
        if self.ckpt_path:
            ckpt_path_list = [Path(self.ckpt_path)]
        else:
            ckpt_path_list = self.ckpt_dir.glob("*.ckpt")

        for ckpt_path in ckpt_path_list:
            logger.info(f"Evaluating checkpoint - {ckpt_path.name}")
            state_dict = load_pl_state_dict(ckpt_path, device=self.device)
            _ = self.sep.model.load_state_dict(state_dict, strict=True)
            metrics = self.run_one_ckpt()

            for instr in metrics:
                logger.info(instr)
                for m in metrics[instr]:
                    logger.info(
                        f"Metric - {m}, mean - {metrics[instr][m].mean()}, std - {metrics[instr][m].std()}, median - {np.median(metrics[instr][m])}"
                    )
        return None


def main(args):
    logger.info("Starting evaluation...")
    args = vars(args)
    program = EvaluateProgram(**args)
    logger.info("Starting evaluation run...")
    program.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--run-dir",
        type=str,
        required=True,
        help="Path to directory checkpoints, configs, etc",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device name - either 'cuda', or 'cpu'.",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=False,
        default=None,
        help="dataset path if differ testset path in config",
    )

    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=False,
        default=None,
        help="checkpoint path if you want centain checkpoint",
    )

    parser.add_argument(
        "--duration",
        type=int,
        required=False,
        default=None,
        help="inference duration",
    )


    args = parser.parse_args()
    log_name = 'test.log' if not args.data_path else f"{args.data_path.split('/')[-1]}_test.log"

    if args.duration:
        log_name = log_name.replace('.log', f'_{args.duration}s.log')

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=f"{args.run_dir}/{log_name}",
        filemode="w",
    )
    main(args)
