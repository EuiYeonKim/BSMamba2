import os
import argparse
import typing as tp
from pathlib import Path

import musdb
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from moisesdb.dataset import MoisesDB

from data import SAD
from torch.utils.data import random_split


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input-dir",
    type=str,
    required=True,
    help="Path to directory with musdb18 dataset",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    required=True,
    help="Path to directory where output .txt file is saved",
)
parser.add_argument(
    '--subset',
    type=str,
    required=False,
    default='train',
    help="Train/test subset of dataset to process"
)
parser.add_argument(
    "--sad-cfg-path",
    type=str,
    required=False,
    default="./conf/sad/default.yaml",
    help="Path to Source Activity Detection config file",
)
parser.add_argument(
    "-t",
    "--targets",
    nargs="+",
    required=False,
    default=["vocals", "drums", "bass", "other"],
    help="Target source. SAD will save salient fragments of vocal audio.",
)
args = parser.parse_args()


def prepare_save_line(
    track_name: str, start_indices: torch.Tensor, window_size: int
) -> tp.Iterable[str]:
    """
    Creates string in format TRACK_NAME START_INDEX END_INDEX.
    """
    for i in start_indices:
        save_line = f"{track_name}\t{i}\t{i + window_size}\n"
        yield save_line


def run_program(
    file_path: Path,
    target: str,
    db: musdb.DB,
    sad: SAD,
) -> None:
    """
    Saves track's name and fragments indices to provided .txt file.
    """
    with open(file_path, "w") as wf:
        for track in tqdm(db):
            print(track)
            # get audio data and transform to torch.Tensor
            y = torch.tensor(track.targets[target].audio.T, dtype=torch.float32)

            # find indices of salient segments
            indices = sad.calculate_salient_indices(y)
            # write to file
            for line in prepare_save_line(track.name, indices, sad.window_size):
                wf.write(line)
    return None


def run_program_moises(
    file_path: Path,
    target: str,
    db,
    sad: SAD,
) -> None:
    """
    Saves track's name and fragments indices to provided .txt file.
    """
    with open(file_path, "w") as wf:
        for track in tqdm(db):
            # get audio data and transform to torch.Tensor
            y = torch.tensor(track.stems[target].audio, dtype=torch.float32)
            # find indices of salient segments
            indices = sad.calculate_salient_indices(y)
            # write to file
            for line in prepare_save_line(track.name, indices, sad.window_size):
                wf.write(line)
    return None


def make_musdb_dataset(
    db_dir: str,
    save_dir: str,
    subset: str,
    targets: tp.List[str],
    sad_cfg_path: DictConfig,
):
    # initialize MUSDB parser
    # split = None if subset == 'test' else split

    db = musdb.DB(
        root=db_dir,
        subsets=subset,
        split=None,
        download=False,
        is_wav=True,
    )
    # initialize Source Activity Detector
    sad_cfg = OmegaConf.load(sad_cfg_path)
    sad = SAD(**sad_cfg)

    # initialize directories where to save indices
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for target in targets:
        if subset == "train":
            file_path = save_dir / f"{target}_train.txt"
        elif subset == "valid":
            file_path = save_dir / f"{target}_valid.txt"
        else:
            file_path = save_dir / f"{target}_test.txt"
        # segment data and save indices to .txt file
        run_program(file_path, target, db, sad)


def make_moises_dataset(args):

    db = MoisesDB(data_path="/data/datasets/moisesdb", sample_rate=44100)

    # initialize Source Activity Detector
    sad_cfg = OmegaConf.load(args.sad_cfg_path)
    sad = SAD(**sad_cfg)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for target in args.targets:
        file_path = save_dir / f"{target}_train.txt"
        run_program(file_path, target, db, sad)


def main(args) -> None:
    make_musdb_dataset(
        args.input_dir,
        args.output_dir,
        args.subset,
        args.targets,
        args.sad_cfg_path,
    )


if __name__ == "__main__":
    main(args)
