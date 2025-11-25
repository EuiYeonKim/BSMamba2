import random
import numpy as np
import torch
from tqdm import tqdm
import torchaudio
import typing as tp


class MixingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_dir: str,
        txt_dir,
        instruments,
        num_steps,
        target: str = "vocals",
        is_training: bool = True,
        silent_prob=0.1,
        augmentation=None,
        batch_size=1,
        mix_prop=1.0,
    ):
        if "no_valid" in txt_dir:
            self.mode = "train" if is_training else "test"
        else:
            self.mode = "train" if is_training else "valid"

        # validation일 때는 aligned sample
        self.random_mix = True if is_training else False

        self.file_dir = file_dir
        self.txt_dir = txt_dir
        self.target = target
        self.instruments = instruments
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.mix_prop = mix_prop

        self.silent_prob = silent_prob
        self.augmentation = augmentation

        # 각 악기의 filelist 불러오기
        self.filelists = {
            "vocals": self.get_filelist("vocals"),
            "drums": self.get_filelist("drums"),
            "bass": self.get_filelist("bass"),
            "other": self.get_filelist("other"),
        }

    def get_filelist(self, source) -> tp.List[tp.Tuple[str, tp.Tuple[int, int]]]:
        filelist = []
        path = f"{self.txt_dir}/{source}_{self.mode}.txt"
        for line in tqdm(open(path, "r").readlines()):
            file_name, start_idx, end_idx = line.split("\t")
            filepath = f"{self.file_dir}/{self.mode}/{file_name}/{source}.wav"
            filelist.append((str(filepath), (int(start_idx), int(end_idx))))
        return filelist

    def __len__(self):
        return self.num_steps

    def load_source(self, filelist):
        track_path, indices = random.choice(filelist)
        offset = indices[0]
        num_frames = indices[1] - indices[0]

        # silent augmentation for each stem
        if random.random() < self.silent_prob:
            return torch.zeros(2, num_frames)
        source, sr = torchaudio.load(
            track_path, frame_offset=offset, num_frames=num_frames, channels_first=True
        )

        return source

    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            # source에서 random한 음성 불러오기
            s1 = self.load_source(self.filelists[instr])

            res.append(s1)

        res = torch.stack(res)

        return res

    def augment(self, mix):
        if self.augmentation["loudness"] is not None:
            loud_values = np.random.uniform(
                low=self.augmentation.loudness["loudness_min"],
                high=self.augmentation.loudness["loudness_max"],
                size=(len(mix),),
            )
            loud_values = torch.tensor(loud_values, dtype=torch.float32)

            mix *= loud_values[:, None, None]

        return mix

    def load_aligned_source(self, filelist, tgt_source):
        track_path, indices = random.choice(filelist)
        offset = indices[0]
        num_frames = indices[1] - indices[0]

        res = []
        for instr in self.instruments:

            # source에서 random한 음성 불러오기
            s1_path = track_path.replace(tgt_source, instr)
            s1, _ = torchaudio.load(
                s1_path,
                frame_offset=offset,
                num_frames=num_frames,
                channels_first=True,
            )

            res.append(s1)

        res = torch.stack(res)

        return res

    def __getitem__(self, index):
        # 원래 random mix는 true or false 이지만 0.5 비율로 임시 변경
        # if self.random_mix:
        if self.random_mix and random.random() < self.mix_prop:
            mix = self.load_random_mix()
        else:
            tgt_source = random.choice([instr for instr in self.target])
            mix = self.load_aligned_source(self.filelists[tgt_source], tgt_source)

        # Augmentation
        if self.augmentation is not None:
            mix = self.augment(mix)

        tgt_index = [self.instruments.index(tgt) if tgt in self.instruments else -1 for tgt in self.target]
        tgt = mix[tgt_index]

        return mix.sum(0).unsqueeze(0), tgt
        

if __name__ == "__main__":

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    dataset = MixingDataset(
        file_dir="/data/datasets/MUSDB18HQ",
        txt_dir="/data/BSRNN/BandSplitRNN-PyTorch/filelist/8seconds/mus18hq",
        is_training=True,
        instruments=["vocals", "other", "bass", "drums"],
        target="vocals",
        num_steps=1000,
        augmentations=dotdict(
            {
                "silent_prob": 0.0,
                "loudness_min": 0.5,
                "loudness_max": 1.5,
                "mixup": True,
                "mixup_probs": [0.0],
            }
        ),
    )

    import torch
    from IPython.display import Audio

    mix, tgt = dataset.__getitem__(0)
    Audio(mix.numpy(), rate=44100)
