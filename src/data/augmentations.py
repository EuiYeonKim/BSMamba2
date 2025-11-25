import random

import torch
import torch.nn as nn


class RandomDrop(nn.Module):
    """
    Randomly drop target chunk from fragment.
    """

    def __init__(
            self,
            p: float = 0.1,
            chunk_size_sec: int = 3,
            window_stft: int = 2048,
            hop_stft: int = 512,
            first_chunk: bool = False,
            sr: int = 44100
    ):
        super().__init__()
        self.p = p

        self.chunk_size = chunk_size_sec * sr
        # additional space to match stft hop size
        pad_chunk = window_stft - self.chunk_size % hop_stft
        self.chunk_size = self.chunk_size + pad_chunk
        self.eval_step = 1 * sr + pad_chunk
        self.first_chunk = first_chunk

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape

        if self.training and random.random() < self.p:
            start = random.randrange(0, T - self.chunk_size) if not self.first_chunk else 0
            end = start + self.chunk_size
            y = y[..., start:end]
        if not self.training:
            y = y.unfold(-1, self.chunk_size, self.eval_step)
            n_chunks = y.shape[-2]
            y = y.permute(0, 3, 1, 2, 4).contiguous().view(B * n_chunks, S, C, self.chunk_size)
        return y


class RandomCrop(nn.Module):
    """
    Randomly selects chunk from fragment.
    """

    def __init__(
            self,
            p: float = 1.,
            chunk_size_sec: int = 3,
            window_stft: int = 2048,
            hop_stft: int = 512,
            first_chunk: bool = False,
            sr: int = 44100
    ):
        super().__init__()
        self.p = p

        self.chunk_size = chunk_size_sec * sr
        # additional space to match stft hop size
        pad_chunk = window_stft - self.chunk_size % hop_stft
        self.chunk_size = self.chunk_size + pad_chunk
        self.eval_step = 1 * sr + pad_chunk
        self.first_chunk = first_chunk

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape

        if self.training and random.random() < self.p:
            start = random.randrange(0, T - self.chunk_size) if not self.first_chunk else 0
            end = start + self.chunk_size
            y = y[..., start:end]
        if not self.training:
            y = y.unfold(-1, self.chunk_size, self.eval_step)
            n_chunks = y.shape[-2]
            y = y.permute(0, 3, 1, 2, 4).contiguous().view(B * n_chunks, S, C, self.chunk_size)
        return y


class GainScale(nn.Module):
    """
    Randomly scales the energy of a chunk in some dB range.
    """

    def __init__(
            self,
            p: float = 1.,
            min_db: float = -10.,
            max_db: float = 10.,
    ):
        super().__init__()
        self.p = p
        self.min_db = min_db
        self.max_db = max_db

    @staticmethod
    def db2amp(db):
        return 10 ** (db / 20)

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape
        device = y.device
    
        if self.training and random.random() < self.p:
            db_scales = torch.empty(
                B, S, 1, 1, device=device
            ).uniform_(self.min_db, self.max_db)
            y *= self.db2amp(db_scales)
        return y


class Mix(nn.Module):
    """
    Mixes random target sources into mixtures.
    """

    def __init__(
            self,
            p: float = 0.5,
            min_db: float = 0.,
            max_db: float = 5.,
    ):
        super().__init__()
        self.p = p
        self.min_db = min_db
        self.max_db = max_db

    @staticmethod
    def db2amp(db):
        return 10 ** (db / 20)

    @staticmethod
    def calc_rms(y: torch.Tensor, keepdim=True) -> torch.Tensor:
        """
        Calculate Power of audio signal.
        """
        return torch.sqrt(
            torch.mean(torch.square(y), dim=-1, keepdim=keepdim)
        )

    def rms_normalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Power-normalize an audio signal.
        """
        rms = self.calc_rms(y, keepdim=True)
        return y / (rms + 1e-8)

    def forward(
            self, y: torch.Tensor
    ) -> torch.Tensor:
        B, S, C, T = y.shape
        device = y.device

        if self.training and random.random() < self.p:
            indices_background = torch.randint(
                0, B, (B,),
            )
            db_scales = torch.empty(
                # 원래 코드
                B, 1, 1, 1, device=device
                # B, S, 1, 1, device=device
            ).uniform_(self.min_db, self.max_db)
            y_targets_only = y[indices_background, 1].unsqueeze(1).repeat_interleave(2, dim=1)
            y_background = self.rms_normalize(y_targets_only)
            rms_background = self.calc_rms(y) / self.db2amp(db_scales)
            y += y_background * rms_background
        return y


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """
    def __init__(self, proba=1, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav):
        B, S, C, T = wav.size()
        device = wav.device

        if self.training and random.random() < self.proba:
            group_size = self.group_size or B
            if B % group_size != 0:
                raise ValueError(f"B size {B} must be divisible by group size {group_size}")
            groups = B // group_size
            wav = wav.view(groups, group_size, S, C, T)
            permutations = torch.argsort(torch.rand(groups, group_size, S, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, C, T))
            wav = wav.view(B, S, C, T)
        # print(wav)
        return wav
    

class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, stems, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = torch.empty(batch, stems, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav
    
class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """
    def forward(self, wav):
        print(wav.shape)
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = torch.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            print(left.shape)
            right = 1 - left
            wav = torch.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        print(wav.shape)
        exit()
        return wav