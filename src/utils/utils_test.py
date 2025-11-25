import numpy as np
import torch
from museval.metrics import bss_eval
import museval
import librosa
from torchaudio.transforms import AmplitudeToDB

import typing as tp

# python evaluate.py -d /data/BandSplitRNN-PyTorch/src/logs/bandsplitrnn/vocals/2024-06-25_10-43


def compute_uSDR(
    y_hat: np.ndarray,
    y_tgt: np.ndarray,
    delta: float = 1e-7,
) -> float:
    """
    Computes SDR metric as in https://arxiv.org/pdf/2108.13559.pdf.
    Taken and slightly rewritten from
    https://github.com/AIcrowd/music-demixing-challenge-starter-kit/blob/master/evaluator/music_demixing.py
    """
    # compute SDR for one song
    num = np.sum(np.square(y_tgt), axis=(1, 2))
    den = np.sum(np.square(y_tgt - y_hat), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def compute_SDRs(y_hat: torch.Tensor, y_tgt: torch.Tensor) -> tp.Tuple[float, float]:
    """
    Computes cSDR and uSDR as defined in paper
    """

    y_hat = y_hat.T.unsqueeze(0).numpy()
    y_tgt = y_tgt.T.unsqueeze(0).numpy()
    # bss_eval way
    cSDR, *_ = bss_eval(y_tgt, y_hat, window=1 * 44100, hop=1 * 44100)
    cSDR = np.nanmedian(cSDR)

    # as in music demixing challenge
    uSDR = compute_uSDR(y_hat, y_tgt)
    return cSDR, uSDR

# MIT License

# Copyright (c) 2024 Roman Solovyev (ZFTurbo)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def bleed_full(
        reference: np.ndarray,
        estimate: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512,
        device: str = 'cpu',
):
    """
    Calculate the 'bleed' and 'fullness' metrics between a reference and an estimated audio signal.

    The 'bleed' metric measures how much the estimated signal bleeds into the reference signal,
    while the 'fullness' metric measures how much the estimated signal retains its distinctiveness
    in relation to the reference signal, both using mel spectrograms and decibel scaling.

    Parameters:
    ----------
    reference : np.ndarray
        The reference audio signal, shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal, shape (channels, time).

    sr : int, optional
        The sample rate of the audio signals. Default is 44100 Hz.

    n_fft : int, optional
        The FFT size used to compute the STFT. Default is 4096.

    hop_length : int, optional
        The hop length for STFT computation. Default is 1024.

    n_mels : int, optional
        The number of mel frequency bins. Default is 512.

    device : str, optional
        The device for computation, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    tuple
        A tuple containing two values:
        - `bleedless` (float): A score indicating how much 'bleeding' the estimated signal has (higher is better).
        - `fullness` (float): A score indicating how 'full' the estimated signal is (higher is better).
    """



    window = torch.hann_window(n_fft).to(device)

    # Compute STFTs with the Hann window
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    diff = S2_db - S1_db

    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    bleedless = 100 * 1 / (average_positive + 1)
    fullness = 100 * 1 / (-average_negative + 1)

    return bleedless.cpu().numpy(), fullness.cpu().numpy()