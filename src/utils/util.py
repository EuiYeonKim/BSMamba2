import torch
import numpy as np
from model.bs_model import DEFAULT_FREQS_PER_BANDS
from einops import rearrange

def get_subband_feats(freq_feats):
    bands = torch.split(freq_feats, DEFAULT_FREQS_PER_BANDS, dim=-2)
    subband_feats = torch.stack([b.mean(dim=-2) for b in bands], dim=-2)  # (K, T)

    return subband_feats

def compute_cIRM(mix_stft, target_stft):
    X_real = mix_stft.real
    X_imag = mix_stft.imag
    S_real = target_stft.real
    S_imag = target_stft.imag

    denom = X_real**2 + X_imag**2 + 1e-10  # 안정성을 위한 작은 값

    M_real = (S_real * X_real + S_imag * X_imag) / denom
    M_imag = (S_imag * X_real - S_real * X_imag) / denom

    cIRM = M_real + 1j * M_imag

    return cIRM

def get_decibel(waveform):
    return 20 * torch.log10(torch.sqrt(torch.mean(waveform**2)) + 1e-12)

def get_vocal_ratio(mixS, tgtS, instrumentsS):
    # compute cIRM of target and non target
    vocal_cirm = compute_cIRM(mixS, tgtS).mean(-3).abs()
    instr_cirm = compute_cIRM(mixS, instrumentsS).mean(-3).abs()
    cirm_ratio = vocal_cirm / (vocal_cirm + instr_cirm + 1e-12)
    cirm_ratio = get_subband_feats(cirm_ratio)

    return cirm_ratio

def get_embedding_labels(mixS, tgtS, instrumentsS, num_bins):
    cirm_ratio = get_vocal_ratio(mixS, tgtS, instrumentsS).flatten()
    # get label 0~9 (integer) 
    labels = (cirm_ratio * num_bins).floor().clamp_(0, num_bins-1).long()
        
    return labels
    
def get_silent_mask(vocals_mag):
    # Generating a silent embeddings mask
    vocals_mag = get_subband_feats(vocals_mag).flatten()
    silent_mask = vocals_mag > -20
    
    return silent_mask