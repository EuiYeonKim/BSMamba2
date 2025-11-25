import torch
import numpy as np

def phase_losses(phase_r, phase_g):

    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=3) - torch.diff(phase_g, dim=3)))

    return ip_loss, gd_loss, iaf_loss

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)
