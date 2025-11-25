import typing as tp

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer, lr_scheduler
from omegaconf import DictConfig
from torch_ema import ExponentialMovingAverage
import torchaudio
from utils.losses import *
from utils.util import *
from einops import rearrange, pack, unpack


class PLModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        featurizer: nn.Module,
        inverse_featurizer: nn.Module,
        multi_featurizer: list,
        augmentations: nn.Module,
        opt: Optimizer,
        sch,
        hparams: DictConfig = None,
    ):
        super().__init__()

        # augmentations
        self.augmentations = augmentations

        # featurizers
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.multi_featurizer = multi_featurizer

        # loss type
        self.complex_error = hparams.complex_error
        self.phase_loss = hparams.phase_loss
        self.mag_loss = hparams.mag_loss
        self.time_loss = hparams.time_loss
        self.multi_stft_loss = hparams.multi_stft_loss
        self.consistency = hparams.consistency

        # model
        self.model = model

        # losses
        self.mae_specR = nn.L1Loss()
        self.mae_specI = nn.L1Loss()
        self.mae_time = nn.L1Loss()
        self.mae_mag = nn.L1Loss()
        self.mae_con_specR = nn.L1Loss()
        self.mae_con_specI = nn.L1Loss()

        # opts
        self.opt = opt
        self.sch = sch

        # logging
        self.save_hyperparameters(hparams)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """

        loss, loss_dict = self.step(batch, is_compute_usdr=False)

        # logging
        for k in loss_dict:
            self.log(f"train/{k}", loss_dict[k].detach(), on_epoch=True, on_step=False, sync_dist=True)
        self.log("train/loss", loss.detach(), on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:

        loss, loss_dict, usdr = self.step(batch, is_compute_usdr=True)

        # logging
        for k in loss_dict:
            self.log(f"val/{k}", loss_dict[k], sync_dist=True, add_dataloader_idx=False)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        self.log("val/usdr", usdr, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

        return loss
        
    def step(
        self, batchT: torch.Tensor, is_compute_usdr=False, is_return_output=False
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor], torch.Tensor]:
        """
        Input shape: [batch_size, n_sources, n_channels, time]
        """
        # augmentations
        # batchT = self.augmentations(batchT)
        if not self.model.stereo:
            batchT = batchT.mean(dim=2, keepdim=True)

        # STFT
        batchS = self.featurizer(batchT)
        mixT, tgtT = batchT
        mixS, tgtS = batchS[0], batchS[1:]

        tgtS = rearrange(tgtS, 'n b c f t -> b n c f t')

        predS = self.model(mixS)
        
        if is_return_output:
            return predS

        # iSTFT
        stemT = self.inverse_featurizer(torch.stack((predS, tgtS), dim=1))
        predT, tgtT = stemT[:, 0], stemT[:, 1]


        # compute loss
        loss, loss_dict = self.compute_losses(predS, tgtS, predT, tgtT, mixT, mixS)

        if is_compute_usdr:
            # compute metrics
            usdr = self.compute_usdr(predT, tgtT)
            return loss, loss_dict, usdr

        return loss, loss_dict

    def compute_losses(
        self,
        predS: torch.Tensor,
        tgtS: torch.Tensor,
        predT: torch.Tensor,
        tgtT: torch.Tensor,
        mixT: torch.Tensor,
        mixS: torch.Tensor,
    ) -> tp.Tuple[torch.Tensor, tp.Dict[str, torch.Tensor]]:

        loss = 0.0
        loss_dict = {}
        

        # Consistency Loss
        if self.consistency:
            conS = self.featurizer(predT)
            lossConR = self.mae_con_specR(conS.real, tgtS.real)
            lossConI = self.mae_con_specI(conS.imag, tgtS.imag)
            loss_dict["lossConR"] = lossConR
            loss_dict["lossConI"] = lossConI

            loss = loss + lossConI + lossConR

        # frequency domain
        if self.complex_error:
            lossR = self.mae_specR(predS.real, tgtS.real)
            lossI = self.mae_specI(predS.imag, tgtS.imag)
            loss_dict["lossR"] = lossR
            loss_dict["lossI"] = lossI

            loss = loss + lossI + lossR
        # phase error
        if self.phase_loss:
            ip_loss, gd_loss, iaf_loss = phase_losses(tgtS.angle(), predS.angle())
            phase_loss = (ip_loss + gd_loss + iaf_loss) * 0.05
            loss_dict["ip_loss"] = ip_loss
            loss_dict["gd_loss"] = gd_loss
            loss_dict["iaf_loss"] = iaf_loss
            loss_dict["phase_loss"] = phase_loss

            loss = loss + phase_loss

        # Magnitude Error
        if self.mag_loss:
            mag_loss = self.mae_mag(predS.abs(), tgtS.abs())
            loss_dict["mag_loss"] = mag_loss

            loss = loss + mag_loss

        # time domain
        if self.time_loss:
            lossT = self.mae_time(predT, tgtT) * 10
            loss_dict["lossTime"] = lossT
            loss = loss + lossT

        if self.multi_stft_loss:
            # multi frequency loss
            lossMultiSpec = 0.0
            for featurizer in self.multi_featurizer:
                reconMS = featurizer(predT)
                targetMS = featurizer(tgtT)

                lossMultiSpec = lossMultiSpec + self.mae_specR(reconMS, targetMS)
            loss_dict["lossMultiSpec"] = lossMultiSpec

            loss = loss + lossMultiSpec

        return loss, loss_dict


    @staticmethod
    def compute_usdr(
        predT: torch.Tensor, tgtT: torch.Tensor, delta: float = 1e-7
    ) -> torch.Tensor:
        
        num = torch.sum(torch.square(tgtT), dim=(-1, -2))
        den = torch.sum(torch.square(tgtT - predT), dim=(-1, -2))
        num += delta
        den += delta
        usdr = 10 * torch.log10(num / den)

        return usdr.mean()

    def on_before_optimizer_step(self, *args, **kwargs):
        norms = pl.utilities.grad_norm(self, norm_type=2)
        norms = dict(filter(lambda elem: "_total" in elem[0], norms.items()))
        self.log_dict(norms)

    def configure_optimizers(self):
        # return {"optimizer": self.opt, "lr_scheduler": self.sch, "monitor": "val/loss"}
        return {"optimizer": self.opt, "monitor": "val/loss"}
    
    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
    #     # peak memory check
    #     peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB 단위
    #     print(peak)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict
