import torch
import typing as tp


def collate_fn(
        batch: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """
    Collates list of mix and target torch.Tensors to torch.Tensor.
    Channels are collapsed to batch dimension.
    Tensor's Input shape: Batch of [n_channels, n_timestamps]
    Tensor's Output shape: [batch_size, n_sources, n_channels, n_timestamps]
    """
    batch_mix = []
    batch_tgt = []
    for mix, tgt in batch:
        batch_mix.append(mix)
        batch_tgt.append(tgt)
    # tgt all때문에 변경
    batch_mix = torch.stack(batch_mix, dim=1)
    batch_tgt = torch.stack(batch_tgt, dim=1)
    # print(batch_mix.shape)

    return torch.cat((batch_mix, batch_tgt), dim=0)