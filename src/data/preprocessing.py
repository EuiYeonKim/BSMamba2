import typing as tp

import torch


class SAD:
    """
    SAD(Source Activity Detector)
    """

    def __init__(
            self,
            sr: int,
            window_size_in_sec: int = 6,
            overlap_ratio: float = 0.5,
            n_chunks_per_segment: int = 10,
            eps: float = 1e-5,
            gamma: float = 1e-3,
            threshold_max_quantile: float = 0.15,
            threshold_segment: float = 0.5,
    ):
        self.sr = sr
        self.n_chunks_per_segment = n_chunks_per_segment
        self.eps = eps
        self.gamma = gamma
        self.threshold_max_quantile = threshold_max_quantile
        self.threshold_segment = threshold_segment

        self.window_size = sr * window_size_in_sec
        self.step_size = int(self.window_size * overlap_ratio)

    def chunk(self, y: torch.Tensor):
        """
        Input shape: [n_channels, n_frames]
        Output shape: []
        """
        # tensor를 window와 step많큼 sliding window방식대로 자른 후 window를 chunk로 또 분할함
        y = y.unfold(-1, self.window_size, self.step_size)
        # 위에서 window를 n_chunks_per_segment개로 나눔 -> 나누어진 것이 tuple로 return됨
        y = y.chunk(self.n_chunks_per_segment, dim=-1)
        
        # chunk된 것들을 다시 쌓아 tensor로 만듦
        y = torch.stack(y, dim=-2)

        return y

    @staticmethod
    def calculate_rms(y: torch.Tensor):
        """
        걍 rms 계산하는거
        """
        y = torch.mean(torch.square(y), dim=-1, keepdim=True)
        y = torch.sqrt(y)
        return y

    def calculate_thresholds(self, rms: torch.Tensor):
        """
        chunk가 threshold보다 큰 비율이 50% 이상일 경우만 사용하기 위한 mask를 구함
        여기서 threshold는 threshold_max_quantile임 (0.15 분위), 즉 chunk들의 15% 구간을 말함
        이 threshold보다 작은 부분이 50% 이상이라면 상대적으로 작은 음성이 50% 이상이라는 의미임
        """
        # rms에는 각 chunk별 rms값이 들어있음, 이게 -50dB값이어야하는데 맞으려나
        rms[rms == 0.] = self.eps
        # 여기서 dim=-2에 대해, 즉 chunk의 갯수의 차원에 대해 0.15 분위의 값을 가져옴
        rms_threshold = torch.quantile(
            rms,
            self.threshold_max_quantile,
            dim=-2,
            keepdim=True,
        )

        # threshold값이 gamma보다 작으면 gamma로 채움
        rms_threshold[rms_threshold < self.gamma] = self.gamma
        # rms가 threshold보다 큰 비율을 구함
        rms_percentage = torch.mean(
            (rms > rms_threshold).float(),
            dim=-2,
            keepdim=True,
        )
        # 이 비율이 0.5 이상이 안되는 부분을 mask하는 mask를 구함, True, False값이 들어있어서 나중에 곱할듯
        rms_mask = torch.all(rms_percentage > self.threshold_segment, dim=0).squeeze()

        return rms_mask

    def calculate_salient(self, y: torch.Tensor, mask: torch.Tensor):
        """
        """
        y = y[:, mask, ...]
        C, D1, D2, D3 = y.shape
        y = y.view(C, D1, D2*D3)
        return y

    def __call__(
            self,
            y: torch.Tensor,
            segment_saliency_mask: tp.Optional[torch.Tensor] = None
    ):
        """
        Stacks signal into segments and filters out silent segments.
        :param y: Input signal.
            Shape [n_channels, n_frames]
               segment_saliency_mask: Optional precomputed mask
            Shape [n_channels, n_segments, 1, 1]
        :return: Salient signal folded into segments of length 'self.window_size' and step 'self.step_size'.
            Shape [n_channels, n_segments, frames_in_segment]
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        if segment_saliency_mask is None:
            segment_saliency_mask = self.calculate_thresholds(rms)
        y_salient = self.calculate_salient(y, segment_saliency_mask)
        return y_salient, segment_saliency_mask

    def calculate_salient_indices(
            self,
            y: torch.Tensor
    ):
        """
        Returns start indices of salient regions of audio
        """
        y = self.chunk(y)
        rms = self.calculate_rms(y)
        mask = self.calculate_thresholds(rms)
        # window들의 index들을 구한 후 mask를 통한 filtering, 이걸 step size와 곱하면 실제 음성 frame의 index가됨
        indices = torch.arange(mask.shape[-1])[mask] * self.step_size
        return indices.tolist()


if __name__ == "__main__":
    import torchaudio

    sr = 44100
    example_path = 'example/example.mp3'

    sad = SAD(sr=sr)
    y, sr = torchaudio.load(example_path)
    y_salience = sad(y)[0]
    print(f"Initial shape: {y.shape}.\nShape after source activity detection: {y_salience.shape}")
