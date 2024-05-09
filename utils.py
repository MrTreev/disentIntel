import numpy as np
import torch
import torchvision
from scipy import signal
global plslog
plslog = False

np.seterr(divide="ignore", invalid="ignore")


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode="reflect")

    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    fft_window = signal.get_window("hann", fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T

    return np.abs(result)


def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    global plslog
    # f0 is logf0
    if plslog:
        print(f"prenormalise min: {f0.min()}")
        print(f"prenormalise max: {f0.max()}")
        print(f"prenormalise ave: {f0.mean()}")
    f0 = f0.astype(float).copy()
    # index_nonzero = f0 != 0
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    if plslog:
        print(f"normalised min: {f0.min()}")
        print(f"normalised max: {f0.max()}")
        print(f"normalised ave: {f0.mean()}")
    return f0


def quantize_f0_numpy(x, num_bins=256):
    global plslog
    # x is logf0
    if plslog:
        print(f"prequantise min: {x.min()}")
        print(f"prequantise max: {x.max()}")
        print(f"prequantise ave: {x.mean()}")
    assert x.ndim == 1
    x = x.astype(float).copy()
    uv = x <= 0
    x[uv] = 0.0
    if plslog:
        print(f"quantise 1 min: {x.min()}")
        print(f"quantise 1 max: {x.max()}")
        print(f"quantise 1 ave: {x.mean()}")
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins + 1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)


def quantize_f0_torch(x, num_bins=256):
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = x <= 0
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins - 1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins + 1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins + 1), x.view(B, -1).long()


def get_mask_from_lengths(lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()
    return mask


def pad_f0(f0, len_out=672):
    length = f0.shape[0]
    if length <= len_out:
        len_pad = len_out - length
        return np.pad(f0, (0, len_pad), "constant", constant_values=(0, 0))
    else:
        start = int(np.floor((length - len_out)) / 2)
        return f0[start : start + len_out]


def pad_seq_to_2(x, len_out=1696):
    len_pad = len_out - x.shape[1]
    assert len_pad >= 0
    return np.pad(x, ((0, 0), (0, len_pad), (0, 0)), "constant"), len_pad


def save_tensor(tensor: torch.Tensor, save_path) -> None:
    image = try_image(tensor)
    if image is not None:
        im_min = image.min()
        im_max = image.max()
        norm_image = 1.0 / (im_max - im_min) * image + 1.0 * im_min / (im_min - im_max)
        torchvision.utils.save_image(norm_image, save_path + ".png")
    else:
        torch.save(tensor, save_path + ".pth")
    return


def try_image(tensor: torch.Tensor) -> torch.Tensor | None:
    if tensor is None:
        return None
    elif tensor.dim() == 2:
        return tensor
    elif tensor.dim() < 2:
        return None
    elif tensor.dim() > 2 and tensor.size(0) == 1:
        for in_tensor in tensor:
            return try_image(in_tensor)
    else:
        return None
