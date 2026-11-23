"""Supplied FFT, audio loading, and scoring utilities."""
import argparse
import json
from pathlib import Path
from tempfile import mkdtemp
import webbrowser

import numpy as np
from scipy.io import wavfile


TRAINING_ERROR_CUTOFF = 0.82


def fft_size(sample_count):
    """Return a power of two that can contain all sample_count samples."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    return 1 << (sample_count - 1).bit_length()


def forward(samples, fft_length):
    """Zero-pad samples and change coordinates."""
    if fft_length < len(samples):
        raise ValueError("FFT grid is too short for these samples")
    return np.fft.rfft(samples, n=fft_length)


def inverse_fft(coefficients, fft_length, output_length):
    """Return the requested prefix of the inverse FFT."""
    if not 0 < output_length <= fft_length:
        raise ValueError("Invalid output length")
    return np.fft.irfft(coefficients, n=fft_length)[:output_length]


def checked_spectrum(values, shape, function_name):
    spectrum = np.asarray(values)
    if spectrum.shape != shape or not np.isfinite(spectrum).all():
        raise ValueError(f"{function_name} must return a finite vector of shape {shape}")
    return spectrum


def training_spectra(clean, distorted):
    """Return X, Y, the common FFT length, and the response length."""
    if len(clean) == 0 or len(clean) != len(distorted):
        raise ValueError("Expected equal nonzero numbers of clean and distorted clips")
    lengths = {len(y) - len(x) + 1 for x, y in zip(clean, distorted)}
    if len(lengths) != 1 or min(lengths) < 1:
        raise ValueError("The training pairs must share a full-convolution response length")
    response_length, = lengths
    size = fft_size(max(map(len, distorted)))
    shape = (size // 2 + 1, len(clean))
    X = np.empty(shape, dtype=complex)
    Y = np.empty(shape, dtype=complex)
    for j, (x, y) in enumerate(zip(clean, distorted)):
        X[:, j] = forward(x, size)
        Y[:, j] = forward(y, size)
    return X, Y, size, response_length


def fit_kernel(clean, distorted, estimate_DH):
    """Convert the submitted diagonal to a kernel on the original sample grid."""
    X, Y, padded_length, response_length = training_spectra(clean, distorted)
    dh = checked_spectrum(estimate_DH(X, Y), (X.shape[0],), "estimate_DH")
    return inverse_fft(dh, padded_length, response_length)


def apply_DH(dh, Y, lam):
    """Solve the diagonal system using the Tikhonov penalty from Q2(b)."""
    return dh.conj() * Y / (np.abs(dh)**2 + lam**2)


def reconstruct(response, distorted, output_length):
    """Reconstruct one clip from the submitted kernel with a fixed penalty."""
    if output_length < 1 or len(distorted) != output_length + len(response) - 1:
        raise ValueError("The distorted clip length must match the full convolution")
    size = fft_size(len(distorted))
    dh = forward(response, size)
    Y = forward(distorted, size)
    X = apply_DH(dh, Y, np.sqrt(0.1))
    return inverse_fft(X, size, output_length)


def read_wav(path):
    fs, data = wavfile.read(path)
    if data.ndim != 1 or data.dtype != np.int16:
        raise ValueError(f"Expected a mono 16-bit WAV: {path}")
    return fs, data.astype(np.float64) / 32768


def play_audio(samples, sample_rate):
    """Save a WAV and open it for playback in the default browser."""
    path = Path(mkdtemp(prefix="mat292-audio-")) / "preview.wav"
    wavfile.write(path, sample_rate, (32767 * np.clip(samples, -1, 1)).astype(np.int16))
    print(f"Audio preview (press Play in your browser): {path}")
    webbrowser.open(path.as_uri())


def play_sounds(library, clips, kernel, training_data, test_folder):
    """Play the selected clips in order, with their reconstructions."""
    train_names, clean, distorted, sample_rate = training_data
    if library == "train":
        names = train_names
        lengths = {name: len(x) for name, x in zip(names, clean)}
    elif library == "test":
        lengths = json.loads((test_folder / "expected_lengths.json").read_text(encoding="utf-8"))
        names = sorted(lengths)
    else:
        raise ValueError("Choose 'train' or 'test' for library")
    if not clips or any(not 0 <= i < len(names) for i in clips):
        raise ValueError(f"Choose clip indices from 0 to {len(names) - 1}")

    pause = np.zeros(sample_rate // 2)
    playlist = []
    for i in clips:
        name = names[i]
        if library == "train":
            y = distorted[i]
            playlist.extend((clean[i], pause))
            print(f"{name}: clean, distorted, recovered")
        else:
            rate, y = read_wav(test_folder / name)
            if rate != sample_rate:
                raise ValueError(f"Sample rate differs from the training library: {name}")
            print(f"{name}: distorted, recovered")
        playlist.extend((y, pause, reconstruct(kernel, y, lengths[name]), pause))
    play_audio(np.concatenate(playlist[:-1]), sample_rate)


def paired_data(folder):
    xs, ys, rates = [], [], set()
    paths = sorted((folder / "clean").glob("*.wav"))
    for path in paths:
        fs, x = read_wav(path)
        fy, y = read_wav(folder / "distorted" / path.name)
        rates.update((fs, fy))
        xs.append(x)
        ys.append(y)
    if not xs or len(rates) != 1:
        raise ValueError("Provide matched audio pairs at a common sample rate")
    return [p.name for p in paths], xs, ys, rates.pop()


def score(truth, estimate):
    estimate = np.asarray(estimate)
    if (estimate.shape != truth.shape or np.iscomplexobj(estimate) or
            not np.isfinite(estimate).all()):
        raise ValueError("Expected a finite real waveform with the clean reference shape")
    reference_norm = np.linalg.norm(truth)
    if reference_norm == 0:
        raise ValueError("Relative error requires a nonzero clean clip")
    return np.linalg.norm(estimate - truth) / reference_norm


def mean_reconstruction_error(clean, distorted, response, clean_signal):
    """Compute the mean relative reconstruction error for one library."""
    return float(np.mean([score(x, clean_signal(response, y, len(x)))
                          for x, y in zip(clean, distorted)]))


def grade(training, testing, fit, clean_signal):
    """Return training error for completeness and private test error for ranking."""
    _, train_x, train_y, train_rate = paired_data(training)
    _, test_x, test_y, test_rate = paired_data(testing)
    if train_rate != test_rate:
        raise ValueError("The training and test libraries must use the same sample rate")

    response = fit(train_x, train_y)
    return (
        mean_reconstruction_error(train_x, train_y, response, clean_signal),
        mean_reconstruction_error(test_x, test_y, response, clean_signal),
    )


def challenge_arguments():
    parser = argparse.ArgumentParser(description="Try the audio reconstruction example.")
    parser.add_argument("--training", type=Path, default=Path(__file__).parent / "training")
    return parser.parse_args()
