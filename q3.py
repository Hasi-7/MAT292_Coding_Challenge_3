"""Apply convolution and recover a short signal."""

import numpy as np

from q1 import save_figure
from q2 import tikhonov


# Q3(a): add reverb to the samples in x using the delay weights h.
def add_reverb(x, h):
    """Return all len(x) + len(h) - 1 samples of the full convolution.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# Q3(a): represent the same convolution as a matrix product.
def convolution_matrix(h, n):
    """Return H with shape (n + len(h) - 1, n).

    For any n-sample signal x, H @ x should match add_reverb(x, h).
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# Q3(b): use the Tikhonov solver from Q2 on the noisy recording.
def reconstruct_signal(H, ydelta, lam):
    """The recovered signal has one sample per column of H."""
    # YOUR CODE HERE
    raise NotImplementedError()


def check_matrix():
    """Compare the two forward calculations on a short example."""
    x = np.array([1.0, 2.0, -1.0, 3.0])
    h = np.array([1.0, 0.5, 0.2])
    y = add_reverb(x, h)
    print("Full convolution:", y)
    H = convolution_matrix(h, len(x))
    print("Convolution matrix:\n", H)
    print("Matrix product agrees with add_reverb:", np.allclose(H @ x, y))


def short_audio(wav_path=None):
    """Return a short mono excerpt and its sample rate."""
    sample_rate = 4000
    sample_count = 400
    if wav_path is None:
        # The demo signal contains tones at 250 Hz and 500 Hz.
        t = np.arange(sample_count) / sample_rate
        x = np.sin(2 * np.pi * 250 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
        return x * np.hanning(sample_count), sample_rate

    from math import gcd
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    original_rate, audio = wavfile.read(wav_path)
    audio = audio.astype(float)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    common = gcd(sample_rate, original_rate)
    audio = resample_poly(audio, sample_rate // common, original_rate // common)

    # Take a short excerpt near a loud sample to avoid leading silence.
    peak_index = int(np.argmax(np.abs(audio)))
    start = max(0, min(peak_index - sample_count // 2, len(audio) - sample_count))
    x = audio[start:start + sample_count]
    return x / max(np.max(np.abs(x)), 1.0), sample_rate


def simulate_recording(x, sample_rate, noise_frequency=None):
    """Return the convolution matrix, clean reverb, and scaled background noise."""
    # Add a strong, delayed copy. The two copies reinforce some frequencies
    # and nearly cancel others.
    h = np.zeros(9)
    h[0], h[-1] = 1.0, 0.95
    H = convolution_matrix(h, len(x))
    y = add_reverb(x, h)
    if noise_frequency is None:
        noise = np.random.default_rng(292).standard_normal(y.shape)
    else:
        time = np.arange(len(y)) / sample_rate
        noise = np.sin(2 * np.pi * noise_frequency * time)
    # Give each noise source the same root-mean-square amplitude.
    noise_rms = np.sqrt(np.mean(noise**2))
    return H, y, 0.03 * np.std(y) * noise / noise_rms


def explore_inverse(x, sample_rate, lam, noise_frequency=None):
    """Plot recovery with random noise or a tone at noise_frequency in Hz."""
    import matplotlib.pyplot as plt

    H, y, noise = simulate_recording(x, sample_rate, noise_frequency)
    ydelta = y + noise
    if noise_frequency is None:
        noise_label = "random noise"
        filename = "q3b_spectra.png"
    else:
        noise_label = f"{noise_frequency:g} Hz tone"
        filename = f"q3c_spectra_{noise_frequency:g}Hz.png"
    x_rec = reconstruct_signal(H, ydelta, lam)
    error = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
    print(f"Relative reconstruction error with {noise_label}: {error:.3g}")

    # Zero-pad to use the same frequency grid in all three panels. Use the
    # same vertical scale so changes in amplitude are visible.
    spectrum_length = 2048
    frequency = np.fft.rfftfreq(spectrum_length, d=1 / sample_rate)
    signals = (x, ydelta, x_rec)
    titles = ("Original", f"Reverb with {noise_label}",
              f"Reconstruction (lambda = {lam:g}, relative error = {error:.1%})")
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, sharey=True)
    for ax, signal, title in zip(axes, signals, titles):
        magnitude = np.abs(np.fft.rfft(signal, n=spectrum_length)) / len(x)
        ax.plot(frequency, magnitude)
        ax.set(title=title, ylabel="Magnitude")
    axes[1].lines[0].set_label(f"Reverb with {noise_label}")
    axes[1].plot(frequency, np.abs(np.fft.rfft(y, n=spectrum_length)) / len(x),
                 "--", label="Reverb only")
    axes[1].legend()
    axes[-1].lines[0].set_label("Reconstruction")
    axes[-1].plot(frequency, np.abs(np.fft.rfft(x, n=spectrum_length)) / len(x),
                  "--", label="Original")
    axes[-1].legend()
    axes[-1].set_xlabel("Frequency (Hz)")
    save_figure(fig, filename)


if __name__ == "__main__":
    check_matrix()
    x, sample_rate = short_audio(wav_path=None)
    # For Q3(c), run once with noise_frequency=250 and once with noise_frequency=500.
    explore_inverse(x, sample_rate, lam=0.01, noise_frequency=None)
