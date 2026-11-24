"""Submit your kernel by implementing estimate_DH, then run this file to hear your result."""
import numpy as np

from challenge_utils import (
    TRAINING_ERROR_CUTOFF, challenge_arguments, checked_spectrum, fit_kernel, inverse_fft,
    mean_reconstruction_error, paired_data, play_sounds, reconstruct, training_spectra,
)

def estimate_DH(X, Y):
    """Return your kernel as the diagonal of D_H, with one entry per row of X.

    Column j of X represents the clean training clip and column j of Y
    represents its distorted version.
    """
    # dh = np.zeros(X.shape[0], dtype=complex)
    # for i in range(len(X)):
    #     dh[i] = np.linalg.lstsq(X[i, :, None], Y[i, :], rcond=None)[0][0]
    # return dh

    alpha = 0
    numerator = np.sum(X.conj() * Y, axis=1)
    denominator = np.sum(np.abs(X) ** 2, axis=1)
    return numerator / (denominator + alpha)
    raise NotImplementedError("Complete estimate_DH to fit the training clips.")

def fit(clean, distorted):
    return fit_kernel(clean, distorted, estimate_DH)


if __name__ == "__main__":
    args = challenge_arguments()
    # load the library and assemble X and Y matrices 
    training_data = paired_data(args.training)
    _, x, y, _ = training_data
    X, Y, padded_length, response_length = training_spectra(x, y)
    print(f"X and Y have shape {X.shape} (coordinates, training clips).")

    # estimate dh from the training library 
    dh = checked_spectrum(estimate_DH(X, Y), (X.shape[0],), "estimate_DH")
    kernel = inverse_fft(dh, padded_length, response_length)

    #  score your reconstruction based on training data 
    training_error = mean_reconstruction_error(x, y, kernel, reconstruct)
    print(f"Mean relative reconstruction error on the training library: {training_error:.4f}")
    if training_error <= TRAINING_ERROR_CUTOFF:
        print(f"Meets the training error cutoff of {TRAINING_ERROR_CUTOFF:.2f}.")
    else:
        print(f"Training error exceeds the cutoff of {TRAINING_ERROR_CUTOFF:.2f}.")

    # Choose the library and the clip indices you want to hear.
    library = "train"  # "train" or "test"
    clips = [0]
    play_sounds(library, clips, kernel, training_data, args.training.parent / "test-inputs")
