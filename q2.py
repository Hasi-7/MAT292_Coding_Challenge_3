"""Inverse Laplace transform and Tikhonov regularization."""

import numpy as np
from q1 import gauss_legendre, matrix_transform, save_figure


# Q2(a): recover the pulse samples from the transform data F.
def inverse_transform(A, F):
    """Find the least-squares fit to F by solving A @ f_rec = F using np.linalg.solve.
    """
    return np.linalg.solve(A, F)
    raise NotImplementedError()


# Q2(b): build and solve the augmented system from the handout.
def tikhonov(A, F, lam):
    """Construct the augmented system and solve it with np.linalg.lstsq.
    """
    n = A.shape[1]
    B = np.vstack([A, lam*np.identity(n)])
    b = np.concatenate([F, np.zeros(n)])

    return np . linalg . lstsq (B , b , rcond = None ) [0]
    raise NotImplementedError()


# Q2(c): reconstruct from the noisy data Fdelta and compare with the clean samples f.
def reconstruction_errors(A, Fdelta, f, lambdas):
    """For each lam, compute f_rec = tikhonov(A, Fdelta, lam).

    The relative error is np.linalg.norm(f_rec - f) / np.linalg.norm(f).
    Return the errors in a 1-D array in the same order as lambdas.
    """
    errors = np.zeros(len(lambdas))
    for i in range(len(lambdas)):
        f_rec = tikhonov(A, Fdelta, lambdas[i])
        errors[i] = np.linalg.norm(f_rec - f) / np.linalg.norm(f)

    return errors
    raise NotImplementedError()


def pulse_problem(n):
    """Set up the unit pulse with n quadrature nodes and n transform samples."""
    t, w = gauss_legendre(n)
    s = np.linspace(0.5, 10, n)
    # All quadrature nodes lie inside the unit pulse.
    A, f, F = matrix_transform(s, t, w, lambda time: 1.0)
    return t, A, f, F


def plot_direct_inversion():
    import matplotlib.pyplot as plt

    # Q2(a): increase the matrix size without adding noise.
    sizes = [6, 8, 10, 12, 14, 16]
    errors = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n in sizes:
        t, A, f, F = pulse_problem(n)
        try:
            frec = inverse_transform(A, F)
        except np.linalg.LinAlgError:
            print(f"n={n}: the matrix is numerically singular.")
            errors.append(np.nan)
            continue
        error = np.linalg.norm(frec-f)/np.linalg.norm(f)
        errors.append(error)
        print(f"n={n}: roundoff reconstruction error {error:.3e}")
        axes[1].plot(t, frec, "o-", label=f"n={n}")
    axes[0].semilogy(sizes, errors, "o-")
    axes[0].set(xlabel="Number of nodes", ylabel="Relative reconstruction error")
    axes[1].axhline(1, color="black", linestyle="--", label="True pulse")
    axes[1].set(xlabel="$t$", ylabel="Recovered function value")
    axes[1].legend(fontsize=8)
    save_figure(fig, "q2a_roundoff.png")


def plot_regularization(t, A, f, F, lambdas):
    import matplotlib.pyplot as plt

    # Q2(b): use the original transform data F for every lambda.
    errors = []
    for lam in lambdas:
        frec = tikhonov(A, F, lam)
        errors.append(np.linalg.norm(frec-f)/np.linalg.norm(f))
    errors = np.array(errors)
    best = np.argmin(errors)
    print(f"Best clean regularized error: {errors[best]:.3e}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(lambdas, errors, label="Tikhonov")
    try:
        direct = inverse_transform(A, F)
        direct_error = np.linalg.norm(direct-f)/np.linalg.norm(f)
        axes[0].axhline(direct_error, color="grey", linestyle="--",
                        label="Direct inverse")
    except np.linalg.LinAlgError:
        pass  # The failure is reported in part (a).
    axes[0].set(xlabel="$\\lambda$", ylabel="Relative reconstruction error")
    axes[0].legend()
    axes[1].plot(t, f, "k--", label="True pulse")
    axes[1].plot(t, tikhonov(A, F, lambdas[best]), "o-", label="Best in sweep")
    for index in (0, len(lambdas) - 1):
        if index != best:
            lam = lambdas[index]
            axes[1].plot(t, tikhonov(A, F, lam), label=f"λ = {lam:.1e}")
    axes[1].set(xlabel="$t$", ylabel="Recovered function value")
    axes[1].legend()
    save_figure(fig, "q2b_regularization.png")


def plot_noisy_recovery(t, A, f, F, lambdas):
    import matplotlib.pyplot as plt

    # Q2(c): scale the same random noise vector to each noise level.
    rng = np.random.default_rng(292)
    z = rng.standard_normal(len(F))
    z = z/np.linalg.norm(z)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[1].plot(t, f, "k--", label="True pulse")
    for eta in [1e-6, 1e-4, 1e-2]:
        Fdelta = F + eta*np.linalg.norm(F)*z
        errors = reconstruction_errors(A, Fdelta, f, lambdas)
        best = np.argmin(errors)
        print(f"Noise {eta:.0e}: best regularized error {errors[best]:.3e}")
        try:
            direct = inverse_transform(A, Fdelta)
            direct_error = np.linalg.norm(direct-f)/np.linalg.norm(f)
            print(f"             direct inverse error {direct_error:.3e}")
        except np.linalg.LinAlgError:
            print("             direct inverse is numerically singular.")
        line, = axes[0].loglog(lambdas, errors, label=f"Noise {eta:.0e}")
        axes[0].plot(lambdas[best], errors[best], "o", color=line.get_color())
        axes[1].plot(t, tikhonov(A, Fdelta, lambdas[best]), "o-",
                     label=f"Noise {eta:.0e}")
    axes[0].set(xlabel="$\\lambda$", ylabel="Relative reconstruction error")
    axes[1].set(xlabel="$t$", ylabel="Recovered function value")
    for ax in axes:
        ax.legend()
    save_figure(fig, "q2c_noise.png")
    print("Best lambdas above use the known pulse for comparison only.")


if __name__ == "__main__":
    plot_direct_inversion()
    t, A, f, F = pulse_problem(16)
    # Use the Frobenius norm to put the sweep on the scale of A.
    lambdas = np.linalg.norm(A) * np.logspace(-13, 0, 105)
    plot_regularization(t, A, f, F, lambdas)
    plot_noisy_recovery(t, A, f, F, lambdas)
