"""Quadrature and the Laplace transform of a rectangular pulse."""

import numpy as np


def save_figure(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def gauss_legendre(q, T=1.0):
    """Return q Gauss--Legendre nodes and weights on [0,T]."""
    # The eigenvalues give the nodes on [-1, 1].
    k = np.arange(1, q)
    beta = k/np.sqrt(4*k**2-1)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    t, V = np.linalg.eigh(J)
    # Map the nodes to [0, T] and scale the weights for that interval.
    t = T*(t+1)/2
    w = T*V[0, :]**2
    return t, w


# Q1(a): evaluate the pulse formula directly for s > 0.
def pulse_transform(s, T):
    """Evaluate the transform of a pulse of duration T.

    In this exercise, s is a 1-D NumPy array of positive transform points.
    Return one value per point, in the same order.
    """
    return np.array((1-(np.e)**(-s*T))/s)
    raise NotImplementedError()


# Q1(b): evaluate the integral by quadrature.
# t contains the time nodes and w contains their quadrature weights.
def quadrature_transform(s, t, w, f):
    """Approximate the transform at each point in s using the samples f(t_j).

    Call f with a single time node to get its value there.
    """
    res = np.zeros(len(s))
    f_tj = [f(time) for time in t]
    for j in range(len(s)):
        res[j] = np.sum(w*f_tj*np.power(np.e, (-1)*s[j]*t))
    return res
    raise NotImplementedError()


# Q1(c): compute all the quadrature sums as F = A @ f.
# Each row of A uses one s value. Each column uses one time node.
def matrix_transform(s, t, w, f):
    """Return (A, fvec, F), where fvec[j] = f(t[j]) and F = A @ fvec."""
    fvec = [f(time) for time in t]
    s_n = s.reshape(len(s), 1)
    A = (-1) * (s_n * t)
    A = np.exp(A)
    A = A * w
    F = A @ fvec
    return (A, fvec, F)
    
    raise NotImplementedError()


def plot_exercises():
    import matplotlib.pyplot as plt

    T = 1.0
    s = np.logspace(-18, -2, 401)
    t, w = gauss_legendre(16, T)

    def pulse(time):
        return 1.0 if 0 <= time <= T else 0.0

    # Q1(a,b): compare direct evaluation with quadrature near zero.
    direct = pulse_transform(s, T)
    quadrature = quadrature_transform(s, t, w, pulse)
    fig, ax = plt.subplots()
    ax.semilogx(s, direct, label="Direct expression")
    ax.semilogx(s, quadrature, "--", label="Gauss--Legendre")
    ax.axhline(T, color="grey", linestyle=":", label="Limit T")
    ax.set(xlabel="$s$", ylabel="$F(s)$", title="Pulse transform near zero")
    ax.legend()
    save_figure(fig, "q1_pulse.png")

    # A stable expression supplies a reference for this particular pulse.
    reference = -np.expm1(-s*T)/s
    fig, ax = plt.subplots()
    ax.loglog(s, abs(direct-reference)/abs(reference), label="Direct expression")
    ax.loglog(s, abs(quadrature-reference)/abs(reference),
              "--", label="Gauss--Legendre")
    ax.set(xlabel="$s$", ylabel="Relative error")
    ax.legend()
    save_figure(fig, "q1_relative_errors.png")

    # Q1(c): compare the matrix product with the quadrature loop.
    s = np.linspace(0.5, 10, 21)
    A, f, F = matrix_transform(s, t, w, pulse)
    F_quadrature = quadrature_transform(s, t, w, pulse)
    print("Q1 matrix/loop difference:", np.linalg.norm(F - F_quadrature))
    fig, ax = plt.subplots()
    ax.plot(s, F_quadrature, label="Quadrature loop")
    ax.plot(s, F, "x", label="Matrix product")
    ax.set(xlabel="$s$", ylabel="$F(s)$", title="The same quadrature in matrix form")
    ax.legend()
    save_figure(fig, "q1_matrix_transform.png")


if __name__ == "__main__":
    plot_exercises()
