import numpy as np
import time as timer

def sample_uniform_ball(N: int, d: int, R: float, seed: int = 0) -> np.ndarray:
    """Uniform samples from the d-dim Euclidean ball of radius R."""
    rng = np.random.default_rng(seed)
    # Random directions
    X = rng.normal(size=(N, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Radii with correct distribution for uniform-in-ball
    r = rng.random(N) ** (1.0 / d)
    return R * X * r[:, None]

def mvee_todd_yildirim(P: np.ndarray, u: np.array = None, tol: float = 1e-6, max_iter: int = 50000):
    """
    Minimum-Volume Enclosing Ellipsoid (MVEE) of points P (N x d) via a
    Khachiyan-style algorithm with Todd–Yıldırım step size.

    Returns:
      c : (d,) center
      A : (d x d) shape matrix such that (x-c)^T A (x-c) <= 1
    (If you prefer (x-c)^T Q^{-1} (x-c) <= 1, then Q = A^{-1}.)
    """
    P = np.asarray(P, dtype=float)
    N, d = P.shape

    # Homogeneous coordinates: q_i = [p_i; 1]
    Q = np.vstack([P.T, np.ones(N)])          # (d+1) x N

    # Initial weights on simplex
    if u is None:
        u = np.ones(N) / N

    for _ in range(max_iter):
        # X(u) = sum_i u_i q_i q_i^T = Q diag(u) Q^T
        # Multiply columns of Q by u, then form X
        # X = Q @ np.diag(u) @ Q.T                      # (d+1) x (d+1)
        X = (Q * u) @ Q.T

        # Compute M_i = q_i^T X^{-1} q_i efficiently without forming X^{-1}
        XinvQ = np.linalg.solve(X, Q)          # (d+1) x N, columns are X^{-1} q_i
        M = np.sum(Q * XinvQ, axis=0)          # length N, columnwise dot products

        j = int(np.argmax(M))
        Mj = M[j]

        # Optimality condition: max_i M_i <= d+1 (up to tolerance)
        if Mj <= (d + 1) * (1.0 + tol):
            break

        # Todd–Yıldırım step size
        alpha = (Mj - (d + 1)) / ((d + 1) * (Mj - 1.0))

        # Update weights toward the worst (largest M_i) point
        u = (1.0 - alpha) * u
        u[j] += alpha

    # Center c = sum u_i p_i
    c = P.T @ u

    # Scatter S = sum u_i (p_i - c)(p_i - c)^T = P^T diag(u) P - c c^T
    S = (P.T * u) @ P - np.outer(c, c)
    S = 0.5 * (S + S.T)  # symmetrize for numerical stability

    # Shape matrix A for ellipsoid (x-c)^T A (x-c) <= 1
    # MVEE: A = (1/d) * S^{-1}
    A = np.linalg.inv(S) / d
    A = 0.5 * (A + A.T)

    return c, A, u


def safe_cov_ellipsoid(P, ridge=1e-12, safety=1.0001):
    """
    Fast safe ellipsoid enclosing all points (N x d) by:
      c = mean(P)
      S = cov(P) + ridge*I
      rho = max_i (p_i-c)^T S^{-1} (p_i-c)
      A = S^{-1} / (rho * safety)

    Returns:
      c : (d,)
      A : (d,d) so that (x-c)^T A (x-c) <= 1 encloses all points (up to safety factor).
    """
    P = np.asarray(P, dtype=float)
    N, d = P.shape

    c = P.mean(axis=0)
    X = P - c

    # Covariance (ML estimate). Any scaling works since we re-inflate with rho.
    S = (X.T @ X) / max(N, 1)

    # Ridge for numerical stability (in case points are near-degenerate)
    S = S + ridge * np.eye(d)

    # Compute rho = max Mahalanobis distance squared
    # Use solve instead of inv
    SinvX_T = np.linalg.solve(S, X.T)                # d x N
    md2 = np.sum(X.T * SinvX_T, axis=0)              # length N
    rho = float(md2.max())

    # Inflate ellipsoid to include all points (and optionally a tiny safety margin)
    A = np.linalg.solve(S, np.eye(d)) / (rho * safety)

    # Symmetrize (numerical)
    A = 0.5 * (A + A.T)
    return c, A

if __name__ == "__main__":
    
    N, d, R = 100, 4, 0.1
    P = sample_uniform_ball(N, d, R, seed=42)
    P1 = sample_uniform_ball(N, d, R, seed=42)

    tic = timer.time()
    c, A, u1 = mvee_todd_yildirim(P, tol=1e-6)
    print('time elapsed:', timer.time() - tic)

    # sanity check: all points should satisfy (x-c)^T A (x-c) <= 1 (up to small slack)
    vals = np.einsum("ni,ij,nj->n", P - c, A, P - c)
    print("max quadratic form:", vals.max())
    print("center c:", c)
    print("A:\n", A)

    tic = timer.time()
    c, A = safe_cov_ellipsoid(P)
    print("center c:", c)
    print("A:\n", A)
    print('time elapsed:', timer.time() - tic)
    vals = np.einsum("ni,ij,nj->n", P - c, A, P - c)
    print("max quadratic form:", vals.max())

    # tic = timer.time()
    # c, A, u2 = mvee_todd_yildirim(P, u = u1, tol=1e-6)
    # print('time elapsed:', timer.time() - tic)
