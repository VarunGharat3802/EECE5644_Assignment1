import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

rng = np.random.default_rng(1)
N = 10000
priors = np.array([0.3, 0.3, 0.4])

# Four Gaussians with means on a regular tetrahedron of side 2.5 (std ~ 1 => ~2.5 sigma apart)
s = 2.5
means = np.array([[0, 0, 0],
                  [s, 0, 0],
                  [s / 2, s * np.sqrt(3) / 2, 0],
                  [s / 2, s * np.sqrt(3) / 6, s * np.sqrt(2 / 3)]])
covs = np.array([np.diag([1.0, 0.8, 1.2]),
                 [[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.9]],
                 [[0.9, -0.2, 0.1], [-0.2, 1.1, 0.0], [0.1, 0.0, 1.0]],
                 np.diag([1.1, 0.9, 1.0])])
avg_std = np.mean([np.sqrt(np.trace(c) / 3) for c in covs])
print(f"mean pairwise distance = {s}, average std = {avg_std:.3f}, ratio = {s/avg_std:.2f}")

# ---------- generate data ----------
L = rng.choice([1, 2, 3], size=N, p=priors)
comp = np.where(L == 1, 0, np.where(L == 2, 1, 2 + rng.integers(0, 2, N)))  # class 3: 50/50 of comps 2,3
X = np.array([rng.multivariate_normal(means[c], covs[c]) for c in comp])
np.savez("q2_data.npz", X=X, L=L)


def class_lik(X):
    p1 = mvn.pdf(X, means[0], covs[0])
    p2 = mvn.pdf(X, means[1], covs[1])
    p3 = 0.5 * mvn.pdf(X, means[2], covs[2]) + 0.5 * mvn.pdf(X, means[3], covs[3])
    return np.vstack([p1, p2, p3])           # 3 x N


lik = class_lik(X)
post = lik * priors[:, None]
post /= post.sum(0)                          # P(L=j | x), 3 x N


def erm(Lambda):
    risk = Lambda @ post                      # R(D=i | x) = sum_j Lambda[i,j] P(L=j|x)
    D = np.argmin(risk, axis=0) + 1
    return D


def confusion(D):
    CM = np.zeros((3, 3))
    for j in range(3):
        for i in range(3):
            CM[i, j] = np.mean(D[L == j + 1] == i + 1)
    return CM                                 # CM[i,j] = P(D=i+1 | L=j+1)


def scatter(D, title, fname):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    markers = {1: ".", 2: "o", 3: "^"}
    for c in [1, 2, 3]:
        for ok, col in [(True, "green"), (False, "red")]:
            idx = (L == c) & ((D == L) == ok)
            ax.scatter(*X[idx].T, marker=markers[c], c=col, s=10 if c == 1 else 8,
                       alpha=.5, label=f"L={c}, {'correct' if ok else 'wrong'}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title); ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout(); plt.savefig(fname, dpi=130); plt.close()


Lambdas = {"0-1 loss (MAP)": 1 - np.eye(3),
           "Lambda10": np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]]),
           "Lambda100": np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])}
fnames = ["q2_partA_map.png", "q2_partB_lambda10.png", "q2_partB_lambda100.png"]

np.set_printoptions(precision=4, suppress=True)
for (name, Lam), fn in zip(Lambdas.items(), fnames):
    D = erm(Lam)
    CM = confusion(D)
    loss = Lam[D - 1, L - 1]
    print(f"\n=== {name} ===")
    print("Confusion matrix P(D=i | L=j) (rows i = decision, cols j = true label):\n", CM)
    print(f"P(error) = {np.mean(D != L):.4f}")
    print(f"Estimated minimum expected risk = {loss.mean():.4f}")
    print("Decision counts:", np.bincount(D, minlength=4)[1:])
    scatter(D, f"{name}: P(err)={np.mean(D != L):.3f}, risk={loss.mean():.3f}", fn)
