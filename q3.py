"""
Q3: Gaussian class-conditional MAP classifiers for Wine Quality (white) and UCI HAR.

Expected local files (download from UCI first):
  winequality-white.csv            (semicolon-separated, 4898 rows)
  UCI HAR Dataset/train/X_train.txt, y_train.txt
  UCI HAR Dataset/test/X_test.txt,  y_test.txt   (train+test = 10299 samples)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

ALPHA = 0.05   # regularization strength: lambda = ALPHA * trace(C) / rank(C)


def fit_gaussian_map(X, y, alpha=ALPHA):
    classes = np.unique(y)
    params = {}
    for c in classes:
        Xc = X[y == c]
        mu = Xc.mean(0)
        C = np.cov(Xc.T, bias=False) if len(Xc) > 1 else np.zeros((X.shape[1],) * 2)
        r = np.linalg.matrix_rank(C)
        lam = alpha * np.trace(C) / max(r, 1)
        lam = max(lam, 1e-6)
        Creg = C + lam * np.eye(X.shape[1])
        params[c] = dict(mu=mu, C=Creg, prior=len(Xc) / len(X), lam=lam, n=len(Xc),
                         cond=np.linalg.cond(C), cond_reg=np.linalg.cond(Creg))
    return params


def predict(X, params):
    classes = np.array(list(params))
    # log p(x|c) + log P(c); MAP decision
    g = np.vstack([mvn.logpdf(X, p["mu"], p["C"]) + np.log(p["prior"]) for p in params.values()])
    return classes[np.argmax(g, axis=0)]


def confusion(y, D, classes):
    CM = np.zeros((len(classes), len(classes)))
    for j, cj in enumerate(classes):
        for i, ci in enumerate(classes):
            CM[i, j] = np.mean(D[y == cj] == ci)
    return CM   # CM[i,j] = P(D=i | L=j)


def pca_plots(X, y, name):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    Z = Xs @ Vt[:3].T
    ev = S ** 2 / np.sum(S ** 2)
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(121)
    for c in np.unique(y):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], s=4, alpha=.5, label=str(c))
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    ax.legend(markerscale=4, fontsize=8); ax.set_title(f"{name}: first 2 PCs")
    ax = fig.add_subplot(122, projection="3d")
    for c in np.unique(y):
        ax.scatter(*Z[y == c].T, s=3, alpha=.4, label=str(c))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(f"{name}: first 3 PCs")
    plt.tight_layout(); plt.savefig(f"q3_{name}_pca.png", dpi=120); plt.close()


def alpha_sweep(X, y, name, alphas=(0.001, 0.01, 0.05, 0.1, 0.3)):
    print(f"--- alpha sweep ({name}) ---")
    for a in alphas:
        D = predict(X, fit_gaussian_map(X, y, a))
        print(f"  alpha = {a:<6} P(error) = {np.mean(D != y):.4f}")


def run(X, y, name, feat_names=None, standardize=False, alpha=ALPHA):
    print(f"\n================ {name} ================")
    u, cnt = np.unique(y, return_counts=True)
    print(f"N = {len(y)}, d = {X.shape[1]}, class counts:", {int(a): int(b) for a, b in zip(u, cnt)})
    print(f"Majority-class baseline P(error) = {1 - cnt.max() / len(y):.4f}")
    Xraw = X
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        print("Features standardized (zero mean, unit variance)")
    params = fit_gaussian_map(X, y, alpha)
    for c, p in params.items():
        print(f"  class {c}: n={p['n']:5d} prior={p['prior']:.4f} lambda={p['lam']:.4g} "
              f"cond(C)={p['cond']:.3g} -> cond(Creg)={p['cond_reg']:.3g}")
    D = predict(X, params)
    classes = np.array(list(params))
    CM = confusion(y, D, classes)
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    print(f"P(error) estimate = {np.mean(D != y):.4f}  ({np.sum(D != y)} errors)")
    print("Confusion matrix P(D=i|L=j), rows=decision, cols=true label, classes =", classes)
    print(CM)
    alpha_sweep(X, y, name)
    pca_plots(Xraw, y, name)
    X = Xraw
    # Feature-subset view: 3 features with the largest between-class mean spread
    mus = np.vstack([p["mu"] for p in params.values()])
    spread = mus.std(0) / (X.std(0) + 1e-12)
    top = np.argsort(-spread)[:3]
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    for c in classes:
        ax.scatter(*X[y == c][:, top].T, s=3, alpha=.4, label=str(c))
    lab = [feat_names[t] if feat_names is not None else f"f{t}" for t in top]
    ax.set_xlabel(lab[0]); ax.set_ylabel(lab[1]); ax.set_zlabel(lab[2])
    ax.legend(markerscale=4, fontsize=8); ax.set_title(f"{name}: 3 most class-separating raw features")
    plt.tight_layout(); plt.savefig(f"q3_{name}_features.png", dpi=120); plt.close()
    return CM, np.mean(D != y)


if __name__ == "__main__":
    # ---------- Wine (white) ----------
    if os.path.exists("winequality-white.csv"):
        df = pd.read_csv("winequality-white.csv", sep=";")
        Xw = df.drop(columns="quality").values.astype(float)
        yw = df["quality"].values
        run(Xw, yw, "wine_raw", list(df.columns[:-1]), standardize=False)
        run(Xw, yw, "wine_std", list(df.columns[:-1]), standardize=True)
    else:
        print("winequality-white.csv not found")

    # ---------- HAR ----------
    base = "UCI HAR Dataset"
    if os.path.isdir(base):
        Xh = np.vstack([np.loadtxt(f"{base}/train/X_train.txt"), np.loadtxt(f"{base}/test/X_test.txt")])
        yh = np.concatenate([np.loadtxt(f"{base}/train/y_train.txt"),
                             np.loadtxt(f"{base}/test/y_test.txt")]).astype(int)
        run(Xh, yh, "har")
    else:
        print("UCI HAR Dataset folder not found")
