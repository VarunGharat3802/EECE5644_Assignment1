import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import eigh

rng = np.random.default_rng(0)
N = 10000
P = np.array([0.35, 0.65])
m0 = np.full(4, -0.5)
C0 = 0.25 * np.array([[2, -.5, .3, 0], [-.5, 1, -.5, 0], [.3, -.5, 1, 0], [0, 0, 0, 2]])
m1 = np.ones(4)
C1 = np.array([[1, .3, -.2, 0], [.3, 2, .3, 0], [-.2, .3, 1, 0], [0, 0, 0, 3]])

# ---------- generate & save data ----------
L = (rng.random(N) >= P[0]).astype(int)
X = np.zeros((N, 4))
X[L == 0] = rng.multivariate_normal(m0, C0, (L == 0).sum())
X[L == 1] = rng.multivariate_normal(m1, C1, (L == 1).sum())
np.savez("q1_data.npz", X=X, L=L)
n0, n1 = (L == 0).sum(), (L == 1).sum()
print(f"N0={n0}, N1={n1}")


def roc(score, L):
    """Decide 1 if score > threshold. Sweeps threshold from -inf to +inf."""
    order = np.argsort(-score)
    s, Ls = score[order], L[order]
    tp = np.concatenate(([0], np.cumsum(Ls))) / (Ls == 1).sum()
    fp = np.concatenate(([0], np.cumsum(1 - Ls))) / (Ls == 0).sum()
    # threshold k: top k samples called class 1 -> threshold between s[k-1] and s[k]
    thr = np.concatenate(([np.inf], (s[:-1] + s[1:]) / 2, [-np.inf]))
    return tp[::-1], fp[::-1], thr[::-1]   # ordered from threshold -inf (1,1) to +inf (0,0)


def analyze(name, score, theo_thr=None, thr_label="log γ"):
    tp, fp, thr = roc(score, L)
    perr = fp * P[0] + (1 - tp) * P[1]
    k = np.argmin(perr)
    out = dict(tp=tp, fp=fp, k=k, perr=perr, thr=thr)
    print(f"\n=== {name} ===")
    print(f"Empirical min P(error) = {perr[k]:.4f} at {thr_label} = {thr[k]:.4f}"
          + (f"  (γ = {np.exp(thr[k]):.4f})" if thr_label == "log γ" else ""))
    print(f"   TPR = {tp[k]:.4f}, FPR = {fp[k]:.4f}")
    if theo_thr is not None:
        D = (score > theo_thr).astype(int)
        pfa = np.mean(D[L == 0] == 1); pd = np.mean(D[L == 1] == 1)
        pe = pfa * P[0] + (1 - pd) * P[1]
        out["theo"] = (pfa, pd, pe)
        print(f"Theoretical threshold {thr_label} = {theo_thr:.4f} (γ = {np.exp(theo_thr):.4f}):"
              f" P(error) = {pe:.4f}, TPR = {pd:.4f}, FPR = {pfa:.4f}")
    return out


def plot_roc(res, title, fname):
    plt.figure(figsize=(6, 5.5))
    plt.plot(res["fp"], res["tp"], lw=1.5, label="ROC")
    k = res["k"]
    plt.plot(res["fp"][k], res["tp"][k], "ro", ms=10,
             label=f"Empirical min P(err) = {res['perr'][k]:.4f}")
    if "theo" in res:
        pfa, pd, pe = res["theo"]
        plt.plot(pfa, pd, "gx", ms=12, mew=2.5, label=f"Theoretical γ, P(err) = {pe:.4f}")
    plt.xlabel("False positive  P(D=1 | L=0)")
    plt.ylabel("True positive  P(D=1 | L=1)")
    plt.title(title); plt.grid(alpha=.3); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(fname, dpi=130); plt.close()


log_gamma_theo = np.log(P[0] / P[1])   # 0-1 loss

# ---------- Part A: true pdf ----------
sA = mvn.logpdf(X, m1, C1) - mvn.logpdf(X, m0, C0)
A = analyze("Part A: true-pdf ERM", sA, log_gamma_theo)
plot_roc(A, "Part A: ERM with true pdf", "q1_partA_roc.png")

# ---------- Part B: Naive Bayes (diagonal covariances) ----------
sB = mvn.logpdf(X, m1, np.diag(np.diag(C1))) - mvn.logpdf(X, m0, np.diag(np.diag(C0)))
B = analyze("Part B: Naive Bayes", sB, log_gamma_theo)
plot_roc(B, "Part B: Naive Bayes (diagonal covariances)", "q1_partB_roc.png")

# ---------- Part C: Fisher LDA ----------
mu0h, mu1h = X[L == 0].mean(0), X[L == 1].mean(0)
S0h, S1h = np.cov(X[L == 0].T), np.cov(X[L == 1].T)
Sb = np.outer(mu1h - mu0h, mu1h - mu0h)
Sw = S0h + S1h
evals, evecs = eigh(Sb, Sw)              # generalized eigenproblem Sb w = λ Sw w
w = evecs[:, np.argmax(evals)]
if w @ (mu1h - mu0h) < 0:
    w = -w                               # class 1 projects to larger values
w = w / np.linalg.norm(w)
print("\nEstimated means:\n", mu0h.round(3), "\n", mu1h.round(3))
print("w_LDA =", w.round(4))
sC = X @ w
C = analyze("Part C: Fisher LDA", sC, thr_label="τ")
plot_roc(C, "Part C: Fisher LDA", "q1_partC_roc.png")

# ---------- comparison ----------
plt.figure(figsize=(6, 5.5))
for r, lab, c in [(A, "True pdf (ERM)", "C0"), (B, "Naive Bayes", "C1"), (C, "Fisher LDA", "C2")]:
    plt.plot(r["fp"], r["tp"], c, label=f"{lab}: min P(err)={r['perr'][r['k']]:.4f}")
    plt.plot(r["fp"][r["k"]], r["tp"][r["k"]], "o", color=c, ms=9, mec="k")
plt.xlabel("P(D=1 | L=0)"); plt.ylabel("P(D=1 | L=1)")
plt.title("ROC comparison"); plt.grid(alpha=.3); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("q1_roc_comparison.png", dpi=130); plt.close()
