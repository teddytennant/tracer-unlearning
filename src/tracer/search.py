"""Erased-concept search: spectral footprint, whitening, coverage, confidence."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def spectral_reshape(M, rho, eps0=1e-3):
    """Reshape a Gram matrix by shrinking small eigenvalues.

    Eigenvalues with lambda/lambda_max >= eps0 are replaced by
    (lambda/lambda_max)**rho. The rest are dropped. The matrix is rebuilt
    from that spectrum.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    M = 0.5 * (M + M.T)
    evals, evecs = jnp.linalg.eigh(M)
    lam_max = jnp.max(evals)
    lam_max_safe = jnp.maximum(lam_max, jnp.float64(1e-30))
    ratio = evals / lam_max_safe
    keep = (lam_max > 0) & (ratio >= eps0)
    safe_ratio = jnp.where(keep, ratio, jnp.float64(1.0))
    # Eq. (1) replaces the kept eigenvalues with (lambda/lambda_max)**rho.
    scaled = jnp.where(keep, safe_ratio ** rho, jnp.float64(0.0))
    scaled = jnp.where(lam_max > 0, scaled, jnp.float64(0.0))
    return (evecs * scaled) @ evecs.T


def whiten(X, gamma, eps_frac=1e-3):
    """Whiten rows of X with Sigma**(-gamma/2) and unit-normalize them.

    Sigma = X.T @ X / n + eps_frac * mean(||x||^2) / d * I.
    Eigenvalues are clipped to at least 1e-12 before the inverse power.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    n, d = X.shape
    mean_norm2 = jnp.mean(jnp.sum(X * X, axis=1))
    eye = jnp.eye(d, dtype=jnp.float64)
    sigma = (X.T @ X) / jnp.float64(n) + (eps_frac * mean_norm2 / jnp.float64(d)) * eye
    sigma = 0.5 * (sigma + sigma.T)
    evals, evecs = jnp.linalg.eigh(sigma)
    evals = jnp.clip(evals, 1e-12, None)
    inv_power = evals ** (-0.5 * jnp.float64(gamma))
    transform = (evecs * inv_power) @ evecs.T
    Y = X @ transform
    rownorm = jnp.linalg.norm(Y, axis=1, keepdims=True)
    rownorm = jnp.maximum(rownorm, jnp.float64(1e-12))
    return Y / rownorm


def coverage(G, X, indices):
    """Coverage phi = tr(P G P) / tr(G) for the span of the indexed rows.

    Empty indices return 0. The span basis comes from an SVD, so dependent
    columns do not go through a Gram inverse.
    """
    G = jnp.asarray(G, dtype=jnp.float64)
    X = jnp.asarray(X, dtype=jnp.float64)
    G = 0.5 * (G + G.T)
    if indices is None or len(list(indices)) == 0:
        return jnp.float64(0.0)
    tr_g = jnp.trace(G)
    if float(jnp.abs(tr_g)) <= 1e-18:
        return jnp.float64(0.0)
    idx = jnp.asarray(list(indices), dtype=jnp.int32)
    rows = X[idx]
    basis = jnp.swapaxes(rows, 0, 1)
    U, S, _vt = jnp.linalg.svd(basis, full_matrices=False)
    if int(S.shape[0]) == 0:
        return jnp.float64(0.0)
    thresh = jnp.maximum(jnp.float64(1e-8) * S[0], jnp.float64(1e-10))
    keep = S >= thresh
    quad_each = jnp.sum(U * (G @ U), axis=0)
    quad = jnp.sum(jnp.where(keep, quad_each, jnp.float64(0.0)))
    return quad / tr_g


def confidence(scores, kappa=0.25, eps=1e-8):
    """(max - median) / (quantile_{1-kappa} - median + eps). Constant field is 0.

    kappa=0.25 is the paper value.
    """
    s = jnp.asarray(scores, dtype=jnp.float64).reshape(-1)
    if int(s.shape[0]) == 0:
        return jnp.float64(0.0)
    if float(jnp.max(s) - jnp.min(s)) <= 0.0:
        return jnp.float64(0.0)
    mx = jnp.max(s)
    med = jnp.median(s)
    q = jnp.quantile(s, jnp.float64(1.0) - jnp.float64(kappa))
    return (mx - med) / (q - med + jnp.float64(eps))


def identify(imprints, encodings, token_sets, groups, grid, J=6, B=3, alpha=0.5, kappa=0.25):
    """Calibrate (beta, gamma, rho), then pursue concepts and estimate how many.

    B=3 and alpha=0.5 are used for this short horizon.
    Imprints are (DeltaW, group_id), or (DeltaW, group_id, ratio). A ratio at or
    below 1e-8 is dropped; otherwise ||DeltaW||_F <= 1e-8 is dropped. Norms used
    as weights are floored at 1e-12. Only selected vectors enter the span.
    """
    groups = [g for g in groups]
    tokens = [frozenset(t) for t in token_sets]
    n_cand = len(tokens)
    kept = []
    for item in imprints:
        delta = jnp.asarray(item[0], dtype=jnp.float64)
        gid = item[1]
        if len(item) >= 3:
            if float(item[2]) <= 1e-8:
                continue
        elif float(jnp.linalg.norm(delta, "fro")) <= 1e-8:
            continue
        kept.append((delta, gid))
    if not kept:
        return [], 0

    def build_grams(beta, rho):
        grams = {}
        for g in groups:
            d = int(jnp.asarray(encodings[g]).shape[1])
            gram = jnp.zeros((d, d), dtype=jnp.float64)
            for delta, gid in kept:
                if gid != g:
                    continue
                nrm = jnp.maximum(jnp.linalg.norm(delta, "fro"), jnp.float64(1e-12))
                weight = nrm ** (-2.0 * jnp.float64(beta))
                foot = spectral_reshape(delta.T @ delta, rho, 1e-3)
                gram = gram + weight * foot
            grams[g] = 0.5 * (gram + gram.T)
        return grams

    def whiten_groups(gamma):
        return {
            g: whiten(jnp.asarray(encodings[g], dtype=jnp.float64), gamma)
            for g in groups
        }

    best_obj = None
    frozen = None
    for theta in grid:
        beta, gamma, rho = (float(theta[0]), float(theta[1]), float(theta[2]))
        grams = build_grams(beta, rho)
        whitened = whiten_groups(gamma)
        fields = {}
        confs = {}
        for g in groups:
            raw = [coverage(grams[g], whitened[g], [i]) for i in range(n_cand)]
            fields[g] = jnp.asarray(raw, dtype=jnp.float64)
            confs[g] = float(confidence(fields[g], kappa=kappa))
        raw_w = {g: max(confs[g], 1e-8) for g in groups}
        znorm = sum(raw_w.values())
        weights = {g: raw_w[g] / znorm for g in groups}
        combined = jnp.zeros((n_cand,), dtype=jnp.float64)
        for g in groups:
            combined = combined + jnp.float64(weights[g]) * fields[g]
        obj = float(confidence(combined, kappa=kappa))
        if best_obj is None or obj > best_obj:
            best_obj = obj
            frozen = (grams, whitened, weights)

    grams, whitened, weights = frozen
    selected = []
    excluded = set()

    def active():
        chosen = set(selected)
        return [i for i in range(n_cand) if i not in excluded and i not in chosen]

    def score_field(cands):
        if not cands:
            return []
        per_group = {}
        for g in groups:
            base = float(coverage(grams[g], whitened[g], selected))
            gains = []
            for i in cands:
                gained = float(coverage(grams[g], whitened[g], selected + [i])) - base
                gains.append(gained)
            arr = jnp.asarray(gains, dtype=jnp.float64)
            mu = float(jnp.mean(arr))
            sd = float(jnp.std(arr))
            if sd < 1e-12:
                per_group[g] = [0.0] * len(cands)
            else:
                per_group[g] = [(float(v) - mu) / sd for v in arr]
        scored = []
        for j in range(len(cands)):
            total = 0.0
            for g in groups:
                total += weights[g] * per_group[g][j]
            scored.append(total)
        return scored

    def field_confidence(cands):
        if not cands:
            return 0.0
        return float(confidence(jnp.asarray(score_field(cands), dtype=jnp.float64), kappa=kappa))

    a = [field_confidence(active())]
    for _step in range(int(J)):
        cands = active()
        if not cands:
            break
        scored = score_field(cands)
        best_i = cands[0]
        best_s = scored[0]
        for i, s in zip(cands, scored):
            if s > best_s or (s == best_s and i < best_i):
                best_s = s
                best_i = i
        selected.append(int(best_i))
        sel_tok = tokens[best_i]
        for j in range(n_cand):
            tj = tokens[j]
            if tj.issubset(sel_tok) or sel_tok.issubset(tj):
                excluded.add(j)
        a.append(field_confidence(active()))

    if len(a) < 2:
        return [], 0
    r = []
    for i in range(1, len(a)):
        prev = max(a[i - 1], 1e-8)
        cur = max(a[i], 1e-8)
        r.append(float(jnp.log(prev) - jnp.log(cur)))
    r_max = max(r)
    # B=3 and alpha=0.5 are used for this short horizon.
    rises = 0
    if len(r) >= B:
        rises = sum(1 for i in range(B) if a[i + 1] > a[i])
    if rises >= (B - 1) and r_max > 0:
        n_hat = next(i for i, ri in enumerate(r, start=1) if ri >= alpha * r_max)
    else:
        n_hat = r.index(r_max) + 1
    n_hat = int(min(n_hat, len(selected)))
    return [int(i) for i in selected[:n_hat]], n_hat
